from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from clawdb.lineage import (
    DM_MIRROR_PUBLIC_PROJECTION_KIND,
    GROUP_PUBLIC_PROJECTION_KIND,
    RAW_PROJECTION_KIND,
)
from clawdb.models import MessageDeleteRequest, MessageEditRequest, MessageIn, SearchRequest, WalRecord


async def _ingest_group_message(service):
    return await service.ingest_message(
        MessageIn(
            tenant_id="default",
            session_id="legacy-group-session",
            role="user",
            content="hello group world",
            channel="feishu",
            platform="feishu",
            chat_type="group",
            account_id="botacct",
            sender_id="ou_user_1",
            from_id="ou_user_1",
            group_id="oc_group_1",
            platform_message_id="om_1001",
            message_id="req-msg-1",
        )
    )


@pytest.mark.asyncio
async def test_group_message_materializes_raw_and_dm_mirror(service):
    ack = await _ingest_group_message(service)

    rows = service.df_store.state.messages_df
    rows = rows[rows["origin_message_id"].astype(str) == str(ack.origin_message_id)]

    assert set(rows["projection_kind"].astype(str)) == {
        RAW_PROJECTION_KIND,
        GROUP_PUBLIC_PROJECTION_KIND,
        DM_MIRROR_PUBLIC_PROJECTION_KIND,
    }

    group_row = rows[rows["projection_kind"].astype(str) == GROUP_PUBLIC_PROJECTION_KIND].iloc[0]
    dm_row = rows[rows["projection_kind"].astype(str) == DM_MIRROR_PUBLIC_PROJECTION_KIND].iloc[0]

    assert str(group_row["projection_scope"]) == "group:feishu_account:botacct:feishu_chat:oc_group_1"
    assert str(dm_row["projection_scope"]) == "dm:feishu_account:botacct:feishu_user:ou_user_1"
    assert str(group_row["visibility"]) == "public"
    assert str(dm_row["visibility"]) == "public"

    text, canonical_path = await service.df_store.virtual_memory_file("memory/legacy-group-session.md")
    assert "hello group world" in text
    assert canonical_path == "memory/group:feishu_account:botacct:feishu_chat:oc_group_1.md"

    dm_text, dm_path = await service.df_store.virtual_memory_file(
        "memory/dm:feishu_account:botacct:feishu_user:ou_user_1.md"
    )
    assert "hello group world" in dm_text
    assert dm_path == "memory/dm:feishu_account:botacct:feishu_user:ou_user_1.md"


@pytest.mark.asyncio
async def test_group_edit_updates_mirror_capsules_and_invalidates_caches(service):
    ack = await _ingest_group_message(service)

    await service.present_capsule_cards("default", "legacy-group-session")
    await service.present_capsule_cards("default", "dm:feishu_account:botacct:feishu_user:ou_user_1")

    service._search_cache["stale"] = [{"x": 1}]
    service._embedding_cache["stale"] = [1.0]

    await service.edit_message(
        MessageEditRequest(
            tenant_id="default",
            origin_message_id=ack.origin_message_id,
            content="edited group body",
        )
    )

    assert service._search_cache == {}
    assert service._embedding_cache == {}

    rows = service.df_store.state.messages_df
    rows = rows[rows["origin_message_id"].astype(str) == str(ack.origin_message_id)]
    assert set(rows["message_state"].astype(str)) == {"edited"}
    assert set(rows["content"].astype(str)) == {"edited group body"}

    group_cards = await service.present_capsule_cards("default", "legacy-group-session")
    dm_cards = await service.present_capsule_cards(
        "default", "dm:feishu_account:botacct:feishu_user:ou_user_1"
    )

    assert "edited group body" in group_cards[0]["summary"]
    assert "edited group body" in dm_cards[0]["summary"]

    search = await service.search(
        SearchRequest(query="edited", tenant_id="default", session_id="legacy-group-session")
    )
    assert any(item.origin_message_id == ack.origin_message_id for item in search.results)


@pytest.mark.asyncio
async def test_group_delete_tombstones_and_removes_visible_projection_views(service):
    ack = await _ingest_group_message(service)

    await service.delete_message(
        MessageDeleteRequest(tenant_id="default", origin_message_id=ack.origin_message_id)
    )

    rows = service.df_store.state.messages_df
    rows = rows[rows["origin_message_id"].astype(str) == str(ack.origin_message_id)]
    assert set(rows["message_state"].astype(str)) == {"deleted"}
    assert set(rows["content"].astype(str)) == {""}
    assert rows["deleted_at"].apply(lambda item: pd.notna(item)).all()

    search = await service.search(
        SearchRequest(query="hello", tenant_id="default", session_id="legacy-group-session")
    )
    assert search.results == []
    assert await service.present_capsule_cards("default", "legacy-group-session") == []

    with pytest.raises(FileNotFoundError):
        await service.df_store.virtual_memory_file("memory/legacy-group-session.md")


@pytest.mark.asyncio
async def test_feishu_identity_prefix_mismatch_is_rejected(service):
    with pytest.raises(ValueError):
        await service.ingest_message(
            MessageIn(
                tenant_id="default",
                session_id="bad-feishu",
                role="user",
                content="bad ids",
                channel="feishu",
                platform="feishu",
                chat_type="group",
                account_id="botacct",
                sender_id="oc_not_a_user",
                group_id="oc_group_1",
            )
        )


@pytest.mark.asyncio
async def test_legacy_wal_message_upsert_backfills_raw_and_projection_rows(service):
    record = WalRecord(
        seq=1,
        ts=datetime.now(timezone.utc),
        event_type="message_upsert",
        payload={
            "tenant_id": "default",
            "session_id": "legacy-direct",
            "message_id": "legacy-msg-1",
            "role": "user",
            "content": "legacy hello",
            "ts": datetime.now(timezone.utc).isoformat(),
            "chat_type": "direct",
            "channel": "feishu",
            "platform": "feishu",
            "account_id": "botacct",
            "sender_id": "ou_user_2",
            "to_id": "ou_user_2",
        },
        checksum=0,
    )

    await service.df_store.apply_wal_record(record)

    rows = service.df_store.state.messages_df
    rows = rows[rows["origin_message_id"].astype(str) == "legacy-msg-1"]
    assert set(rows["projection_kind"].astype(str)) == {RAW_PROJECTION_KIND, "private_dm"}
    assert rows.shape[0] == 2
