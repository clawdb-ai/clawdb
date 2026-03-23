from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from clawdb.config import ClawDBConfig
from clawdb.lineage import (
    DM_MIRROR_PUBLIC_PROJECTION_KIND,
    GROUP_PUBLIC_PROJECTION_KIND,
    RAW_PROJECTION_KIND,
)
from clawdb.models import MessageEditRequest, MessageIn, SearchRequest
from clawdb.service import ClawDBService


def _storage_config(tmp_path: Path) -> ClawDBConfig:
    data_root = tmp_path / "data"
    return ClawDBConfig(
        data_root=data_root,
        wal_dir=data_root / "wal",
        parquet_dir=data_root / "parquet",
        checkpoints_dir=data_root / "checkpoints",
        metadata_parquet_path=data_root / "checkpoints" / "metadata.parquet",
        wal_sync_policy="always",
        wal_sync_interval_ms=25,
        queue_backend="inmemory",
        queue_topic="clawdb.test.events",
        queue_zeromq_endpoint="inproc://clawdb-storage-test",
        queue_consumer_count=1,
        ingest_backpressure_lag_threshold=1000,
        ingest_backpressure_max_wait_ms=25,
        ingest_backpressure_poll_interval_ms=1,
        idempotency_dedupe_enabled=True,
        topic_auto_classify_enabled=False,
        topic_gep_dim=32,
        topic_gep_concentration=0.8,
        topic_gep_sigma2=0.7,
        topic_gep_prior_sigma2=1.2,
        openclaw_require_signature=False,
        flush_interval_seconds=60,
        lock_timeout_seconds=1.0,
        lock_watchdog_seconds=10.0,
        cache_hit_ratio_alert_threshold=0.8,
        search_log_enabled=False,
    )


async def _build_service(tmp_path: Path) -> ClawDBService:
    service = ClawDBService(config=_storage_config(tmp_path))
    await service.startup()
    return service


async def _ingest_group_message(service: ClawDBService, *, content: str):
    return await service.ingest_message(
        MessageIn(
            tenant_id="default",
            session_id="legacy-group-session",
            role="user",
            content=content,
            channel="feishu",
            platform="feishu",
            chat_type="group",
            account_id="botacct",
            sender_id="ou_user_1",
            from_id="ou_user_1",
            group_id="oc_group_1",
            platform_message_id="om_storage_1001",
            message_id="req-storage-1",
            topic_id="storage_topic",
        )
    )


@pytest.mark.asyncio
async def test_rebuild_indexes_restores_projection_and_derived_layers_from_authoritative_raw(tmp_path: Path):
    service = await _build_service(tmp_path)
    try:
        ack = await _ingest_group_message(service, content="storage rebuild original body")
        await service.edit_message(
            MessageEditRequest(
                tenant_id="default",
                origin_message_id=ack.origin_message_id,
                content="storage rebuild edited body",
            )
        )

        state = service.df_store.state
        raw_only = state.messages_df[
            state.messages_df["projection_kind"].astype(str) == RAW_PROJECTION_KIND
        ].reset_index(drop=True)
        service.df_store.state.messages_df = raw_only
        service.df_store.state.session_rollups_df = state.session_rollups_df.iloc[0:0].copy()
        service.df_store.state.topics_df = state.topics_df.iloc[0:0].copy()
        service.df_store.state.capsules_df = state.capsules_df.iloc[0:0].copy()
        service.df_store.state.sessions_df = state.sessions_df.iloc[0:0].copy()
        service.df_store._invalidate_all_indexes_locked()

        rebuild = await service.rebuild_indexes()

        rows = service.df_store.state.messages_df
        rows = rows[rows["origin_message_id"].astype(str) == str(ack.origin_message_id)]
        assert set(rows["projection_kind"].astype(str)) == {
            RAW_PROJECTION_KIND,
            GROUP_PUBLIC_PROJECTION_KIND,
            DM_MIRROR_PUBLIC_PROJECTION_KIND,
        }
        raw_row = rows[rows["projection_kind"].astype(str) == RAW_PROJECTION_KIND].iloc[0]
        assert str(raw_row["native_session_id"]) == "legacy-group-session"
        assert set(rows["content"].astype(str)) == {"storage rebuild edited body"}

        assert rebuild.authoritative_raw_messages == 1
        assert rebuild.rebuilt_projection_messages == 2
        assert rebuild.rebuilt_session_rollups > 0
        assert rebuild.rebuilt_topics == 1
        assert rebuild.rebuilt_capsules == 1
        assert rebuild.rebuilt_messages == 3

        text, canonical_path = await service.df_store.virtual_memory_file("memory/legacy-group-session.md")
        assert "storage rebuild edited body" in text
        assert canonical_path == "memory/group:feishu_account:botacct:feishu_chat:oc_group_1.md"

        cards = await service.present_capsule_cards("default", "legacy-group-session")
        assert len(cards) == 1
        assert "storage rebuild edited body" in cards[0]["summary"]

        search = await service.search(
            SearchRequest(
                query="edited body",
                tenant_id="default",
                session_id="legacy-group-session",
            )
        )
        assert any(item.origin_message_id == ack.origin_message_id for item in search.results)
    finally:
        await service.shutdown()


@pytest.mark.asyncio
async def test_startup_recovers_from_raw_only_parquet_and_flush_compacts_partitions(tmp_path: Path):
    service = await _build_service(tmp_path)
    config = service.config
    try:
        ack = await _ingest_group_message(service, content="raw only rebuild startup body")
        await service.flush_now()
        await service.flush_now()

        message_files = sorted(config.parquet_dir.glob("messages/dt=*/part-*.parquet"))
        rollup_files = sorted(config.parquet_dir.glob("session_rollups/dt=*/part-*.parquet"))
        topic_files = sorted(config.parquet_dir.glob("topics/dt=*/part-*.parquet"))
        capsule_files = sorted(config.parquet_dir.glob("capsules/dt=*/part-*.parquet"))
        assert len(message_files) == 1
        assert len(rollup_files) == 1
        assert len(topic_files) == 1
        assert len(capsule_files) == 1
    finally:
        await service.shutdown()

    persisted_messages = pd.concat(
        [pd.read_parquet(file_path) for file_path in config.parquet_dir.glob("messages/dt=*/part-*.parquet")],
        ignore_index=True,
    )
    raw_only = persisted_messages[
        persisted_messages["projection_kind"].astype(str) == RAW_PROJECTION_KIND
    ].reset_index(drop=True)
    shutil.rmtree(config.parquet_dir / "messages")
    raw_only["dt"] = pd.to_datetime(raw_only["ts"], utc=True).dt.strftime("%Y-%m-%d")
    for dt, part in raw_only.groupby("dt"):
        target = config.parquet_dir / "messages" / f"dt={dt}"
        target.mkdir(parents=True, exist_ok=True)
        part.drop(columns=["dt"]).to_parquet(target / "part-raw-only.parquet", index=False)
    for table in ["session_rollups", "topics", "capsules", "sessions"]:
        shutil.rmtree(config.parquet_dir / table)

    fresh = ClawDBService(config=config)
    await fresh.startup()
    try:
        rows = fresh.df_store.state.messages_df
        rows = rows[rows["origin_message_id"].astype(str) == str(ack.origin_message_id)]
        assert set(rows["projection_kind"].astype(str)) == {
            RAW_PROJECTION_KIND,
            GROUP_PUBLIC_PROJECTION_KIND,
            DM_MIRROR_PUBLIC_PROJECTION_KIND,
        }
        assert str(
            rows[rows["projection_kind"].astype(str) == RAW_PROJECTION_KIND].iloc[0]["native_session_id"]
        ) == "legacy-group-session"
        assert not fresh.df_store.state.session_rollups_df.empty
        assert not fresh.df_store.state.topics_df.empty
        assert not fresh.df_store.state.capsules_df.empty

        text, canonical_path = await fresh.df_store.virtual_memory_file("memory/legacy-group-session.md")
        assert "raw only rebuild startup body" in text
        assert canonical_path == "memory/group:feishu_account:botacct:feishu_chat:oc_group_1.md"
    finally:
        await fresh.shutdown()
