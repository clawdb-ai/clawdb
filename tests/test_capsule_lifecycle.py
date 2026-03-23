from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from clawdb.config import ClawDBConfig
from clawdb.models import MessageEditRequest, MessageIn
from clawdb.service import ClawDBService


def _test_config(tmp_path: Path) -> ClawDBConfig:
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
        queue_zeromq_endpoint="inproc://clawdb-test",
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


def _body(char: str, length: int) -> str:
    return str(char) * int(length)


async def _ingest_capsule_message(
    service: ClawDBService,
    *,
    session_id: str,
    topic_id: str,
    message_id: str,
    content: str,
    minute_offset: int,
):
    base_ts = datetime(2026, 3, 21, tzinfo=timezone.utc)
    return await service.ingest_message(
        MessageIn(
            tenant_id="default",
            session_id=session_id,
            role="user",
            content=content,
            topic_id=topic_id,
            message_id=message_id,
            ts=base_ts + timedelta(minutes=minute_offset),
        )
    )


def _capsule_rows(service: ClawDBService):
    return service.df_store.state.capsules_df.sort_values(
        ["tenant_id", "topic_id", "capsule_ordinal"],
        kind="stable",
    ).reset_index(drop=True)


@pytest.mark.asyncio
async def test_capsule_lifecycle_rolls_over_at_exact_threshold_and_builds_links(service):
    message_ids: list[str] = []
    lengths = [30_000, 30_000, 40_000, 15_000]
    for idx, (char, length) in enumerate(zip(("a", "b", "c", "d"), lengths), start=1):
        ack = await _ingest_capsule_message(
            service,
            session_id="capsule-session",
            topic_id="capsule_topic",
            message_id=f"capsule-msg-{idx}",
            content=_body(char, length),
            minute_offset=idx,
        )
        message_ids.append(str(ack.origin_message_id))

    rows = _capsule_rows(service)
    assert rows.shape[0] == 2

    first = rows.iloc[0]
    second = rows.iloc[1]

    assert str(first["topic_id"]) == "capsule_topic"
    assert str(first["capsule_state"]) == "sealed"
    assert int(first["source_body_char_count"]) == 100_000
    assert int(first["source_message_count"]) == 3
    assert json.loads(str(first["source_message_ids_json"])) == message_ids[:3]
    assert str(first["next_capsule_id"]) == str(second["capsule_id"])
    assert json.loads(str(first["forward_link_ids_json"])) == [str(second["capsule_id"])]
    assert json.loads(str(first["back_link_ids_json"])) == []

    assert str(second["capsule_state"]) == "open"
    assert int(second["source_body_char_count"]) == 15_000
    assert int(second["source_message_count"]) == 1
    assert json.loads(str(second["source_message_ids_json"])) == message_ids[3:]
    assert str(second["prev_capsule_id"]) == str(first["capsule_id"])
    assert json.loads(str(second["back_link_ids_json"])) == [str(first["capsule_id"])]
    assert json.loads(str(second["forward_link_ids_json"])) == []

    second_pointer = json.loads(str(second["pointer_json"]))
    assert second_pointer["capsule_ordinal"] == 2
    assert second_pointer["prev_capsule_id"] == str(first["capsule_id"])
    assert second_pointer["next_capsule_id"] is None

    cards = await service.present_capsule_cards("default", "capsule-session")
    cards_by_id = {str(item["capsule_id"]): item for item in cards}
    assert cards_by_id[str(first["capsule_id"])]["capsule_state"] == "sealed"
    assert cards_by_id[str(second["capsule_id"])]["source_body_char_count"] == 15_000
    assert cards_by_id[str(second["capsule_id"])]["back_link_ids"] == [str(first["capsule_id"])]


@pytest.mark.asyncio
async def test_capsule_lifecycle_refreshes_vectors_when_source_rows_change(service):
    acks = []
    for idx, (char, length) in enumerate((("a", 30_000), ("b", 30_000), ("c", 40_000), ("d", 15_000)), start=1):
        ack = await _ingest_capsule_message(
            service,
            session_id="capsule-session",
            topic_id="capsule_topic",
            message_id=f"refresh-msg-{idx}",
            content=_body(char, length),
            minute_offset=idx,
        )
        acks.append(ack)

    before = _capsule_rows(service)
    first_before = before.iloc[0]

    await service.edit_message(
        MessageEditRequest(
            tenant_id="default",
            origin_message_id=acks[0].origin_message_id,
            content=_body("z", 30_000),
            ts=datetime(2026, 3, 21, 1, 30, tzinfo=timezone.utc),
        )
    )

    after = _capsule_rows(service)
    first_after = after.iloc[0]

    assert str(first_after["capsule_id"]) == str(first_before["capsule_id"])
    assert int(first_after["source_body_char_count"]) == int(first_before["source_body_char_count"])
    assert str(first_after["source_hash"]) != str(first_before["source_hash"])
    assert str(first_after["vector_ref"]) != str(first_before["vector_ref"])
    assert len(json.loads(str(first_after["vector_json"]))) == service.config.topic_gep_dim
    assert "zzzz" in str(first_after["summary"])


@pytest.mark.asyncio
async def test_schema_migration_backfills_capsules_from_raw_messages(tmp_path: Path):
    config = _test_config(tmp_path)
    service_one = ClawDBService(config=config)
    await service_one.startup()
    try:
        for idx, (char, length) in enumerate((("a", 30_000), ("b", 30_000), ("c", 40_000), ("d", 15_000)), start=1):
            await _ingest_capsule_message(
                service_one,
                session_id="capsule-session",
                topic_id="capsule_topic",
                message_id=f"backfill-msg-{idx}",
                content=_body(char, length),
                minute_offset=idx,
            )
        await service_one.flush_now()
        expected = _capsule_rows(service_one).copy()
    finally:
        await service_one.shutdown()

    shutil.rmtree(config.parquet_dir / "capsules")

    service_two = ClawDBService(config=config)
    await service_two.startup()
    try:
        actual = _capsule_rows(service_two)
        assert not actual.empty
        assert actual["capsule_id"].astype(str).tolist() == expected["capsule_id"].astype(str).tolist()
        assert actual["capsule_state"].astype(str).tolist() == expected["capsule_state"].astype(str).tolist()
        assert actual["source_body_char_count"].astype(int).tolist() == expected["source_body_char_count"].astype(int).tolist()
        assert actual["source_message_count"].astype(int).tolist() == expected["source_message_count"].astype(int).tolist()
        assert actual["prev_capsule_id"].astype(str).tolist() == expected["prev_capsule_id"].astype(str).tolist()
        assert actual["next_capsule_id"].astype(str).tolist() == expected["next_capsule_id"].astype(str).tolist()
        assert actual["vector_ref"].astype(str).tolist() == expected["vector_ref"].astype(str).tolist()
    finally:
        await service_two.shutdown()
