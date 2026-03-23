from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from clawdb.config import ClawDBConfig
from clawdb.dataframes import DataFrameStore
from clawdb.models import MessageDeleteRequest, MessageEditRequest, MessageIn
from clawdb.service import ClawDBService


DM_SCOPE = "dm:feishu_account:botacct:feishu_user:ou_user_rollup"


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


async def _ingest_direct_message(
    service: ClawDBService,
    *,
    content: str,
    ts: datetime,
    message_suffix: str,
):
    return await service.ingest_message(
        MessageIn(
            tenant_id="default",
            session_id="window-session",
            role="user",
            content=content,
            channel="feishu",
            platform="feishu",
            chat_type="direct",
            account_id="botacct",
            sender_id="ou_user_rollup",
            from_id="ou_user_rollup",
            to_id="ou_user_rollup",
            platform_message_id=f"pm-{message_suffix}",
            message_id=f"req-{message_suffix}",
            ts=ts,
        )
    )


def _rollup_rows(service: ClawDBService):
    df = service.df_store.state.session_rollups_df
    return df[
        (df["tenant_id"].astype(str) == "default")
        & (df["session_id"].astype(str) == DM_SCOPE)
    ].copy()


@pytest.mark.asyncio
async def test_session_rollups_materialize_all_summary_windows_and_vectors(service):
    await _ingest_direct_message(
        service,
        content="year end hello",
        ts=datetime(2025, 12, 31, 23, 55, tzinfo=timezone.utc),
        message_suffix="a",
    )
    await _ingest_direct_message(
        service,
        content="new year hello",
        ts=datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc),
        message_suffix="b",
    )
    await _ingest_direct_message(
        service,
        content="monday followup",
        ts=datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc),
        message_suffix="c",
    )
    await _ingest_direct_message(
        service,
        content="quarter close",
        ts=datetime(2026, 3, 31, 8, 30, tzinfo=timezone.utc),
        message_suffix="d",
    )
    await _ingest_direct_message(
        service,
        content="quarter open",
        ts=datetime(2026, 4, 1, 9, 0, tzinfo=timezone.utc),
        message_suffix="e",
    )

    rollups = _rollup_rows(service)
    assert set(rollups["window_kind"].astype(str)) == {
        "daily",
        "weekly",
        "monthly",
        "quarterly",
        "yearly",
        "lifetime",
    }
    assert set(rollups[rollups["window_kind"].astype(str) == "daily"]["window_key"].astype(str)) == {
        "2025-12-31",
        "2026-01-01",
        "2026-01-05",
        "2026-03-31",
        "2026-04-01",
    }
    assert set(rollups[rollups["window_kind"].astype(str) == "weekly"]["window_key"].astype(str)) == {
        "2026-W01",
        "2026-W02",
        "2026-W14",
    }
    assert set(rollups[rollups["window_kind"].astype(str) == "monthly"]["window_key"].astype(str)) == {
        "2025-12",
        "2026-01",
        "2026-03",
        "2026-04",
    }
    assert set(rollups[rollups["window_kind"].astype(str) == "quarterly"]["window_key"].astype(str)) == {
        "2025-Q4",
        "2026-Q1",
        "2026-Q2",
    }
    assert set(rollups[rollups["window_kind"].astype(str) == "yearly"]["window_key"].astype(str)) == {
        "2025",
        "2026",
    }
    lifetime = rollups[rollups["window_kind"].astype(str) == "lifetime"]
    assert lifetime.shape[0] == 1
    assert int(lifetime.iloc[0]["message_count"]) == 5
    assert "quarter open" in str(lifetime.iloc[0]["summary"])
    assert rollups["rollup_id"].astype(str).is_unique
    assert (rollups["vector_text"].astype(str) == rollups["summary"].astype(str)).all()
    assert set(rollups["vector_dim"].astype(int)) == {service.config.topic_gep_dim}
    assert rollups["vector_ref"].astype(str).str.startswith("session_rollup:").all()
    assert all(
        len(json.loads(value)) == service.config.topic_gep_dim
        for value in rollups["vector_json"].astype(str).tolist()
    )


@pytest.mark.asyncio
async def test_session_rollups_recompute_on_edit_and_delete(service):
    first = await _ingest_direct_message(
        service,
        content="older day",
        ts=datetime(2026, 3, 22, 8, 0, tzinfo=timezone.utc),
        message_suffix="edit-a",
    )
    target = await _ingest_direct_message(
        service,
        content="needs rewrite",
        ts=datetime(2026, 3, 23, 9, 0, tzinfo=timezone.utc),
        message_suffix="edit-b",
    )
    await _ingest_direct_message(
        service,
        content="next month survivor",
        ts=datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc),
        message_suffix="edit-c",
    )

    before = _rollup_rows(service).set_index(["window_kind", "window_key"])

    await service.edit_message(
        MessageEditRequest(
            tenant_id="default",
            origin_message_id=target.origin_message_id,
            content="edited march day",
            ts=datetime(2026, 3, 23, 10, 0, tzinfo=timezone.utc),
        )
    )

    after_edit = _rollup_rows(service)
    changed = after_edit[after_edit["summary"].astype(str).str.contains("edited march day", regex=False)]
    assert set(changed["window_kind"].astype(str)) == {
        "daily",
        "weekly",
        "monthly",
        "quarterly",
        "yearly",
        "lifetime",
    }
    after_edit_indexed = after_edit.set_index(["window_kind", "window_key"])
    for key in [
        ("daily", "2026-03-23"),
        ("weekly", "2026-W13"),
        ("monthly", "2026-03"),
        ("quarterly", "2026-Q1"),
        ("yearly", "2026"),
        ("lifetime", "lifetime"),
    ]:
        assert (
            after_edit_indexed.at[key, "vector_ref"]
            != before.at[key, "vector_ref"]
        )

    await service.delete_message(
        MessageDeleteRequest(
            tenant_id="default",
            origin_message_id=target.origin_message_id,
            ts=datetime(2026, 3, 23, 11, 0, tzinfo=timezone.utc),
        )
    )

    after_delete = _rollup_rows(service)
    assert not after_delete["summary"].astype(str).str.contains("edited march day", regex=False).any()
    assert "2026-03-23" not in set(
        after_delete[after_delete["window_kind"].astype(str) == "daily"]["window_key"].astype(str)
    )
    lifetime = after_delete[after_delete["window_kind"].astype(str) == "lifetime"].iloc[0]
    assert int(lifetime["message_count"]) == 2
    assert "older day" in str(lifetime["summary"])
    assert "next month survivor" in str(lifetime["summary"])
    assert first.origin_message_id != target.origin_message_id


@pytest.mark.asyncio
async def test_session_rollups_persist_across_parquet_reload(service):
    await _ingest_direct_message(
        service,
        content="persist me",
        ts=datetime(2026, 2, 10, 8, 15, tzinfo=timezone.utc),
        message_suffix="persist-a",
    )
    await _ingest_direct_message(
        service,
        content="persist me too",
        ts=datetime(2026, 2, 11, 9, 30, tzinfo=timezone.utc),
        message_suffix="persist-b",
    )
    expected = _rollup_rows(service).sort_values(["window_kind", "window_key"], kind="stable").reset_index(drop=True)

    await service.flush_now()

    fresh = DataFrameStore()
    await fresh.load_parquet(service.config.parquet_dir)
    actual = fresh.state.session_rollups_df
    actual = actual[
        (actual["tenant_id"].astype(str) == "default")
        & (actual["session_id"].astype(str) == DM_SCOPE)
    ].sort_values(["window_kind", "window_key"], kind="stable").reset_index(drop=True)

    assert actual.shape[0] == expected.shape[0]
    assert actual["rollup_id"].astype(str).tolist() == expected["rollup_id"].astype(str).tolist()
    assert actual["summary"].astype(str).tolist() == expected["summary"].astype(str).tolist()
    assert actual["vector_ref"].astype(str).tolist() == expected["vector_ref"].astype(str).tolist()


@pytest.mark.asyncio
async def test_schema_migration_backfills_session_rollups_from_messages(tmp_path: Path):
    config = _test_config(tmp_path)

    service_one = ClawDBService(config=config)
    await service_one.startup()
    try:
        await _ingest_direct_message(
            service_one,
            content="migration seed one",
            ts=datetime(2026, 3, 1, 7, 0, tzinfo=timezone.utc),
            message_suffix="migrate-a",
        )
        await _ingest_direct_message(
            service_one,
            content="migration seed two",
            ts=datetime(2026, 3, 2, 7, 0, tzinfo=timezone.utc),
            message_suffix="migrate-b",
        )
        await service_one.flush_now()
    finally:
        await service_one.shutdown()

    shutil.rmtree(config.parquet_dir / "session_rollups")

    service_two = ClawDBService(config=config)
    await service_two.startup()
    try:
        assert (config.parquet_dir / "session_rollups").exists()
        rollups = _rollup_rows(service_two)
        assert not rollups.empty
        assert set(rollups["window_kind"].astype(str)) == {
            "daily",
            "weekly",
            "monthly",
            "quarterly",
            "yearly",
            "lifetime",
        }
    finally:
        await service_two.shutdown()
