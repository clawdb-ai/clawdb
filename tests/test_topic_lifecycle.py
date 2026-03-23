from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from clawdb.config import ClawDBConfig
from clawdb.models import MessageDeleteRequest, MessageEditRequest, MessageIn
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


async def _ingest_topic_message(
    service: ClawDBService,
    *,
    topic_id: str,
    content: str,
    message_id: str,
    minute_offset: int,
    topic_parent_id: str | None = None,
    topic_path: str | None = None,
):
    base_ts = datetime(2026, 3, 20, tzinfo=timezone.utc)
    return await service.ingest_message(
        MessageIn(
            tenant_id="default",
            session_id="topic-batch",
            role="user",
            content=content,
            topic_id=topic_id,
            topic_parent_id=topic_parent_id,
            topic_path=topic_path,
            message_id=message_id,
            ts=base_ts + timedelta(minutes=minute_offset),
        )
    )


def _topic_rows(service: ClawDBService):
    return service.df_store.state.topics_df.sort_values(["tenant_id", "topic_id"], kind="stable").reset_index(
        drop=True
    )


def _topic_row(service: ClawDBService, topic_id: str):
    rows = _topic_rows(service)
    match = rows[rows["topic_id"].astype(str) == str(topic_id)]
    assert match.shape[0] == 1
    return match.iloc[0]


@pytest.mark.asyncio
async def test_topic_lifecycle_materializes_merge_split_reparent_and_drift(service):
    messages = [
        ("billing", "billing invoice refund status for march invoice dispute"),
        ("billing", "billing refund invoice investigation and invoice dispute update"),
        ("invoice_ops", "invoice refund dispute for billing invoice update"),
        ("invoice_ops", "billing invoice dispute refund approved by finance"),
        ("project_alpha", "orchard apple harvest pruning seedling soil"),
        ("project_alpha", "orchard apple harvest irrigation soil branch"),
        ("project_alpha", "kernel api latency crash service pager deploy"),
        ("project_alpha", "service kernel outage latency rollback deploy api"),
        ("support", "refund escalation invoice billing queue customer issue"),
        ("support", "invoice billing exception refund queue support agent"),
        ("support", "customer refund invoice appeal support queue"),
        ("refund_escalations", "refund escalation invoice queue urgent billing"),
        ("refund_escalations", "invoice refund escalation queue for billing issue"),
        ("drift_topic", "travel itinerary passport hotel luggage airport"),
        ("drift_topic", "travel hotel boarding luggage airport gate"),
        ("drift_topic", "ml training gpu gradient checkpoint optimizer batch"),
    ]
    for idx, (topic_id, content) in enumerate(messages, start=1):
        await _ingest_topic_message(
            service,
            topic_id=topic_id,
            content=content,
            message_id=f"topic-msg-{idx}",
            minute_offset=idx,
        )

    billing = _topic_row(service, "billing")
    assert str(billing["status"]) == "active"
    assert str(billing["canonical_topic_id"]) == "billing"
    assert int(billing["message_count"]) == 4
    assert json.loads(str(billing["merged_topic_ids_json"])) == ["billing", "invoice_ops"]
    assert str(billing["vector_ref"]).startswith("topic:")

    invoice_ops = _topic_row(service, "invoice_ops")
    assert str(invoice_ops["status"]) == "merged"
    assert str(invoice_ops["canonical_topic_id"]) == "billing"
    assert str(invoice_ops["topic_path"]).startswith("billing/")

    project_alpha = _topic_row(service, "project_alpha")
    split_ids = json.loads(str(project_alpha["split_topic_ids_json"]))
    assert str(project_alpha["status"]) == "split"
    assert len(split_ids) == 2
    for split_id in split_ids:
        child = _topic_row(service, split_id)
        assert str(child["status"]) == "active"
        assert str(child["topic_parent_id"]) == "project_alpha"
        assert str(child["topic_path"]).startswith("project_alpha/")

    refund_escalations = _topic_row(service, "refund_escalations")
    assert str(refund_escalations["topic_parent_id"]) == "support"
    assert str(refund_escalations["topic_path"]) == "support/refund_escalations"

    drift_topic = _topic_row(service, "drift_topic")
    assert float(drift_topic["drift_score"]) >= 0.6
    assert pd.notna(drift_topic["drift_corrected_at"])
    assert "ml training gpu gradient checkpoint optimizer batch" in str(drift_topic["summary"])
    assert len(json.loads(str(drift_topic["vector_json"]))) == service.config.topic_gep_dim


@pytest.mark.asyncio
async def test_topic_lifecycle_refreshes_vectors_and_compacts_deleted_topics(service):
    await _ingest_topic_message(
        service,
        topic_id="drift_topic",
        content="travel itinerary passport hotel luggage airport",
        message_id="drift-a",
        minute_offset=1,
    )
    await _ingest_topic_message(
        service,
        topic_id="drift_topic",
        content="travel hotel boarding luggage airport gate",
        message_id="drift-b",
        minute_offset=2,
    )
    drift_target = await _ingest_topic_message(
        service,
        topic_id="drift_topic",
        content="ml training gpu gradient checkpoint optimizer batch",
        message_id="drift-c",
        minute_offset=3,
    )
    retired = await _ingest_topic_message(
        service,
        topic_id="retired_topic",
        content="one shot archival capsule message",
        message_id="retired-a",
        minute_offset=4,
    )

    before_drift = _topic_row(service, "drift_topic")
    before_retired = _topic_row(service, "retired_topic")

    await service.edit_message(
        MessageEditRequest(
            tenant_id="default",
            origin_message_id=drift_target.origin_message_id,
            content="vector refresh semantic reroute embeddings recall",
            ts=datetime(2026, 3, 20, 1, 0, tzinfo=timezone.utc),
        )
    )

    after_drift = _topic_row(service, "drift_topic")
    assert str(after_drift["vector_ref"]) != str(before_drift["vector_ref"])
    assert "vector refresh semantic reroute embeddings recall" in str(after_drift["summary"])
    assert pd.notna(after_drift["drift_corrected_at"])

    await service.delete_message(
        MessageDeleteRequest(
            tenant_id="default",
            origin_message_id=retired.origin_message_id,
            ts=datetime(2026, 3, 20, 2, 0, tzinfo=timezone.utc),
        )
    )

    after_retired = _topic_row(service, "retired_topic")
    assert str(after_retired["status"]) == "compacted"
    assert int(after_retired["message_count"]) == 0
    assert int(after_retired["deleted_message_count"]) == 1
    assert str(after_retired["vector_ref"]) != str(before_retired["vector_ref"])


@pytest.mark.asyncio
async def test_schema_migration_backfills_topics_from_messages(tmp_path: Path):
    config = _test_config(tmp_path)
    service_one = ClawDBService(config=config)
    await service_one.startup()
    try:
        await _ingest_topic_message(
            service_one,
            topic_id="billing",
            content="billing invoice refund status for march invoice dispute",
            message_id="migrate-a",
            minute_offset=1,
        )
        await _ingest_topic_message(
            service_one,
            topic_id="billing",
            content="billing refund invoice investigation and invoice dispute update",
            message_id="migrate-b",
            minute_offset=2,
        )
        await _ingest_topic_message(
            service_one,
            topic_id="invoice_ops",
            content="invoice refund dispute for billing invoice update",
            message_id="migrate-c",
            minute_offset=3,
        )
        await _ingest_topic_message(
            service_one,
            topic_id="invoice_ops",
            content="billing invoice dispute refund approved by finance",
            message_id="migrate-d",
            minute_offset=4,
        )
        await service_one.flush_now()
    finally:
        await service_one.shutdown()

    shutil.rmtree(config.parquet_dir / "topics")

    service_two = ClawDBService(config=config)
    await service_two.startup()
    try:
        assert (config.parquet_dir / "topics").exists()
        billing = _topic_row(service_two, "billing")
        invoice_ops = _topic_row(service_two, "invoice_ops")
        assert str(billing["status"]) == "active"
        assert int(billing["message_count"]) == 4
        assert json.loads(str(billing["merged_topic_ids_json"])) == ["billing", "invoice_ops"]
        assert str(invoice_ops["status"]) == "merged"
        assert str(invoice_ops["canonical_topic_id"]) == "billing"
    finally:
        await service_two.shutdown()
