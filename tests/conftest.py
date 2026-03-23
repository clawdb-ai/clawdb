from __future__ import annotations

from pathlib import Path

import pytest

from clawdb.config import ClawDBConfig
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


@pytest.fixture
async def service(tmp_path: Path):
    svc = ClawDBService(config=_test_config(tmp_path))
    await svc.startup()
    try:
        yield svc
    finally:
        await svc.shutdown()
