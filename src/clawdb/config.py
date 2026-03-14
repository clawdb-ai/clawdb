from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClawDBConfig:
    data_root: Path
    wal_dir: Path
    parquet_dir: Path
    checkpoints_dir: Path
    metadata_parquet_path: Path
    wal_sync_policy: str
    wal_sync_interval_ms: int
    queue_backend: str
    queue_topic: str
    queue_zeromq_endpoint: str
    queue_consumer_count: int
    ingest_backpressure_lag_threshold: int
    ingest_backpressure_max_wait_ms: int
    ingest_backpressure_poll_interval_ms: int
    openclaw_require_signature: bool
    flush_interval_seconds: int
    lock_timeout_seconds: float
    lock_watchdog_seconds: float
    cache_hit_ratio_alert_threshold: float
    search_log_enabled: bool

    @classmethod
    def from_env(cls) -> "ClawDBConfig":
        root = Path(os.getenv("CLAWDB_DATA_ROOT", "data")).resolve()
        wal_dir = root / "wal"
        parquet_dir = root / "parquet"
        checkpoints_dir = root / "checkpoints"
        require_signature_raw = os.getenv("CLAWDB_OPENCLAW_REQUIRE_SIGNATURE", "true")
        require_signature = require_signature_raw.strip().lower() in {"1", "true", "yes", "on"}
        search_log_enabled_raw = os.getenv("CLAWDB_SEARCH_LOG_ENABLED", "true")
        search_log_enabled = search_log_enabled_raw.strip().lower() in {"1", "true", "yes", "on"}
        return cls(
            data_root=root,
            wal_dir=wal_dir,
            parquet_dir=parquet_dir,
            checkpoints_dir=checkpoints_dir,
            metadata_parquet_path=Path(
                os.getenv("CLAWDB_METADATA_PARQUET_PATH", str(checkpoints_dir / "metadata.parquet"))
            ).resolve(),
            wal_sync_policy=os.getenv("CLAWDB_WAL_SYNC", "always").strip().lower(),
            wal_sync_interval_ms=int(os.getenv("CLAWDB_WAL_SYNC_INTERVAL_MS", "25")),
            queue_backend=os.getenv("CLAWDB_QUEUE_BACKEND", "zeromq").strip().lower(),
            queue_topic=os.getenv("CLAWDB_QUEUE_TOPIC", "clawdb.memory.events"),
            queue_zeromq_endpoint=os.getenv(
                "CLAWDB_QUEUE_ZEROMQ_ENDPOINT", "inproc://clawdb-memory-events"
            ),
            queue_consumer_count=int(os.getenv("CLAWDB_QUEUE_CONSUMERS", "2")),
            ingest_backpressure_lag_threshold=int(
                os.getenv("CLAWDB_INGEST_BACKPRESSURE_LAG_THRESHOLD", "20000")
            ),
            ingest_backpressure_max_wait_ms=int(
                os.getenv("CLAWDB_INGEST_BACKPRESSURE_MAX_WAIT_MS", "250")
            ),
            ingest_backpressure_poll_interval_ms=int(
                os.getenv("CLAWDB_INGEST_BACKPRESSURE_POLL_INTERVAL_MS", "10")
            ),
            openclaw_require_signature=require_signature,
            flush_interval_seconds=int(os.getenv("CLAWDB_FLUSH_INTERVAL_SECONDS", "10")),
            lock_timeout_seconds=float(os.getenv("CLAWDB_LOCK_TIMEOUT_SECONDS", "1.5")),
            lock_watchdog_seconds=float(os.getenv("CLAWDB_LOCK_WATCHDOG_SECONDS", "10")),
            cache_hit_ratio_alert_threshold=float(
                os.getenv("CLAWDB_CACHE_HIT_RATIO_ALERT_THRESHOLD", "0.80")
            ),
            search_log_enabled=search_log_enabled,
        )

    def ensure_dirs(self) -> None:
        self.wal_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
