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
    idempotency_dedupe_enabled: bool
    topic_auto_classify_enabled: bool
    topic_gep_dim: int
    topic_gep_concentration: float
    topic_gep_sigma2: float
    topic_gep_prior_sigma2: float
    openclaw_require_signature: bool
    flush_interval_seconds: int
    lock_timeout_seconds: float
    lock_watchdog_seconds: float
    cache_hit_ratio_alert_threshold: float
    search_log_enabled: bool

    @classmethod
    def from_env(cls) -> "ClawDBConfig":
        def _env_bool(name: str, default: str) -> bool:
            raw = os.getenv(name, default).strip().lower()
            return raw in {"1", "true", "yes", "on"}

        root = Path(os.getenv("CLAWDB_DATA_ROOT", "data")).resolve()
        wal_dir = root / "wal"
        parquet_dir = root / "parquet"
        checkpoints_dir = root / "checkpoints"
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
            idempotency_dedupe_enabled=_env_bool("CLAWDB_IDEMPOTENCY_DEDUPE_ENABLED", "false"),
            topic_auto_classify_enabled=_env_bool("CLAWDB_TOPIC_AUTO_CLASSIFY_ENABLED", "true"),
            topic_gep_dim=max(8, int(os.getenv("CLAWDB_TOPIC_GEP_DIM", "64"))),
            topic_gep_concentration=float(os.getenv("CLAWDB_TOPIC_GEP_CONCENTRATION", "0.8")),
            topic_gep_sigma2=float(os.getenv("CLAWDB_TOPIC_GEP_SIGMA2", "0.7")),
            topic_gep_prior_sigma2=float(os.getenv("CLAWDB_TOPIC_GEP_PRIOR_SIGMA2", "1.2")),
            openclaw_require_signature=_env_bool("CLAWDB_OPENCLAW_REQUIRE_SIGNATURE", "true"),
            flush_interval_seconds=int(os.getenv("CLAWDB_FLUSH_INTERVAL_SECONDS", "10")),
            lock_timeout_seconds=float(os.getenv("CLAWDB_LOCK_TIMEOUT_SECONDS", "1.5")),
            lock_watchdog_seconds=float(os.getenv("CLAWDB_LOCK_WATCHDOG_SECONDS", "10")),
            cache_hit_ratio_alert_threshold=float(
                os.getenv("CLAWDB_CACHE_HIT_RATIO_ALERT_THRESHOLD", "0.80")
            ),
            search_log_enabled=_env_bool("CLAWDB_SEARCH_LOG_ENABLED", "true"),
        )

    def ensure_dirs(self) -> None:
        self.wal_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
