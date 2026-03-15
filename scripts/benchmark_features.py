#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic
from typing import Dict, List

from clawdb.config import ClawDBConfig
from clawdb.models import MessageIn, SearchRequest
from clawdb.service import ClawDBService


WORDS = [
    "dragon",
    "vector",
    "cache",
    "wal",
    "async",
    "queue",
    "capsule",
    "memory",
    "tenant",
    "session",
    "checkpoint",
    "openclaw",
    "parquet",
    "pandas",
    "search",
]


@dataclass
class BenchConfig:
    messages: int
    sessions: int
    tenants: int
    ingest_concurrency: int
    cold_searches: int
    hot_searches: int
    queue_backend: str
    random_seed: int


@dataclass
class BenchResult:
    ingest_ops: int
    ingest_p50_ms: float
    ingest_p95_ms: float
    ingest_ops_per_sec: float
    cold_search_ops: int
    cold_search_p50_ms: float
    cold_search_p95_ms: float
    hot_search_ops: int
    hot_search_p50_ms: float
    hot_search_p95_ms: float
    cache_hit_ratio_1m: float
    cache_hit_ratio_5m: float
    cache_hits_total: int
    cache_misses_total: int
    replay_startup_ms: float
    wal_last_seq: int
    queue_backend: str


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    index = int(round((len(values_sorted) - 1) * p))
    index = max(0, min(index, len(values_sorted) - 1))
    return float(values_sorted[index])


def make_service_config(data_root: Path, queue_backend: str) -> ClawDBConfig:
    endpoint_suffix = random.randint(100_000, 999_999)
    return ClawDBConfig(
        data_root=data_root,
        wal_dir=data_root / "wal",
        parquet_dir=data_root / "parquet",
        checkpoints_dir=data_root / "checkpoints",
        metadata_parquet_path=data_root / "checkpoints" / "metadata.parquet",
        wal_sync_policy="always",
        wal_sync_interval_ms=10,
        queue_backend=queue_backend,
        queue_topic="clawdb.memory.events",
        queue_zeromq_endpoint=f"inproc://bench-clawdb-memory-events-{endpoint_suffix}",
        queue_consumer_count=2,
        ingest_backpressure_lag_threshold=1_000_000,
        ingest_backpressure_max_wait_ms=0,
        ingest_backpressure_poll_interval_ms=1,
        idempotency_dedupe_enabled=False,
        topic_auto_classify_enabled=True,
        topic_gep_dim=64,
        topic_gep_concentration=0.8,
        topic_gep_sigma2=0.7,
        topic_gep_prior_sigma2=1.2,
        openclaw_require_signature=False,
        flush_interval_seconds=60,
        lock_timeout_seconds=2.0,
        lock_watchdog_seconds=20.0,
        cache_hit_ratio_alert_threshold=0.80,
        search_log_enabled=False,
    )


def build_message_content(idx: int, rng: random.Random) -> str:
    chosen = rng.sample(WORDS, k=4)
    return f"message-{idx} {' '.join(chosen)}"


async def run_ingest(
    svc: ClawDBService,
    bench_cfg: BenchConfig,
    rng: random.Random,
) -> Dict[str, object]:
    semaphore = asyncio.Semaphore(max(1, bench_cfg.ingest_concurrency))
    latencies_ms: List[float] = []
    start_all = monotonic()

    async def one(idx: int) -> None:
        tenant = f"tenant-{idx % bench_cfg.tenants}"
        session = f"session-{idx % bench_cfg.sessions}"
        content = build_message_content(idx, rng)
        payload = MessageIn(
            tenant_id=tenant,
            session_id=session,
            role="user",
            content=content,
            topic_id=f"topic-{idx % 8}",
            capsule_level="L0",
        )
        async with semaphore:
            started = monotonic()
            await svc.ingest_message(payload)
            latencies_ms.append((monotonic() - started) * 1000.0)

    await asyncio.gather(*(one(i) for i in range(bench_cfg.messages)))
    duration = max(0.001, monotonic() - start_all)
    return {
        "latencies_ms": latencies_ms,
        "ops_per_sec": float(bench_cfg.messages / duration),
    }


async def run_cold_search(
    svc: ClawDBService,
    bench_cfg: BenchConfig,
    rng: random.Random,
) -> List[float]:
    latencies_ms: List[float] = []
    for i in range(bench_cfg.cold_searches):
        tenant = f"tenant-{i % bench_cfg.tenants}"
        session = f"session-{i % bench_cfg.sessions}"
        query = f"{WORDS[i % len(WORDS)]} q{i}"
        started = monotonic()
        await svc.search(
            SearchRequest(
                query=query,
                tenant_id=tenant,
                session_id=session,
                max_results=5,
            )
        )
        latencies_ms.append((monotonic() - started) * 1000.0)
    return latencies_ms


async def run_hot_search(
    svc: ClawDBService,
    bench_cfg: BenchConfig,
) -> List[float]:
    latencies_ms: List[float] = []
    req = SearchRequest(
        query="memory vector cache",
        tenant_id="tenant-0",
        session_id="session-0",
        max_results=5,
    )
    await svc.search(req)  # cache warm
    for _ in range(bench_cfg.hot_searches):
        started = monotonic()
        await svc.search(req)
        latencies_ms.append((monotonic() - started) * 1000.0)
    return latencies_ms


async def run_benchmark(bench_cfg: BenchConfig) -> Dict[str, object]:
    rng = random.Random(bench_cfg.random_seed)
    with tempfile.TemporaryDirectory(prefix="clawdb-bench-") as tmp:
        data_root = Path(tmp)
        cfg = make_service_config(data_root=data_root, queue_backend=bench_cfg.queue_backend)
        svc = ClawDBService(config=cfg)
        await svc.startup()
        try:
            ingest = await run_ingest(svc, bench_cfg, rng)
            cold_lat = await run_cold_search(svc, bench_cfg, rng)
            hot_lat = await run_hot_search(svc, bench_cfg)
            cache = await svc.cache_hit_report()
            await svc.flush_now()
        finally:
            await svc.shutdown()

        replay_svc = ClawDBService(config=cfg)
        replay_started = monotonic()
        await replay_svc.startup()
        replay_ms = (monotonic() - replay_started) * 1000.0
        wal_last_seq = replay_svc.wal.last_seq
        await replay_svc.shutdown()

    result = BenchResult(
        ingest_ops=bench_cfg.messages,
        ingest_p50_ms=percentile(ingest["latencies_ms"], 0.50),
        ingest_p95_ms=percentile(ingest["latencies_ms"], 0.95),
        ingest_ops_per_sec=float(ingest["ops_per_sec"]),
        cold_search_ops=bench_cfg.cold_searches,
        cold_search_p50_ms=percentile(cold_lat, 0.50),
        cold_search_p95_ms=percentile(cold_lat, 0.95),
        hot_search_ops=bench_cfg.hot_searches,
        hot_search_p50_ms=percentile(hot_lat, 0.50),
        hot_search_p95_ms=percentile(hot_lat, 0.95),
        cache_hit_ratio_1m=float(cache.memory_cache_hit_ratio_1m),
        cache_hit_ratio_5m=float(cache.memory_cache_hit_ratio_5m),
        cache_hits_total=int(cache.memory_cache_hits_total),
        cache_misses_total=int(cache.memory_cache_misses_total),
        replay_startup_ms=float(replay_ms),
        wal_last_seq=int(wal_last_seq),
        queue_backend=bench_cfg.queue_backend,
    )
    return {
        "config": asdict(bench_cfg),
        "result": asdict(result),
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark existing clawdb features")
    parser.add_argument("--messages", type=int, default=2000)
    parser.add_argument("--sessions", type=int, default=40)
    parser.add_argument("--tenants", type=int, default=4)
    parser.add_argument("--ingest-concurrency", type=int, default=64)
    parser.add_argument("--cold-searches", type=int, default=500)
    parser.add_argument("--hot-searches", type=int, default=500)
    parser.add_argument("--queue-backend", choices=["zeromq", "inmemory"], default="zeromq")
    parser.add_argument("--seed", type=int, default=20260315)
    parser.add_argument("--out", type=Path, default=Path("Docs/benchmark_report_latest.json"))
    args = parser.parse_args()

    bench_cfg = BenchConfig(
        messages=max(1, args.messages),
        sessions=max(1, args.sessions),
        tenants=max(1, args.tenants),
        ingest_concurrency=max(1, args.ingest_concurrency),
        cold_searches=max(1, args.cold_searches),
        hot_searches=max(1, args.hot_searches),
        queue_backend=args.queue_backend,
        random_seed=args.seed,
    )
    payload = await run_benchmark(bench_cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
