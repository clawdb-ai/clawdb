# ClawDB Implementation Audit Checklist

## Purpose
Execution checklist to verify clawdb can fully replace OpenClaw memory-related modules and OpenViking+QMD memory paths.

## A. Core Backend Coverage
- [x] `clawdb.api` exposes all blueprint endpoints.
- [x] `clawdb.service` enforces async task orchestration and backpressure hooks.
- [x] `clawdb.wal` provides append, checksum validation, replay, and sequence guarantees.
- [x] `clawdb.dataframes` provides pandas hot-state and Parquet persistence.
- [x] `clawdb.mq` provides high-throughput queue integration path (Kafka/Redpanda class available).
- [x] `clawdb.mq` provides ZeroMQ backend option for low-latency single-node mode.
- [x] default queue backend is `zeromq` unless explicitly overridden by environment.
- [x] `clawdb.locks` enforces lock ordering and deadlock watchdog telemetry.
- [x] `clawdb.metrics` exposes cache-hit and latency signals.
- [x] `clawdb.metadata` persists checkpoint metadata as DataFrame + Parquet for replay coordination.
- [x] message ingest supports idempotency keys.
- [x] tenant/session isolation is enforced for search and virtual memory read paths.

## B. OpenClaw Memory Masking Coverage
- [x] OpenClaw plugin slot `memory` is set to `memory-clawdb`.
- [x] OpenClaw `memory-core` slot provider is disabled.
- [x] `memory_search` tool calls route to `/v1/openclaw/memory/search`.
- [x] `memory_get` tool calls route to `/v1/openclaw/memory/get`.
- [x] plugin CLI `clawdb-memory status/search/get` resolves to clawdb responses.
- [x] plugin resolves embedding credentials using OpenClaw runtime auth/profile resolution.
- [x] plugin requests include OpenClaw-compatible signed headers accepted by clawdb.
- [x] clawdb enforces signed OpenClaw adapter requests by default (configurable for local debug only).

## C. Durability And Recovery
- [x] Acknowledged writes only after WAL append success.
- [x] Startup replay from checkpoint + WAL tail works.
- [x] metadata checkpoint and file checkpoint are both written on flush.
- [x] Parquet load-out and rehydrate paths pass smoke tests.

## D. Cache-Hit Enforcement
- [x] `memory_cache_hit_ratio_1m` emitted and queryable.
- [x] `memory_cache_hit_ratio_5m` emitted and queryable.
- [x] Hits/misses/evictions counters emitted.
- [x] release gate checks cache-hit metrics pipeline integrity.

## E. Integration Readiness
- [x] `scripts/bootstrap_openclaw.sh` completes successfully.
- [x] `scripts/install_openclaw_integration.sh` installs plugin and config template.
- [x] `scripts/smoke_test_integration.sh` passes end-to-end.
- [x] `Docs/openclaw_integration_test.md` smoke test completes.
- [x] End-to-end query from OpenClaw reaches clawdb and returns citation-bearing results.

## F. Benchmark Coverage
- [x] `scripts/benchmark_features.py` reports ingest/search/replay metrics.
- [x] benchmark report generated for `zeromq` backend.
- [x] benchmark report generated for `inmemory` backend.
- [x] benchmark summary saved at `Docs/benchmark_summary_2026-03-15.md`.
