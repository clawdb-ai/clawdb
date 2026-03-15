# ClawDB Benchmark Summary (2026-03-15)

## Scope
Benchmark existing implemented features using in-process service benchmarks:
- ingest ACK latency and throughput
- cold/hot search latency
- cache-hit behavior
- restart replay startup time

Benchmark runner:
- `scripts/benchmark_features.py`

Report artifacts:
- `Docs/benchmark_report_latest.json` (`zeromq`)
- `Docs/benchmark_report_inmemory.json` (`inmemory`)

## Benchmark Configuration
- messages: `6000`
- sessions: `120`
- tenants: `8`
- ingest concurrency: `96`
- cold searches: `1200`
- hot searches: `1200`

## Results

### ZeroMQ backend
- ingest p50: `155.08 ms`
- ingest p95: `175.81 ms`
- ingest throughput: `621.41 ops/s`
- cold search p50: `3.06 ms`
- cold search p95: `3.43 ms`
- hot search p50: `0.57 ms`
- hot search p95: `0.65 ms`
- cache-hit ratio (1m): `0.4998`
- replay startup: `18.97 ms`

### In-memory backend
- ingest p50: `143.63 ms`
- ingest p95: `165.41 ms`
- ingest throughput: `673.81 ops/s`
- cold search p50: `3.04 ms`
- cold search p95: `3.41 ms`
- hot search p50: `0.57 ms`
- hot search p95: `0.64 ms`
- cache-hit ratio (1m): `0.4998`
- replay startup: `19.36 ms`

## Blueprint target comparison
- Read latency target (`P50 < 20 ms`, `P95 < 80 ms` mixed): met in this benchmark for search path.
- Write ACK target (`P95 < 10 ms`): not met under current durability settings (`WAL sync=always`).

## Notes
- Benchmarks are service-level (not full network/API path), focused on current implementation behavior.
- Ingest latency is dominated by `fsync` durability cost in `sync=always` mode.
- To trade durability for lower write latency, evaluate `CLAWDB_WAL_SYNC=interval` with controlled risk.
