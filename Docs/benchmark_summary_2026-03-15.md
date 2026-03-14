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
- ingest p50: `82.79 ms`
- ingest p95: `106.86 ms`
- ingest throughput: `1113.70 ops/s`
- cold search p50: `2.41 ms`
- cold search p95: `2.96 ms`
- hot search p50: `0.57 ms`
- hot search p95: `0.77 ms`
- cache-hit ratio (1m): `0.4998`
- replay startup: `19.53 ms`

### In-memory backend
- ingest p50: `71.77 ms`
- ingest p95: `87.66 ms`
- ingest throughput: `1287.62 ops/s`
- cold search p50: `2.42 ms`
- cold search p95: `2.95 ms`
- hot search p50: `0.58 ms`
- hot search p95: `0.76 ms`
- cache-hit ratio (1m): `0.4998`
- replay startup: `20.30 ms`

## Blueprint target comparison
- Read latency target (`P50 < 20 ms`, `P95 < 80 ms` mixed): met in this benchmark for search path.
- Write ACK target (`P95 < 10 ms`): not met under current durability settings (`WAL sync=always`).

## Notes
- Benchmarks are service-level (not full network/API path), focused on current implementation behavior.
- Ingest latency is dominated by `fsync` durability cost in `sync=always` mode.
- To trade durability for lower write latency, evaluate `CLAWDB_WAL_SYNC=interval` with controlled risk.
