# clawdb

Fast memory is exciting. Trusted memory is useful. ⚡  
`clawdb` is built to deliver both for OpenClaw.

`clawdb` is an async Python memory backend that can replace OpenClaw memory modules with WAL-first durability, DataFrame hot-state performance, Parquet persistence, and production-grade integration controls.

## Why ClawDB 🚀

Most memory systems force a painful choice: low latency or reliable recovery. `clawdb` is designed so you can keep speed and still trust what was acknowledged.

- Async-by-default service and worker model.
- Append-only WAL semantics for write durability.
- In-memory `pandas` for hot retrieval paths.
- Parquet snapshots for load-in/load-out recovery.
- Signed OpenClaw adapter routes by default.
- Tenant isolation, idempotent ingest, deadlock safeguards, and backpressure controls.

## Core Capabilities 🧩

- OpenClaw memory compatibility:
  - `memory_search`
  - `memory_get`
- Plugin-backed memory masking:
  - `integration/openclaw/memory-clawdb/`
- Queue backends:
  - `zeromq` (default)
  - `kafka`
  - `inmemory` (test/dev)
- Durability:
  - WAL append + checksum replay validation
  - checkpoint metadata persisted as DataFrame + Parquet
- Safety:
  - lock ordering + timeout + watchdog cycle detection
  - ingest backpressure admission control with `503` fail-fast

## Performance Snapshot 📊

Benchmark date: `2026-03-15`  
Workload: `6000` ingests, `1200` cold searches, `1200` hot searches, `120` sessions, `8` tenants, concurrency `96`.

Reports:
- `Docs/benchmark_report_latest.json`
- `Docs/benchmark_report_inmemory.json`
- `Docs/benchmark_summary_2026-03-15.md`

| Stack | Ingest P50 | Ingest P95 | Throughput (ops/s) | Cold Search P50 | Cold Search P95 | Hot Search P50 | Hot Search P95 | Replay Startup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Official Memory | Not benchmarked in this repository run | Not benchmarked in this repository run | Not benchmarked in this repository run | Not benchmarked in this repository run | Not benchmarked in this repository run | Not benchmarked in this repository run | Not benchmarked in this repository run | Not benchmarked in this repository run |
| OpenViking + QMD | Not benchmarked in this repository run | Not benchmarked in this repository run | Not benchmarked in this repository run | Not benchmarked in this repository run | Not benchmarked in this repository run | Not benchmarked in this repository run | Not benchmarked in this repository run | Not benchmarked in this repository run |
| ClawDB (`zeromq`) | `155.08 ms` | `175.81 ms` | `621.41` | `3.06 ms` | `3.43 ms` | `0.57 ms` | `0.65 ms` | `18.97 ms` |
| ClawDB (`inmemory`) | `143.63 ms` | `165.41 ms` | `673.81` | `3.04 ms` | `3.41 ms` | `0.57 ms` | `0.64 ms` | `19.36 ms` |

Interpretation:
- Search latency is already strong.
- Ingest P95 is dominated by `fsync` under `WAL sync=always`.
- For lower write latency, evaluate `CLAWDB_WAL_SYNC=interval` with explicit durability trade-off acceptance.

## Architecture At A Glance 🏗️

```text
OpenClaw / API clients
  -> clawdb.api (FastAPI)
    -> clawdb.service (async orchestration)
      -> clawdb.wal (append + replay)
      -> clawdb.dataframes (hot-state DataFrames)
      -> clawdb.mq (ZeroMQ/Kafka/InMemory)
      -> clawdb.metadata (DataFrame + Parquet checkpoint metadata)
      -> clawdb.metrics (cache-hit + latency telemetry)
      -> clawdb.locks (deadlock-safe lock manager)
```

## Quick Start ⚙️

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
uvicorn clawdb.api:app --reload
```

Run tests:

```bash
python3 -m pytest -q tests
```

Run OpenClaw integration smoke:

```bash
./scripts/smoke_test_integration.sh
```

Run benchmark:

```bash
python3 scripts/benchmark_features.py \
  --messages 6000 \
  --sessions 120 \
  --tenants 8 \
  --ingest-concurrency 96 \
  --cold-searches 1200 \
  --hot-searches 1200 \
  --queue-backend zeromq \
  --out Docs/benchmark_report_latest.json
```

## OpenClaw Integration 🔌

- Plugin source: `integration/openclaw/memory-clawdb/`
- Install helper: `scripts/install_openclaw_integration.sh`
- Bootstrap helper: `scripts/bootstrap_openclaw.sh`
- Smoke test: `scripts/smoke_test_integration.sh`
- Integration guide: `Docs/openclaw_integration_test.md`

OpenClaw adapter routes are signed by default:
- `POST /v1/openclaw/memory/search`
- `POST /v1/openclaw/memory/get`

## Configuration 🛠️

Core runtime:
- `CLAWDB_QUEUE_BACKEND` (`zeromq` default)
- `CLAWDB_QUEUE_CONSUMERS`
- `CLAWDB_WAL_SYNC` (`always` or `interval`)
- `CLAWDB_WAL_SYNC_INTERVAL_MS`
- `CLAWDB_OPENCLAW_REQUIRE_SIGNATURE` (`true` default)

Backpressure:
- `CLAWDB_INGEST_BACKPRESSURE_LAG_THRESHOLD` (`20000` default)
- `CLAWDB_INGEST_BACKPRESSURE_MAX_WAIT_MS` (`250` default)
- `CLAWDB_INGEST_BACKPRESSURE_POLL_INTERVAL_MS` (`10` default)

Logging:
- `CLAWDB_SEARCH_LOG_ENABLED` (`true` default)

Storage:
- `CLAWDB_DATA_ROOT`
- `CLAWDB_METADATA_PARQUET_PATH` (`<data_root>/checkpoints/metadata.parquet` default)

## Reliability Notes 🛡️

- WAL records are append-only in normal operation.
- Acknowledged writes are WAL-first.
- Replay uses checkpoint + WAL tail reconstruction.
- Idempotency key support prevents duplicate ingest.
- Tenant/session isolation is enforced in search and memory-get paths.

## Project Status ✅

Active implementation with passing tests and integration smoke in this repository state.

If you are evaluating memory backends for OpenClaw, `clawdb` gives you a practical path: predictable integration now, and measurable optimization headroom next. 🌟
