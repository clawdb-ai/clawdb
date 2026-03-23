# ClawDB

[![OpenClaw Backend](https://img.shields.io/badge/OpenClaw-Memory_Backend-0A7EA4)](https://github.com/clawdb-ai/clawdb)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-0.1.0-2E8B57)](./pyproject.toml)

> **Fast memory is exciting. Trusted memory is useful. ClawDB is built to deliver both.**

[Quick Start](#quick-start) | [Benchmarks](#performance-snapshot-2026-03-15) | [OpenClaw Integration](#openclaw-integration)

---

## Why ClawDB

Most memory systems force a painful choice: low latency or reliable recovery.  
`clawdb` is an async Python memory backend for OpenClaw designed to keep both.

### What ClawDB Offers

| Capability | Description |
|---|---|
| **Async Service Model** | Async-by-default API/service/worker flow tuned for concurrent ingest + search |
| **WAL-First Durability** | Append-only WAL semantics with checksum-aware replay validation |
| **Hot-State Performance** | In-memory `pandas` DataFrames for low-latency retrieval paths |
| **Persistent Recovery** | DataFrame + Parquet checkpoint metadata for restart reconstruction |
| **Queue Flexibility** | `zeromq` (default), `kafka`, and `inmemory` backends |
| **Production Safety** | Tenant isolation, idempotent ingest, deadlock-safe locking, backpressure controls |
| **OpenClaw Compatibility** | Supports OpenClaw `memory_search` and `memory_get` adapter flows |

---

## Architecture

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

---

## Performance Snapshot (2026-03-15)

Workload: `6000` ingests, `1200` cold searches, `1200` hot searches, `120` sessions, `8` tenants, concurrency `96`.

Reports are generated locally under gitignored `Docs/`.

| Stack | Ingest P50 | Ingest P95 | Throughput (ops/s) | Cold Search P50 | Cold Search P95 | Hot Search P50 | Hot Search P95 | Replay Startup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Official Memory | Not benchmarked in this repository run | Not benchmarked in this repository run | Not benchmarked in this repository run | Not benchmarked in this repository run | Not benchmarked in this repository run | Not benchmarked in this repository run | Not benchmarked in this repository run | Not benchmarked in this repository run |
| OpenViking + QMD | Not benchmarked in this repository run | Not benchmarked in this repository run | Not benchmarked in this repository run | Not benchmarked in this repository run | Not benchmarked in this repository run | Not benchmarked in this repository run | Not benchmarked in this repository run | Not benchmarked in this repository run |
| ClawDB (`zeromq`) | `155.08 ms` | `175.81 ms` | `621.41` | `3.06 ms` | `3.43 ms` | `0.57 ms` | `0.65 ms` | `18.97 ms` |
| ClawDB (`inmemory`) | `143.63 ms` | `165.41 ms` | `673.81` | `3.04 ms` | `3.41 ms` | `0.57 ms` | `0.64 ms` | `19.36 ms` |

Observations:
- Search latency is already strong.
- Ingest P95 is dominated by `fsync` under `CLAWDB_WAL_SYNC=always`.
- Lower write latency can be traded with `CLAWDB_WAL_SYNC=interval`.

---

## Quick Start

```bash
uv sync --extra dev
uv run python -m uvicorn clawdb.api:app --reload
```

### Python Dependencies

`pyproject.toml` is the source of truth for Python dependencies and is installable with `uv sync --extra dev`.

- Runtime: `fastapi`, `uvicorn`, `pydantic`, `numpy`, `scipy`, `pandas`, `pyarrow`, `prometheus-client`, `httpx`, `pyzmq`, `typing-extensions`
- Dev extra: `pytest`, `pytest-asyncio`

Local validation note:
- `Docs/`, `.venv/`, and `tests/` are local development artifacts and stay gitignored.
- If your local checkout includes `tests/`, run:

```bash
uv run pytest -q tests
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

---

## OpenClaw Integration

### Paths and Helpers

- Plugin source: `integration/openclaw/memory-clawdb/`
- Skill pack: `integration/openclaw/skills/clawdb-first-memory/`
- Install helper: `scripts/install_openclaw_integration.sh`
- Bootstrap helper: `scripts/bootstrap_openclaw.sh`
- Smoke test: `scripts/smoke_test_integration.sh`
- Local integration guide: `Docs/openclaw_integration_test.md` (gitignored local doc)

### Signed Adapter Routes

- `POST /v1/openclaw/memory/search`
- `POST /v1/openclaw/memory/get`

### Skill-Driven Setup

```bash
bash integration/openclaw/skills/clawdb-first-memory/scripts/setup_first_choice_memory.sh \
  --repo-root "$(pwd)" \
  --openclaw-dir "$(pwd)/external/openclaw" \
  --profile clawdb-test \
  --base-url http://127.0.0.1:8080 \
  --verify
```

---

## Configuration

### Core Runtime

- `CLAWDB_QUEUE_BACKEND` (`zeromq` default)
- `CLAWDB_QUEUE_CONSUMERS`
- `CLAWDB_WAL_SYNC` (`always` or `interval`)
- `CLAWDB_WAL_SYNC_INTERVAL_MS`
- `CLAWDB_OPENCLAW_REQUIRE_SIGNATURE` (`true` default)

### Backpressure

- `CLAWDB_INGEST_BACKPRESSURE_LAG_THRESHOLD` (`20000` default)
- `CLAWDB_INGEST_BACKPRESSURE_MAX_WAIT_MS` (`250` default)
- `CLAWDB_INGEST_BACKPRESSURE_POLL_INTERVAL_MS` (`10` default)

### Logging and Storage

- `CLAWDB_SEARCH_LOG_ENABLED` (`true` default)
- `CLAWDB_DATA_ROOT`
- `CLAWDB_METADATA_PARQUET_PATH` (`<data_root>/checkpoints/metadata.parquet` default)

---

## Reliability Notes

- WAL records are append-only in normal operation.
- Acknowledged writes are WAL-first.
- Replay uses checkpoint + WAL tail reconstruction.
- Idempotency keys prevent duplicate ingest.
- Tenant/session isolation is enforced in search and memory-get paths.
- Schema migrations are versioned and WAL-preserving via `python -m clawdb.migrate`.
- Startup can auto-migrate older parquet schemas (`CLAWDB_SCHEMA_AUTO_MIGRATE=true` by default).

---

## Project Status

Active implementation with passing tests and integration smoke in this repository state.

*Predictable integration now, measurable optimization headroom next.*
