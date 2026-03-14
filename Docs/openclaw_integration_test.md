# OpenClaw Integration Test Guide (clawdb memory mask)

## Goal
Use official OpenClaw with `memory-clawdb` plugin so all memory tool calls are routed to clawdb endpoints instead of builtin memory modules.

## 1) Bootstrap OpenClaw + plugin

```bash
./scripts/bootstrap_openclaw.sh
```

This performs:
- clone official `openclaw/openclaw` into `external/openclaw` (if missing)
- install Node dependencies via `pnpm`
- install `integration/openclaw/memory-clawdb` into `external/openclaw/extensions/memory-clawdb`
- generate `external/openclaw/openclaw.clawdb.config.example.json`

## 2) Start clawdb service

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
./scripts/run_local.sh
```

Service default endpoint: `http://127.0.0.1:8080`

## 3) Validate clawdb health

```bash
curl -s http://127.0.0.1:8080/v1/memory/health | jq
curl -s http://127.0.0.1:8080/v1/memory/metrics/cache-hit | jq
```

Default behavior:
- Queue backend defaults to `zeromq` for low-latency single-node event processing.
- OpenClaw adapter routes (`/v1/openclaw/memory/*`) require request signatures by default.

## 4) Point OpenClaw to memory-clawdb plugin

Use `external/openclaw/openclaw.clawdb.config.example.json` as your OpenClaw config baseline and ensure:
- `plugins.slots.memory = "memory-clawdb"`
- `plugins.entries.memory-core.enabled = false`
- `plugins.entries.memory-clawdb.enabled = true`
- `plugins.entries.memory-clawdb.config.baseUrl = "http://127.0.0.1:8080"`
- set `plugins.entries.memory-clawdb.config.apiKey` or export `CLAWDB_API_KEY` so plugin requests are signed even without embedding-provider keys

## 5) Functional smoke

1. Ingest memory content via clawdb API:
```bash
curl -s -X POST http://127.0.0.1:8080/v1/memory/messages \
  -H 'content-type: application/json' \
  -d '{"session_id":"demo","role":"user","content":"we decided to use clawdb as memory backend"}' | jq
```

2. Verify OpenClaw-compatible search path:
```bash
curl -s -X POST http://127.0.0.1:8080/v1/openclaw/memory/search \
  -H 'content-type: application/json' \
  -d '{"query":"memory backend","sessionKey":"demo"}' | jq
```

Note:
- Direct `curl` calls to OpenClaw adapter routes are expected to fail with `401` unless signed.
- Use the `memory-clawdb` plugin/CLI for normal integration tests, or set `CLAWDB_OPENCLAW_REQUIRE_SIGNATURE=false` only in local debug mode.

3. Verify OpenClaw-compatible get path:
```bash
curl -s -X POST http://127.0.0.1:8080/v1/openclaw/memory/get \
  -H 'content-type: application/json' \
  -d '{"relPath":"memory/demo.md","from":1,"lines":20}' | jq
```

## 6) Expected masking behavior
- Memory slot is owned by `memory-clawdb`.
- `memory_search` and `memory_get` tools resolve through clawdb HTTP endpoints.
- `clawdb-memory` CLI command (from plugin) uses clawdb for status/search/get.
- `clawdb-memory search` supports `--tenant-id` for tenant-scoped retrieval.
- Builtin memory-core is disabled in slot selection.
- Embedding auth is resolved through OpenClaw runtime auth profiles/provider config.
- Requests are signed (`x-openclaw-signature` + timestamp) and verified by clawdb.

## 7) One-command smoke test

```bash
./scripts/smoke_test_integration.sh
```

The script starts clawdb, writes a sample message, and validates OpenClaw plugin path through `clawdb-memory`.
The script also sets `CLAWDB_API_KEY` (default `clawdb-smoke-signing-key`) to satisfy strict signature checks.

## 8) Operational checks
- Monitor `/v1/memory/metrics/cache-hit` continuously.
- Confirm WAL progression in `data/wal/wal-00000001.log`.
- Confirm Parquet load-out in `data/parquet/...` partitions.
- For low-latency single-node mode, set `CLAWDB_QUEUE_BACKEND=zeromq`.
