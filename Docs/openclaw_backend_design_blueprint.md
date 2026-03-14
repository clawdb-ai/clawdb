# OpenClaw Backend Design Blueprint (Integrated, English)

## Document Status
- Version: `v1.2.0-blueprint`
- Date: `2026-03-14`
- Status: `Review Draft (Pre-Implementation)`
- Source Baseline: `~/Downloads/openclaw_backend_design.md` (imported conceptually)

## Scope And Intent
This document is the implementation blueprint to validate before coding. It preserves the architectural intent of the imported OpenClaw backend design while applying the required constraints:

1. Documentation language is English and future patches are expected in English.
2. All Go/Rust implementation assumptions are replaced with Python implementations.
3. Runtime data path prioritizes in-memory pandas DataFrames and Parquet load-in/load-out.
4. WAL remains mandatory for durability and recovery.
5. Design aligns with official OpenClaw memory contracts and can replace memory-related modules, including a migration path that can also replace OpenViking + QMD memory orchestration.
6. Cache-hit reporting is consistently measured, checked, and enforced operationally.
7. All task execution paths are asynchronous by default (sync work only for minimal critical sections).
8. High-performance message queue is required for event-driven decoupling and throughput.
9. Deadlock prevention, detection, and recovery controls are mandatory.

If this document conflicts with the original imported markdown, this document takes precedence.

---

## 1. Executive Summary

OpenClaw Backend provides a high-throughput, low-latency memory system for agent workflows with layered context (`L0/L1/L2`), hybrid retrieval (lexical + semantic), and durable write semantics.

Core strategy:
- Use Python services for orchestration and retrieval logic.
- Keep hot working sets in pandas DataFrames.
- Persist snapshots and segments as Parquet.
- Protect writes with WAL-first semantics.
- Execute application tasks asynchronously with backpressure controls.
- Use a high-performance message queue for internal event fan-out.
- Apply strict deadlock-avoidance rules (lock ordering, timeouts, watchdogs).
- Maintain strict compatibility with OpenClaw memory module contracts.
- Track cache-hit quality as a first-class SLI.

Target outcomes:
- P50 read latency: `< 20 ms` for hot memory queries.
- P95 read latency: `< 80 ms` for mixed hot/cold queries.
- Write ingest acknowledgment (WAL append): `< 10 ms` P95.
- Zero acknowledged-write loss under process crash.

---

## 2. Architecture Overview

## 2.1 Layered System

```text
Client/API
  -> Memory Service API (Python/FastAPI)
    -> Async Task Orchestrator (asyncio task groups / worker pools)
    -> Domain Layer
       - Topic Router (Trie + Bayesian topic update)
       - Capsule Manager (L0/L1/L2)
       - Retrieval Engine (BM25 + Vector + Fusion)
    -> Data Layer
       - Message Queue Adapter (Kafka/Redpanda/NATS JetStream)
       - Cache Manager (in-memory DataFrames)
       - WAL Manager (append, fsync policy, replay)
       - Parquet Store (segment snapshots/load)
       - Metadata Store (PostgreSQL)
```

## 2.2 Non-Negotiable Runtime Rules
- No production Go or Rust services in this blueprint.
- All non-trivial tasks (index updates, compaction, capsule refresh, replication events) run asynchronously.
- High-performance MQ is mandatory for async event distribution and replay-safe pipelines.
- All mutable memory state is reconstructed from WAL + Parquet checkpoints.
- In-memory state format is pandas DataFrame for all high-traffic memory tables.
- Deadlock safeguards (global lock hierarchy + timeout + watchdog) are release-blocking requirements.
- Any API compatibility gap with official OpenClaw memory contract is treated as a release blocker.

---

## 3. Python Technology Stack

## 3.1 Core Runtime
- Python: `3.11+`
- API framework: `FastAPI` + `uvicorn`/`gunicorn`
- Validation: `pydantic v2`
- Dataframe engine: `pandas`
- Parquet IO: `pyarrow` (preferred) or `fastparquet`
- WAL serialization: JSONL or MsgPack records with monotonic sequence IDs
- Async runtime: `asyncio` + `anyio`
- Async I/O clients: `httpx`, async DB driver variants as needed
- DB metadata: PostgreSQL (`psycopg` / `SQLAlchemy` optional)
- Background jobs: async worker pools; distributed workers only if queue backlog requires horizontal fan-out
- Message queue: `Redpanda/Kafka` (primary) or `NATS JetStream` (low-latency alternative)
- Metrics: Prometheus client + OpenTelemetry

## 3.2 Retrieval Components
- Lexical retrieval: Python BM25 implementation (or `rank_bm25`)
- Vector retrieval: adapter interface (FAISS/Qdrant/Milvus selectable)
- Fusion: weighted score fusion + rerank hooks

## 3.3 Async Task And Queue Model
- Producer path: API writes WAL, then publishes immutable event envelopes to MQ.
- Consumer path: idempotent async workers apply downstream jobs (indexing, cache warming, compaction triggers).
- Queue contracts include: `event_id`, `wal_seq`, `tenant_id`, `session_id`, `event_type`, `created_at`.
- Exactly-once is approximated via at-least-once delivery + idempotent consumers keyed by `event_id`.
- Backpressure policy:
  - Pause non-critical consumers when lag exceeds threshold.
  - Prioritize write-path integrity and replay consumers.

---

## 4. Data Model (DataFrame + Parquet First)

## 4.1 Canonical In-Memory Tables

### `messages_df`
- `message_id` (str)
- `session_id` (str)
- `role` (category)
- `content` (str)
- `ts` (datetime64[ns, UTC])
- `topic_id` (str)
- `embedding_ref` (str | null)
- `capsule_level` (`L0|L1|L2`)

### `capsules_df`
- `capsule_id` (str)
- `session_id` (str)
- `topic_id` (str)
- `summary` (str)
- `level` (`L0|L1|L2`)
- `score` (float64)
- `updated_at` (datetime64[ns, UTC])

### `cache_index_df`
- `key` (str)
- `entity_type` (str)
- `entity_id` (str)
- `last_access` (datetime64[ns, UTC])
- `hit_count` (int64)
- `miss_count` (int64)

## 4.2 Parquet Layout

```text
data/
  parquet/
    messages/dt=YYYY-MM-DD/part-*.parquet
    capsules/dt=YYYY-MM-DD/part-*.parquet
    cache_index/dt=YYYY-MM-DD/part-*.parquet
  wal/
    wal-00000001.log
    wal-00000002.log
  checkpoints/
    ckpt-<seq>.json
```

Rules:
- Partition by date for efficient bounded reads.
- Column pruning and predicate pushdown are mandatory for cold loads.
- Every checkpoint records the last applied WAL sequence.

---

## 5. Durability: WAL Kept Intact

## 5.1 Write Path (Authoritative)
1. Validate request.
2. Create WAL record with sequence ID and checksum.
3. Append WAL record.
4. Apply mutation to in-memory DataFrames.
5. Publish post-commit event (`wal_seq`) to high-performance MQ.
6. Asynchronously flush segment updates to Parquet.
7. Return ACK only after successful WAL append (and configured fsync policy).

## 5.2 Recovery Path
1. Load latest checkpoint metadata.
2. Load DataFrames from referenced Parquet snapshots.
3. Replay WAL records with sequence `> checkpoint_seq`.
4. Rebuild cache indexes and retrieval indexes.
5. Expose readiness only after replay integrity checks pass.

## 5.3 WAL Policies
- `sync=always` for strongest durability mode.
- `sync=interval` (e.g., 10-50 ms) only if product explicitly accepts bounded risk.
- WAL compaction allowed only after checkpoint confirms durable Parquet snapshot and metadata transaction.

---

## 6. OpenClaw Compatibility And Replacement Design

## 6.1 Compatibility Goal
The backend must behave as a drop-in replacement for official OpenClaw memory-related modules from the API and behavior perspective.

## 6.2 Adapter Boundaries

### Memory Contract Adapter
- Maps OpenClaw memory API payloads to internal Python domain models.
- Enforces required fields, status codes, and error semantics.
- Keeps versioned compatibility profiles (e.g., `openclaw/v1`).

### Retrieval Contract Adapter
- Exposes equivalent query semantics for top-k retrieval, filters, and score explainability.

### Session/Context Adapter
- Ensures session scoping and tiered memory behavior (`L0/L1/L2`) are compatible with upstream expectations.

## 6.3 OpenViking + QMD Replacement Path
- Step 1: Deploy Python memory service behind contract adapter in shadow mode.
- Step 2: Mirror traffic and compare outputs (`hit@k`, latency, cache-hit ratio).
- Step 3: Switch read path to Python service.
- Step 4: Switch write path with WAL durability validation.
- Step 5: Decommission old memory pipeline after parity SLO window is met.

---

## 7. Cache-Hit Reporting (Must Be Consistently Checked)

## 7.1 Required Metrics
- `memory_cache_hit_ratio_1m`
- `memory_cache_hit_ratio_5m`
- `memory_cache_hits_total`
- `memory_cache_misses_total`
- `memory_cache_evictions_total`
- `memory_cache_lookup_latency_ms`

## 7.2 Definitions
- `cache_hit_ratio = hits / (hits + misses)`
- Separate by key dimensions: `session_id`, `tenant_id`, `capsule_level`, `query_type`

## 7.3 Operational Rules
- Emit hit/miss metric on every cache lookup.
- Add per-request debug field in logs: `cache_hit=true|false`.
- Reject release if cache-hit metric pipeline is missing or malformed.
- Alert when 5-minute cache-hit ratio drops below agreed threshold (default `0.80`) for sustained window.

## 7.4 Verification During Rollout
- Compare cache-hit ratios between legacy and replacement systems.
- Track divergence report daily until parity is stable.

---

## 8. API Surface (Python Service)

## 8.1 Endpoints (Blueprint)
- `POST /v1/memory/messages`
- `POST /v1/memory/search`
- `POST /v1/memory/capsules/refresh`
- `GET /v1/memory/sessions/{session_id}`
- `GET /v1/memory/health`
- `GET /v1/memory/metrics/cache-hit`

## 8.2 Response Guarantees
- WAL sequence number included in write acknowledgments.
- Retrieval responses include source tier (`L0/L1/L2`) and score breakdown.
- Health endpoint reports `wal_replay_lag`, `checkpoint_seq`, and `cache_hit_ratio_5m`.

---

## 9. Performance Strategy

## 9.1 Hot Path
- Keep recent session slices in memory-resident DataFrames.
- Use vectorized filtering for session/topic/time predicates.
- Avoid row-wise Python loops in retrieval preprocessing.

## 9.2 Cold Path
- Use Parquet predicate pushdown on partition columns.
- Materialize only required columns before join/fusion steps.
- Maintain background compaction to reduce small-file overhead.

## 9.3 Concurrency
- Async request handling with bounded worker pools.
- Single-writer or partitioned-writer WAL strategy to avoid sequence conflicts.
- Event consumers use partition affinity to preserve ordering where required.
- Avoid blocking calls on event loop; isolate CPU-heavy tasks in process pools.
- Enforce bounded retries with jittered backoff to prevent retry storms.
- Backpressure on ingest when WAL queue depth exceeds threshold.

---

## 10. Deadlock Prevention Strategy
- Define and enforce global lock acquisition order:
  - `session_lock -> topic_lock -> capsule_lock -> index_lock`
- Prefer lock-free or compare-and-swap patterns for cache counters where feasible.
- Use timeout-based lock acquisition with structured failure telemetry.
- Use fine-grained locks; avoid coarse global mutexes in hot paths.
- For cross-resource operations, use transactional outbox/event choreography instead of nested blocking waits.
- Deadlock watchdog scans lock wait graphs and emits critical alerts when cycle signatures are detected.

## 10.1 Failure Handling
- On suspected deadlock, fail fast for non-critical tasks and reschedule safely.
- Critical writer path enters degraded mode with strict admission control rather than indefinite blocking.
- Incident payload must include lock owners, waiters, and operation IDs.

---

## 11. Security And Reliability
- WAL files checksum-validated on replay.
- Idempotency key support for message ingest.
- Tenant/session isolation enforced in every query path.
- Structured audit logs for memory mutation operations.
- Queue ACLs and topic-level authz enforced for producer/consumer identities.
- Optional encryption-at-rest for Parquet and WAL directories.

---

## 12. Test And Validation Matrix

## 12.1 Functional
- Contract tests against official OpenClaw memory API behavior.
- Snapshot parity tests vs legacy OpenViking+QMD outputs.

## 12.2 Durability
- Crash-after-append simulation: acknowledged writes must survive.
- WAL corruption simulation: detect, isolate, and fail-safe.

## 12.3 Performance
- Load tests on million-message datasets.
- Cache-hit sensitivity benchmarks across session churn profiles.
- Async queue lag benchmarks under burst traffic.

## 12.4 Observability
- Validate all cache-hit and WAL health metrics in staging.
- Enforce dashboards + alerts before production cutover.
- Validate deadlock watchdog metrics and alert routing.

---

## 13. Implementation Guardrails
- Do not start coding until this blueprint is approved.
- Keep docs, code comments, and commit messages in English.
- Any new module must declare:
  - OpenClaw compatibility scope
  - WAL interaction policy
  - DataFrame/Parquet data ownership
  - Cache-hit metric emission points
  - Async execution model and queue topics
  - Lock usage and deadlock-safety guarantees

---

## 14. Review Checklist (For Your Double-Check)
- [ ] English-only wording is acceptable for your team workflow.
- [ ] Python-only replacement is acceptable (no Go/Rust runtime dependency).
- [ ] DataFrame + Parquet approach fits expected memory size and cost envelope.
- [ ] WAL durability policy matches your risk tolerance.
- [ ] OpenClaw memory contract coverage is complete enough for drop-in replacement.
- [ ] OpenViking+QMD replacement sequencing is acceptable.
- [ ] Cache-hit report definitions/thresholds are acceptable.
- [ ] Async-by-default task model is acceptable.
- [ ] MQ technology choice and throughput targets are acceptable.
- [ ] Deadlock prevention and watchdog policy are acceptable.

---

## 15. Explicit Delta From Imported Source
- Language changed from mixed Chinese to English.
- Go/Rust-centric implementation details replaced by Python stack and module contracts.
- Storage strategy made explicit as DataFrame hot state + Parquet persistence + WAL durability.
- Compatibility/migration framing tightened to official OpenClaw replacement goals.
- Cache-hit reporting elevated from optional monitoring to mandatory release criterion.
- Async-first execution and high-performance MQ were made explicit architecture constraints.
- Deadlock prevention, detection, and recovery controls were promoted to mandatory design requirements.
