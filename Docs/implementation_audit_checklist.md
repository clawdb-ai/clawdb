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

## G. Source Section 7.1 Roadmap Compliance (Strict)

### G1. Phase 1: Core Storage Layer
- [x] Buffer Layer implemented with in-memory pandas (`messages_df`, `capsules_df`, `cache_index_df`, `sessions_df`, `snapshots_df`).
- [x] WAL Engine implemented and append-only semantics preserved.
- [x] DB schema represented as DataFrame schemas + Parquet partitions (no SQLite runtime dependency).
- [x] High-performance async MQ configured (`zeromq` default, additional backends available).

### G2. Phase 2: Trie + Topic Detection
- [x] Trie tree implementation exists and is updated during ingest/replay.
- [x] Capsule manager flow exists (refresh/materialize/present capsule cards).
- [x] Auto topic classification uses Gauss-Ewens process implementation.
- [x] Folder judger assigns capsule level (`L0/L1/L2`) from topic growth.

### G3. Phase 3: Vector Retrieval
- [x] HNSW-style vector index path implemented.
- [x] BM25 lexical retrieval implemented.
- [x] n-top-k candidate gather implemented.
- [x] Hybrid fusion implemented and used by search path.

### G4. Phase 4: IM Presentation Layer
- [x] Linear IM presentation endpoint available.
- [x] Capsule cards presentation endpoint available.
- [x] Forum-style presentation endpoint available.
- [x] Index management endpoints (status/rebuild) available.

### G5. Phase 5: Session Lifecycle
- [x] Session manager implemented (session table and lifecycle operations).
- [x] Snapshot chain with WAL sequence traceability implemented.
- [x] Fork logic implemented and exposed via API.
- [x] Spawn logic implemented and exposed via API.

### G6. Phase 6: Validation Coverage
- [x] Unit tests cover ingest/search/replay/MQ/OpenClaw adapter.
- [x] Integration smoke test covers OpenClaw plugin path end-to-end.
- [x] Load test script exists and runs successfully.
- [x] Chaos test script exists and verifies recovery/read consistency.

### G7. Additional Mandatory Constraints
- [x] Async execution model is used across API/service/queue interfaces.
- [x] Deadlock prevention includes lock ordering, timeout, and watchdog scanning.
- [x] Every ingested message is stored in DataFrame memory and persisted to Parquet on flush/checkpoint.
- [x] Cache-hit report endpoint and telemetry counters are present and queryable.

## H. Official OpenClaw IM Channel And Schema Coverage
- [x] Official IM channel capability matrix extracted from `external/openclaw/extensions/*` plugin capabilities (`chatTypes`).
- [x] DM/group/thread canonical schema derived from OpenClaw `MsgContext` and routing/session-key core modules.
- [x] ClawDB ingest schema expanded for OpenClaw channel metadata:
  - [x] `channel`, `chat_type`, `account_id`
  - [x] `from_id`, `to_id`, `sender_id`, `sender_name`, `sender_username`, `sender_e164`
  - [x] `group_id`, `group_subject`, `group_channel`, `group_space`
  - [x] `native_channel_id`, `message_thread_id`, `thread_parent_id`, `reply_to_id`
- [x] Topic organization schema expanded and persisted:
  - [x] `topic_parent_id`, `topic_path`, `topic_source`, `topic_confidence`
- [x] Search/filter schema expanded with IM metadata dimensions:
  - [x] `channel`, `chat_type`, `group_id`, `topic_id`, `message_thread_id`

## I. Skill-Based Onboarding Coverage
- [x] OpenClaw/Codex-compatible skill package exists: `integration/openclaw/skills/clawdb-first-memory/SKILL.md`.
- [x] Skill includes deterministic setup script for first-choice memory slot selection.
- [x] Skill includes verification script for `status/search/get` flow validation.
- [x] Skill includes installer script to publish skill into OpenClaw state skills directory.
- [x] Skill references migration workflow and OpenClaw auth/signing compatibility notes.

## J. Schema Migration And Upgrade Safety
- [x] Versioned schema migration module exists (`src/clawdb/migrate.py`).
- [x] CLI path is available for users (`python -m clawdb.migrate`).
- [x] Dry-run planning mode emits actionable migration plan output.
- [x] Migration supports backup-before-write and configurable backup path.
- [x] Migration rewrites all parquet tables to canonical schema shape (`messages/capsules/cache_index/sessions/snapshots`).
- [x] Migration updates schema metadata version checkpoint in metadata parquet.
- [x] Migration is idempotent on rerun.
- [x] Startup auto-migration preflight is integrated and test-covered.
- [x] WAL remains intact during migration (no reset/truncate in migration flow).
- [x] Post-migration history remains readable from all memory/context presentation levels.
