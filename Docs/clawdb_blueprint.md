# ClawDB Blueprint

Updated: 2026-03-23

## Purpose

This document captures the intended ClawDB architecture discussed in this thread, plus the gap analysis against:

- `volcengine/OpenViking`
- `tobi/qmd`
- current `clawdb` implementation

The goal is to preserve the core ClawDB design while borrowing proven engineering practice from OpenViking and QMD.

## Core Principle

ClawDB should not reuse OpenViking's L0/L1/L2 semantics.

OpenViking defines:

- L0 = abstract
- L1 = overview
- L2 = full details

ClawDB defines:

- L0 = global raw-message plus belief/soul layer
- L1 = DM and group session layer, including public/private projections and time-window summaries
- L2 = topic and capsule layer organized over all raw messages

These are different ontologies. Reusing OpenViking naming without changing ClawDB storage semantics will encode the wrong system.

## Intended ClawDB Tier Model

### L0

L0 is the total-system layer.

It is not only `agent.md` or `soul.md`. It also includes final stable beliefs in soul, plus access to the full raw-message base.

L0 should contain:

- one global raw-message store
- system-level beliefs
- system-level soul / final beliefs
- global abstracts and higher-level summaries over the whole system

Operationally:

- L0 has all raw messages available
- L0 also has abstractions over those raw messages
- L0 is not only a summary tier; it is the top global control and worldview layer

### L1

L1 is the session layer.

It contains:

- DM sessions
- group sessions

For group sessions:

- if a particular user ID interacts with the bot inside a group, that interaction is also appended into that user's 1:1 DM-with-bot dataframe
- therefore some interactions are intentionally stored twice as projections

For the per-user 1:1 dataframe:

- direct DM messages must be marked `private`
- group-origin mirrored messages must be marked `public`
- each mirrored message must also carry the group chat ID
- for Feishu this means user identity such as `ou_*` and group identity such as `oc_*` must be preserved

Each L1 session should have summarization windows:

- daily
- weekly
- monthly
- quarterly
- yearly
- lifetime

These time-span summaries should be vectorized independently.

### L2

L2 is the topic and capsule layer built after L1 is well organized.

All messages should be automatically organized into small topics.

For each new message:

- it joins an existing topic, or
- it creates a new topic

The core topic process should use the ClawDB Gauss-Ewens design, not OpenViking's or QMD's ontology.

If a topic accumulates over 100K raw-message information, where 100K excludes metadata columns and IDs, then:

- that 100K raw-message body forms a capsule
- capsules under the same topic should have mutual pointers
- one topic should live in one dataframe
- each topic should have its own vectorization
- each capsule should have its own vectorization

## Retrieval Design

Any search should use hybrid retrieval:

- BM25 on raw text with weight `0.3`
- vectorization with weight `0.7`

A fuzzy search may hit:

- L0
- L1 sessions
- L2 topics
- specific capsules

Retrieval should be able to return the best match across all these layers.

## Engineering Guidance To Borrow

Use OpenViking and QMD as engineering references, not as the semantic model for ClawDB tiers.

Take from OpenViking:

- source-of-truth vs derived-index discipline
- async semantic pipelines
- crash recovery / consistency patterns
- memory extraction organization

Take from QMD:

- persistent FTS/vector storage layout
- BM25 + vector + fusion + rerank engineering
- chunking discipline
- measurable evaluation and retrieval tests

Do not take from either:

- their ontology for L0/L1/L2

## Current Machine Observations

As of 2026-03-23, the live machine state shows:

- OpenViking is running from the official upstream checkout with local machine-side service/config glue
- QMD is running from the official upstream checkout with local machine-side service/config glue
- the current active ClawDB message parquet tree is tiny and currently contains only one stored message
- the OpenViking mirror on this machine also currently contains only one message line for the active session export

This means the current local live dataset is not the full intended historical corpus.

## Current ClawDB Implementation Reality

Current ClawDB is not yet the above design.

### What exists now

- one in-memory global `messages_df`
- parquet persistence grouped by channel/session file path
- a lightweight topic auto-assignment path
- a lightweight in-process hybrid retrieval path
- WAL/checkpoint machinery

### What does not exist yet

- a true L0 global belief/soul layer
- L1 DM/group projection fan-out
- `public` / `private` lineage projection
- time-window rollups
- topic-owned persistent dataframes
- real capsule rollover and capsule linking
- persistent lexical and vector indexes for raw/topic/capsule/session layers
- robust retrieval evaluation suite

## Critical Findings

### 1. ClawDB does not implement the intended tier semantics

Current `capsule_level` is assigned by message-count cutoffs, not by the intended global/session/topic meaning.

### 2. ClawDB has no true L0 store

There is no dedicated global belief table, soul table, or belief finalization pipeline.

### 3. L1 public/private mirror rules are missing

Current ingest schema has raw identity fields such as:

- `from_id`
- `to_id`
- `sender_id`
- `group_id`

But it does not implement:

- DM/group dual-projection writes
- `public` vs `private` lineage
- group-to-DM mirrored session views

### 4. Time-window summarization is missing

There is no implemented pipeline for:

- daily
- weekly
- monthly
- quarterly
- yearly
- lifetime

summary generation or vectorization.

### 5. The current Gauss-Ewens implementation is only a lightweight placeholder

The current topic model uses hashed token vectors and per-topic running means. It is not yet a complete topic lifecycle system.

Missing pieces include:

- topic repair
- topic merge/split
- topic reparenting
- persistent topic-owned storage
- topic-level vector index management

### 6. Current capsules are not real capsules

Current capsule refresh logic is a small session summary, not a topic-based 100K raw-message capsule lifecycle.

Missing pieces include:

- per-topic capsule accumulation
- threshold rollover
- capsule mutual pointers
- capsule lineage
- capsule vectorization

### 7. Retrieval is materially weaker than target design

Current ClawDB retrieval:

- builds transient retrieval docs from messages
- rebuilds in-process BM25/vector state per query
- does not maintain a persistent lexical index
- does not maintain a persistent vector index for topics/capsules/sessions
- does not yet use the exact target weighting contract as the system default

### 8. Validation maturity is weak

Current ClawDB tests mainly cover:

- export
- storage layout

It does not currently have a strong retrieval evaluation harness comparable to QMD's search-quality tests.

## Compare Matrix

| Concern | OpenViking | QMD | ClawDB now | Judgment |
|---|---|---|---|---|
| Tier semantics | Implemented, but as abstract/overview/details | No L0/L1/L2 ontology | Misapplied as size buckets | Redesign required |
| Global L0 raw+belief layer | Partial memory model, not one global raw DF | No | No | Missing |
| L1 DM/group projections | Session structures exist | Collection/document oriented | Raw fields only | Missing |
| Public/private group-to-DM duplication | No native chat-platform fan-out | No | No | Missing |
| Time-window summaries | No native daily/week/month rollups | No native rollups | No | Missing |
| Topic layer | No ClawDB-style topic layer | No ClawDB-style topic layer | Placeholder only | Replace |
| Capsule lifecycle | No ClawDB-style capsules | No ClawDB-style capsules | Trivial session summary only | Replace |
| Persistent lexical index | Not the main focus | Yes | No | Missing |
| Persistent vector index | Yes, derived index model | Yes | No durable multi-tier store | Missing |
| Hybrid retrieval | Hierarchical vector + rerank | Strong BM25 + vector + fusion + rerank | Lightweight in-process hybrid | Inadequate |
| Async semantic/index pipeline | Strong | Strong index/embed pipeline | Queue exists, downstream work thin | Partial |
| Crash recovery / consistency | Strong | Durable SQLite-based store | WAL/checkpoint only | Partial |
| Research depth | Very deep | Solid and measurable | Weak | Weak |
| Evaluation maturity | Strong docs/research | Strong tests with thresholds | Minimal | Weak |

## What OpenViking Already Solves Well

- source-of-truth vs derived-index philosophy
- async semantic queue architecture
- memory extraction organization
- crash recovery and lock discipline
- hierarchical retrieval over structured context trees

## What QMD Already Solves Well

- persistent FTS5 lexical index
- persistent vector index
- hybrid retrieval with rank fusion
- reranking on chunked content
- measurable retrieval quality tests

## What ClawDB Must Own Itself

ClawDB needs its own first-class schema and should not be modeled as a thin OpenViking or QMD wrapper.

Required first-class entities include:

- `raw_global_messages`
- `l0_beliefs`
- `l0_soul`
- `l1_session_messages`
- `l1_session_rollups`
- `topics`
- `topic_messages`
- `capsules`
- `capsule_links`
- `embedding_index_metadata`

## Missing Specification Items

The following points were not fully specified yet and must be defined before implementation can be trusted.

### Identity and projection

- [x] Canonical identity model for mirrored writes
  Each inbound message now materializes one authoritative `raw_global` row plus deterministic session projections keyed from the same origin.
- [x] Stable `origin_message_id` for all projections
  `origin_message_id` resolves in this order: explicit origin, platform/account/platform message ID, then request `message_id`.
- [x] `projection_kind` and `projection_scope`
  Supported projection kinds are `raw_global`, `private_dm`, `group_public`, and `dm_mirror_public`; scopes are `global`, `dm:<account>:<user>`, and `group:<account>:<chat>`.
- [x] Exact Feishu `ou_*` / `oc_*` mapping rules
  Feishu user fields normalize to `feishu_user:<raw>` and reject `oc_*`; Feishu chat fields normalize to `feishu_chat:<raw>` and reject `ou_*`; account keys normalize to `feishu_account:<raw>`.

### Edit and delete semantics

- [x] What happens when a raw group message is edited
  Editing a raw group message rewrites the raw row and every live projection row sharing its `origin_message_id`, preserves the original `ts`, and advances `updated_at`.
- [x] What happens when a raw group message is deleted
  Deleting a raw group message tombstones the raw row and every projection row by setting `message_state=deleted`, blanking visible content, and recording `deleted_at`.
- [x] How mirrored DM projections update
  Group-origin mirror rows in the per-user DM scope track the same `origin_message_id` as the group view and receive the same edit/delete mutations.
- [x] How summaries, topics, capsules, and vectors invalidate and rebuild
  Message mutations clear search/vector caches, rebuild the in-memory topic/trie state from authoritative raw rows, and refresh affected session capsule summaries.

### Summary lifecycle

- [x] Recompute triggers for daily summaries
  Any message upsert, edit, or delete touching an L1 projection session rebuilds that session's daily rollup rows from the current live projection rows.
- [x] Recompute triggers for weekly summaries
  The same projection-session mutation path also rebuilds the affected session's ISO-week rollup rows on Monday-based UTC boundaries.
- [x] Recompute triggers for monthly summaries
  Monthly rollups rebuild from the same mutation trigger using UTC calendar-month boundaries per L1 session.
- [x] Recompute triggers for quarterly summaries
  Quarterly rollups rebuild from the same mutation trigger using UTC calendar-quarter boundaries per L1 session.
- [x] Recompute triggers for yearly summaries
  Yearly rollups rebuild from the same mutation trigger using UTC calendar-year boundaries per L1 session.
- [x] Recompute triggers for lifetime summaries
  Lifetime rollups rebuild from the same mutation trigger over the full live projection history of the affected L1 session.
- [x] Exact summary storage and vectorization contract
  L1 rollups are materialized as `session_rollups` rows keyed by `(tenant_id, session_id, window_kind, window_key)` with UTC bucket bounds, source coverage timestamps, message and character counts, deterministic summary text, and independent vector payload fields `vector_text`, `vector_ref`, `vector_dim`, and `vector_json`.

### Topic lifecycle

- [x] Topic merge policy
- [x] Topic split policy
- [x] Topic drift correction
- [x] Topic reparenting
- [x] Topic compaction policy
- [x] Topic vector refresh policy

### Capsule lifecycle

- [x] Exact 100K threshold accounting rule
  Capsule accounting uses only normalized raw `content` characters from authoritative `raw_global` rows; metadata columns, IDs, vectors, summaries, and separator text do not count toward the `100000` threshold, and raw messages are never split across capsules.
- [x] Capsule rollover behavior
  Each canonical topic maintains one chronological open capsule tail; the tail seals as soon as its cumulative raw-body count reaches or exceeds `100000`, and the next raw message starts the next capsule. Topics below threshold still expose one open tail capsule.
- [x] Capsule pointer structure
  Every capsule row is keyed by stable `(tenant_id, canonical topic_id, capsule_ordinal)` and stores `capsule_id`, `topic_path`, first/last origin IDs, source coverage timestamps, ordered source message IDs, and a structured `pointer_json`.
- [x] Capsule back-links and forward-links
  Capsule lineage is explicit: `prev_capsule_id` and `next_capsule_id` link adjacent capsules, while `back_link_ids_json` and `forward_link_ids_json` carry the full ordered topic-local lineage.
- [x] Capsule vector refresh policy
  Capsule vectors are derived from deterministic capsule summary text; any edit, delete, or boundary change that alters the ordered source raw rows regenerates `source_hash`, `vector_ref`, `vector_json`, and `updated_at`.
- [x] Capsule rebuild contract from source raw messages
  `capsules` are a derived table rebuilt only from authoritative raw rows plus the current canonical topic mapping; deleting or backfilling the capsule parquet must reproduce the same capsule boundaries, lineage links, summaries, and vector references from raw source.

### Retrieval contract

Search uses one candidate pool spanning:

- a synthesized L0 scope abstract over the authoritative raw-message scope
- L1 session rollup windows
- L2 canonical topics
- L2 capsules
- raw messages from the authoritative global raw store

Cross-tier ranking uses one normalized score for every candidate. Sort by final score descending; exact ties break toward the more specific surviving object in this order:

- raw messages
- capsules
- topics
- L1 session rollups
- L0 scope abstracts

Retrieval modes are:

- `hybrid`: lexical `0.3`, vector `0.7`
- `lexical`: lexical `1.0`, vector `0.0`
- `vector`: lexical `0.0`, vector `1.0`

Reranking is optional, not mandatory. It is applied only when an external embedding context is available and the caller leaves rerank in `auto`; otherwise first-pass ranking is final.

Final results cite the returned entity itself. Derived results additionally carry up to two raw `origin_message_id` anchors; raw-message hits cite only their own `origin_message_id`.

- [x] Exact cross-tier ranking between L0, L1, topics, capsules, and raw messages
- [x] Exact lexical/vector weighting by retrieval mode
- [x] Whether reranking is mandatory or optional
- [x] What gets cited in final results

### Storage and rebuild contract

- [x] One authoritative raw source of truth
  Authoritative state lives only in `messages` rows where `projection_kind == raw_global`; each raw row also retains the original `native_session_id` so projection aliases can be rebuilt without consulting any derived table.
- [x] Which layers are derived only
  `projection_messages`, `session_rollups`, `topics`, and `capsules` are derived layers; they may be persisted for speed, but they are always rebuildable from authoritative raw rows and are never treated as source-of-truth state.
- [x] Full rebuild procedure from raw source
  Startup, explicit index rebuild, and schema migration all run the same raw-first rebuild path: extract authoritative raw rows, backfill missing raw `native_session_id` values from surviving projections when available, rebuild canonical projections, preserve only non-canonical extra projection copies, and then rematerialize session rollups, topics, and capsules from that rebuilt state.
- [x] Compaction and retention rules
  Each flush rewrites the parquet tree as one compacted snapshot with one retained `part-*.parquet` per table/date partition; stale derived parquet parts are discarded, while authoritative raw rows and their tombstones are retained in the compacted snapshot until superseded by newer raw state or WAL replay.

### Metrics and acceptance

- [ ] Hit@k targets
- [ ] NDCG targets
- [ ] latency targets
- [ ] memory budget / DDR budget
- [ ] rebuild-time targets

## Concrete Gap Checklist

This section is the implementation checklist to take seriously.

### A. Items not yet fully specified by the blueprint

- [ ] canonical mirrored-write identity model
- [ ] edit/delete/tombstone semantics
- [ ] summary invalidation policy
- [ ] topic repair policy
- [ ] capsule lifecycle invariants
- [ ] cross-tier retrieval contract
- [ ] embedding provenance and re-embed policy
- [ ] rebuild contract
- [ ] platform identity normalization
- [ ] measurable acceptance metrics

### B. OpenViking and QMD have research and implementation, ClawDB currently does not match them

- [ ] OpenViking-level source-of-truth discipline
- [ ] OpenViking-level async semantic pipeline
- [ ] OpenViking-level crash recovery and lock discipline
- [ ] QMD-level persistent lexical index
- [ ] QMD-level persistent vector index
- [ ] QMD-level hybrid retrieval and fusion logic
- [ ] QMD-level evaluation harness and quality thresholds
- [ ] a local ClawDB research corpus comparable in depth to the implementation surface

### C. ClawDB areas that exist in name only or are currently too weak

- [ ] `topic_id` should become a real topic lifecycle, not just a label
- [ ] `capsule_level` should stop meaning message-count bucket
- [ ] `capsules_df` should become real topic capsule state
- [ ] session search should not be transient in-memory only
- [ ] topic and capsule vector indexes should be durable
- [ ] L1 rollups should be materialized and vectorized
- [ ] L0 beliefs/soul should be first-class data structures
- [ ] mirrored DM/group projections should be explicit and queryable

## Implementation Direction

Recommended direction:

1. Keep one authoritative raw-message source of truth.
2. Materialize L1 session projections from raw ingest.
3. Materialize time-window summaries over L1 sessions.
4. Materialize L2 topics over all raw messages using the ClawDB Gauss-Ewens design.
5. Materialize capsules from topic growth.
6. Build durable lexical and vector indexes over:
   - raw messages
   - L1 summaries
   - topics
   - capsules
   - L0 beliefs
7. Make every derived layer rebuildable from authoritative raw data.

## Final Judgment

Current ClawDB is not a completed implementation of the intended ClawDB design.

It is mostly not that design yet.

OpenViking and QMD should be used as engineering references, but ClawDB must own:

- its own L0/L1/L2 semantics
- its own topic and capsule lifecycle
- its own session projection model
- its own rebuild and retrieval contract
