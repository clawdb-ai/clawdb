from __future__ import annotations

import asyncio
import hashlib
import json
import math
import sys
from collections import deque
from time import monotonic
from typing import Dict, List, Literal, Optional, Sequence, Set
from uuid import uuid4

from .config import ClawDBConfig
from .dataframes import DataFrameStore
from .embeddings import EmbeddingAuthContext, EmbeddingRouter, embedding_backend_signature
from .folder_judger import FolderJudger
from .lineage import materialize_message_bundle, normalize_platform
from .locks import DeadlockSafeLockManager, LockRank
from .metrics import (
    CacheTelemetry,
    aggregate_ranked_relevance,
    hit_at_k,
    ndcg_at_k,
    percentile,
    record_acceptance_benchmark,
)
from .metadata import DataFrameMetadataStore
from .migrate import auto_migrate_if_needed
from .models import (
    AcceptanceBenchmarkRequest,
    AcceptanceBenchmarkResponse,
    AcceptanceCaseReport,
    AcceptanceCheck,
    CacheHitReportResponse,
    CapsuleRefreshRequest,
    CapsuleRefreshResponse,
    HealthResponse,
    MessageAck,
    MessageDeleteRequest,
    MessageEditRequest,
    MessageIn,
    IndexRebuildResponse,
    IndexStatusResponse,
    OpenClawMemoryReadResponse,
    OpenClawMemorySearchRequest,
    ResearchBenchmarkRequest,
    ResearchBenchmarkResponse,
    ResearchCorpusCoverage,
    SearchResult,
    SessionForkRequest,
    SessionForkResponse,
    SessionSnapshotRequest,
    SessionSnapshotResponse,
    SessionSpawnRequest,
    SessionSpawnResponse,
    SearchRequest,
    SearchResponse,
)
from .mq import AsyncMessageQueue, build_event, create_queue
from .research import get_research_corpus
from .retrieval import HybridRetrievalEngine, RetrievalDoc, resolve_retrieval_weights
from .trie import TopicTrie
from .topics import GaussianEwensTopicModel
from .wal import WalManager


class ClawDBService:
    def __init__(self, config: Optional[ClawDBConfig] = None) -> None:
        self.config = config or ClawDBConfig.from_env()
        self.config.ensure_dirs()
        self.df_store = DataFrameStore()
        self.wal = WalManager(
            wal_dir=self.config.wal_dir,
            sync_policy=self.config.wal_sync_policy,
            sync_interval_ms=self.config.wal_sync_interval_ms,
        )
        self.queue: AsyncMessageQueue = create_queue(
            self.config.queue_backend,
            self.config.queue_topic,
            self.config.queue_zeromq_endpoint,
        )
        self.lock_manager = DeadlockSafeLockManager(
            lock_timeout_seconds=self.config.lock_timeout_seconds,
            watchdog_seconds=self.config.lock_watchdog_seconds,
        )
        self.embedding_router = EmbeddingRouter()
        self.telemetry = CacheTelemetry()
        self.metadata = DataFrameMetadataStore(self.config.metadata_parquet_path)
        self.topic_model = GaussianEwensTopicModel(
            dim=self.config.topic_gep_dim,
            concentration=self.config.topic_gep_concentration,
            sigma2=self.config.topic_gep_sigma2,
            prior_sigma2=self.config.topic_gep_prior_sigma2,
        )
        self.topic_trie = TopicTrie()
        self.folder_judger = FolderJudger()
        self.retrieval_engine = HybridRetrievalEngine(dim=self.config.topic_gep_dim)
        self._search_cache: Dict[str, List[dict]] = {}
        self._embedding_cache: Dict[str, List[float]] = {}
        self._idempotency_index: Dict[str, MessageAck] = {}
        self._cache_cap = 10_000
        self._flush_task: Optional[asyncio.Task] = None
        self._queue_tasks: List[asyncio.Task] = []
        self._watchdog_task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._checkpoint_seq = 0
        self._semantic_pipeline_async_enabled = (
            str(self.config.semantic_pipeline_mode or "inline").strip().lower() == "async"
        )

    class BackpressureRejectedError(RuntimeError):
        pass

    async def startup(self) -> None:
        await auto_migrate_if_needed(
            data_root=self.config.data_root,
            parquet_dir=self.config.parquet_dir,
            metadata_parquet_path=self.config.metadata_parquet_path,
        )
        await self._load_checkpoint_and_replay()
        recovered_semantic_jobs = await self.df_store.recover_semantic_jobs_for_startup()
        self._stop.clear()
        self._flush_task = asyncio.create_task(self._periodic_flush_loop(), name="clawdb-flush")
        self._queue_tasks = []
        if self._semantic_pipeline_async_enabled:
            consumers = max(1, int(self.config.queue_consumer_count))
            for idx in range(consumers):
                self._queue_tasks.append(
                    asyncio.create_task(
                        self._queue_consumer_loop(idx),
                        name=f"clawdb-queue-{idx}",
                    )
                )
        self._watchdog_task = asyncio.create_task(self._watchdog_loop(), name="clawdb-watchdog")
        if recovered_semantic_jobs.total > 0:
            await self._wake_semantic_pipeline(
                wal_seq=recovered_semantic_jobs.max_wal_seq,
                tenant_id="*",
                session_ids=[],
                cause="startup_recovery",
            )

    async def shutdown(self) -> None:
        self._stop.set()
        tasks: List[asyncio.Task] = []
        if self._flush_task is not None:
            tasks.append(self._flush_task)
        tasks.extend(self._queue_tasks)
        if self._watchdog_task is not None:
            tasks.append(self._watchdog_task)
        for task in tasks:
            if task is None:
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await self.queue.close()
        await self.flush_now()

    async def _load_checkpoint_and_replay(self) -> None:
        ckpt_path = self.config.checkpoints_dir / "latest.json"
        start_seq = 0
        parquet_loaded = False
        checkpoint = await self.metadata.load_checkpoint(slot="wal")
        if checkpoint is not None:
            start_seq = int(checkpoint.last_seq)
            self._checkpoint_seq = start_seq
            try:
                await self.df_store.load_parquet(self.config.parquet_dir)
                parquet_loaded = True
            except Exception:
                start_seq = 0
                self._checkpoint_seq = 0
                parquet_loaded = False
        if ckpt_path.exists():
            try:
                data = json.loads(ckpt_path.read_text(encoding="utf-8"))
                file_start_seq = int(data.get("last_seq", 0))
                if file_start_seq > start_seq:
                    start_seq = file_start_seq
                    self._checkpoint_seq = start_seq
                    await self.df_store.load_parquet(self.config.parquet_dir)
                    parquet_loaded = True
            except Exception:
                start_seq = 0
                self._checkpoint_seq = 0
                parquet_loaded = False
        if not parquet_loaded:
            try:
                await self.df_store.load_parquet(self.config.parquet_dir)
                parquet_loaded = True
            except Exception:
                parquet_loaded = False
        for record in self.wal.replay(from_seq_exclusive=start_seq):
            await self.df_store.apply_wal_record(record)
            if record.event_type == "message_upsert":
                key = str(record.payload.get("idempotency_key") or "")
                if key:
                    scoped = self._idempotency_scope(
                        str(record.payload.get("tenant_id") or "default"),
                        str(record.payload.get("session_id") or "default"),
                        key,
                    )
                    request_message_id = str(
                        record.payload.get("request_message_id")
                        or record.payload.get("message_id")
                        or record.payload.get("origin_message_id")
                        or ""
                    )
                    origin_message_id = str(
                        record.payload.get("origin_message_id")
                        or request_message_id
                        or ""
                    )
                    self._idempotency_index[scoped] = MessageAck(
                        wal_seq=record.seq,
                        message_id=request_message_id,
                        origin_message_id=origin_message_id,
                        affected_projections=len(list(record.payload.get("projections") or [])),
                    )
        await self.df_store.rebuild_storage_from_authoritative_raw(
            vector_dim=self.config.topic_gep_dim
        )
        await self._rebuild_topic_state_from_store(rebuild_materialized_topics=False)

    async def flush_now(self) -> None:
        await self.df_store.save_parquet(self.config.parquet_dir)
        self._checkpoint_seq = self.wal.last_seq
        payload = {"last_seq": self._checkpoint_seq}
        await self.metadata.save_checkpoint(self._checkpoint_seq, slot="wal")
        (self.config.checkpoints_dir / "latest.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    async def _periodic_flush_loop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(self.config.flush_interval_seconds)
            await self.flush_now()

    async def _queue_consumer_loop(self, worker_id: int) -> None:
        async for event in self.queue.consume():
            if self._stop.is_set():
                break
            if event.event_type != "semantic_refresh":
                continue
            await self._drain_semantic_pipeline_worker(
                worker_id=f"queue-{worker_id}",
                raise_on_error=False,
            )

    async def _watchdog_loop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(max(1, int(self.config.lock_watchdog_seconds / 2)))
            alerts = await self.lock_manager.watchdog_once()
            for alert in alerts:
                # Keep logging lightweight and structured.
                print(alert)
            if not self._semantic_pipeline_async_enabled:
                continue
            stats = await self.df_store.semantic_job_stats()
            if stats.pending <= 0 or stats.running > 0:
                continue
            await self._wake_semantic_pipeline(
                wal_seq=stats.max_wal_seq,
                tenant_id="*",
                session_ids=[],
                cause="watchdog",
                fallback_inline=True,
            )

    async def _wake_semantic_pipeline(
        self,
        *,
        wal_seq: int,
        tenant_id: str,
        session_ids: Sequence[str],
        cause: str,
        fallback_inline: bool = True,
    ) -> None:
        if not self._semantic_pipeline_async_enabled:
            await self._drain_semantic_pipeline_worker(
                worker_id="inline",
                raise_on_error=True,
            )
            return
        payload = {
            "tenant_id": str(tenant_id or "default"),
            "session_ids": [str(item) for item in session_ids if str(item)],
            "cause": str(cause or ""),
        }
        try:
            await self.queue.publish(build_event(int(wal_seq), "semantic_refresh", payload))
        except Exception:
            if fallback_inline:
                await self._drain_semantic_pipeline_worker(
                    worker_id="fallback",
                    raise_on_error=True,
                )
            else:
                raise

    async def _run_semantic_job(
        self,
        *,
        worker_id: str,
        claim,
        raise_on_error: bool,
    ) -> None:
        try:
            async with self.lock_manager.acquire("semantic:index", LockRank.INDEX):
                await self.df_store.rebuild_all_topics(vector_dim=self.config.topic_gep_dim)
                await self._rebuild_topic_state_from_store(rebuild_materialized_topics=False)
            await self._refresh_impacted_sessions(claim.tenant_id, claim.impacted_sessions)
            async with self.lock_manager.acquire("semantic:index", LockRank.INDEX):
                await self.df_store.rebuild_search_indexes(vector_dim=self.config.topic_gep_dim)
                self._invalidate_query_state()
            await self.df_store.complete_semantic_job(
                job_id=claim.job_id,
                worker_id=worker_id,
                claimed_wal_seq=claim.claimed_wal_seq,
                retry_delay_seconds=self.config.semantic_retry_delay_seconds,
            )
        except Exception as exc:
            await self.df_store.complete_semantic_job(
                job_id=claim.job_id,
                worker_id=worker_id,
                claimed_wal_seq=claim.claimed_wal_seq,
                retry_delay_seconds=self.config.semantic_retry_delay_seconds,
                error=str(exc),
            )
            if raise_on_error:
                raise
            print(
                json.dumps(
                    {
                        "event": "semantic.pipeline.error",
                        "worker_id": worker_id,
                        "tenant_id": claim.tenant_id,
                        "job_id": claim.job_id,
                        "error": str(exc),
                    }
                )
            )

    async def _drain_semantic_pipeline_worker(
        self,
        *,
        worker_id: str,
        raise_on_error: bool,
        timeout_seconds: Optional[float] = None,
    ) -> int:
        processed = 0
        deadline = (
            monotonic() + max(0.1, float(timeout_seconds))
            if timeout_seconds is not None
            else None
        )
        while True:
            claim = await self.df_store.claim_next_semantic_job(
                worker_id=worker_id,
                lease_seconds=self.config.semantic_job_lease_seconds,
            )
            if claim is None:
                stats = await self.df_store.semantic_job_stats()
                if stats.running == 0:
                    return processed
                if deadline is not None and monotonic() >= deadline:
                    raise TimeoutError("semantic pipeline drain timed out while jobs were still running")
                await asyncio.sleep(0.01)
                continue
            await self._run_semantic_job(
                worker_id=worker_id,
                claim=claim,
                raise_on_error=raise_on_error,
            )
            processed += 1
            if deadline is not None and monotonic() >= deadline:
                stats = await self.df_store.semantic_job_stats()
                if stats.total > 0:
                    raise TimeoutError("semantic pipeline drain timed out before backlog reached zero")
                return processed

    async def drain_semantic_pipeline(self, timeout_seconds: float = 15.0) -> int:
        return await self._drain_semantic_pipeline_worker(
            worker_id=f"drain-{uuid4().hex[:8]}",
            raise_on_error=True,
            timeout_seconds=timeout_seconds,
        )

    def _invalidate_query_state(self) -> None:
        self._search_cache.clear()
        self._embedding_cache.clear()

    async def _refresh_impacted_sessions(self, tenant_id: str, session_ids: Sequence[str]) -> None:
        for session_id in sorted({str(item) for item in session_ids if str(item)}):
            await self.df_store.refresh_session_rollups(
                tenant_id,
                session_id,
                vector_dim=self.config.topic_gep_dim,
            )
            await self.df_store.refresh_capsules(
                tenant_id,
                session_id,
                vector_dim=self.config.topic_gep_dim,
            )

    async def _sync_semantic_state_for_read(
        self,
        *,
        tenant_id: str,
        session_id: Optional[str],
    ) -> None:
        if not self._semantic_pipeline_async_enabled:
            return
        requires_refresh = await self.df_store.has_pending_semantic_refresh(
            tenant_id=tenant_id,
            session_id=session_id,
        )
        if not requires_refresh:
            return
        await self.drain_semantic_pipeline()

    async def _rebuild_topic_state_from_store(self, *, rebuild_materialized_topics: bool = True) -> None:
        if rebuild_materialized_topics:
            await self.df_store.rebuild_all_topics(vector_dim=self.config.topic_gep_dim)
        self.topic_trie = TopicTrie()
        self.topic_model = GaussianEwensTopicModel(
            dim=self.config.topic_gep_dim,
            concentration=self.config.topic_gep_concentration,
            sigma2=self.config.topic_gep_sigma2,
            prior_sigma2=self.config.topic_gep_prior_sigma2,
        )
        docs = await self.df_store.message_documents(tenant_id="*", session_id=None, row_mode="raw")
        for item in docs:
            topic_id = str(item.get("topic_id") or "default")
            content = str(item.get("content") or "")
            if not content:
                continue
            self.topic_trie.insert(topic_id, content)
            self.topic_model.observe_replay(topic_id, content)

    async def _resolve_origin_or_raise(
        self,
        *,
        tenant_id: str,
        origin_message_id: Optional[str],
        platform: Optional[str],
        account_id: Optional[str],
        platform_message_id: Optional[str],
    ) -> str:
        resolved_origin = await self.df_store.resolve_origin_message_id(
            tenant_id=tenant_id,
            origin_message_id=origin_message_id,
            platform=(normalize_platform(platform) if platform else None),
            account_id=account_id,
            platform_message_id=platform_message_id,
        )
        if not resolved_origin:
            raise KeyError("message origin not found")
        return resolved_origin

    def _cache_key(self, req: SearchRequest) -> str:
        return (
            f"{req.tenant_id or 'default'}::{req.session_id or '_'}::"
            f"{req.query.strip().lower()}::{req.max_results}::{req.min_score}::"
            f"{req.channel or '_'}::{req.chat_type or '_'}::{req.group_id or '_'}::"
            f"{req.topic_id or '_'}::{req.message_thread_id or '_'}::"
            f"{req.retrieval_mode}::{req.rerank}"
        )

    def _idempotency_scope(self, tenant_id: str, session_id: str, idempotency_key: str) -> str:
        return f"{tenant_id}::{session_id}::{idempotency_key}"

    async def _enforce_ingest_backpressure(self) -> None:
        threshold = int(self.config.ingest_backpressure_lag_threshold)
        if threshold <= 0:
            return
        max_wait_ms = max(0, int(self.config.ingest_backpressure_max_wait_ms))
        poll_interval_ms = max(1, int(self.config.ingest_backpressure_poll_interval_ms))
        started = monotonic()
        while True:
            queue_lag = await self.queue.lag()
            semantic_stats = await self.df_store.semantic_job_stats()
            backlog = max(queue_lag, semantic_stats.pending + semantic_stats.running)
            if backlog <= threshold:
                return
            elapsed_ms = (monotonic() - started) * 1000.0
            if elapsed_ms >= max_wait_ms:
                raise ClawDBService.BackpressureRejectedError(
                    f"ingest backpressure: semantic backlog {backlog} exceeds threshold {threshold}"
                )
            await asyncio.sleep(poll_interval_ms / 1000.0)

    async def ingest_message(self, req: MessageIn) -> MessageAck:
        await self._enforce_ingest_backpressure()
        normalized_channel = (req.channel or "").strip().lower() or None
        normalized_chat_type = (req.chat_type or "").strip().lower() or None
        normalized_platform = normalize_platform(req.platform, normalized_channel)
        resolved_topic_id = req.topic_id
        auto_topic_assigned = False
        if self.config.topic_auto_classify_enabled and not resolved_topic_id:
            resolved_topic_id = self.topic_model.propose_topic(req.content)
            auto_topic_assigned = bool(resolved_topic_id)
        topic_id = resolved_topic_id or req.topic_id or "default"
        topic_source = req.topic_source or ("gauss_ewens" if auto_topic_assigned else "explicit")
        topic_path = req.topic_path or (
            f"{req.topic_parent_id}/{topic_id}" if req.topic_parent_id else topic_id
        )
        topic_confidence = req.topic_confidence
        if topic_confidence is None:
            topic_confidence = 0.6 if auto_topic_assigned else 1.0
        req_with_topic = req.model_copy(
            update={
                "channel": normalized_channel,
                "platform": normalized_platform,
                "chat_type": normalized_chat_type,
                "topic_id": topic_id,
                "topic_path": topic_path,
                "topic_source": topic_source,
                "topic_confidence": topic_confidence,
            }
        )
        payload = materialize_message_bundle(req_with_topic.model_dump(mode="json"))
        origin_message_id = str(payload["origin_message_id"])
        lock_key = f"message:{req_with_topic.tenant_id}:{origin_message_id}"
        async with self.lock_manager.acquire(lock_key, LockRank.SESSION):
            if self.config.idempotency_dedupe_enabled and req_with_topic.idempotency_key:
                scope = self._idempotency_scope(
                    req_with_topic.tenant_id,
                    req_with_topic.session_id,
                    req_with_topic.idempotency_key,
                )
                cached_ack = self._idempotency_index.get(scope)
                if cached_ack is not None:
                    return cached_ack
            record = await self.wal.append("message_upsert", payload)
            upsert_result = await self.df_store.apply_message_bundle(
                payload,
                vector_dim=self.config.topic_gep_dim,
            )
            await self.df_store.enqueue_semantic_refresh(
                tenant_id=req_with_topic.tenant_id,
                wal_seq=record.seq,
                session_ids=upsert_result.affected_sessions,
                cause="message_upsert",
            )
            ack = MessageAck(
                wal_seq=record.seq,
                message_id=req_with_topic.message_id,
                origin_message_id=origin_message_id,
                affected_projections=upsert_result.affected_projections,
            )
            if self.config.idempotency_dedupe_enabled and req_with_topic.idempotency_key:
                self._idempotency_index[scope] = ack
                if len(self._idempotency_index) > self._cache_cap:
                    self._idempotency_index.pop(next(iter(self._idempotency_index)))
        self._invalidate_query_state()
        await self._wake_semantic_pipeline(
            wal_seq=record.seq,
            tenant_id=req_with_topic.tenant_id,
            session_ids=upsert_result.affected_sessions,
            cause="message_upsert",
        )
        return ack

    async def edit_message(self, req: MessageEditRequest) -> MessageAck:
        resolved_origin = await self._resolve_origin_or_raise(
            tenant_id=req.tenant_id,
            origin_message_id=req.origin_message_id,
            platform=req.platform,
            account_id=req.account_id,
            platform_message_id=req.platform_message_id,
        )
        payload = {
            "tenant_id": req.tenant_id,
            "origin_message_id": resolved_origin,
            "platform": normalize_platform(req.platform) if req.platform else "",
            "account_id": req.account_id or "",
            "platform_message_id": req.platform_message_id or "",
            "content": req.content,
            "ts": req.ts.isoformat(),
        }
        lock_key = f"message:{req.tenant_id}:{resolved_origin}"
        async with self.lock_manager.acquire(lock_key, LockRank.SESSION):
            record = await self.wal.append("message_edit", payload)
            result = await self.df_store.edit_message(
                tenant_id=req.tenant_id,
                origin_message_id=resolved_origin,
                content=req.content,
                edited_at=req.ts,
                vector_dim=self.config.topic_gep_dim,
            )
            if not result.found:
                raise KeyError("message origin not found")
            await self.df_store.enqueue_semantic_refresh(
                tenant_id=req.tenant_id,
                wal_seq=record.seq,
                session_ids=result.affected_sessions,
                cause="message_edit",
            )
        self._invalidate_query_state()
        await self._wake_semantic_pipeline(
            wal_seq=record.seq,
            tenant_id=req.tenant_id,
            session_ids=result.affected_sessions,
            cause="message_edit",
        )
        return MessageAck(
            wal_seq=record.seq,
            message_id=resolved_origin,
            origin_message_id=resolved_origin,
            affected_projections=result.affected_projections,
        )

    async def delete_message(self, req: MessageDeleteRequest) -> MessageAck:
        resolved_origin = await self._resolve_origin_or_raise(
            tenant_id=req.tenant_id,
            origin_message_id=req.origin_message_id,
            platform=req.platform,
            account_id=req.account_id,
            platform_message_id=req.platform_message_id,
        )
        payload = {
            "tenant_id": req.tenant_id,
            "origin_message_id": resolved_origin,
            "platform": normalize_platform(req.platform) if req.platform else "",
            "account_id": req.account_id or "",
            "platform_message_id": req.platform_message_id or "",
            "ts": req.ts.isoformat(),
        }
        lock_key = f"message:{req.tenant_id}:{resolved_origin}"
        async with self.lock_manager.acquire(lock_key, LockRank.SESSION):
            record = await self.wal.append("message_delete", payload)
            result = await self.df_store.delete_message(
                tenant_id=req.tenant_id,
                origin_message_id=resolved_origin,
                deleted_at=req.ts,
                vector_dim=self.config.topic_gep_dim,
            )
            if not result.found:
                raise KeyError("message origin not found")
            await self.df_store.enqueue_semantic_refresh(
                tenant_id=req.tenant_id,
                wal_seq=record.seq,
                session_ids=result.affected_sessions,
                cause="message_delete",
            )
        self._invalidate_query_state()
        await self._wake_semantic_pipeline(
            wal_seq=record.seq,
            tenant_id=req.tenant_id,
            session_ids=result.affected_sessions,
            cause="message_delete",
        )
        return MessageAck(
            wal_seq=record.seq,
            message_id=resolved_origin,
            origin_message_id=resolved_origin,
            affected_projections=result.affected_projections,
        )

    def _embedding_cache_key(self, ctx: EmbeddingAuthContext, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{embedding_backend_signature(ctx)}:{digest}"

    async def _embed_texts_cached(
        self,
        ctx: EmbeddingAuthContext,
        texts: Sequence[str],
    ) -> List[List[float]]:
        cached_vectors: List[Optional[List[float]]] = []
        misses: List[str] = []
        miss_indexes: List[int] = []
        for i, text in enumerate(texts):
            key = self._embedding_cache_key(ctx, text)
            vector = self._embedding_cache.get(key)
            cached_vectors.append(vector)
            if vector is None:
                misses.append(text)
                miss_indexes.append(i)

        if misses:
            generated = await self.embedding_router.embed_texts(ctx, misses)
            for idx, vector in zip(miss_indexes, generated):
                text = texts[idx]
                key = self._embedding_cache_key(ctx, text)
                self._embedding_cache[key] = vector
                cached_vectors[idx] = vector

        return [vector or [] for vector in cached_vectors]

    def _cosine_similarity(self, left: Sequence[float], right: Sequence[float]) -> float:
        if not left or not right:
            return 0.0
        dims = min(len(left), len(right))
        if dims == 0:
            return 0.0
        dot = 0.0
        left_norm = 0.0
        right_norm = 0.0
        for i in range(dims):
            lv = float(left[i])
            rv = float(right[i])
            dot += lv * rv
            left_norm += lv * lv
            right_norm += rv * rv
        if left_norm <= 0.0 or right_norm <= 0.0:
            return 0.0
        return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))

    def _result_sort_key(self, item: SearchResult) -> tuple[float, int, float, float, str]:
        entity_priority = {
            "raw_message": 0,
            "capsule": 1,
            "topic": 2,
            "session_rollup": 3,
            "l0_abstract": 4,
        }
        semantic_component = float(item.score_semantic or 0.0) if item.reranked else 0.0
        vector_component = semantic_component if item.reranked else float(item.score_vector or 0.0)
        return (
            -float(item.score),
            entity_priority.get(str(item.entity_type), 99),
            -float(item.score_lexical or 0.0),
            -vector_component,
            str(item.entity_id or item.path),
        )

    def _acceptance_result_keys(self, item: SearchResult) -> List[str]:
        keys: List[str] = []
        if item.entity_id:
            keys.append(f"entity:{item.entity_type}:{item.entity_id}")
        if item.origin_message_id:
            keys.append(f"origin:{item.origin_message_id}")
        for citation in [item.citation, *item.citations]:
            if citation:
                keys.append(str(citation))
        deduped: List[str] = []
        seen: Set[str] = set()
        for key in keys:
            if key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        return deduped

    def _estimate_python_object_bytes(self, value: object, seen: Optional[Set[int]] = None) -> int:
        if seen is None:
            seen = set()
        obj_id = id(value)
        if obj_id in seen:
            return 0
        seen.add(obj_id)
        size = sys.getsizeof(value)
        if isinstance(value, dict):
            for key, item in value.items():
                size += self._estimate_python_object_bytes(key, seen)
                size += self._estimate_python_object_bytes(item, seen)
            return size
        if isinstance(value, (list, tuple, set, frozenset, deque)):
            for item in value:
                size += self._estimate_python_object_bytes(item, seen)
            return size
        if hasattr(value, "model_dump") and callable(getattr(value, "model_dump")):
            return size + self._estimate_python_object_bytes(value.model_dump(mode="python"), seen)
        if hasattr(value, "__dict__"):
            return size + self._estimate_python_object_bytes(vars(value), seen)
        return size

    def _estimate_working_set_breakdown(self) -> tuple[int, int, int]:
        state = self.df_store.state
        dataframe_bytes = sum(
            int(frame.memory_usage(index=True, deep=True).sum())
            for frame in [
                state.messages_df,
                state.capsules_df,
                state.beliefs_df,
                state.projections_df,
                state.session_rollups_df,
                state.topics_df,
                state.search_docs_df,
                state.lexical_index_df,
                state.vector_index_df,
                state.cache_index_df,
                state.sessions_df,
                state.snapshots_df,
            ]
        )
        cache_bytes = self._estimate_python_object_bytes(
            {
                "search_cache": self._search_cache,
                "embedding_cache": self._embedding_cache,
                "idempotency_index": self._idempotency_index,
            }
        )
        index_bytes = self._estimate_python_object_bytes(
            {
                "topic_trie": self.topic_trie,
                "topic_model": self.topic_model,
                "telemetry": self.telemetry,
                "retrieval_engine": self.retrieval_engine,
            }
        )
        return dataframe_bytes, cache_bytes, index_bytes

    def _acceptance_check(
        self,
        *,
        name: str,
        actual: float,
        target: float,
        comparator: Literal["gte", "lte"],
        unit: str,
    ) -> AcceptanceCheck:
        if comparator == "gte":
            passed = float(actual) >= float(target)
        else:
            passed = float(actual) <= float(target)
        return AcceptanceCheck(
            name=name,
            actual=float(actual),
            target=float(target),
            comparator=comparator,
            passed=bool(passed),
            unit=unit,
        )

    async def search(
        self,
        req: SearchRequest,
        embedding_ctx: Optional[EmbeddingAuthContext] = None,
    ) -> SearchResponse:
        await self._sync_semantic_state_for_read(
            tenant_id=req.tenant_id,
            session_id=req.session_id,
        )
        started = monotonic()
        key = self._cache_key(req)
        if embedding_ctx:
            key = f"{key}::emb:{embedding_backend_signature(embedding_ctx)}"
        cached = self._search_cache.get(key)
        cache_hit = cached is not None
        if cache_hit:
            results = cached
        else:
            docs_raw = await self.df_store.retrieval_documents(
                tenant_id=req.tenant_id,
                session_id=req.session_id,
                channel=req.channel,
                chat_type=req.chat_type,
                group_id=req.group_id,
                topic_id=req.topic_id,
                message_thread_id=req.message_thread_id,
            )
            docs = [
                RetrievalDoc(doc_id=str(item["doc_id"]), text=str(item["text"]))
                for item in docs_raw
            ]
            doc_map = {str(item["doc_id"]): item for item in docs_raw}
            lexical_postings, vector_entries = await self.df_store.search_index_entries(
                tenant_id=req.tenant_id,
                doc_ids=list(doc_map.keys()),
            )
            retrieval = self.retrieval_engine.search(
                query=req.query,
                docs=docs,
                top_k=max(req.max_results * 5, req.max_results),
                retrieval_mode=req.retrieval_mode,
                lexical_postings=lexical_postings,
                vector_entries=vector_entries,
            )
            raw_results: List[SearchResult] = []
            for score in retrieval:
                item = doc_map.get(score.doc_id)
                if item is None:
                    continue
                raw_results.append(
                    SearchResult(
                        path=str(item["path"]),
                        start_line=int(item.get("start_line") or 1),
                        end_line=int(item.get("end_line") or item.get("start_line") or 1),
                        score=round(float(score.score), 6),
                        score_lexical=round(float(score.score_lexical), 6),
                        score_semantic=0.0,
                        score_vector=round(float(score.score_vector), 6),
                        snippet=str(item["snippet"])[:700],
                        source="memory",
                        source_tier=str(item.get("source_tier") or "L0"),
                        entity_type=str(item.get("entity_type") or "raw_message"),
                        entity_id=str(item.get("entity_id") or item["doc_id"]),
                        retrieval_mode=req.retrieval_mode,
                        reranked=False,
                        citation=str(item.get("citation") or "") or None,
                        citations=[str(citation) for citation in list(item.get("citations") or [])],
                        channel=str(item.get("channel") or "") or None,
                        chat_type=str(item.get("chat_type") or "") or None,
                        account_id=str(item.get("account_id") or "") or None,
                        group_id=str(item.get("group_id") or "") or None,
                        topic_id=str(item.get("topic_id") or "") or None,
                        topic_path=str(item.get("topic_path") or "") or None,
                        message_thread_id=str(item.get("message_thread_id") or "") or None,
                        sender_id=str(item.get("sender_id") or "") or None,
                        origin_message_id=str(item.get("origin_message_id") or "") or None,
                        projection_kind=str(item.get("projection_kind") or "") or None,
                        projection_scope=str(item.get("projection_scope") or "") or None,
                    )
                )
            rescored = sorted(raw_results, key=self._result_sort_key)
            if embedding_ctx and rescored and req.rerank != "off":
                try:
                    lexical_weight, vector_weight = resolve_retrieval_weights(req.retrieval_mode)
                    texts = [req.query, *[item.snippet for item in rescored]]
                    vectors = await self._embed_texts_cached(embedding_ctx, texts)
                    query_vec = vectors[0]
                    msg_vecs = vectors[1:]
                    merged = []
                    for item, vector in zip(rescored, msg_vecs):
                        lexical_score = float(item.score_lexical or 0.0)
                        semantic_score = max(0.0, self._cosine_similarity(query_vec, vector))
                        combined = (lexical_weight * lexical_score) + (vector_weight * semantic_score)
                        updated = item.model_copy(
                            update={
                                "score": round(float(combined), 6),
                                "score_semantic": round(float(semantic_score), 6),
                                "reranked": True,
                            }
                        )
                        merged.append(updated)
                    rescored = sorted(merged, key=self._result_sort_key)
                except Exception:
                    rescored = sorted(raw_results, key=self._result_sort_key)
            filtered = [item for item in rescored if item.score >= req.min_score]
            results = [item.model_dump() for item in filtered[: max(1, req.max_results)]]
            self._search_cache[key] = results
            if len(self._search_cache) > self._cache_cap:
                self._search_cache.pop(next(iter(self._search_cache)))
                self.telemetry.observe_eviction()
        latency_ms = (monotonic() - started) * 1000.0
        session_id = req.session_id or "_"
        self.telemetry.observe_lookup(
            cache_hit,
            latency_ms,
            tenant_id=req.tenant_id or "default",
            session_id=session_id,
            query_type="memory_search",
            capsule_level="mixed",
        )
        await self.df_store.record_cache_lookup(
            key=key,
            tenant_id=req.tenant_id or "default",
            session_id=session_id,
            query_type="memory_search",
            capsule_level="mixed",
            hit=cache_hit,
        )
        if self.config.search_log_enabled:
            print(
                json.dumps(
                    {
                        "event": "memory.search",
                        "tenant_id": req.tenant_id or "default",
                        "session_id": session_id,
                        "cache_hit": cache_hit,
                        "query_type": "memory_search",
                        "latency_ms": round(float(latency_ms), 3),
                    }
                )
            )
        return SearchResponse(wal_seq=self.wal.last_seq, cache_hit=cache_hit, results=results)

    async def create_snapshot(self, req: SessionSnapshotRequest) -> SessionSnapshotResponse:
        snapshot_id = f"snap-{uuid4()}"
        payload = req.model_dump(mode="json")
        payload["snapshot_id"] = snapshot_id
        payload["wal_seq"] = self.wal.last_seq
        lock_key = f"session:{req.tenant_id}:{req.session_id}"
        async with self.lock_manager.acquire(lock_key, LockRank.SESSION):
            record = await self.wal.append("session_snapshot", payload)
            await self.df_store.create_snapshot(
                tenant_id=req.tenant_id,
                session_id=req.session_id,
                snapshot_id=snapshot_id,
                wal_seq=record.seq,
                note=req.note or "",
            )
        return SessionSnapshotResponse(snapshot_id=snapshot_id, wal_seq=record.seq)

    async def fork_session(self, req: SessionForkRequest) -> SessionForkResponse:
        target = req.target_session_id or f"{req.source_session_id}-fork-{uuid4().hex[:8]}"
        snapshot_id = f"snap-{uuid4()}"
        payload = req.model_dump(mode="json")
        payload["target_session_id"] = target
        payload["snapshot_id"] = snapshot_id
        lock_key = f"session:{req.tenant_id}:{req.source_session_id}"
        async with self.lock_manager.acquire(lock_key, LockRank.SESSION):
            record = await self.wal.append("session_fork", payload)
            await self.df_store.fork_session(
                tenant_id=req.tenant_id,
                source_session_id=req.source_session_id,
                target_session_id=target,
            )
            await self.df_store.rebuild_search_indexes(vector_dim=self.config.topic_gep_dim)
            await self.df_store.create_snapshot(
                tenant_id=req.tenant_id,
                session_id=req.source_session_id,
                snapshot_id=snapshot_id,
                wal_seq=record.seq,
                note=req.note or "fork",
            )
        return SessionForkResponse(
            source_session_id=req.source_session_id,
            target_session_id=target,
            snapshot_id=snapshot_id,
            wal_seq=record.seq,
        )

    async def spawn_session(self, req: SessionSpawnRequest) -> SessionSpawnResponse:
        target = req.session_id or f"spawn-{uuid4().hex[:12]}"
        payload = req.model_dump(mode="json")
        payload["session_id"] = target
        lock_key = f"session:{req.tenant_id}:{target}"
        async with self.lock_manager.acquire(lock_key, LockRank.SESSION):
            record = await self.wal.append("session_spawn", payload)
            await self.df_store.spawn_session(
                tenant_id=req.tenant_id,
                session_id=target,
                parent_session_id=req.seed_session_id,
            )
        return SessionSpawnResponse(
            session_id=target,
            parent_session_id=req.seed_session_id,
            wal_seq=record.seq,
        )

    async def list_session_snapshots(self, tenant_id: str, session_id: str) -> List[dict]:
        return await self.df_store.list_snapshots(tenant_id=tenant_id, session_id=session_id)

    async def present_linear_im(self, tenant_id: str, session_id: str) -> OpenClawMemoryReadResponse:
        path = f"memory/{session_id}.md" if tenant_id == "default" else f"memory/{tenant_id}/{session_id}.md"
        return await self.openclaw_memory_get(path, from_line=1, lines=5000)

    async def present_capsule_cards(self, tenant_id: str, session_id: str) -> List[dict]:
        await self._sync_semantic_state_for_read(
            tenant_id=tenant_id,
            session_id=session_id,
        )
        cards = await self.df_store.capsule_cards(tenant_id=tenant_id, session_id=session_id)
        if cards:
            return cards
        await self.refresh_capsules(CapsuleRefreshRequest(tenant_id=tenant_id, session_id=session_id))
        return await self.df_store.capsule_cards(tenant_id=tenant_id, session_id=session_id)

    async def present_forum_style(self, tenant_id: str, session_id: str) -> List[dict]:
        return await self.df_store.forum_view(tenant_id=tenant_id, session_id=session_id)

    async def list_projection_state(
        self,
        *,
        tenant_id: str = "default",
        session_id: Optional[str] = None,
        projection_kind: Optional[str] = None,
        origin_message_id: Optional[str] = None,
        group_id: Optional[str] = None,
        include_deleted: bool = False,
    ) -> List[dict]:
        return await self.df_store.list_projection_state(
            tenant_id=tenant_id,
            session_id=session_id,
            projection_kind=projection_kind,
            origin_message_id=origin_message_id,
            group_id=group_id,
            include_deleted=include_deleted,
        )

    async def list_belief_state(
        self,
        *,
        tenant_id: str = "default",
        scope_type: Optional[str] = None,
        session_id: Optional[str] = None,
        topic_id: Optional[str] = None,
    ) -> List[dict]:
        return await self.df_store.list_belief_state(
            tenant_id=tenant_id,
            scope_type=scope_type,
            session_id=session_id,
            topic_id=topic_id,
        )

    async def index_status(self) -> IndexStatusResponse:
        sessions = await self.df_store.session_count()
        snapshots = await self.df_store.snapshot_count()
        semantic_stats = await self.df_store.semantic_job_stats()
        return IndexStatusResponse(
            trie_topics=self.topic_trie.topic_count,
            session_count=sessions,
            snapshot_count=snapshots,
            wal_seq=self.wal.last_seq,
            semantic_job_backlog=semantic_stats.pending,
            semantic_jobs_running=semantic_stats.running,
        )

    async def rebuild_indexes(self) -> IndexRebuildResponse:
        await self.drain_semantic_pipeline()
        async with self.lock_manager.acquire("semantic:index", LockRank.INDEX):
            await self.df_store.clear_semantic_jobs()
            self._invalidate_query_state()
            rebuild = await self.df_store.rebuild_storage_from_authoritative_raw(
                vector_dim=self.config.topic_gep_dim
            )
            await self._rebuild_topic_state_from_store(rebuild_materialized_topics=False)
        return IndexRebuildResponse(
            wal_seq=self.wal.last_seq,
            rebuilt_topics=rebuild.topic_count,
            rebuilt_messages=rebuild.raw_message_count + rebuild.projection_message_count,
            authoritative_raw_messages=rebuild.raw_message_count,
            rebuilt_projection_messages=rebuild.projection_message_count,
            rebuilt_session_rollups=rebuild.session_rollup_count,
            rebuilt_capsules=rebuild.capsule_count,
            rebuilt_embedding_metadata=rebuild.embedding_metadata_count,
        )

    async def refresh_capsules(self, req: CapsuleRefreshRequest) -> CapsuleRefreshResponse:
        payload = req.model_dump(mode="json")
        lock_key = f"session:{req.tenant_id}:{req.session_id}"
        async with self.lock_manager.acquire(lock_key, LockRank.SESSION):
            record = await self.wal.append("capsule_refresh", payload)
            count = await self.df_store.refresh_capsules(
                req.tenant_id,
                req.session_id,
                vector_dim=self.config.topic_gep_dim,
            )
            await self.df_store.rebuild_search_indexes(vector_dim=self.config.topic_gep_dim)
        return CapsuleRefreshResponse(wal_seq=record.seq, capsule_count=count)

    async def health(self) -> HealthResponse:
        lag = await self.queue.lag()
        ratio = self.telemetry.hit_ratio(300)
        semantic_stats = await self.df_store.semantic_job_stats()
        status = "ok"
        if lag > self.config.ingest_backpressure_lag_threshold:
            status = "degraded"
        if ratio < self.config.cache_hit_ratio_alert_threshold and (self.telemetry.hits_total + self.telemetry.misses_total) > 20:
            status = "degraded"
        if semantic_stats.pending > max(1, self.config.queue_consumer_count * 4):
            status = "degraded"
        return HealthResponse(
            status=status,
            wal_replay_lag=max(0, self.wal.last_seq - self._checkpoint_seq),
            checkpoint_seq=self._checkpoint_seq,
            cache_hit_ratio_5m=ratio,
            queue_backend=self.config.queue_backend,
            queue_lag=lag,
            semantic_mode="async" if self._semantic_pipeline_async_enabled else "inline",
            semantic_job_backlog=semantic_stats.pending,
            semantic_jobs_running=semantic_stats.running,
        )

    async def cache_hit_report(self) -> CacheHitReportResponse:
        return CacheHitReportResponse(
            memory_cache_hit_ratio_1m=self.telemetry.hit_ratio(60),
            memory_cache_hit_ratio_5m=self.telemetry.hit_ratio(300),
            memory_cache_hits_total=self.telemetry.hits_total,
            memory_cache_misses_total=self.telemetry.misses_total,
            memory_cache_evictions_total=self.telemetry.evictions_total,
            memory_cache_lookup_latency_ms_p50=self.telemetry.p50_lookup_latency_ms(),
        )

    async def evaluate_research_benchmark(
        self,
        req: ResearchBenchmarkRequest,
    ) -> ResearchBenchmarkResponse:
        corpus = get_research_corpus(req.corpus_name)
        tenant_id = str(req.tenant_id or "research-benchmark")
        for message in corpus.messages:
            await self.ingest_message(
                message.model_copy(
                    deep=True,
                    update={"tenant_id": tenant_id},
                )
            )
        for edit in corpus.edits:
            await self.edit_message(
                edit.model_copy(
                    deep=True,
                    update={"tenant_id": tenant_id},
                )
            )
        for delete in corpus.deletes:
            await self.delete_message(
                delete.model_copy(
                    deep=True,
                    update={"tenant_id": tenant_id},
                )
            )
        benchmark = await self.evaluate_acceptance(
            corpus.build_acceptance_request(
                tenant_id=tenant_id,
                latency_repetitions=req.latency_repetitions,
                targets=req.targets,
            )
        )
        coverage_payload = corpus.coverage()
        coverage_payload["entity_types_seen"] = sorted(
            {
                str(entity_type)
                for case in benchmark.cases
                for entity_type in case.result_entity_types
                if str(entity_type)
            }
        )
        return ResearchBenchmarkResponse(
            **benchmark.model_dump(),
            corpus_name=corpus.name,
            corpus_version=corpus.version,
            corpus_description=corpus.description,
            seeded_messages=len(corpus.messages),
            seeded_edits=len(corpus.edits),
            seeded_deletes=len(corpus.deletes),
            coverage=ResearchCorpusCoverage(**coverage_payload),
        )

    async def evaluate_acceptance(
        self,
        req: AcceptanceBenchmarkRequest,
    ) -> AcceptanceBenchmarkResponse:
        if not req.cases:
            raise ValueError("at least one acceptance case is required")
        await self.drain_semantic_pipeline()

        hit_targets = {max(1, int(k)): float(v) for k, v in req.targets.hit_at.items()}
        ndcg_targets = {max(1, int(k)): float(v) for k, v in req.targets.ndcg_at.items()}
        max_requested_results = max(
            [1, *hit_targets.keys(), *ndcg_targets.keys(), *[case.search.max_results for case in req.cases]]
        )

        hit_totals = {k: 0.0 for k in hit_targets}
        ndcg_totals = {k: 0.0 for k in ndcg_targets}
        case_reports: List[AcceptanceCaseReport] = []

        prepared_searches = []
        for case in req.cases:
            search_req = case.search.model_copy(
                update={"max_results": max(int(case.search.max_results), max_requested_results)}
            )
            prepared_searches.append(search_req)
            judgments = {
                str(judgment.match_key): float(judgment.relevance)
                for judgment in case.judgments
                if float(judgment.relevance) > 0.0
            }
            result = await self.search(search_req)
            ranked_keys = [self._acceptance_result_keys(item) for item in result.results]
            ranked_relevance = aggregate_ranked_relevance(ranked_keys, judgments)
            ideal_relevance = sorted(judgments.values(), reverse=True)

            case_hit = {k: hit_at_k(ranked_relevance, k) for k in hit_targets}
            case_ndcg = {k: ndcg_at_k(ranked_relevance, ideal_relevance, k) for k in ndcg_targets}
            for k, value in case_hit.items():
                hit_totals[k] += float(value)
            for k, value in case_ndcg.items():
                ndcg_totals[k] += float(value)

            top_match_keys = [
                keys[0] if keys else str(item.entity_id or item.path)
                for keys, item in zip(ranked_keys, result.results)
            ]
            case_reports.append(
                AcceptanceCaseReport(
                    label=case.label or search_req.query,
                    query=search_req.query,
                    hit_at=case_hit,
                    ndcg_at=case_ndcg,
                    top_match_keys=top_match_keys,
                    matched_relevance=ranked_relevance,
                    result_entity_types=[str(item.entity_type) for item in result.results],
                )
            )

        case_count = max(1, len(req.cases))
        hit_summary = {k: float(hit_totals[k] / case_count) for k in hit_targets}
        ndcg_summary = {k: float(ndcg_totals[k] / case_count) for k in ndcg_targets}

        latency_repetitions = max(1, int(req.latency_repetitions))
        cold_samples: List[float] = []
        warm_samples: List[float] = []
        for _ in range(latency_repetitions):
            for search_req in prepared_searches:
                self._search_cache.clear()
                started = monotonic()
                await self.search(search_req)
                cold_samples.append((monotonic() - started) * 1000.0)

                started = monotonic()
                await self.search(search_req)
                warm_samples.append((monotonic() - started) * 1000.0)

        cold_latency_p50 = percentile(cold_samples, 50.0)
        cold_latency_p95 = percentile(cold_samples, 95.0)
        warm_latency_p50 = percentile(warm_samples, 50.0)
        warm_latency_p95 = percentile(warm_samples, 95.0)

        rebuild_started = monotonic()
        rebuild = await self.rebuild_indexes()
        rebuild_time_ms = (monotonic() - rebuild_started) * 1000.0

        dataframe_bytes, cache_bytes, index_bytes = self._estimate_working_set_breakdown()
        working_set_bytes = int(dataframe_bytes + cache_bytes + index_bytes)

        checks: List[AcceptanceCheck] = []
        for k, target in hit_targets.items():
            checks.append(
                self._acceptance_check(
                    name=f"hit@{k}",
                    actual=hit_summary[k],
                    target=target,
                    comparator="gte",
                    unit="ratio",
                )
            )
        for k, target in ndcg_targets.items():
            checks.append(
                self._acceptance_check(
                    name=f"ndcg@{k}",
                    actual=ndcg_summary[k],
                    target=target,
                    comparator="gte",
                    unit="ratio",
                )
            )
        checks.append(
            self._acceptance_check(
                name="cold-latency-p95",
                actual=cold_latency_p95,
                target=req.targets.cold_latency_p95_ms,
                comparator="lte",
                unit="ms",
            )
        )
        checks.append(
            self._acceptance_check(
                name="warm-latency-p95",
                actual=warm_latency_p95,
                target=req.targets.warm_latency_p95_ms,
                comparator="lte",
                unit="ms",
            )
        )
        checks.append(
            self._acceptance_check(
                name="working-set-memory",
                actual=float(working_set_bytes),
                target=float(req.targets.max_working_set_bytes),
                comparator="lte",
                unit="bytes",
            )
        )
        checks.append(
            self._acceptance_check(
                name="rebuild-time",
                actual=rebuild_time_ms,
                target=req.targets.max_rebuild_time_ms,
                comparator="lte",
                unit="ms",
            )
        )

        passed = all(check.passed for check in checks)
        record_acceptance_benchmark(
            passed=passed,
            case_count=case_count,
            cold_latency_sample_count=len(cold_samples),
            warm_latency_sample_count=len(warm_samples),
            checks=[check.model_dump() for check in checks],
        )

        return AcceptanceBenchmarkResponse(
            passed=passed,
            case_count=case_count,
            hit_at=hit_summary,
            ndcg_at=ndcg_summary,
            cold_latency_ms_p50=cold_latency_p50,
            cold_latency_ms_p95=cold_latency_p95,
            cold_latency_sample_count=len(cold_samples),
            warm_latency_ms_p50=warm_latency_p50,
            warm_latency_ms_p95=warm_latency_p95,
            warm_latency_sample_count=len(warm_samples),
            dataframe_bytes=dataframe_bytes,
            cache_bytes=cache_bytes,
            index_bytes=index_bytes,
            working_set_bytes=working_set_bytes,
            working_set_mebibytes=round(float(working_set_bytes / (1024 * 1024)), 6),
            rebuild_time_ms=rebuild_time_ms,
            authoritative_raw_messages=rebuild.authoritative_raw_messages,
            rebuilt_projection_messages=rebuild.rebuilt_projection_messages,
            rebuilt_session_rollups=rebuild.rebuilt_session_rollups,
            rebuilt_topics=rebuild.rebuilt_topics,
            rebuilt_capsules=rebuild.rebuilt_capsules,
            checks=checks,
            cases=case_reports,
        )

    async def openclaw_memory_search(
        self,
        req: OpenClawMemorySearchRequest,
        embedding_ctx: Optional[EmbeddingAuthContext] = None,
    ):
        internal = SearchRequest(
            query=req.query,
            tenant_id=req.tenantId or "default",
            session_id=req.sessionKey,
            max_results=req.maxResults or 6,
            min_score=req.minScore or 0.0,
        )
        result = await self.search(internal, embedding_ctx=embedding_ctx)
        return [
            {
                "path": item.path,
                "startLine": item.start_line,
                "endLine": item.end_line,
                "score": item.score,
                "scoreLexical": item.score_lexical,
                "scoreSemantic": item.score_semantic,
                "scoreVector": item.score_vector,
                "snippet": item.snippet,
                "source": item.source,
                "sourceTier": item.source_tier,
                "entityType": item.entity_type,
                "entityId": item.entity_id,
                "retrievalMode": item.retrieval_mode,
                "reranked": item.reranked,
                "citation": item.citation,
                "citations": item.citations,
                "channel": item.channel,
                "chatType": item.chat_type,
                "accountId": item.account_id,
                "groupId": item.group_id,
                "topicId": item.topic_id,
                "topicPath": item.topic_path,
                "threadId": item.message_thread_id,
                "senderId": item.sender_id,
                "originMessageId": item.origin_message_id,
                "projectionKind": item.projection_kind,
                "projectionScope": item.projection_scope,
            }
            for item in result.results
        ]

    async def openclaw_memory_get(
        self,
        rel_path: str,
        from_line: int = 1,
        lines: int = 200,
    ) -> OpenClawMemoryReadResponse:
        text, canonical_path = await self.df_store.virtual_memory_file(rel_path)
        all_lines = text.splitlines()
        start_idx = max(0, from_line - 1)
        end_idx = min(len(all_lines), start_idx + max(1, lines))
        sliced = "\n".join(all_lines[start_idx:end_idx])
        return OpenClawMemoryReadResponse(text=sliced, path=canonical_path)
