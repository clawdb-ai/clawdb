from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class MessageIn(BaseModel):
    tenant_id: str = "default"
    session_id: str
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    channel: Optional[str] = None
    platform: Optional[str] = None
    chat_type: Optional[Literal["direct", "group", "channel", "thread"]] = None
    account_id: Optional[str] = None
    from_id: Optional[str] = None
    to_id: Optional[str] = None
    sender_id: Optional[str] = None
    sender_name: Optional[str] = None
    sender_username: Optional[str] = None
    sender_e164: Optional[str] = None
    group_id: Optional[str] = None
    group_subject: Optional[str] = None
    group_channel: Optional[str] = None
    group_space: Optional[str] = None
    native_channel_id: Optional[str] = None
    platform_message_id: Optional[str] = None
    origin_message_id: Optional[str] = None
    message_thread_id: Optional[str] = None
    thread_parent_id: Optional[str] = None
    reply_to_id: Optional[str] = None
    topic_id: Optional[str] = None
    topic_parent_id: Optional[str] = None
    topic_path: Optional[str] = None
    topic_confidence: Optional[float] = None
    topic_source: Optional[Literal["explicit", "gauss_ewens", "trie", "manual", "replay"]] = None
    capsule_level: Literal["L0", "L1", "L2"] = "L0"
    idempotency_key: Optional[str] = None
    message_id: str = Field(default_factory=lambda: str(uuid4()))
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MessageAck(BaseModel):
    status: Literal["ok"] = "ok"
    wal_seq: int
    message_id: str
    origin_message_id: Optional[str] = None
    affected_projections: int = 0


class MessageEditRequest(BaseModel):
    tenant_id: str = "default"
    origin_message_id: Optional[str] = None
    platform: Optional[str] = None
    account_id: Optional[str] = None
    platform_message_id: Optional[str] = None
    content: str
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MessageDeleteRequest(BaseModel):
    tenant_id: str = "default"
    origin_message_id: Optional[str] = None
    platform: Optional[str] = None
    account_id: Optional[str] = None
    platform_message_id: Optional[str] = None
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SearchRequest(BaseModel):
    query: str
    tenant_id: str = "default"
    session_id: Optional[str] = None
    channel: Optional[str] = None
    chat_type: Optional[Literal["direct", "group", "channel", "thread"]] = None
    group_id: Optional[str] = None
    topic_id: Optional[str] = None
    message_thread_id: Optional[str] = None
    retrieval_mode: Literal["hybrid", "lexical", "vector"] = "hybrid"
    rerank: Literal["auto", "off"] = "auto"
    max_results: int = 6
    min_score: float = 0.0


class SearchResult(BaseModel):
    path: str
    start_line: int
    end_line: int
    score: float
    score_lexical: float = 0.0
    score_semantic: float = 0.0
    score_vector: Optional[float] = None
    snippet: str
    source: Literal["memory", "sessions"] = "memory"
    source_tier: Literal["L0", "L1", "L2"] = "L0"
    entity_type: Literal["l0_abstract", "session_rollup", "topic", "capsule", "raw_message"] = (
        "raw_message"
    )
    entity_id: Optional[str] = None
    retrieval_mode: Literal["hybrid", "lexical", "vector"] = "hybrid"
    reranked: bool = False
    citation: Optional[str] = None
    citations: List[str] = Field(default_factory=list)
    channel: Optional[str] = None
    chat_type: Optional[str] = None
    account_id: Optional[str] = None
    group_id: Optional[str] = None
    topic_id: Optional[str] = None
    topic_path: Optional[str] = None
    message_thread_id: Optional[str] = None
    sender_id: Optional[str] = None
    origin_message_id: Optional[str] = None
    projection_kind: Optional[str] = None
    projection_scope: Optional[str] = None


class SearchResponse(BaseModel):
    wal_seq: int
    cache_hit: bool
    results: List[SearchResult]


class CapsuleRefreshRequest(BaseModel):
    tenant_id: str = "default"
    session_id: str


class CapsuleRefreshResponse(BaseModel):
    status: Literal["ok"] = "ok"
    wal_seq: int
    capsule_count: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    wal_replay_lag: int
    checkpoint_seq: int
    cache_hit_ratio_5m: float
    queue_backend: str
    queue_lag: int


class OpenClawMemorySearchRequest(BaseModel):
    query: str
    tenantId: Optional[str] = "default"
    maxResults: Optional[int] = 6
    minScore: Optional[float] = 0.0
    sessionKey: Optional[str] = None


class OpenClawMemoryReadRequest(BaseModel):
    relPath: str
    fromLine: Optional[int] = Field(default=1, alias="from")
    lines: Optional[int] = 200


class OpenClawMemoryReadResponse(BaseModel):
    text: str
    path: str


class SessionSnapshotRequest(BaseModel):
    tenant_id: str = "default"
    session_id: str
    note: Optional[str] = None


class SessionSnapshotResponse(BaseModel):
    status: Literal["ok"] = "ok"
    snapshot_id: str
    wal_seq: int


class SessionForkRequest(BaseModel):
    tenant_id: str = "default"
    source_session_id: str
    target_session_id: Optional[str] = None
    note: Optional[str] = None


class SessionForkResponse(BaseModel):
    status: Literal["ok"] = "ok"
    source_session_id: str
    target_session_id: str
    snapshot_id: str
    wal_seq: int


class SessionSpawnRequest(BaseModel):
    tenant_id: str = "default"
    seed_session_id: Optional[str] = None
    session_id: Optional[str] = None
    note: Optional[str] = None


class SessionSpawnResponse(BaseModel):
    status: Literal["ok"] = "ok"
    session_id: str
    parent_session_id: Optional[str] = None
    wal_seq: int


class IndexStatusResponse(BaseModel):
    status: Literal["ok"] = "ok"
    trie_topics: int
    session_count: int
    snapshot_count: int
    wal_seq: int


class IndexRebuildResponse(BaseModel):
    status: Literal["ok"] = "ok"
    wal_seq: int
    rebuilt_topics: int
    rebuilt_messages: int


class CacheHitReportResponse(BaseModel):
    memory_cache_hit_ratio_1m: float
    memory_cache_hit_ratio_5m: float
    memory_cache_hits_total: int
    memory_cache_misses_total: int
    memory_cache_evictions_total: int
    memory_cache_lookup_latency_ms_p50: float


class WalRecord(BaseModel):
    seq: int
    ts: datetime
    event_type: str
    payload: Dict[str, Any]
    checksum: int
