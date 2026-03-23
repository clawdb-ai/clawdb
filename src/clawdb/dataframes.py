from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import pandas as pd

from .lineage import (
    DM_MIRROR_PUBLIC_PROJECTION_KIND,
    MESSAGE_STATE_DELETED,
    RAW_PROJECTION_KIND,
    materialize_message_bundle,
)
from .models import SearchResult, WalRecord
from .topics import _vectorize


MESSAGES_COLUMNS = [
    "message_id",
    "origin_message_id",
    "tenant_id",
    "session_id",
    "role",
    "content",
    "ts",
    "channel",
    "chat_type",
    "account_id",
    "from_id",
    "to_id",
    "sender_id",
    "sender_name",
    "sender_username",
    "sender_e164",
    "group_id",
    "group_subject",
    "group_channel",
    "group_space",
    "native_channel_id",
    "message_thread_id",
    "thread_parent_id",
    "reply_to_id",
    "topic_id",
    "topic_parent_id",
    "topic_path",
    "topic_confidence",
    "topic_source",
    "embedding_ref",
    "capsule_level",
    "idempotency_key",
    "projection_kind",
    "projection_scope",
    "visibility",
    "platform",
    "platform_message_id",
    "native_session_id",
    "message_state",
    "updated_at",
    "deleted_at",
]

CAPSULES_COLUMNS = [
    "capsule_id",
    "tenant_id",
    "session_id",
    "topic_id",
    "summary",
    "level",
    "score",
    "updated_at",
]

SESSION_ROLLUPS_COLUMNS = [
    "rollup_id",
    "tenant_id",
    "session_id",
    "window_kind",
    "window_key",
    "bucket_start",
    "bucket_end",
    "source_first_ts",
    "source_last_ts",
    "message_count",
    "content_char_count",
    "summary",
    "vector_text",
    "vector_ref",
    "vector_dim",
    "vector_json",
    "updated_at",
]

CACHE_INDEX_COLUMNS = [
    "key",
    "tenant_id",
    "session_id",
    "query_type",
    "capsule_level",
    "entity_type",
    "entity_id",
    "last_access",
    "hit_count",
    "miss_count",
]

SESSIONS_COLUMNS = [
    "tenant_id",
    "session_id",
    "parent_session_id",
    "origin",
    "created_at",
]

SNAPSHOTS_COLUMNS = [
    "snapshot_id",
    "tenant_id",
    "session_id",
    "wal_seq",
    "note",
    "created_at",
]


MESSAGE_MULTIINDEX_LEVELS = [
    "tenant_id",
    "session_id",
    "channel",
    "chat_type",
    "group_id",
    "topic_id",
    "message_thread_id",
    "ts",
    "message_id",
]

CAPSULE_MULTIINDEX_LEVELS = [
    "tenant_id",
    "session_id",
    "capsule_id",
]

SESSION_ROLLUP_MULTIINDEX_LEVELS = [
    "tenant_id",
    "session_id",
    "window_kind",
    "window_key",
]

SESSION_MULTIINDEX_LEVELS = [
    "tenant_id",
    "session_id",
]

SNAPSHOT_MULTIINDEX_LEVELS = [
    "tenant_id",
    "session_id",
    "snapshot_id",
]

CACHE_LOOKUP_MULTIINDEX_LEVELS = [
    "key",
    "tenant_id",
    "session_id",
    "query_type",
    "capsule_level",
]


@dataclass
class DataFramesState:
    messages_df: pd.DataFrame
    capsules_df: pd.DataFrame
    session_rollups_df: pd.DataFrame
    cache_index_df: pd.DataFrame
    sessions_df: pd.DataFrame
    snapshots_df: pd.DataFrame


@dataclass(frozen=True)
class MessageMutationResult:
    origin_message_id: str
    affected_sessions: List[str]
    affected_projections: int
    found: bool


@dataclass(frozen=True)
class MessageUpsertResult:
    origin_message_id: str
    affected_sessions: List[str]
    affected_projections: int
    replaced_existing: bool


ROLLUP_WINDOW_KINDS = (
    "daily",
    "weekly",
    "monthly",
    "quarterly",
    "yearly",
    "lifetime",
)
ROLLUP_SUMMARY_MAX_CHARS = 4000
DEFAULT_ROLLUP_VECTOR_DIM = 64


def _utc_timestamp(value: object) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return pd.Timestamp.now(tz="UTC")
    return ts


def _rollup_period_start(
    ts: pd.Timestamp,
    window_kind: str,
    lifetime_start: pd.Timestamp,
) -> pd.Timestamp:
    normalized = _utc_timestamp(ts)
    if window_kind == "daily":
        return normalized.floor("D")
    if window_kind == "weekly":
        day_start = normalized.floor("D")
        return day_start - pd.Timedelta(days=int(day_start.weekday()))
    if window_kind == "monthly":
        return pd.Timestamp(year=normalized.year, month=normalized.month, day=1, tz="UTC")
    if window_kind == "quarterly":
        month = ((int(normalized.month) - 1) // 3) * 3 + 1
        return pd.Timestamp(year=normalized.year, month=month, day=1, tz="UTC")
    if window_kind == "yearly":
        return pd.Timestamp(year=normalized.year, month=1, day=1, tz="UTC")
    return _utc_timestamp(lifetime_start)


def _rollup_period_end(
    bucket_start: pd.Timestamp,
    window_kind: str,
    source_last_ts: pd.Timestamp,
) -> pd.Timestamp:
    start = _utc_timestamp(bucket_start)
    if window_kind == "daily":
        return start + pd.Timedelta(days=1)
    if window_kind == "weekly":
        return start + pd.Timedelta(days=7)
    if window_kind == "monthly":
        if start.month == 12:
            return pd.Timestamp(year=start.year + 1, month=1, day=1, tz="UTC")
        return pd.Timestamp(year=start.year, month=start.month + 1, day=1, tz="UTC")
    if window_kind == "quarterly":
        month = start.month + 3
        year = start.year
        if month > 12:
            month -= 12
            year += 1
        return pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    if window_kind == "yearly":
        return pd.Timestamp(year=start.year + 1, month=1, day=1, tz="UTC")
    return _utc_timestamp(source_last_ts)


def _rollup_window_key(bucket_start: pd.Timestamp, window_kind: str) -> str:
    start = _utc_timestamp(bucket_start)
    if window_kind == "daily":
        return start.strftime("%Y-%m-%d")
    if window_kind == "weekly":
        iso = start.isocalendar()
        return f"{int(iso.year):04d}-W{int(iso.week):02d}"
    if window_kind == "monthly":
        return start.strftime("%Y-%m")
    if window_kind == "quarterly":
        quarter = ((int(start.month) - 1) // 3) + 1
        return f"{int(start.year):04d}-Q{quarter}"
    if window_kind == "yearly":
        return start.strftime("%Y")
    return "lifetime"


def _render_rollup_summary(
    *,
    window_kind: str,
    window_key: str,
    ordered: pd.DataFrame,
    source_first_ts: pd.Timestamp,
    source_last_ts: pd.Timestamp,
    message_count: int,
    content_char_count: int,
) -> str:
    header = (
        f"{window_kind}:{window_key} "
        f"messages={message_count} "
        f"chars={content_char_count} "
        f"coverage={_utc_timestamp(source_first_ts).isoformat()}..{_utc_timestamp(source_last_ts).isoformat()}"
    )
    lines: List[str] = []
    for _, row in ordered.iterrows():
        content = str(row.get("content") or "")
        if not content:
            continue
        lines.append(f"[{str(row.get('role') or 'user')}] {content}")
    if not lines:
        return header
    body = "\n".join(lines)
    summary = f"{header}\n{body}"
    if len(summary) <= ROLLUP_SUMMARY_MAX_CHARS:
        return summary
    ellipsis = "\n...\n"
    budget = max(0, ROLLUP_SUMMARY_MAX_CHARS - len(header) - len(ellipsis))
    if budget <= 0:
        return header[:ROLLUP_SUMMARY_MAX_CHARS]
    return f"{header}{ellipsis}{body[-budget:]}"


def _serialize_vector(text: str, dim: int) -> str:
    vec = [round(float(item), 8) for item in _vectorize(text, max(8, int(dim)))]
    return json.dumps(vec, separators=(",", ":"))


def materialize_session_rollups(
    messages_frame: pd.DataFrame,
    *,
    vector_dim: int = DEFAULT_ROLLUP_VECTOR_DIM,
) -> pd.DataFrame:
    if messages_frame.empty:
        return pd.DataFrame(columns=SESSION_ROLLUPS_COLUMNS)
    scoped = messages_frame.copy().reset_index(drop=True)
    if "projection_kind" not in scoped.columns or "message_state" not in scoped.columns:
        return pd.DataFrame(columns=SESSION_ROLLUPS_COLUMNS)
    scoped = scoped[scoped["projection_kind"].astype(str) != RAW_PROJECTION_KIND]
    scoped = scoped[scoped["message_state"].astype(str) != MESSAGE_STATE_DELETED]
    scoped = scoped[scoped["session_id"].astype(str) != ""]
    if scoped.empty:
        return pd.DataFrame(columns=SESSION_ROLLUPS_COLUMNS)
    scoped["tenant_id"] = scoped["tenant_id"].fillna("default").astype(str)
    scoped["session_id"] = scoped["session_id"].fillna("").astype(str)
    scoped["role"] = scoped["role"].fillna("user").astype(str)
    scoped["content"] = scoped["content"].fillna("").astype(str)
    scoped["ts"] = pd.to_datetime(scoped["ts"], utc=True, errors="coerce")
    scoped = scoped[scoped["ts"].notna()]
    if scoped.empty:
        return pd.DataFrame(columns=SESSION_ROLLUPS_COLUMNS)

    rows: List[Dict[str, object]] = []
    materialized_at = pd.Timestamp.utcnow()
    resolved_vector_dim = max(8, int(vector_dim))
    for (tenant_id, session_id), session_subset in scoped.groupby(["tenant_id", "session_id"], sort=True):
        ordered = session_subset.sort_values("ts", kind="stable").reset_index(drop=True)
        lifetime_start = _utc_timestamp(ordered["ts"].min())
        for window_kind in ROLLUP_WINDOW_KINDS:
            bucketed = ordered.copy()
            bucketed["_bucket_start"] = bucketed["ts"].apply(
                lambda value: _rollup_period_start(
                    _utc_timestamp(value),
                    window_kind,
                    lifetime_start,
                )
            )
            for bucket_start, bucket in bucketed.groupby("_bucket_start", sort=True):
                bucket_start_ts = _utc_timestamp(bucket_start)
                bucket_ordered = bucket.sort_values("ts", kind="stable").reset_index(drop=True)
                source_first_ts = _utc_timestamp(bucket_ordered["ts"].min())
                source_last_ts = _utc_timestamp(bucket_ordered["ts"].max())
                window_key = _rollup_window_key(bucket_start_ts, window_kind)
                summary = _render_rollup_summary(
                    window_kind=window_kind,
                    window_key=window_key,
                    ordered=bucket_ordered,
                    source_first_ts=source_first_ts,
                    source_last_ts=source_last_ts,
                    message_count=int(bucket_ordered.shape[0]),
                    content_char_count=int(bucket_ordered["content"].astype(str).map(len).sum()),
                )
                vector_text = summary
                rows.append(
                    {
                        "rollup_id": f"rollup:{tenant_id}:{session_id}:{window_kind}:{window_key}",
                        "tenant_id": str(tenant_id),
                        "session_id": str(session_id),
                        "window_kind": window_kind,
                        "window_key": window_key,
                        "bucket_start": bucket_start_ts,
                        "bucket_end": _rollup_period_end(bucket_start_ts, window_kind, source_last_ts),
                        "source_first_ts": source_first_ts,
                        "source_last_ts": source_last_ts,
                        "message_count": int(bucket_ordered.shape[0]),
                        "content_char_count": int(bucket_ordered["content"].astype(str).map(len).sum()),
                        "summary": summary,
                        "vector_text": vector_text,
                        "vector_ref": (
                            "session_rollup:"
                            f"{hashlib.sha256(vector_text.encode('utf-8')).hexdigest()}"
                        ),
                        "vector_dim": resolved_vector_dim,
                        "vector_json": _serialize_vector(vector_text, resolved_vector_dim),
                        "updated_at": materialized_at,
                    }
                )
    if not rows:
        return pd.DataFrame(columns=SESSION_ROLLUPS_COLUMNS)
    frame = pd.DataFrame(rows, columns=SESSION_ROLLUPS_COLUMNS)
    for col in ["bucket_start", "bucket_end", "source_first_ts", "source_last_ts", "updated_at"]:
        frame[col] = pd.to_datetime(frame[col], utc=True, errors="coerce")
    for col in ["message_count", "content_char_count", "vector_dim"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0).astype(int)
    return frame[SESSION_ROLLUPS_COLUMNS]


class DataFrameStore:
    def __init__(self) -> None:
        self._state = DataFramesState(
            messages_df=pd.DataFrame(columns=MESSAGES_COLUMNS),
            capsules_df=pd.DataFrame(columns=CAPSULES_COLUMNS),
            session_rollups_df=pd.DataFrame(columns=SESSION_ROLLUPS_COLUMNS),
            cache_index_df=pd.DataFrame(columns=CACHE_INDEX_COLUMNS),
            sessions_df=pd.DataFrame(columns=SESSIONS_COLUMNS),
            snapshots_df=pd.DataFrame(columns=SNAPSHOTS_COLUMNS),
        )
        self._state.messages_df = self._state.messages_df.astype(
            {
                "message_id": "string",
                "origin_message_id": "string",
                "tenant_id": "string",
                "session_id": "string",
                "role": "string",
                "content": "string",
                "channel": "string",
                "chat_type": "string",
                "account_id": "string",
                "from_id": "string",
                "to_id": "string",
                "sender_id": "string",
                "sender_name": "string",
                "sender_username": "string",
                "sender_e164": "string",
                "group_id": "string",
                "group_subject": "string",
                "group_channel": "string",
                "group_space": "string",
                "native_channel_id": "string",
                "message_thread_id": "string",
                "thread_parent_id": "string",
                "reply_to_id": "string",
                "topic_id": "string",
                "topic_parent_id": "string",
                "topic_path": "string",
                "topic_source": "string",
                "embedding_ref": "string",
                "capsule_level": "string",
                "idempotency_key": "string",
                "projection_kind": "string",
                "projection_scope": "string",
                "visibility": "string",
                "platform": "string",
                "platform_message_id": "string",
                "native_session_id": "string",
                "message_state": "string",
            }
        )
        self._state.session_rollups_df = self._state.session_rollups_df.astype(
            {
                "rollup_id": "string",
                "tenant_id": "string",
                "session_id": "string",
                "window_kind": "string",
                "window_key": "string",
                "summary": "string",
                "vector_text": "string",
                "vector_ref": "string",
                "vector_dim": "int64",
                "vector_json": "string",
                "message_count": "int64",
                "content_char_count": "int64",
            }
        )
        self._state.sessions_df = self._state.sessions_df.astype(
            {
                "tenant_id": "string",
                "session_id": "string",
                "parent_session_id": "string",
                "origin": "string",
            }
        )
        self._state.snapshots_df = self._state.snapshots_df.astype(
            {
                "snapshot_id": "string",
                "tenant_id": "string",
                "session_id": "string",
                "wal_seq": "int64",
                "note": "string",
            }
        )
        self._lock = asyncio.Lock()
        self._messages_indexed_df: Optional[pd.DataFrame] = None
        self._messages_index_dirty = True
        self._capsules_indexed_df: Optional[pd.DataFrame] = None
        self._capsules_index_dirty = True
        self._session_rollups_indexed_df: Optional[pd.DataFrame] = None
        self._session_rollups_index_dirty = True
        self._sessions_indexed_df: Optional[pd.DataFrame] = None
        self._sessions_index_dirty = True
        self._snapshots_indexed_df: Optional[pd.DataFrame] = None
        self._snapshots_index_dirty = True
        self._cache_lookup_indexed_df: Optional[pd.DataFrame] = None
        self._cache_lookup_index_dirty = True

    def _invalidate_messages_index_locked(self) -> None:
        self._messages_indexed_df = None
        self._messages_index_dirty = True

    def _invalidate_capsules_index_locked(self) -> None:
        self._capsules_indexed_df = None
        self._capsules_index_dirty = True

    def _invalidate_session_rollups_index_locked(self) -> None:
        self._session_rollups_indexed_df = None
        self._session_rollups_index_dirty = True

    def _invalidate_sessions_index_locked(self) -> None:
        self._sessions_indexed_df = None
        self._sessions_index_dirty = True

    def _invalidate_snapshots_index_locked(self) -> None:
        self._snapshots_indexed_df = None
        self._snapshots_index_dirty = True

    def _invalidate_cache_lookup_index_locked(self) -> None:
        self._cache_lookup_indexed_df = None
        self._cache_lookup_index_dirty = True

    def _invalidate_all_indexes_locked(self) -> None:
        self._invalidate_messages_index_locked()
        self._invalidate_capsules_index_locked()
        self._invalidate_session_rollups_index_locked()
        self._invalidate_sessions_index_locked()
        self._invalidate_snapshots_index_locked()
        self._invalidate_cache_lookup_index_locked()

    def _build_messages_index_locked(self) -> pd.DataFrame:
        df = self._state.messages_df
        if df.empty:
            empty = df.copy()
            self._messages_indexed_df = empty.set_index(MESSAGE_MULTIINDEX_LEVELS, drop=False)
            self._messages_index_dirty = False
            return self._messages_indexed_df
        indexed = df.copy()
        indexed["tenant_id"] = indexed["tenant_id"].fillna("default").astype(str)
        indexed["session_id"] = indexed["session_id"].fillna("default").astype(str)
        indexed["channel"] = indexed["channel"].fillna("").astype(str)
        indexed["chat_type"] = indexed["chat_type"].fillna("").astype(str)
        indexed["group_id"] = indexed["group_id"].fillna("").astype(str)
        indexed["topic_id"] = indexed["topic_id"].fillna("default").astype(str)
        indexed["message_thread_id"] = indexed["message_thread_id"].fillna("").astype(str)
        indexed["message_id"] = indexed["message_id"].fillna("").astype(str)
        indexed["origin_message_id"] = indexed["origin_message_id"].fillna("").astype(str)
        indexed["projection_kind"] = indexed["projection_kind"].fillna("").astype(str)
        indexed["projection_scope"] = indexed["projection_scope"].fillna("").astype(str)
        indexed["native_session_id"] = indexed["native_session_id"].fillna("").astype(str)
        indexed["message_state"] = indexed["message_state"].fillna("active").astype(str)
        indexed["ts"] = pd.to_datetime(indexed["ts"], utc=True, errors="coerce")
        indexed["ts"] = indexed["ts"].fillna(pd.Timestamp.now(tz="UTC"))
        indexed["updated_at"] = pd.to_datetime(indexed["updated_at"], utc=True, errors="coerce")
        indexed["updated_at"] = indexed["updated_at"].fillna(indexed["ts"])
        indexed["deleted_at"] = pd.to_datetime(indexed["deleted_at"], utc=True, errors="coerce")
        indexed = indexed.set_index(MESSAGE_MULTIINDEX_LEVELS, drop=False).sort_index(kind="stable")
        self._messages_indexed_df = indexed
        self._messages_index_dirty = False
        return indexed

    def _resolve_session_ids_locked(
        self,
        tenant_id: str,
        session_id: str,
        *,
        include_mirrors: bool = False,
    ) -> List[str]:
        if self._messages_index_dirty or self._messages_indexed_df is None:
            indexed = self._build_messages_index_locked()
        else:
            indexed = self._messages_indexed_df
        if indexed.empty:
            return []
        tenant_mask = (
            indexed["tenant_id"].astype(str) == str(tenant_id)
            if tenant_id != "*"
            else pd.Series([True] * len(indexed), index=indexed.index)
        )
        exact = indexed[tenant_mask & (indexed["session_id"].astype(str) == str(session_id))]
        if not exact.empty:
            return [str(session_id)]
        alias = indexed[tenant_mask & (indexed["native_session_id"].astype(str) == str(session_id))]
        if not include_mirrors:
            alias = alias[alias["projection_kind"].astype(str) != DM_MIRROR_PUBLIC_PROJECTION_KIND]
        resolved = alias["session_id"].astype(str).dropna().unique().tolist()
        return sorted([item for item in resolved if item])

    def _filter_message_rows(
        self,
        df: pd.DataFrame,
        *,
        row_mode: Literal["projection", "raw", "all"],
        include_deleted: bool,
    ) -> pd.DataFrame:
        if df.empty:
            return df
        out = df
        if row_mode == "projection":
            out = out[out["projection_kind"].astype(str) != RAW_PROJECTION_KIND]
        elif row_mode == "raw":
            out = out[out["projection_kind"].astype(str) == RAW_PROJECTION_KIND]
        if not include_deleted:
            out = out[out["message_state"].astype(str) != MESSAGE_STATE_DELETED]
        return out

    def _messages_for_query_locked(
        self,
        *,
        tenant_id: str,
        session_id: Optional[str],
        channel: Optional[str] = None,
        chat_type: Optional[str] = None,
        group_id: Optional[str] = None,
        topic_id: Optional[str] = None,
        message_thread_id: Optional[str] = None,
        row_mode: Literal["projection", "raw", "all"] = "projection",
        include_deleted: bool = False,
    ) -> pd.DataFrame:
        if self._messages_index_dirty or self._messages_indexed_df is None:
            indexed = self._build_messages_index_locked()
        else:
            indexed = self._messages_indexed_df
        if indexed.empty:
            return indexed
        session_ids: List[str] = []
        if session_id is not None:
            session_ids = self._resolve_session_ids_locked(tenant_id, str(session_id))
            if not session_ids:
                return indexed.iloc[0:0].copy()
        tenant_mask = (
            indexed["tenant_id"].astype(str) == str(tenant_id)
            if tenant_id != "*"
            else pd.Series([True] * len(indexed), index=indexed.index)
        )
        scoped = indexed[tenant_mask]
        if session_ids:
            scoped = scoped[scoped["session_id"].astype(str).isin(session_ids)]
        if channel is not None:
            scoped = scoped[scoped["channel"].astype(str) == str(channel)]
        if chat_type is not None:
            scoped = scoped[scoped["chat_type"].astype(str) == str(chat_type)]
        if group_id is not None:
            scoped = scoped[scoped["group_id"].astype(str) == str(group_id)]
        if topic_id is not None:
            scoped = scoped[scoped["topic_id"].astype(str) == str(topic_id)]
        if message_thread_id is not None:
            scoped = scoped[scoped["message_thread_id"].astype(str) == str(message_thread_id)]
        return self._filter_message_rows(
            scoped,
            row_mode=row_mode,
            include_deleted=include_deleted,
        )

    def _build_capsules_index_locked(self) -> pd.DataFrame:
        df = self._state.capsules_df
        if df.empty:
            empty = df.copy()
            self._capsules_indexed_df = empty.set_index(CAPSULE_MULTIINDEX_LEVELS, drop=False)
            self._capsules_index_dirty = False
            return self._capsules_indexed_df
        indexed = df.copy()
        indexed["tenant_id"] = indexed["tenant_id"].fillna("default").astype(str)
        indexed["session_id"] = indexed["session_id"].fillna("default").astype(str)
        indexed["capsule_id"] = indexed["capsule_id"].fillna("").astype(str)
        indexed = indexed.set_index(CAPSULE_MULTIINDEX_LEVELS, drop=False).sort_index(kind="stable")
        self._capsules_indexed_df = indexed
        self._capsules_index_dirty = False
        return indexed

    def _capsules_for_query_locked(self, tenant_id: str, session_id: str) -> pd.DataFrame:
        if self._capsules_index_dirty or self._capsules_indexed_df is None:
            indexed = self._build_capsules_index_locked()
        else:
            indexed = self._capsules_indexed_df
        if indexed.empty:
            return indexed
        session_ids = self._resolve_session_ids_locked(tenant_id, str(session_id))
        if not session_ids:
            session_ids = [str(session_id)]
        scoped = indexed[
            (indexed["tenant_id"].astype(str) == str(tenant_id))
            & (indexed["session_id"].astype(str).isin(session_ids))
        ]
        if scoped.empty:
            return indexed.iloc[0:0].copy()
        return scoped

    def _build_session_rollups_index_locked(self) -> pd.DataFrame:
        df = self._state.session_rollups_df
        if df.empty:
            empty = df.copy()
            self._session_rollups_indexed_df = empty.set_index(
                SESSION_ROLLUP_MULTIINDEX_LEVELS,
                drop=False,
            )
            self._session_rollups_index_dirty = False
            return self._session_rollups_indexed_df
        indexed = df.copy()
        indexed["tenant_id"] = indexed["tenant_id"].fillna("default").astype(str)
        indexed["session_id"] = indexed["session_id"].fillna("default").astype(str)
        indexed["window_kind"] = indexed["window_kind"].fillna("").astype(str)
        indexed["window_key"] = indexed["window_key"].fillna("").astype(str)
        indexed["bucket_start"] = pd.to_datetime(indexed["bucket_start"], utc=True, errors="coerce")
        indexed["updated_at"] = pd.to_datetime(indexed["updated_at"], utc=True, errors="coerce")
        indexed = indexed.set_index(SESSION_ROLLUP_MULTIINDEX_LEVELS, drop=False).sort_index(kind="stable")
        self._session_rollups_indexed_df = indexed
        self._session_rollups_index_dirty = False
        return indexed

    def _session_rollups_for_query_locked(
        self,
        tenant_id: str,
        session_id: str,
        window_kind: Optional[str] = None,
    ) -> pd.DataFrame:
        if self._session_rollups_index_dirty or self._session_rollups_indexed_df is None:
            indexed = self._build_session_rollups_index_locked()
        else:
            indexed = self._session_rollups_indexed_df
        if indexed.empty:
            return indexed
        session_ids = self._resolve_session_ids_locked(tenant_id, str(session_id))
        if not session_ids:
            session_ids = [str(session_id)]
        scoped = indexed[
            (indexed["tenant_id"].astype(str) == str(tenant_id))
            & (indexed["session_id"].astype(str).isin(session_ids))
        ]
        if window_kind is not None:
            scoped = scoped[scoped["window_kind"].astype(str) == str(window_kind)]
        if scoped.empty:
            return indexed.iloc[0:0].copy()
        return scoped

    def _build_sessions_index_locked(self) -> pd.DataFrame:
        df = self._state.sessions_df
        if df.empty:
            empty = df.copy()
            self._sessions_indexed_df = empty.set_index(SESSION_MULTIINDEX_LEVELS, drop=False)
            self._sessions_index_dirty = False
            return self._sessions_indexed_df
        indexed = df.copy()
        indexed["tenant_id"] = indexed["tenant_id"].fillna("default").astype(str)
        indexed["session_id"] = indexed["session_id"].fillna("default").astype(str)
        indexed = indexed.set_index(SESSION_MULTIINDEX_LEVELS, drop=False).sort_index(kind="stable")
        self._sessions_indexed_df = indexed
        self._sessions_index_dirty = False
        return indexed

    def _session_exists_locked(self, tenant_id: str, session_id: str) -> bool:
        if self._sessions_index_dirty or self._sessions_indexed_df is None:
            indexed = self._build_sessions_index_locked()
        else:
            indexed = self._sessions_indexed_df
        if indexed.empty:
            return False
        return (str(tenant_id), str(session_id)) in indexed.index

    def _build_snapshots_index_locked(self) -> pd.DataFrame:
        df = self._state.snapshots_df
        if df.empty:
            empty = df.copy()
            self._snapshots_indexed_df = empty.set_index(SNAPSHOT_MULTIINDEX_LEVELS, drop=False)
            self._snapshots_index_dirty = False
            return self._snapshots_indexed_df
        indexed = df.copy()
        indexed["tenant_id"] = indexed["tenant_id"].fillna("default").astype(str)
        indexed["session_id"] = indexed["session_id"].fillna("default").astype(str)
        indexed["snapshot_id"] = indexed["snapshot_id"].fillna("").astype(str)
        indexed = indexed.set_index(SNAPSHOT_MULTIINDEX_LEVELS, drop=False).sort_index(kind="stable")
        self._snapshots_indexed_df = indexed
        self._snapshots_index_dirty = False
        return indexed

    def _snapshots_for_query_locked(self, tenant_id: str, session_id: str) -> pd.DataFrame:
        if self._snapshots_index_dirty or self._snapshots_indexed_df is None:
            indexed = self._build_snapshots_index_locked()
        else:
            indexed = self._snapshots_indexed_df
        if indexed.empty:
            return indexed
        key = (str(tenant_id), str(session_id), slice(None))
        try:
            scoped = indexed.loc[key]
        except KeyError:
            return indexed.iloc[0:0].copy()
        if isinstance(scoped, pd.Series):
            return scoped.to_frame().T
        return scoped

    def _build_cache_lookup_index_locked(self) -> pd.DataFrame:
        df = self._state.cache_index_df
        if df.empty:
            empty = df.copy()
            empty["_row_id"] = pd.Series(dtype="int64")
            self._cache_lookup_indexed_df = empty.set_index(CACHE_LOOKUP_MULTIINDEX_LEVELS, drop=False)
            self._cache_lookup_index_dirty = False
            return self._cache_lookup_indexed_df
        indexed = df.copy()
        indexed["key"] = indexed["key"].fillna("").astype(str)
        indexed["tenant_id"] = indexed["tenant_id"].fillna("default").astype(str)
        indexed["session_id"] = indexed["session_id"].fillna("_").astype(str)
        indexed["query_type"] = indexed["query_type"].fillna("memory_search").astype(str)
        indexed["capsule_level"] = indexed["capsule_level"].fillna("mixed").astype(str)
        indexed["_row_id"] = indexed.index.astype(int)
        indexed = indexed.set_index(CACHE_LOOKUP_MULTIINDEX_LEVELS, drop=False).sort_index(kind="stable")
        self._cache_lookup_indexed_df = indexed
        self._cache_lookup_index_dirty = False
        return indexed

    def _cache_lookup_row_id_locked(
        self,
        *,
        key: str,
        tenant_id: str,
        session_id: str,
        query_type: str,
        capsule_level: str,
    ) -> Optional[int]:
        if self._cache_lookup_index_dirty or self._cache_lookup_indexed_df is None:
            indexed = self._build_cache_lookup_index_locked()
        else:
            indexed = self._cache_lookup_indexed_df
        if indexed.empty:
            return None
        lookup_key = (
            str(key),
            str(tenant_id),
            str(session_id),
            str(query_type),
            str(capsule_level),
        )
        try:
            hit = indexed.loc[lookup_key]
        except KeyError:
            return None
        if isinstance(hit, pd.Series):
            return int(hit["_row_id"])
        return int(hit.iloc[0]["_row_id"])

    def _chronological_messages(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        return df.reset_index(drop=True).sort_values("ts", kind="stable")

    @property
    def state(self) -> DataFramesState:
        return self._state

    async def ensure_session(
        self,
        tenant_id: str,
        session_id: str,
        parent_session_id: Optional[str] = None,
        origin: str = "normal",
    ) -> None:
        async with self._lock:
            if self._session_exists_locked(tenant_id, session_id):
                return
            df = self._state.sessions_df
            row = {
                "tenant_id": tenant_id,
                "session_id": session_id,
                "parent_session_id": parent_session_id or "",
                "origin": origin,
                "created_at": pd.Timestamp.utcnow(),
            }
            row_df = pd.DataFrame([row], columns=SESSIONS_COLUMNS)
            if df.empty:
                self._state.sessions_df = row_df
            else:
                self._state.sessions_df = pd.concat(
                    [df, row_df],
                    ignore_index=True,
                )
            self._invalidate_sessions_index_locked()

    async def add_message(self, payload: Dict[str, object]) -> None:
        await self.apply_message_bundle(materialize_message_bundle(payload))

    async def apply_message_bundle(self, bundle: Dict[str, object]) -> MessageUpsertResult:
        tenant_id = str(bundle.get("tenant_id") or "default")
        projections = list(bundle.get("projections") or [])
        for row in projections:
            session_id = str(row.get("session_id") or "")
            if session_id:
                await self.ensure_session(tenant_id=tenant_id, session_id=session_id)
        async with self._lock:
            origin_message_id = str(bundle.get("origin_message_id") or "")
            existing = self._state.messages_df[
                self._state.messages_df["origin_message_id"].astype(str) == origin_message_id
            ]
            replaced_existing = not existing.empty
            if replaced_existing:
                self._state.messages_df = self._state.messages_df[
                    self._state.messages_df["origin_message_id"].astype(str) != origin_message_id
                ]
            rows = [dict(bundle["raw_message"]), *[dict(item) for item in projections]]
            row_df = pd.DataFrame(rows, columns=MESSAGES_COLUMNS)
            if self._state.messages_df.empty:
                self._state.messages_df = row_df
            else:
                self._state.messages_df = pd.concat(
                    [self._state.messages_df, row_df],
                    ignore_index=True,
                )
            self._invalidate_messages_index_locked()
            affected_sessions = sorted(
                {
                    str(row.get("session_id") or "")
                    for row in projections
                    if str(row.get("session_id") or "")
                }
            )
            return MessageUpsertResult(
                origin_message_id=origin_message_id,
                affected_sessions=affected_sessions,
                affected_projections=len(projections),
                replaced_existing=replaced_existing,
            )

    async def resolve_origin_message_id(
        self,
        *,
        tenant_id: str,
        origin_message_id: Optional[str] = None,
        platform: Optional[str] = None,
        account_id: Optional[str] = None,
        platform_message_id: Optional[str] = None,
    ) -> Optional[str]:
        explicit = str(origin_message_id or "").strip()
        async with self._lock:
            if explicit:
                mask = (
                    (self._state.messages_df["tenant_id"].astype(str) == str(tenant_id))
                    & (self._state.messages_df["origin_message_id"].astype(str) == explicit)
                )
                if mask.any():
                    return explicit
                return None
            platform_message_id_text = str(platform_message_id or "").strip()
            if not platform_message_id_text:
                return None
            df = self._state.messages_df
            mask = (
                (df["tenant_id"].astype(str) == str(tenant_id))
                & (df["projection_kind"].astype(str) == RAW_PROJECTION_KIND)
                & (df["platform_message_id"].astype(str) == platform_message_id_text)
            )
            if platform is not None:
                mask &= df["platform"].astype(str) == str(platform)
            if account_id is not None and str(account_id).strip():
                mask &= df["account_id"].astype(str) == str(account_id)
            matches = df[mask]["origin_message_id"].astype(str).dropna().unique().tolist()
            if not matches:
                return None
            if len(matches) > 1:
                raise ValueError(
                    f"platform_message_id resolved to multiple origin_message_id values: {platform_message_id_text}"
                )
            return str(matches[0])

    async def edit_message(
        self,
        *,
        tenant_id: str,
        origin_message_id: str,
        content: str,
        edited_at: datetime,
    ) -> MessageMutationResult:
        async with self._lock:
            mask = (
                (self._state.messages_df["tenant_id"].astype(str) == str(tenant_id))
                & (self._state.messages_df["origin_message_id"].astype(str) == str(origin_message_id))
            )
            if not mask.any():
                return MessageMutationResult(
                    origin_message_id=origin_message_id,
                    affected_sessions=[],
                    affected_projections=0,
                    found=False,
                )
            rows = self._state.messages_df[mask]
            if (rows["message_state"].astype(str) == MESSAGE_STATE_DELETED).all():
                return MessageMutationResult(
                    origin_message_id=origin_message_id,
                    affected_sessions=[],
                    affected_projections=0,
                    found=False,
                )
            projection_rows = rows[rows["projection_kind"].astype(str) != RAW_PROJECTION_KIND]
            edited_at_text = pd.to_datetime(edited_at, utc=True).isoformat()
            self._state.messages_df.loc[mask, "content"] = str(content)
            self._state.messages_df.loc[mask, "updated_at"] = edited_at_text
            self._state.messages_df.loc[mask, "message_state"] = "edited"
            self._state.messages_df.loc[mask, "deleted_at"] = None
            self._invalidate_messages_index_locked()
            return MessageMutationResult(
                origin_message_id=origin_message_id,
                affected_sessions=sorted(
                    {
                        str(item)
                        for item in projection_rows["session_id"].astype(str).tolist()
                        if str(item)
                    }
                ),
                affected_projections=int(projection_rows.shape[0]),
                found=True,
            )

    async def delete_message(
        self,
        *,
        tenant_id: str,
        origin_message_id: str,
        deleted_at: datetime,
    ) -> MessageMutationResult:
        async with self._lock:
            mask = (
                (self._state.messages_df["tenant_id"].astype(str) == str(tenant_id))
                & (self._state.messages_df["origin_message_id"].astype(str) == str(origin_message_id))
            )
            if not mask.any():
                return MessageMutationResult(
                    origin_message_id=origin_message_id,
                    affected_sessions=[],
                    affected_projections=0,
                    found=False,
                )
            rows = self._state.messages_df[mask]
            projection_rows = rows[rows["projection_kind"].astype(str) != RAW_PROJECTION_KIND]
            self._state.messages_df.loc[mask, "content"] = ""
            self._state.messages_df.loc[mask, "message_state"] = MESSAGE_STATE_DELETED
            deleted_at_text = pd.to_datetime(deleted_at, utc=True).isoformat()
            self._state.messages_df.loc[mask, "updated_at"] = deleted_at_text
            self._state.messages_df.loc[mask, "deleted_at"] = deleted_at_text
            self._invalidate_messages_index_locked()
            return MessageMutationResult(
                origin_message_id=origin_message_id,
                affected_sessions=sorted(
                    {
                        str(item)
                        for item in projection_rows["session_id"].astype(str).tolist()
                        if str(item)
                    }
                ),
                affected_projections=int(projection_rows.shape[0]),
                found=True,
            )

    async def count_topic_messages(self, tenant_id: str, topic_id: str) -> int:
        async with self._lock:
            scoped = self._messages_for_query_locked(
                tenant_id=tenant_id,
                session_id=None,
                topic_id=topic_id,
                row_mode="raw",
            )
            return int(scoped.shape[0])

    async def rebuild_all_session_rollups(self, *, vector_dim: int = DEFAULT_ROLLUP_VECTOR_DIM) -> int:
        async with self._lock:
            self._state.session_rollups_df = materialize_session_rollups(
                self._state.messages_df,
                vector_dim=vector_dim,
            )
            self._invalidate_session_rollups_index_locked()
            return int(self._state.session_rollups_df.shape[0])

    async def refresh_session_rollups(
        self,
        tenant_id: str,
        session_id: str,
        *,
        vector_dim: int = DEFAULT_ROLLUP_VECTOR_DIM,
    ) -> int:
        async with self._lock:
            session_ids = self._resolve_session_ids_locked(tenant_id, str(session_id))
            if not session_ids:
                session_ids = [str(session_id)]
            self._state.session_rollups_df = self._state.session_rollups_df[
                ~(
                    (self._state.session_rollups_df["tenant_id"].astype(str) == str(tenant_id))
                    & (self._state.session_rollups_df["session_id"].astype(str).isin(session_ids))
                )
            ]
            subset = self._messages_for_query_locked(
                tenant_id=tenant_id,
                session_id=session_id,
                row_mode="projection",
            )
            materialized = materialize_session_rollups(subset, vector_dim=vector_dim)
            if materialized.empty:
                self._invalidate_session_rollups_index_locked()
                return 0
            self._state.session_rollups_df = pd.concat(
                [self._state.session_rollups_df, materialized[SESSION_ROLLUPS_COLUMNS]],
                ignore_index=True,
            )
            self._invalidate_session_rollups_index_locked()
            return int(materialized.shape[0])

    async def refresh_capsules(self, tenant_id: str, session_id: str) -> int:
        async with self._lock:
            session_ids = self._resolve_session_ids_locked(tenant_id, str(session_id))
            if not session_ids:
                session_ids = [str(session_id)]
            self._state.capsules_df = self._state.capsules_df[
                ~(
                    (self._state.capsules_df["tenant_id"].astype(str) == str(tenant_id))
                    & (self._state.capsules_df["session_id"].astype(str).isin(session_ids))
                )
            ]
            self._invalidate_capsules_index_locked()
            subset = self._messages_for_query_locked(
                tenant_id=tenant_id,
                session_id=session_id,
                row_mode="projection",
            )
            if subset.empty:
                return 0
            rows: List[Dict[str, object]] = []
            for resolved_session_id in sorted(subset["session_id"].astype(str).unique().tolist()):
                session_subset = subset[subset["session_id"].astype(str) == resolved_session_id]
                lifetime_rollups = self._session_rollups_for_query_locked(
                    tenant_id=tenant_id,
                    session_id=resolved_session_id,
                    window_kind="lifetime",
                )
                if lifetime_rollups.empty:
                    ordered = self._chronological_messages(session_subset)
                    summary = " ".join(ordered["content"].astype(str).tail(20).tolist())[:2000]
                else:
                    latest = lifetime_rollups.sort_values("updated_at", kind="stable").iloc[-1]
                    summary = str(latest.get("summary") or "")
                rows.append(
                    {
                        "capsule_id": f"caps-{tenant_id}-{resolved_session_id}",
                        "tenant_id": tenant_id,
                        "session_id": resolved_session_id,
                        "topic_id": "default",
                        "summary": summary,
                        "level": "L1",
                        "score": 1.0,
                        "updated_at": pd.Timestamp.utcnow(),
                    }
                )
            self._state.capsules_df = pd.concat(
                [self._state.capsules_df, pd.DataFrame(rows, columns=CAPSULES_COLUMNS)],
                ignore_index=True,
            )
            self._invalidate_capsules_index_locked()
            return len(rows)

    async def create_snapshot(
        self,
        tenant_id: str,
        session_id: str,
        snapshot_id: str,
        wal_seq: int,
        note: str = "",
    ) -> None:
        await self.ensure_session(tenant_id=tenant_id, session_id=session_id)
        async with self._lock:
            row = {
                "snapshot_id": snapshot_id,
                "tenant_id": tenant_id,
                "session_id": session_id,
                "wal_seq": int(wal_seq),
                "note": note,
                "created_at": pd.Timestamp.utcnow(),
            }
            row_df = pd.DataFrame([row], columns=SNAPSHOTS_COLUMNS)
            if self._state.snapshots_df.empty:
                self._state.snapshots_df = row_df
            else:
                self._state.snapshots_df = pd.concat(
                    [self._state.snapshots_df, row_df],
                    ignore_index=True,
                )
            self._invalidate_snapshots_index_locked()

    async def fork_session(
        self,
        tenant_id: str,
        source_session_id: str,
        target_session_id: str,
    ) -> None:
        await self.ensure_session(
            tenant_id=tenant_id,
            session_id=target_session_id,
            parent_session_id=source_session_id,
            origin="fork",
        )
        async with self._lock:
            src_msg = self._messages_for_query_locked(
                tenant_id=tenant_id,
                session_id=source_session_id,
            )
            if not src_msg.empty:
                copied = src_msg.copy()
                copied["session_id"] = target_session_id
                self._state.messages_df = pd.concat(
                    [self._state.messages_df, copied[MESSAGES_COLUMNS]],
                    ignore_index=True,
                )
                self._invalidate_messages_index_locked()
            src_caps = self._state.capsules_df[
                (self._state.capsules_df["tenant_id"].astype(str) == tenant_id)
                & (self._state.capsules_df["session_id"].astype(str) == source_session_id)
            ]
            if not src_caps.empty:
                copied_caps = src_caps.copy()
                copied_caps["session_id"] = target_session_id
                copied_caps["capsule_id"] = copied_caps["capsule_id"].astype(str) + f"-fork-{target_session_id}"
                self._state.capsules_df = pd.concat(
                    [self._state.capsules_df, copied_caps[CAPSULES_COLUMNS]],
                    ignore_index=True,
                )
                self._invalidate_capsules_index_locked()
            src_rollups = self._state.session_rollups_df[
                (self._state.session_rollups_df["tenant_id"].astype(str) == tenant_id)
                & (self._state.session_rollups_df["session_id"].astype(str) == source_session_id)
            ]
            if not src_rollups.empty:
                copied_rollups = src_rollups.copy()
                copied_rollups["session_id"] = target_session_id
                copied_rollups["rollup_id"] = copied_rollups.apply(
                    lambda row: (
                        f"rollup:{tenant_id}:{target_session_id}:"
                        f"{str(row.get('window_kind') or '')}:{str(row.get('window_key') or '')}"
                    ),
                    axis=1,
                )
                self._state.session_rollups_df = pd.concat(
                    [self._state.session_rollups_df, copied_rollups[SESSION_ROLLUPS_COLUMNS]],
                    ignore_index=True,
                )
                self._invalidate_session_rollups_index_locked()

    async def spawn_session(
        self,
        tenant_id: str,
        session_id: str,
        parent_session_id: Optional[str],
    ) -> None:
        await self.ensure_session(
            tenant_id=tenant_id,
            session_id=session_id,
            parent_session_id=parent_session_id,
            origin="spawn",
        )

    async def list_snapshots(self, tenant_id: str, session_id: str) -> List[Dict[str, object]]:
        async with self._lock:
            df = self._snapshots_for_query_locked(tenant_id, session_id)
            df = df.reset_index(drop=True).sort_values("created_at", kind="stable")
            out: List[Dict[str, object]] = []
            for _, row in df.iterrows():
                out.append(
                    {
                        "snapshot_id": str(row["snapshot_id"]),
                        "tenant_id": str(row["tenant_id"]),
                        "session_id": str(row["session_id"]),
                        "wal_seq": int(row["wal_seq"]),
                        "note": str(row.get("note") or ""),
                        "created_at": pd.to_datetime(row["created_at"], utc=True).isoformat(),
                    }
                )
            return out

    async def session_count(self) -> int:
        async with self._lock:
            return int(self._state.sessions_df.shape[0])

    async def snapshot_count(self) -> int:
        async with self._lock:
            return int(self._state.snapshots_df.shape[0])

    async def apply_wal_record(self, record: WalRecord) -> None:
        if record.event_type == "message_upsert":
            payload = record.payload
            if "raw_message" in payload and "projections" in payload:
                await self.apply_message_bundle(payload)
            else:
                await self.apply_message_bundle(materialize_message_bundle(payload))
        elif record.event_type == "message_edit":
            resolved_origin = await self.resolve_origin_message_id(
                tenant_id=str(record.payload.get("tenant_id") or "default"),
                origin_message_id=str(record.payload.get("origin_message_id") or "") or None,
                platform=str(record.payload.get("platform") or "") or None,
                account_id=str(record.payload.get("account_id") or "") or None,
                platform_message_id=str(record.payload.get("platform_message_id") or "") or None,
            )
            if resolved_origin:
                await self.edit_message(
                    tenant_id=str(record.payload.get("tenant_id") or "default"),
                    origin_message_id=resolved_origin,
                    content=str(record.payload.get("content") or ""),
                    edited_at=pd.to_datetime(record.payload.get("ts"), utc=True).to_pydatetime(),
                )
        elif record.event_type == "message_delete":
            resolved_origin = await self.resolve_origin_message_id(
                tenant_id=str(record.payload.get("tenant_id") or "default"),
                origin_message_id=str(record.payload.get("origin_message_id") or "") or None,
                platform=str(record.payload.get("platform") or "") or None,
                account_id=str(record.payload.get("account_id") or "") or None,
                platform_message_id=str(record.payload.get("platform_message_id") or "") or None,
            )
            if resolved_origin:
                await self.delete_message(
                    tenant_id=str(record.payload.get("tenant_id") or "default"),
                    origin_message_id=resolved_origin,
                    deleted_at=pd.to_datetime(record.payload.get("ts"), utc=True).to_pydatetime(),
                )
        elif record.event_type == "capsule_refresh":
            await self.refresh_capsules(
                str(record.payload.get("tenant_id") or "default"),
                str(record.payload["session_id"]),
            )
        elif record.event_type == "session_snapshot":
            await self.create_snapshot(
                tenant_id=str(record.payload.get("tenant_id") or "default"),
                session_id=str(record.payload["session_id"]),
                snapshot_id=str(record.payload["snapshot_id"]),
                wal_seq=int(record.payload["wal_seq"]),
                note=str(record.payload.get("note") or ""),
            )
        elif record.event_type == "session_fork":
            await self.fork_session(
                tenant_id=str(record.payload.get("tenant_id") or "default"),
                source_session_id=str(record.payload["source_session_id"]),
                target_session_id=str(record.payload["target_session_id"]),
            )
            if record.payload.get("snapshot_id"):
                await self.create_snapshot(
                    tenant_id=str(record.payload.get("tenant_id") or "default"),
                    session_id=str(record.payload["source_session_id"]),
                    snapshot_id=str(record.payload["snapshot_id"]),
                    wal_seq=int(record.seq),
                    note=str(record.payload.get("note") or "fork"),
                )
        elif record.event_type == "session_spawn":
            await self.spawn_session(
                tenant_id=str(record.payload.get("tenant_id") or "default"),
                session_id=str(record.payload["session_id"]),
                parent_session_id=str(record.payload.get("parent_session_id") or "") or None,
            )

    async def record_cache_lookup(
        self,
        *,
        key: str,
        tenant_id: str,
        session_id: str,
        query_type: str,
        capsule_level: str,
        hit: bool,
    ) -> None:
        async with self._lock:
            now = pd.Timestamp.utcnow()
            row_id = self._cache_lookup_row_id_locked(
                key=key,
                tenant_id=tenant_id,
                session_id=session_id,
                query_type=query_type,
                capsule_level=capsule_level,
            )
            cache_df = self._state.cache_index_df
            if row_id is None:
                row = {
                    "key": key,
                    "tenant_id": tenant_id,
                    "session_id": session_id,
                    "query_type": query_type,
                    "capsule_level": capsule_level,
                    "entity_type": "search",
                    "entity_id": key,
                    "last_access": now,
                    "hit_count": 1 if hit else 0,
                    "miss_count": 0 if hit else 1,
                }
                if cache_df.empty:
                    self._state.cache_index_df = pd.DataFrame([row], columns=CACHE_INDEX_COLUMNS)
                else:
                    self._state.cache_index_df = pd.concat(
                        [cache_df, pd.DataFrame([row], columns=CACHE_INDEX_COLUMNS)],
                        ignore_index=True,
                    )
                self._invalidate_cache_lookup_index_locked()
                return

            def _safe_int(value: object) -> int:
                if value is None or pd.isna(value):
                    return 0
                return int(value)

            self._state.cache_index_df.at[row_id, "last_access"] = now
            if hit:
                self._state.cache_index_df.at[row_id, "hit_count"] = (
                    _safe_int(self._state.cache_index_df.at[row_id, "hit_count"]) + 1
                )
            else:
                self._state.cache_index_df.at[row_id, "miss_count"] = (
                    _safe_int(self._state.cache_index_df.at[row_id, "miss_count"]) + 1
                )

    def _token_score(self, text: str, query_tokens: List[str]) -> float:
        if not text or not query_tokens:
            return 0.0
        lower = text.lower()
        overlap = 0
        for token in query_tokens:
            if token in lower:
                overlap += 1
        return overlap / max(len(query_tokens), 1)

    def _semantic_score(self, text: str, query: str) -> float:
        text_set = set(text.lower().split())
        query_set = set(query.lower().split())
        union = text_set | query_set
        if not union:
            return 0.0
        return len(text_set & query_set) / len(union)

    async def message_documents(
        self,
        tenant_id: str,
        session_id: Optional[str],
        channel: Optional[str] = None,
        chat_type: Optional[str] = None,
        group_id: Optional[str] = None,
        topic_id: Optional[str] = None,
        message_thread_id: Optional[str] = None,
        row_mode: Literal["projection", "raw", "all"] = "projection",
    ) -> List[Dict[str, object]]:
        async with self._lock:
            df = self._messages_for_query_locked(
                tenant_id=tenant_id,
                session_id=session_id,
                channel=channel,
                chat_type=chat_type,
                group_id=group_id,
                topic_id=topic_id,
                message_thread_id=message_thread_id,
                row_mode=row_mode,
            )
            if df.empty:
                return []
            grouped: Dict[str, int] = {}
            docs: List[Dict[str, object]] = []
            for _, row in self._chronological_messages(df).iterrows():
                tid = str(row["tenant_id"])
                sid = str(row["session_id"])
                key = f"{tid}:{sid}"
                grouped[key] = grouped.get(key, 0) + 1
                line_no = grouped[key]
                path = f"memory/{sid}.md" if tid == "default" else f"memory/{tid}/{sid}.md"
                doc_id = (
                    str(row.get("origin_message_id") or row["message_id"])
                    if row_mode == "raw"
                    else str(row["message_id"])
                )
                docs.append(
                    {
                        "doc_id": doc_id,
                        "message_id": str(row["message_id"]),
                        "origin_message_id": str(row.get("origin_message_id") or row["message_id"]),
                        "tenant_id": tid,
                        "session_id": sid,
                        "channel": str(row.get("channel") or ""),
                        "chat_type": str(row.get("chat_type") or ""),
                        "account_id": str(row.get("account_id") or ""),
                        "group_id": str(row.get("group_id") or ""),
                        "topic_id": str(row.get("topic_id") or "default"),
                        "topic_path": str(row.get("topic_path") or row.get("topic_id") or "default"),
                        "message_thread_id": str(row.get("message_thread_id") or ""),
                        "sender_id": str(row.get("sender_id") or ""),
                        "capsule_level": str(row.get("capsule_level") or "L0"),
                        "projection_kind": str(row.get("projection_kind") or ""),
                        "projection_scope": str(row.get("projection_scope") or ""),
                        "content": str(row["content"]),
                        "path": path,
                        "line_no": line_no,
                    }
                )
            return docs

    async def hybrid_search(
        self,
        query: str,
        tenant_id: str,
        session_id: Optional[str],
        channel: Optional[str],
        chat_type: Optional[str],
        group_id: Optional[str],
        topic_id: Optional[str],
        message_thread_id: Optional[str],
        max_results: int,
        min_score: float,
    ) -> List[SearchResult]:
        query_clean = query.strip()
        if not query_clean:
            return []
        query_tokens = [t for t in query_clean.lower().split() if t]
        async with self._lock:
            df = self._messages_for_query_locked(
                tenant_id=tenant_id,
                session_id=session_id,
                channel=channel,
                chat_type=chat_type,
                group_id=group_id,
                topic_id=topic_id,
                message_thread_id=message_thread_id,
            )
            if df.empty:
                return []
            scored: List[Tuple[float, SearchResult]] = []
            grouped: Dict[str, int] = {}
            for _, row in self._chronological_messages(df).iterrows():
                sid = str(row["session_id"])
                tid = str(row.get("tenant_id") or "default")
                grouped_key = f"{tid}:{sid}"
                grouped[grouped_key] = grouped.get(grouped_key, 0) + 1
                content = str(row["content"])
                bm25 = self._token_score(content, query_tokens)
                semantic = self._semantic_score(content, query_clean)
                score = (0.6 * bm25) + (0.4 * semantic)
                if score < min_score:
                    continue
                line_no = grouped[grouped_key]
                path = f"memory/{sid}.md" if tid == "default" else f"memory/{tid}/{sid}.md"
                result = SearchResult(
                    path=path,
                    start_line=line_no,
                    end_line=line_no,
                    score=round(float(score), 6),
                    score_lexical=round(float(bm25), 6),
                    score_semantic=round(float(semantic), 6),
                    snippet=content[:700],
                    source="memory",
                    source_tier=str(row.get("capsule_level") or "L0"),
                    citation=f"origin:{row.get('origin_message_id') or row['message_id']}",
                    channel=str(row.get("channel") or "") or None,
                    chat_type=str(row.get("chat_type") or "") or None,
                    account_id=str(row.get("account_id") or "") or None,
                    group_id=str(row.get("group_id") or "") or None,
                    topic_id=str(row.get("topic_id") or "default"),
                    topic_path=str(row.get("topic_path") or row.get("topic_id") or "default"),
                    message_thread_id=str(row.get("message_thread_id") or "") or None,
                    sender_id=str(row.get("sender_id") or "") or None,
                    origin_message_id=str(row.get("origin_message_id") or row["message_id"]),
                    projection_kind=str(row.get("projection_kind") or "") or None,
                    projection_scope=str(row.get("projection_scope") or "") or None,
                )
                scored.append((score, result))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[: max(1, max_results)]]

    async def forum_view(self, tenant_id: str, session_id: str) -> List[Dict[str, object]]:
        async with self._lock:
            df = self._chronological_messages(
                self._messages_for_query_locked(tenant_id=tenant_id, session_id=session_id)
            )
            if df.empty:
                return []
            grouped: Dict[str, List[Dict[str, object]]] = {}
            for _, row in df.iterrows():
                topic = str(row.get("topic_id") or "default")
                grouped.setdefault(topic, []).append(
                    {
                        "message_id": str(row["message_id"]),
                        "origin_message_id": str(row.get("origin_message_id") or row["message_id"]),
                        "role": str(row["role"]),
                        "content": str(row["content"]),
                        "ts": pd.to_datetime(row["ts"], utc=True).isoformat(),
                        "projection_kind": str(row.get("projection_kind") or ""),
                        "projection_scope": str(row.get("projection_scope") or ""),
                    }
                )
            out = [{"topic_id": topic, "messages": msgs} for topic, msgs in grouped.items()]
            return out

    async def capsule_cards(self, tenant_id: str, session_id: str) -> List[Dict[str, object]]:
        async with self._lock:
            df = self._capsules_for_query_locked(tenant_id, session_id)
            df = df.reset_index(drop=True).sort_values("updated_at", kind="stable", ascending=False)
            out: List[Dict[str, object]] = []
            for _, row in df.iterrows():
                out.append(
                    {
                        "capsule_id": str(row["capsule_id"]),
                        "topic_id": str(row.get("topic_id") or "default"),
                        "summary": str(row.get("summary") or ""),
                        "level": str(row.get("level") or "L1"),
                        "score": float(row.get("score") or 0.0),
                        "updated_at": pd.to_datetime(row["updated_at"], utc=True).isoformat(),
                    }
                )
            return out

    async def session_rollup_cards(self, tenant_id: str, session_id: str) -> List[Dict[str, object]]:
        async with self._lock:
            df = self._session_rollups_for_query_locked(tenant_id, session_id)
            if df.empty:
                return []
            df = df.reset_index(drop=True).sort_values(
                ["bucket_start", "window_kind"],
                kind="stable",
                ascending=[True, True],
            )
            out: List[Dict[str, object]] = []
            for _, row in df.iterrows():
                out.append(
                    {
                        "rollup_id": str(row["rollup_id"]),
                        "window_kind": str(row["window_kind"]),
                        "window_key": str(row["window_key"]),
                        "bucket_start": _utc_timestamp(row["bucket_start"]).isoformat(),
                        "bucket_end": _utc_timestamp(row["bucket_end"]).isoformat(),
                        "source_first_ts": _utc_timestamp(row["source_first_ts"]).isoformat(),
                        "source_last_ts": _utc_timestamp(row["source_last_ts"]).isoformat(),
                        "message_count": int(row.get("message_count") or 0),
                        "content_char_count": int(row.get("content_char_count") or 0),
                        "summary": str(row.get("summary") or ""),
                        "vector_ref": str(row.get("vector_ref") or ""),
                        "vector_dim": int(row.get("vector_dim") or 0),
                    }
                )
            return out

    async def virtual_memory_file(self, rel_path: str) -> Tuple[str, str]:
        normalized = rel_path.replace("\\", "/").lstrip("/")
        if not normalized.startswith("memory/") or not normalized.endswith(".md"):
            raise FileNotFoundError(f"unsupported memory path: {rel_path}")
        parts = normalized.split("/")
        tenant_id = "default"
        if len(parts) == 2:
            session_id = parts[-1].replace(".md", "")
        elif len(parts) == 3:
            tenant_id = parts[1]
            session_id = parts[2].replace(".md", "")
        else:
            raise FileNotFoundError(f"unsupported memory path: {rel_path}")
        async with self._lock:
            df = self._chronological_messages(
                self._messages_for_query_locked(tenant_id=tenant_id, session_id=session_id)
            )
            if df.empty:
                raise FileNotFoundError(f"no session memory found for tenant={tenant_id} session={session_id}")
            lines = []
            for _, row in df.iterrows():
                role = str(row["role"])
                content = str(row["content"])
                lines.append(f"- [{role}] {content}")
            resolved_ids = self._resolve_session_ids_locked(tenant_id, session_id)
            canonical_session_id = resolved_ids[0] if resolved_ids else session_id
        canonical = (
            f"memory/{canonical_session_id}.md"
            if tenant_id == "default"
            else f"memory/{tenant_id}/{canonical_session_id}.md"
        )
        return "\n".join(lines), canonical

    async def save_parquet(self, parquet_dir: Path) -> None:
        async with self._lock:
            state = self._state
            await asyncio.to_thread(self._save_parquet_sync, parquet_dir, state)

    def _save_parquet_sync(self, parquet_dir: Path, state: DataFramesState) -> None:
        parquet_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")

        def _write_partitioned(df: pd.DataFrame, name: str) -> None:
            if df.empty:
                target = parquet_dir / name / "dt=empty"
                target.mkdir(parents=True, exist_ok=True)
                (target / f"part-{timestamp}.parquet").touch(exist_ok=True)
                return
            write_df = df.copy()
            if "ts" in write_df.columns:
                write_df["dt"] = pd.to_datetime(write_df["ts"], utc=True).dt.strftime("%Y-%m-%d")
            elif "updated_at" in write_df.columns:
                write_df["dt"] = pd.to_datetime(write_df["updated_at"], utc=True).dt.strftime("%Y-%m-%d")
            elif "created_at" in write_df.columns:
                write_df["dt"] = pd.to_datetime(write_df["created_at"], utc=True).dt.strftime("%Y-%m-%d")
            else:
                write_df["dt"] = datetime.utcnow().strftime("%Y-%m-%d")
            for dt, part in write_df.groupby("dt"):
                target = parquet_dir / name / f"dt={dt}"
                target.mkdir(parents=True, exist_ok=True)
                part.drop(columns=["dt"]).to_parquet(target / f"part-{timestamp}.parquet", index=False)

        _write_partitioned(state.messages_df, "messages")
        _write_partitioned(state.capsules_df, "capsules")
        _write_partitioned(state.session_rollups_df, "session_rollups")
        _write_partitioned(state.cache_index_df, "cache_index")
        _write_partitioned(state.sessions_df, "sessions")
        _write_partitioned(state.snapshots_df, "snapshots")

    async def load_parquet(self, parquet_dir: Path) -> None:
        async with self._lock:
            await asyncio.to_thread(self._load_parquet_sync, parquet_dir)

    def _load_parquet_sync(self, parquet_dir: Path) -> None:
        def _read_all(name: str, columns: List[str]) -> pd.DataFrame:
            base = parquet_dir / name
            if not base.exists():
                return pd.DataFrame(columns=columns)
            files = sorted(base.glob("dt=*/part-*.parquet"))
            if not files:
                return pd.DataFrame(columns=columns)
            parts = []
            for file in files:
                try:
                    part = pd.read_parquet(file)
                    for col in columns:
                        if col not in part.columns:
                            part[col] = None
                    parts.append(part[columns])
                except Exception:
                    continue
            if not parts:
                return pd.DataFrame(columns=columns)
            return pd.concat(parts, ignore_index=True)

        self._state = DataFramesState(
            messages_df=_read_all("messages", MESSAGES_COLUMNS),
            capsules_df=_read_all("capsules", CAPSULES_COLUMNS),
            session_rollups_df=_read_all("session_rollups", SESSION_ROLLUPS_COLUMNS),
            cache_index_df=_read_all("cache_index", CACHE_INDEX_COLUMNS),
            sessions_df=_read_all("sessions", SESSIONS_COLUMNS),
            snapshots_df=_read_all("snapshots", SNAPSHOTS_COLUMNS),
        )
        self._invalidate_all_indexes_locked()
