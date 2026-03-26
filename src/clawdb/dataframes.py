from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import pandas as pd

from .beliefs import BELIEFS_COLUMNS, materialize_l0_beliefs
from .capsules import CAPSULES_COLUMNS, materialize_capsule_lifecycle
from .embeddings import (
    DETERMINISTIC_EMBEDDING_MODEL,
    DETERMINISTIC_EMBEDDING_PROVIDER,
    deterministic_embedding_ref,
    embedding_source_hash,
)
from .lineage import (
    CANONICAL_PROJECTION_KINDS,
    DM_MIRROR_PUBLIC_PROJECTION_KIND,
    MESSAGE_STATE_ACTIVE,
    MESSAGE_STATE_DELETED,
    PLATFORM_IDENTITY_COLUMNS,
    RAW_PROJECTION_KIND,
    materialize_message_bundle,
    materialize_projection_rows,
    normalize_identity,
    normalize_platform,
    projection_message_id,
)
from .models import SearchResult, WalRecord
from .projections import PROJECTIONS_COLUMNS, materialize_projection_state
from .search_index import (
    LEXICAL_INDEX_COLUMNS,
    SEARCH_DOC_COLUMNS,
    VECTOR_INDEX_COLUMNS,
    LexicalPosting,
    VectorEntry,
    materialize_lexical_index,
    materialize_vector_index,
    parse_vector_json,
)
from .storage_layout import MessageChannelFile, table_parquet_files
from .textsize import utf8_text_size
from .topics import (
    DEFAULT_TOPIC_VECTOR_DIM,
    TOPICS_COLUMNS,
    _vectorize,
    materialize_topic_lifecycle,
)


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
    "account_key",
    "from_id",
    "from_user_key",
    "to_id",
    "to_user_key",
    "projection_target_user_key",
    "sender_id",
    "sender_user_key",
    "sender_name",
    "sender_username",
    "sender_e164",
    "group_id",
    "group_chat_key",
    "group_subject",
    "group_channel",
    "group_space",
    "native_channel_id",
    "message_thread_id",
    "thread_parent_id",
    "reply_to_id",
    "topic_id",
    "source_topic_id",
    "topic_parent_id",
    "topic_path",
    "source_topic_path",
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

TOPIC_MULTIINDEX_LEVELS = [
    "tenant_id",
    "topic_id",
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

SEMANTIC_JOBS_COLUMNS = [
    "job_id",
    "tenant_id",
    "status",
    "latest_wal_seq",
    "claimed_wal_seq",
    "impacted_sessions_json",
    "claimed_sessions_json",
    "cause",
    "attempt_count",
    "last_error",
    "enqueued_at",
    "started_at",
    "updated_at",
    "lease_owner",
    "lease_expires_at",
    "available_at",
]

EMBEDDING_INDEX_METADATA_COLUMNS = [
    "tenant_id",
    "entity_type",
    "entity_id",
    "embedding_ref",
    "embedding_provider",
    "embedding_model",
    "source_hash",
    "reembed_policy",
    "updated_at",
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

MESSAGE_IDENTITY_COLUMNS = [
    "tenant_id",
    "origin_message_id",
    "projection_kind",
    "projection_scope",
]

RAW_COMPAT_PROJECTION_KINDS = {RAW_PROJECTION_KIND, "raw"}

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

SEARCH_DOC_MULTIINDEX_LEVELS = [
    "tenant_id",
    "doc_id",
]

LEXICAL_INDEX_MULTIINDEX_LEVELS = [
    "tenant_id",
    "token",
    "doc_id",
]

VECTOR_INDEX_MULTIINDEX_LEVELS = [
    "tenant_id",
    "doc_id",
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

SEMANTIC_JOB_STATUS_PENDING = "pending"
SEMANTIC_JOB_STATUS_RUNNING = "running"


@dataclass
class DataFramesState:
    messages_df: pd.DataFrame
    capsules_df: pd.DataFrame
    beliefs_df: pd.DataFrame
    projections_df: pd.DataFrame
    session_rollups_df: pd.DataFrame
    topics_df: pd.DataFrame
    search_docs_df: pd.DataFrame
    lexical_index_df: pd.DataFrame
    vector_index_df: pd.DataFrame
    embedding_index_metadata_df: pd.DataFrame
    cache_index_df: pd.DataFrame
    sessions_df: pd.DataFrame
    snapshots_df: pd.DataFrame
    semantic_jobs_df: pd.DataFrame


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


@dataclass(frozen=True)
class StorageRebuildResult:
    raw_message_count: int
    projection_message_count: int
    projection_state_count: int
    session_count: int
    session_rollup_count: int
    topic_count: int
    capsule_count: int
    belief_count: int
    embedding_metadata_count: int


@dataclass(frozen=True)
class SemanticJobClaim:
    job_id: str
    tenant_id: str
    claimed_wal_seq: int
    latest_wal_seq: int
    impacted_sessions: List[str]
    attempt_count: int
    cause: str


@dataclass(frozen=True)
class SemanticJobStats:
    pending: int
    running: int
    total: int
    max_wal_seq: int


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
RETRIEVAL_ABSTRACT_MAX_CHARS = 4000
AUTHORITATIVE_RAW_MESSAGE_SOURCE = "messages.raw_global"
DERIVED_ONLY_LAYERS = (
    "projection_messages",
    "session_rollups",
    "topics",
    "capsules",
)
FULL_REBUILD_SEQUENCE = DERIVED_ONLY_LAYERS


def _utc_timestamp(value: object) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return pd.Timestamp.now(tz="UTC")
    return ts


def _iso_timestamp_or_none(value: object) -> Optional[str]:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.isoformat()


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


def _trim_text(value: object, limit: int) -> str:
    text = str(value or "")
    if len(text) <= max(0, int(limit)):
        return text
    return text[: max(0, int(limit) - 3)] + "..."


def _safe_path_fragment(value: object) -> str:
    raw = str(value or "").strip() or "_"
    out = []
    for char in raw:
        if char.isalnum() or char in {"-", "_", "."}:
            out.append(char)
        else:
            out.append("_")
    return "".join(out)


def _dedupe_citations(values: List[str], limit: int = 3) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        citation = str(value or "").strip()
        if not citation or citation in seen:
            continue
        out.append(citation)
        seen.add(citation)
        if len(out) >= max(1, int(limit)):
            break
    return out


def _json_string_list(value: object) -> List[str]:
    if value is None or value == "" or value == "[]":
        return []
    try:
        raw = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    seen = set()
    for item in raw:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        out.append(text)
        seen.add(text)
    return out


def _group_identity_mask(frame: pd.DataFrame, group_id: object) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    group_text = str(group_id or "").strip()
    if not group_text:
        return pd.Series([False] * len(frame), index=frame.index, dtype=bool)
    group_ids = frame["group_id"].fillna("").astype(str)
    if "group_chat_key" in frame.columns:
        group_chat_keys = frame["group_chat_key"].fillna("").astype(str)
    else:
        group_chat_keys = pd.Series([""] * len(frame), index=frame.index, dtype="string").astype(str)
    suffix = group_text.rsplit(":", 1)[-1].strip()
    canonical_suffix = group_chat_keys.str.rsplit(":", n=1).str[-1]
    mask = (group_ids == group_text) | (group_chat_keys == group_text)
    if suffix:
        mask |= (group_ids == suffix) | (canonical_suffix == suffix)
    return mask


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
    materialized_at = pd.Timestamp.now(tz="UTC")
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
                    content_char_count=int(bucket_ordered["content"].astype(str).map(utf8_text_size).sum()),
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
                        "content_char_count": int(
                            bucket_ordered["content"].astype(str).map(utf8_text_size).sum()
                        ),
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


def authoritative_raw_messages(messages_frame: pd.DataFrame) -> pd.DataFrame:
    if messages_frame.empty:
        return pd.DataFrame(columns=MESSAGES_COLUMNS)
    scoped = messages_frame.copy().reset_index(drop=True)
    for col in MESSAGES_COLUMNS:
        if col not in scoped.columns:
            scoped[col] = None
    scoped = scoped[MESSAGES_COLUMNS]
    scoped["tenant_id"] = scoped["tenant_id"].fillna("default").astype(str)
    scoped["message_id"] = scoped["message_id"].fillna("").astype(str)
    scoped["origin_message_id"] = scoped["origin_message_id"].fillna(scoped["message_id"]).astype(str)
    scoped["session_id"] = scoped["session_id"].fillna("").astype(str)
    scoped["projection_kind"] = scoped["projection_kind"].fillna("").astype(str)
    scoped["projection_scope"] = scoped["projection_scope"].fillna("").astype(str)
    scoped["visibility"] = scoped["visibility"].fillna("").astype(str)
    scoped["native_session_id"] = scoped["native_session_id"].fillna("").astype(str)
    scoped["ts"] = pd.to_datetime(scoped["ts"], utc=True, errors="coerce")
    scoped["updated_at"] = pd.to_datetime(scoped["updated_at"], utc=True, errors="coerce")
    scoped["updated_at"] = scoped["updated_at"].fillna(scoped["ts"])
    raw_rows = scoped[scoped["projection_kind"].astype(str) == RAW_PROJECTION_KIND].copy()
    if raw_rows.empty:
        return pd.DataFrame(columns=MESSAGES_COLUMNS)
    projection_rows = scoped[
        (scoped["projection_kind"].astype(str) != RAW_PROJECTION_KIND)
        & (scoped["native_session_id"].astype(str) != "")
    ].copy()
    if not projection_rows.empty:
        projection_rows = projection_rows.sort_values(
            ["ts", "session_id", "message_id"],
            ascending=[True, True, True],
            kind="stable",
        )
        native_session_lookup = (
            projection_rows.groupby(["tenant_id", "origin_message_id"], sort=False)["native_session_id"].first()
        )
        blank_mask = raw_rows["native_session_id"].astype(str) == ""
        for row_id, row in raw_rows[blank_mask].iterrows():
            native_session_id = native_session_lookup.get(
                (str(row["tenant_id"]), str(row["origin_message_id"])),
                "",
            )
            if native_session_id:
                raw_rows.at[row_id, "native_session_id"] = str(native_session_id)
    raw_rows.loc[raw_rows["projection_scope"].astype(str) == "", "projection_scope"] = "global"
    raw_rows.loc[raw_rows["visibility"].astype(str) == "", "visibility"] = "raw"
    raw_rows = raw_rows.sort_values(
        ["ts", "updated_at", "origin_message_id", "message_id"],
        ascending=[True, True, True, True],
        kind="stable",
    )
    raw_rows = raw_rows.drop_duplicates(
        subset=["tenant_id", "origin_message_id"],
        keep="last",
    )
    return raw_rows[MESSAGES_COLUMNS].reset_index(drop=True)


def materialize_projection_messages_from_raw(raw_messages_frame: pd.DataFrame) -> pd.DataFrame:
    if raw_messages_frame.empty:
        return pd.DataFrame(columns=MESSAGES_COLUMNS)
    ordered = raw_messages_frame.copy().reset_index(drop=True)
    for col in MESSAGES_COLUMNS:
        if col not in ordered.columns:
            ordered[col] = None
    ordered = ordered[MESSAGES_COLUMNS]
    ordered["tenant_id"] = ordered["tenant_id"].fillna("default").astype(str)
    ordered["message_id"] = ordered["message_id"].fillna("").astype(str)
    ordered["origin_message_id"] = ordered["origin_message_id"].fillna(ordered["message_id"]).astype(str)
    ordered["ts"] = pd.to_datetime(ordered["ts"], utc=True, errors="coerce")
    ordered = ordered.sort_values(
        ["ts", "origin_message_id", "message_id"],
        ascending=[True, True, True],
        kind="stable",
    )
    rows: List[Dict[str, object]] = []
    for _, row in ordered.iterrows():
        rows.extend(materialize_projection_rows(row.to_dict()))
    if not rows:
        return pd.DataFrame(columns=MESSAGES_COLUMNS)
    materialized = pd.DataFrame(rows, columns=MESSAGES_COLUMNS)
    materialized["tenant_id"] = materialized["tenant_id"].fillna("default").astype(str)
    materialized["message_id"] = materialized["message_id"].fillna("").astype(str)
    materialized["origin_message_id"] = materialized["origin_message_id"].fillna(
        materialized["message_id"]
    ).astype(str)
    materialized["session_id"] = materialized["session_id"].fillna("").astype(str)
    materialized["ts"] = pd.to_datetime(materialized["ts"], utc=True, errors="coerce")
    materialized = materialized.sort_values(
        ["ts", "session_id", "message_id"],
        ascending=[True, True, True],
        kind="stable",
    )
    materialized = materialized.drop_duplicates(
        subset=["tenant_id", "origin_message_id", "message_id", "session_id"],
        keep="last",
    )
    return materialized[MESSAGES_COLUMNS].reset_index(drop=True)


def _preserved_projection_messages(
    messages_frame: pd.DataFrame,
    raw_messages_frame: pd.DataFrame,
    canonical_projection_messages: pd.DataFrame,
) -> pd.DataFrame:
    if messages_frame.empty or raw_messages_frame.empty:
        return pd.DataFrame(columns=MESSAGES_COLUMNS)
    existing = messages_frame.copy().reset_index(drop=True)
    for col in MESSAGES_COLUMNS:
        if col not in existing.columns:
            existing[col] = None
    existing = existing[MESSAGES_COLUMNS]
    existing["tenant_id"] = existing["tenant_id"].fillna("default").astype(str)
    existing["message_id"] = existing["message_id"].fillna("").astype(str)
    existing["origin_message_id"] = existing["origin_message_id"].fillna(existing["message_id"]).astype(str)
    existing["session_id"] = existing["session_id"].fillna("").astype(str)
    existing["projection_kind"] = existing["projection_kind"].fillna("").astype(str)
    existing["projection_scope"] = existing["projection_scope"].fillna("").astype(str)
    existing["visibility"] = existing["visibility"].fillna("").astype(str)
    existing["native_session_id"] = existing["native_session_id"].fillna("").astype(str)
    existing = existing[existing["projection_kind"].astype(str) != RAW_PROJECTION_KIND]
    existing = existing[
        ~existing.apply(
            lambda row: (
                str(row["projection_kind"]) in CANONICAL_PROJECTION_KINDS
                and str(row["message_id"])
                == projection_message_id(
                    str(row["origin_message_id"]),
                    str(row["projection_kind"]),
                    str(row["projection_scope"]),
                )
            ),
            axis=1,
        )
    ]
    if existing.empty:
        return pd.DataFrame(columns=MESSAGES_COLUMNS)
    raw_lookup = {
        (str(row["tenant_id"]), str(row["origin_message_id"])): row.to_dict()
        for _, row in raw_messages_frame.iterrows()
    }
    canonical_keys = {
        (
            str(row["tenant_id"]),
            str(row["origin_message_id"]),
            str(row["message_id"]),
            str(row["session_id"]),
        )
        for _, row in canonical_projection_messages.iterrows()
    }
    rows: List[Dict[str, object]] = []
    for _, row in existing.iterrows():
        key = (
            str(row["tenant_id"]),
            str(row["origin_message_id"]),
            str(row["message_id"]),
            str(row["session_id"]),
        )
        if key in canonical_keys:
            continue
        raw_row = raw_lookup.get((str(row["tenant_id"]), str(row["origin_message_id"])))
        if raw_row is None:
            continue
        rebuilt = dict(raw_row)
        rebuilt.update(
            {
                "message_id": str(row["message_id"]),
                "session_id": str(row["session_id"]),
                "projection_kind": str(row["projection_kind"]),
                "projection_scope": str(row["projection_scope"]),
                "visibility": str(row["visibility"] or raw_row.get("visibility") or ""),
                "native_session_id": str(row["native_session_id"]),
            }
        )
        rows.append(rebuilt)
    if not rows:
        return pd.DataFrame(columns=MESSAGES_COLUMNS)
    preserved = pd.DataFrame(rows, columns=MESSAGES_COLUMNS)
    preserved["tenant_id"] = preserved["tenant_id"].fillna("default").astype(str)
    preserved["message_id"] = preserved["message_id"].fillna("").astype(str)
    preserved["origin_message_id"] = preserved["origin_message_id"].fillna(preserved["message_id"]).astype(str)
    preserved["session_id"] = preserved["session_id"].fillna("").astype(str)
    preserved["ts"] = pd.to_datetime(preserved["ts"], utc=True, errors="coerce")
    preserved = preserved.sort_values(
        ["ts", "session_id", "message_id"],
        ascending=[True, True, True],
        kind="stable",
    )
    preserved = preserved.drop_duplicates(
        subset=["tenant_id", "origin_message_id", "message_id", "session_id"],
        keep="last",
    )
    return preserved[MESSAGES_COLUMNS].reset_index(drop=True)


def materialize_projection_sessions(projection_messages_frame: pd.DataFrame) -> pd.DataFrame:
    if projection_messages_frame.empty:
        return pd.DataFrame(columns=SESSIONS_COLUMNS)
    scoped = projection_messages_frame.copy().reset_index(drop=True)
    for col in SESSIONS_COLUMNS:
        if col not in scoped.columns:
            scoped[col] = None
    scoped["tenant_id"] = scoped["tenant_id"].fillna("default").astype(str)
    scoped["session_id"] = scoped["session_id"].fillna("").astype(str)
    scoped["ts"] = pd.to_datetime(scoped["ts"], utc=True, errors="coerce")
    scoped = scoped[scoped["session_id"].astype(str) != ""].copy()
    if scoped.empty:
        return pd.DataFrame(columns=SESSIONS_COLUMNS)
    rows: List[Dict[str, object]] = []
    for (tenant_id, session_id), group in scoped.groupby(["tenant_id", "session_id"], sort=True):
        created_at = pd.to_datetime(group["ts"], utc=True, errors="coerce").min()
        rows.append(
            {
                "tenant_id": str(tenant_id),
                "session_id": str(session_id),
                "parent_session_id": "",
                "origin": "projection",
                "created_at": _utc_timestamp(created_at),
            }
        )
    return pd.DataFrame(rows, columns=SESSIONS_COLUMNS)


def _merge_session_rows(existing_sessions: pd.DataFrame, derived_sessions: pd.DataFrame) -> pd.DataFrame:
    if existing_sessions.empty:
        return derived_sessions.reset_index(drop=True)
    existing = existing_sessions.copy().reset_index(drop=True)
    for col in SESSIONS_COLUMNS:
        if col not in existing.columns:
            existing[col] = None
    existing = existing[SESSIONS_COLUMNS]
    existing["tenant_id"] = existing["tenant_id"].fillna("default").astype(str)
    existing["session_id"] = existing["session_id"].fillna("").astype(str)
    existing = existing[existing["session_id"].astype(str) != ""].copy()
    if derived_sessions.empty:
        return existing.reset_index(drop=True)
    derived = derived_sessions.copy().reset_index(drop=True)
    derived["tenant_id"] = derived["tenant_id"].fillna("default").astype(str)
    derived["session_id"] = derived["session_id"].fillna("").astype(str)
    existing_keys = {
        (str(row["tenant_id"]), str(row["session_id"]))
        for _, row in existing.iterrows()
    }
    derived = derived[
        ~derived.apply(
            lambda row: (str(row["tenant_id"]), str(row["session_id"])) in existing_keys,
            axis=1,
        )
    ].reset_index(drop=True)
    if derived.empty:
        return existing.reset_index(drop=True)
    return pd.concat([existing, derived[SESSIONS_COLUMNS]], ignore_index=True)


def rebuild_materialized_storage_from_raw(
    messages_frame: pd.DataFrame,
    sessions_frame: pd.DataFrame,
    *,
    vector_dim: int = DEFAULT_TOPIC_VECTOR_DIM,
) -> Dict[str, pd.DataFrame]:
    raw_messages = authoritative_raw_messages(messages_frame)
    canonical_projection_messages = materialize_projection_messages_from_raw(raw_messages)
    preserved_projection_messages = _preserved_projection_messages(
        messages_frame,
        raw_messages,
        canonical_projection_messages,
    )
    projection_parts = [
        frame
        for frame in (canonical_projection_messages, preserved_projection_messages)
        if not frame.empty
    ]
    projection_messages = (
        pd.concat(projection_parts, ignore_index=True)[MESSAGES_COLUMNS]
        if projection_parts
        else pd.DataFrame(columns=MESSAGES_COLUMNS)
    )
    message_parts = [raw_messages]
    if not projection_messages.empty:
        message_parts.append(projection_messages)
    rebuilt_messages = (
        pd.concat(message_parts, ignore_index=True)[MESSAGES_COLUMNS]
        if message_parts and any(not frame.empty for frame in message_parts)
        else pd.DataFrame(columns=MESSAGES_COLUMNS)
    )
    derived_sessions = materialize_projection_sessions(projection_messages)
    rebuilt_sessions = _merge_session_rows(sessions_frame, derived_sessions)
    rebuilt_rollups = materialize_session_rollups(rebuilt_messages, vector_dim=vector_dim)
    rebuilt_topics = materialize_topic_lifecycle(raw_messages, vector_dim=vector_dim)
    rebuilt_capsules = materialize_capsule_lifecycle(
        raw_messages,
        topics_frame=rebuilt_topics,
        vector_dim=vector_dim,
    )
    rebuilt_projection_state = materialize_projection_state(
        rebuilt_messages,
        vector_dim=vector_dim,
    )
    rebuilt_beliefs = materialize_l0_beliefs(
        rebuilt_messages,
        vector_dim=vector_dim,
    )
    rebuilt_embedding_metadata = materialize_embedding_index_metadata(
        rebuilt_messages,
        rebuilt_rollups,
        rebuilt_topics,
        rebuilt_capsules,
        rebuilt_beliefs,
        rebuilt_projection_state,
    )
    return {
        "messages": rebuilt_messages,
        "projection_messages": projection_messages,
        "projections": rebuilt_projection_state,
        "beliefs": rebuilt_beliefs,
        "sessions": rebuilt_sessions,
        "session_rollups": rebuilt_rollups,
        "topics": rebuilt_topics,
        "capsules": rebuilt_capsules,
        "embedding_index_metadata": rebuilt_embedding_metadata,
    }


def materialize_embedding_index_metadata(
    messages_frame: pd.DataFrame,
    session_rollups_frame: pd.DataFrame,
    topics_frame: pd.DataFrame,
    capsules_frame: pd.DataFrame,
    beliefs_frame: pd.DataFrame,
    projections_frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    def _append_entity_rows(
        frame: pd.DataFrame,
        *,
        entity_type: str,
        entity_id_col: str,
        text_col: str,
        ref_col: str,
    ) -> None:
        if frame.empty:
            return
        scoped = frame.copy().reset_index(drop=True)
        scoped["tenant_id"] = scoped["tenant_id"].fillna("default").astype(str)
        scoped[entity_id_col] = scoped[entity_id_col].fillna("").astype(str)
        scoped[text_col] = scoped[text_col].fillna("").astype(str)
        scoped[ref_col] = scoped[ref_col].fillna("").astype(str)
        scoped["updated_at"] = pd.to_datetime(scoped["updated_at"], utc=True, errors="coerce")
        for _, row in scoped.iterrows():
            entity_id = str(row.get(entity_id_col) or "")
            if not entity_id:
                continue
            text = str(row.get(text_col) or "")
            rows.append(
                {
                    "tenant_id": str(row.get("tenant_id") or "default"),
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "embedding_ref": str(row.get(ref_col) or deterministic_embedding_ref(entity_type, text)),
                    "embedding_provider": DETERMINISTIC_EMBEDDING_PROVIDER,
                    "embedding_model": DETERMINISTIC_EMBEDDING_MODEL,
                    "source_hash": embedding_source_hash(text),
                    "reembed_policy": "content_hash_change",
                    "updated_at": _utc_timestamp(row.get("updated_at")),
                }
            )

    raw_messages = authoritative_raw_messages(messages_frame)
    if not raw_messages.empty:
        scoped = raw_messages.copy().reset_index(drop=True)
        scoped["tenant_id"] = scoped["tenant_id"].fillna("default").astype(str)
        scoped["origin_message_id"] = scoped["origin_message_id"].fillna(scoped["message_id"]).astype(str)
        scoped["content"] = scoped["content"].fillna("").astype(str)
        scoped["embedding_ref"] = scoped["embedding_ref"].fillna("").astype(str)
        scoped["updated_at"] = pd.to_datetime(scoped["updated_at"], utc=True, errors="coerce")
        scoped["ts"] = pd.to_datetime(scoped["ts"], utc=True, errors="coerce")
        for _, row in scoped.iterrows():
            content = str(row.get("content") or "")
            rows.append(
                {
                    "tenant_id": str(row.get("tenant_id") or "default"),
                    "entity_type": "raw_message",
                    "entity_id": str(row.get("origin_message_id") or row.get("message_id") or ""),
                    "embedding_ref": str(
                        row.get("embedding_ref") or deterministic_embedding_ref("raw_message", content)
                    ),
                    "embedding_provider": DETERMINISTIC_EMBEDDING_PROVIDER,
                    "embedding_model": DETERMINISTIC_EMBEDDING_MODEL,
                    "source_hash": embedding_source_hash(content),
                    "reembed_policy": "content_hash_change",
                    "updated_at": _utc_timestamp(
                        row.get("updated_at") if pd.notna(row.get("updated_at")) else row.get("ts")
                    ),
                }
            )

    _append_entity_rows(
        session_rollups_frame,
        entity_type="session_rollup",
        entity_id_col="rollup_id",
        text_col="vector_text",
        ref_col="vector_ref",
    )
    _append_entity_rows(
        topics_frame,
        entity_type="topic",
        entity_id_col="topic_id",
        text_col="vector_text",
        ref_col="vector_ref",
    )
    _append_entity_rows(
        capsules_frame,
        entity_type="capsule",
        entity_id_col="capsule_id",
        text_col="vector_text",
        ref_col="vector_ref",
    )
    _append_entity_rows(
        beliefs_frame,
        entity_type="l0_abstract",
        entity_id_col="belief_id",
        text_col="vector_text",
        ref_col="vector_ref",
    )
    _append_entity_rows(
        projections_frame,
        entity_type="projection",
        entity_id_col="projection_id",
        text_col="vector_text",
        ref_col="vector_ref",
    )
    if not rows:
        return pd.DataFrame(columns=EMBEDDING_INDEX_METADATA_COLUMNS)
    frame = pd.DataFrame(rows, columns=EMBEDDING_INDEX_METADATA_COLUMNS)
    frame["tenant_id"] = frame["tenant_id"].fillna("default").astype(str)
    frame["entity_type"] = frame["entity_type"].fillna("").astype(str)
    frame["entity_id"] = frame["entity_id"].fillna("").astype(str)
    frame["embedding_ref"] = frame["embedding_ref"].fillna("").astype(str)
    frame["embedding_provider"] = frame["embedding_provider"].fillna("").astype(str)
    frame["embedding_model"] = frame["embedding_model"].fillna("").astype(str)
    frame["source_hash"] = frame["source_hash"].fillna("").astype(str)
    frame["reembed_policy"] = frame["reembed_policy"].fillna("").astype(str)
    frame["updated_at"] = pd.to_datetime(frame["updated_at"], utc=True, errors="coerce")
    frame = frame.sort_values(
        ["tenant_id", "entity_type", "entity_id"],
        ascending=[True, True, True],
        kind="stable",
    ).drop_duplicates(
        subset=["tenant_id", "entity_type", "entity_id"],
        keep="last",
    )
    return frame[EMBEDDING_INDEX_METADATA_COLUMNS].reset_index(drop=True)


def _infer_projection_target_user_key(rows: pd.DataFrame) -> Optional[str]:
    if rows.empty:
        return None
    scoped = rows.copy()
    scoped["ts"] = pd.to_datetime(scoped["ts"], utc=True, errors="coerce")
    scoped["updated_at"] = pd.to_datetime(scoped["updated_at"], utc=True, errors="coerce")
    scoped["origin_message_id"] = scoped["origin_message_id"].fillna(scoped["message_id"]).astype(str)
    scoped = scoped.sort_values(
        ["ts", "updated_at", "origin_message_id"],
        ascending=[False, False, False],
        kind="stable",
    )

    def _row_user_key(row: pd.Series) -> str:
        explicit_target = str(row.get("projection_target_user_key") or "").strip()
        if explicit_target:
            return explicit_target
        role = str(row.get("role") or "").strip().lower()
        if role == "assistant":
            normalized_columns = ["to_user_key", "sender_user_key", "from_user_key"]
            raw_columns = ["to_id", "sender_id", "from_id"]
        else:
            normalized_columns = ["sender_user_key", "from_user_key", "to_user_key"]
            raw_columns = ["sender_id", "from_id", "to_id"]
        for column in normalized_columns:
            value = str(row.get(column) or "").strip()
            if value:
                return value
        platform = normalize_platform(str(row.get("platform") or "") or None)
        for column in raw_columns:
            try:
                value = normalize_identity(platform, str(row.get(column) or ""), "user")
            except ValueError:
                value = ""
            if value:
                return value
        return ""

    for include_assistant_rows in (False, True):
        for _, row in scoped.iterrows():
            role = str(row.get("role") or "").strip().lower()
            if not include_assistant_rows and role == "assistant":
                continue
            user_key = _row_user_key(row)
            if user_key:
                return user_key
    return None


class DataFrameStore:
    def __init__(self) -> None:
        self._state = DataFramesState(
            messages_df=pd.DataFrame(columns=MESSAGES_COLUMNS),
            capsules_df=pd.DataFrame(columns=CAPSULES_COLUMNS),
            beliefs_df=pd.DataFrame(columns=BELIEFS_COLUMNS),
            projections_df=pd.DataFrame(columns=PROJECTIONS_COLUMNS),
            session_rollups_df=pd.DataFrame(columns=SESSION_ROLLUPS_COLUMNS),
            topics_df=pd.DataFrame(columns=TOPICS_COLUMNS),
            search_docs_df=pd.DataFrame(columns=SEARCH_DOC_COLUMNS),
            lexical_index_df=pd.DataFrame(columns=LEXICAL_INDEX_COLUMNS),
            vector_index_df=pd.DataFrame(columns=VECTOR_INDEX_COLUMNS),
            embedding_index_metadata_df=pd.DataFrame(columns=EMBEDDING_INDEX_METADATA_COLUMNS),
            cache_index_df=pd.DataFrame(columns=CACHE_INDEX_COLUMNS),
            sessions_df=pd.DataFrame(columns=SESSIONS_COLUMNS),
            snapshots_df=pd.DataFrame(columns=SNAPSHOTS_COLUMNS),
            semantic_jobs_df=pd.DataFrame(columns=SEMANTIC_JOBS_COLUMNS),
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
                "account_key": "string",
                "from_id": "string",
                "from_user_key": "string",
                "to_id": "string",
                "to_user_key": "string",
                "projection_target_user_key": "string",
                "sender_id": "string",
                "sender_user_key": "string",
                "sender_name": "string",
                "sender_username": "string",
                "sender_e164": "string",
                "group_id": "string",
                "group_chat_key": "string",
                "group_subject": "string",
                "group_channel": "string",
                "group_space": "string",
                "native_channel_id": "string",
                "message_thread_id": "string",
                "thread_parent_id": "string",
                "reply_to_id": "string",
                "topic_id": "string",
                "source_topic_id": "string",
                "topic_parent_id": "string",
                "topic_path": "string",
                "source_topic_path": "string",
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
        self._state.capsules_df = self._state.capsules_df.astype(
            {
                "capsule_id": "string",
                "tenant_id": "string",
                "session_id": "string",
                "topic_id": "string",
                "topic_path": "string",
                "capsule_ordinal": "int64",
                "capsule_state": "string",
                "summary": "string",
                "level": "string",
                "score": "float64",
                "source_message_count": "int64",
                "source_body_char_count": "int64",
                "threshold_body_char_count": "int64",
                "first_origin_message_id": "string",
                "last_origin_message_id": "string",
                "source_message_ids_json": "string",
                "source_session_ids_json": "string",
                "source_topic_ids_json": "string",
                "active_message_count": "int64",
                "edited_message_count": "int64",
                "topic_message_count": "int64",
                "topic_body_char_count": "int64",
                "prev_capsule_id": "string",
                "next_capsule_id": "string",
                "back_link_ids_json": "string",
                "forward_link_ids_json": "string",
                "pointer_json": "string",
                "vector_text": "string",
                "vector_ref": "string",
                "vector_dim": "int64",
                "vector_json": "string",
                "source_hash": "string",
            }
        )
        self._state.beliefs_df = self._state.beliefs_df.astype(
            {
                "belief_id": "string",
                "tenant_id": "string",
                "scope_type": "string",
                "scope_key": "string",
                "session_id": "string",
                "topic_id": "string",
                "group_id": "string",
                "projection_kind": "string",
                "projection_scope": "string",
                "raw_message_count": "int64",
                "first_origin_message_id": "string",
                "last_origin_message_id": "string",
                "source_message_ids_json": "string",
                "source_session_ids_json": "string",
                "topic_ids_json": "string",
                "summary": "string",
                "vector_text": "string",
                "vector_ref": "string",
                "vector_dim": "int64",
                "vector_json": "string",
                "source_hash": "string",
            }
        )
        self._state.projections_df = self._state.projections_df.astype(
            {
                "projection_id": "string",
                "tenant_id": "string",
                "session_id": "string",
                "projection_kind": "string",
                "projection_scope": "string",
                "visibility": "string",
                "chat_type": "string",
                "native_session_id": "string",
                "native_session_ids_json": "string",
                "paired_projection_ids_json": "string",
                "paired_session_ids_json": "string",
                "paired_projection_scopes_json": "string",
                "account_id": "string",
                "account_key": "string",
                "group_id": "string",
                "group_chat_key": "string",
                "sender_id": "string",
                "sender_user_key": "string",
                "topic_ids_json": "string",
                "origin_message_count": "int64",
                "active_message_count": "int64",
                "deleted_message_count": "int64",
                "first_origin_message_id": "string",
                "last_origin_message_id": "string",
                "origin_message_ids_json": "string",
                "summary": "string",
                "vector_text": "string",
                "vector_ref": "string",
                "vector_dim": "int64",
                "vector_json": "string",
                "source_hash": "string",
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
        self._state.topics_df = self._state.topics_df.astype(
            {
                "topic_id": "string",
                "tenant_id": "string",
                "canonical_topic_id": "string",
                "topic_parent_id": "string",
                "topic_path": "string",
                "source_topic_id": "string",
                "status": "string",
                "historical_message_count": "int64",
                "message_count": "int64",
                "deleted_message_count": "int64",
                "content_char_count": "int64",
                "keywords_json": "string",
                "merged_topic_ids_json": "string",
                "split_topic_ids_json": "string",
                "drift_score": "float64",
                "summary": "string",
                "vector_text": "string",
                "vector_ref": "string",
                "vector_dim": "int64",
                "vector_json": "string",
            }
        )
        self._state.search_docs_df = self._state.search_docs_df.astype(
            {
                "tenant_id": "string",
                "doc_id": "string",
                "entity_type": "string",
                "entity_id": "string",
                "source_tier": "string",
                "session_id": "string",
                "text": "string",
                "path": "string",
                "start_line": "int64",
                "end_line": "int64",
                "snippet": "string",
                "citation": "string",
                "citations_json": "string",
                "channel": "string",
                "chat_type": "string",
                "account_id": "string",
                "group_id": "string",
                "topic_id": "string",
                "topic_path": "string",
                "message_thread_id": "string",
                "sender_id": "string",
                "origin_message_id": "string",
                "projection_kind": "string",
                "projection_scope": "string",
                "vector_ref": "string",
                "vector_dim": "int64",
                "vector_json": "string",
            }
        )
        self._state.lexical_index_df = self._state.lexical_index_df.astype(
            {
                "tenant_id": "string",
                "doc_id": "string",
                "token": "string",
                "term_freq": "int64",
                "doc_len": "int64",
            }
        )
        self._state.vector_index_df = self._state.vector_index_df.astype(
            {
                "tenant_id": "string",
                "doc_id": "string",
                "vector_dim": "int64",
                "vector_json": "string",
                "vector_norm": "float64",
            }
        )
        self._state.embedding_index_metadata_df = self._state.embedding_index_metadata_df.astype(
            {
                "tenant_id": "string",
                "entity_type": "string",
                "entity_id": "string",
                "embedding_ref": "string",
                "embedding_provider": "string",
                "embedding_model": "string",
                "source_hash": "string",
                "reembed_policy": "string",
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
        self._state.semantic_jobs_df = self._state.semantic_jobs_df.astype(
            {
                "job_id": "string",
                "tenant_id": "string",
                "status": "string",
                "latest_wal_seq": "int64",
                "claimed_wal_seq": "int64",
                "impacted_sessions_json": "string",
                "claimed_sessions_json": "string",
                "cause": "string",
                "attempt_count": "int64",
                "last_error": "string",
                "lease_owner": "string",
            }
        )
        self._lock = asyncio.Lock()
        self._messages_indexed_df: Optional[pd.DataFrame] = None
        self._messages_index_dirty = True
        self._capsules_indexed_df: Optional[pd.DataFrame] = None
        self._capsules_index_dirty = True
        self._session_rollups_indexed_df: Optional[pd.DataFrame] = None
        self._session_rollups_index_dirty = True
        self._search_docs_indexed_df: Optional[pd.DataFrame] = None
        self._search_docs_index_dirty = True
        self._lexical_index_indexed_df: Optional[pd.DataFrame] = None
        self._lexical_index_dirty = True
        self._vector_indexed_df: Optional[pd.DataFrame] = None
        self._vector_index_dirty = True
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

    def _invalidate_search_docs_index_locked(self) -> None:
        self._search_docs_indexed_df = None
        self._search_docs_index_dirty = True

    def _invalidate_lexical_index_locked(self) -> None:
        self._lexical_index_indexed_df = None
        self._lexical_index_dirty = True

    def _invalidate_vector_index_locked(self) -> None:
        self._vector_indexed_df = None
        self._vector_index_dirty = True

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
        self._invalidate_search_docs_index_locked()
        self._invalidate_lexical_index_locked()
        self._invalidate_vector_index_locked()
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
        if "source_topic_id" not in indexed.columns:
            indexed["source_topic_id"] = indexed["topic_id"]
        indexed["source_topic_id"] = indexed["source_topic_id"].fillna(indexed["topic_id"]).astype(str)
        if "source_topic_path" not in indexed.columns:
            indexed["source_topic_path"] = indexed.get("topic_path", pd.Series(["default"] * len(indexed)))
        indexed["source_topic_path"] = indexed["source_topic_path"].fillna(
            indexed.get("topic_path", indexed["topic_id"])
        ).astype(str)
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
            out = out[~out["projection_kind"].astype(str).isin(RAW_COMPAT_PROJECTION_KINDS)]
        elif row_mode == "raw":
            out = out[out["projection_kind"].astype(str).isin(RAW_COMPAT_PROJECTION_KINDS)]
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
            scoped = scoped[_group_identity_mask(scoped, group_id)]
        if topic_id is not None:
            canonical_ids = self._canonical_topic_ids_locked(tenant_id, [str(topic_id)])
            if canonical_ids:
                scoped = scoped[scoped["topic_id"].astype(str).isin(canonical_ids)]
            else:
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
        indexed["topic_id"] = indexed["topic_id"].fillna("default").astype(str)
        indexed["capsule_ordinal"] = pd.to_numeric(indexed["capsule_ordinal"], errors="coerce").fillna(0).astype(int)
        indexed["capsule_state"] = indexed["capsule_state"].fillna("").astype(str)
        indexed["updated_at"] = pd.to_datetime(indexed["updated_at"], utc=True, errors="coerce")
        indexed = indexed.set_index(CAPSULE_MULTIINDEX_LEVELS, drop=False).sort_index(kind="stable")
        self._capsules_indexed_df = indexed
        self._capsules_index_dirty = False
        return indexed

    def _canonical_topic_ids_locked(
        self,
        tenant_id: str,
        topic_ids: List[str],
    ) -> List[str]:
        if not topic_ids:
            return []
        if self._state.topics_df.empty:
            return sorted({str(item) for item in topic_ids if str(item)})
        topics = self._state.topics_df.copy()
        topics["tenant_id"] = topics["tenant_id"].fillna("default").astype(str)
        topics["topic_id"] = topics["topic_id"].fillna("default").astype(str)
        topics["canonical_topic_id"] = topics["canonical_topic_id"].fillna(topics["topic_id"]).astype(str)
        scoped = topics[topics["tenant_id"].astype(str) == str(tenant_id)]
        canonical_lookup = {
            (str(row["tenant_id"]), str(row["topic_id"])): str(row["canonical_topic_id"] or row["topic_id"])
            for _, row in scoped.iterrows()
        }
        return sorted(
            {
                canonical_lookup.get((str(tenant_id), str(topic_id)), str(topic_id))
                for topic_id in topic_ids
                if str(topic_id)
            }
        )

    def _apply_materialized_topics_to_messages_locked(self) -> None:
        if self._state.messages_df.empty:
            return
        scoped = self._state.messages_df.copy().reset_index(drop=True)
        scoped["tenant_id"] = scoped["tenant_id"].fillna("default").astype(str)
        scoped["topic_id"] = scoped["topic_id"].fillna("default").astype(str)
        if "source_topic_id" not in scoped.columns:
            scoped["source_topic_id"] = scoped["topic_id"]
        scoped["source_topic_id"] = scoped["source_topic_id"].fillna(scoped["topic_id"]).astype(str)
        scoped["topic_path"] = scoped["topic_path"].fillna(scoped["topic_id"]).astype(str)
        if "source_topic_path" not in scoped.columns:
            scoped["source_topic_path"] = scoped["topic_path"]
        scoped["source_topic_path"] = scoped["source_topic_path"].fillna(scoped["topic_path"]).astype(str)
        scoped["topic_parent_id"] = scoped["topic_parent_id"].fillna("").astype(str)
        scoped["projection_kind"] = scoped["projection_kind"].fillna("").astype(str)

        if self._state.topics_df.empty:
            scoped["topic_id"] = scoped["source_topic_id"]
            scoped["topic_path"] = scoped["source_topic_path"]
        else:
            topics = self._state.topics_df.copy().reset_index(drop=True)
            topics["tenant_id"] = topics["tenant_id"].fillna("default").astype(str)
            topics["topic_id"] = topics["topic_id"].fillna("default").astype(str)
            topics["canonical_topic_id"] = topics["canonical_topic_id"].fillna(topics["topic_id"]).astype(str)
            topics["topic_path"] = topics["topic_path"].fillna(topics["canonical_topic_id"]).astype(str)
            topics["topic_parent_id"] = topics["topic_parent_id"].fillna("").astype(str)

            canonical_lookup = {
                (str(row["tenant_id"]), str(row["topic_id"])): str(row["canonical_topic_id"] or row["topic_id"])
                for _, row in topics.iterrows()
            }
            canonical_path_lookup: Dict[Tuple[str, str], str] = {}
            canonical_parent_lookup: Dict[Tuple[str, str], str] = {}
            for _, row in topics.iterrows():
                tenant_key = str(row["tenant_id"])
                canonical_topic_id = str(row["canonical_topic_id"] or row["topic_id"])
                topic_key = str(row["topic_id"])
                key = (tenant_key, canonical_topic_id)
                if key not in canonical_path_lookup or topic_key == canonical_topic_id:
                    canonical_path_lookup[key] = str(row["topic_path"] or canonical_topic_id)
                    canonical_parent_lookup[key] = str(row["topic_parent_id"] or "")

            canonical_ids = [
                canonical_lookup.get((tenant_id, source_topic_id), source_topic_id)
                for tenant_id, source_topic_id in zip(
                    scoped["tenant_id"].astype(str).tolist(),
                    scoped["source_topic_id"].astype(str).tolist(),
                )
            ]
            scoped["topic_id"] = canonical_ids
            scoped["topic_path"] = [
                source_topic_path
                or canonical_path_lookup.get(
                    (tenant_id, canonical_topic_id),
                    canonical_topic_id,
                )
                for tenant_id, canonical_topic_id, source_topic_path in zip(
                    scoped["tenant_id"].astype(str).tolist(),
                    scoped["topic_id"].astype(str).tolist(),
                    scoped["source_topic_path"].astype(str).tolist(),
                )
            ]
            scoped["topic_parent_id"] = [
                canonical_parent_lookup.get((tenant_id, canonical_topic_id), topic_parent_id)
                for tenant_id, canonical_topic_id, topic_parent_id in zip(
                    scoped["tenant_id"].astype(str).tolist(),
                    scoped["topic_id"].astype(str).tolist(),
                    scoped["topic_parent_id"].astype(str).tolist(),
                )
            ]

        scoped.loc[scoped["projection_kind"].astype(str) == RAW_PROJECTION_KIND, "capsule_level"] = "L0"
        scoped.loc[scoped["projection_kind"].astype(str) != RAW_PROJECTION_KIND, "capsule_level"] = "L1"
        self._state.messages_df = scoped[MESSAGES_COLUMNS]
        self._invalidate_messages_index_locked()

    def _topic_ids_for_session_locked(
        self,
        tenant_id: str,
        session_id: str,
        *,
        include_deleted: bool,
    ) -> List[str]:
        rows = self._messages_for_query_locked(
            tenant_id=tenant_id,
            session_id=session_id,
            row_mode="all",
            include_deleted=include_deleted,
        )
        topic_ids = sorted({str(item) for item in rows["topic_id"].astype(str).tolist() if str(item)}) if not rows.empty else []
        if not topic_ids and str(session_id).startswith("topic:"):
            fallback_topic_id = str(session_id).split(":", 1)[1].strip()
            if fallback_topic_id:
                topic_ids = [fallback_topic_id]
        return self._canonical_topic_ids_locked(tenant_id, topic_ids)

    def _capsules_for_query_locked(self, tenant_id: str, session_id: str) -> pd.DataFrame:
        if self._capsules_index_dirty or self._capsules_indexed_df is None:
            indexed = self._build_capsules_index_locked()
        else:
            indexed = self._capsules_indexed_df
        if indexed.empty:
            return indexed
        topic_ids = self._topic_ids_for_session_locked(tenant_id, str(session_id), include_deleted=False)
        if not topic_ids:
            return indexed.iloc[0:0].copy()
        scoped = indexed[
            (indexed["tenant_id"].astype(str) == str(tenant_id))
            & (indexed["topic_id"].astype(str).isin(topic_ids))
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

    def _raw_messages_for_query_locked(
        self,
        *,
        tenant_id: str,
        session_id: Optional[str],
        channel: Optional[str] = None,
        chat_type: Optional[str] = None,
        group_id: Optional[str] = None,
        topic_id: Optional[str] = None,
        message_thread_id: Optional[str] = None,
        include_deleted: bool = False,
    ) -> pd.DataFrame:
        if session_id is None:
            return self._messages_for_query_locked(
                tenant_id=tenant_id,
                session_id=None,
                channel=channel,
                chat_type=chat_type,
                group_id=group_id,
                topic_id=topic_id,
                message_thread_id=message_thread_id,
                row_mode="raw",
                include_deleted=include_deleted,
            )
        scoped = self._messages_for_query_locked(
            tenant_id=tenant_id,
            session_id=session_id,
            channel=channel,
            chat_type=chat_type,
            group_id=group_id,
            topic_id=topic_id,
            message_thread_id=message_thread_id,
            row_mode="all",
            include_deleted=include_deleted,
        )
        if scoped.empty:
            return scoped.iloc[0:0].copy()
        if self._messages_index_dirty or self._messages_indexed_df is None:
            indexed = self._build_messages_index_locked()
        else:
            indexed = self._messages_indexed_df
        origin_ids = {
            str(item)
            for item in scoped["origin_message_id"].astype(str).tolist()
            if str(item)
        }
        if not origin_ids:
            return indexed.iloc[0:0].copy()
        raw_rows = indexed[
            (indexed["tenant_id"].astype(str) == str(tenant_id))
            & (indexed["projection_kind"].astype(str) == RAW_PROJECTION_KIND)
            & (indexed["origin_message_id"].astype(str).isin(origin_ids))
        ]
        if not include_deleted:
            raw_rows = raw_rows[raw_rows["message_state"].astype(str) != MESSAGE_STATE_DELETED]
        if raw_rows.empty:
            return indexed.iloc[0:0].copy()
        return self._chronological_messages(raw_rows)

    def _topic_rows_for_query_locked(
        self,
        *,
        tenant_id: str,
        session_id: Optional[str],
        topic_id: Optional[str],
    ) -> pd.DataFrame:
        topics = self._state.topics_df
        if topics.empty:
            return topics.iloc[0:0].copy()
        scoped = topics.copy().reset_index(drop=True)
        scoped["tenant_id"] = scoped["tenant_id"].fillna("default").astype(str)
        scoped["topic_id"] = scoped["topic_id"].fillna("default").astype(str)
        scoped["canonical_topic_id"] = scoped["canonical_topic_id"].fillna(scoped["topic_id"]).astype(str)
        scoped["topic_path"] = scoped["topic_path"].fillna(scoped["canonical_topic_id"]).astype(str)
        scoped["status"] = scoped["status"].fillna("").astype(str)
        scoped["updated_at"] = pd.to_datetime(scoped["updated_at"], utc=True, errors="coerce")
        scoped = scoped[scoped["tenant_id"].astype(str) == str(tenant_id)]
        if topic_id is not None:
            canonical_ids = self._canonical_topic_ids_locked(tenant_id, [str(topic_id)])
            scoped = scoped[scoped["canonical_topic_id"].astype(str).isin(canonical_ids)]
        elif session_id is not None:
            canonical_ids = self._topic_ids_for_session_locked(
                tenant_id,
                str(session_id),
                include_deleted=False,
            )
            scoped = scoped[scoped["canonical_topic_id"].astype(str).isin(canonical_ids)]
        if scoped.empty:
            return topics.iloc[0:0].copy()
        scoped = scoped[scoped["status"].astype(str) != "compacted"].copy()
        if scoped.empty:
            return topics.iloc[0:0].copy()
        scoped["_canonical_priority"] = (
            scoped["topic_id"].astype(str) != scoped["canonical_topic_id"].astype(str)
        ).astype(int)
        scoped = scoped.sort_values(
            ["canonical_topic_id", "_canonical_priority", "updated_at", "topic_id"],
            ascending=[True, True, False, True],
            kind="stable",
        )
        scoped = scoped.groupby("canonical_topic_id", sort=True, as_index=False).head(1)
        return scoped.drop(columns=["_canonical_priority"]).reset_index(drop=True)

    def _session_rollup_rows_for_scope_locked(
        self,
        *,
        tenant_id: str,
        session_id: Optional[str],
        channel: Optional[str] = None,
        chat_type: Optional[str] = None,
        group_id: Optional[str] = None,
        topic_id: Optional[str] = None,
        message_thread_id: Optional[str] = None,
    ) -> pd.DataFrame:
        if self._session_rollups_index_dirty or self._session_rollups_indexed_df is None:
            indexed = self._build_session_rollups_index_locked()
        else:
            indexed = self._session_rollups_indexed_df
        if indexed.empty:
            return indexed.iloc[0:0].copy()
        if session_id is not None:
            return self._session_rollups_for_query_locked(tenant_id, str(session_id))
        scoped = indexed[indexed["tenant_id"].astype(str) == str(tenant_id)]
        projection_rows = self._messages_for_query_locked(
            tenant_id=tenant_id,
            session_id=None,
            channel=channel,
            chat_type=chat_type,
            group_id=group_id,
            topic_id=topic_id,
            message_thread_id=message_thread_id,
            row_mode="projection",
            include_deleted=False,
        )
        if not projection_rows.empty:
            session_ids = sorted(
                {
                    str(item)
                    for item in projection_rows["session_id"].astype(str).tolist()
                    if str(item)
                }
            )
            scoped = scoped[scoped["session_id"].astype(str).isin(session_ids)]
        if scoped.empty:
            return indexed.iloc[0:0].copy()
        return scoped

    def _capsule_rows_for_scope_locked(
        self,
        *,
        tenant_id: str,
        session_id: Optional[str],
        topic_id: Optional[str],
    ) -> pd.DataFrame:
        if self._capsules_index_dirty or self._capsules_indexed_df is None:
            indexed = self._build_capsules_index_locked()
        else:
            indexed = self._capsules_indexed_df
        if indexed.empty:
            return indexed.iloc[0:0].copy()
        if session_id is not None:
            return self._capsules_for_query_locked(tenant_id, str(session_id))
        scoped = indexed[indexed["tenant_id"].astype(str) == str(tenant_id)]
        if topic_id is not None:
            canonical_ids = self._canonical_topic_ids_locked(tenant_id, [str(topic_id)])
            scoped = scoped[scoped["topic_id"].astype(str).isin(canonical_ids)]
        if scoped.empty:
            return indexed.iloc[0:0].copy()
        return scoped

    def _materialize_search_docs_locked(self) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []

        def _json_citations(values: Sequence[str]) -> str:
            seen: set[str] = set()
            ordered: List[str] = []
            for item in values:
                value = str(item or "").strip()
                if not value or value in seen:
                    continue
                seen.add(value)
                ordered.append(value)
            return json.dumps(ordered, separators=(",", ":"))

        def _origin_bounds(frame: pd.DataFrame) -> List[str]:
            if frame.empty:
                return []
            ordered = self._chronological_messages(frame)
            first_origin = str(ordered.iloc[0].get("origin_message_id") or ordered.iloc[0]["message_id"])
            last_origin = str(ordered.iloc[-1].get("origin_message_id") or ordered.iloc[-1]["message_id"])
            citations = [f"origin:{first_origin}"]
            if last_origin and last_origin != first_origin:
                citations.append(f"origin:{last_origin}")
            return citations

        raw_rows = authoritative_raw_messages(self._state.messages_df)
        if not raw_rows.empty:
            scoped_raw = self._chronological_messages(raw_rows).reset_index(drop=True)
            for _, row in scoped_raw.iterrows():
                origin_id = str(row.get("origin_message_id") or row.get("message_id") or "")
                if not origin_id:
                    continue
                updated_at = row.get("updated_at") if pd.notna(row.get("updated_at")) else row.get("ts")
                citations = [f"origin:{origin_id}"]
                rows.append(
                    {
                        "tenant_id": str(row.get("tenant_id") or "default"),
                        "doc_id": f"raw:{origin_id}",
                        "entity_type": "raw_message",
                        "entity_id": origin_id,
                        "source_tier": "L0",
                        "session_id": str(row.get("native_session_id") or ""),
                        "updated_at": _utc_timestamp(updated_at),
                        "text": str(row.get("content") or ""),
                        "path": f"memory/raw/{_safe_path_fragment(row.get('tenant_id') or 'default')}/{origin_id}.md",
                        "start_line": 1,
                        "end_line": 1,
                        "snippet": _trim_text(row.get("content") or "", 700),
                        "citation": citations[0],
                        "citations_json": _json_citations(citations),
                        "channel": str(row.get("channel") or ""),
                        "chat_type": str(row.get("chat_type") or ""),
                        "account_id": str(row.get("account_id") or ""),
                        "group_id": str(row.get("group_id") or ""),
                        "topic_id": str(row.get("topic_id") or "default"),
                        "topic_path": str(row.get("topic_path") or row.get("topic_id") or "default"),
                        "message_thread_id": str(row.get("message_thread_id") or ""),
                        "sender_id": str(row.get("sender_id") or ""),
                        "origin_message_id": origin_id,
                        "projection_kind": str(row.get("projection_kind") or ""),
                        "projection_scope": str(row.get("projection_scope") or ""),
                        "vector_ref": "",
                        "vector_dim": 0,
                        "vector_json": "[]",
                    }
                )

        if not self._state.beliefs_df.empty:
            beliefs = self._state.beliefs_df.copy().reset_index(drop=True)
            beliefs["tenant_id"] = beliefs["tenant_id"].fillna("default").astype(str)
            beliefs["belief_id"] = beliefs["belief_id"].fillna("").astype(str)
            beliefs["scope_type"] = beliefs["scope_type"].fillna("").astype(str)
            beliefs["scope_key"] = beliefs["scope_key"].fillna("").astype(str)
            beliefs["session_id"] = beliefs["session_id"].fillna("").astype(str)
            beliefs["topic_id"] = beliefs["topic_id"].fillna("").astype(str)
            beliefs["group_id"] = beliefs["group_id"].fillna("").astype(str)
            beliefs["projection_kind"] = beliefs["projection_kind"].fillna("").astype(str)
            beliefs["projection_scope"] = beliefs["projection_scope"].fillna("").astype(str)
            beliefs["summary"] = beliefs["summary"].fillna("").astype(str)
            beliefs["updated_at"] = pd.to_datetime(beliefs["updated_at"], utc=True, errors="coerce")
            beliefs["vector_ref"] = beliefs["vector_ref"].fillna("").astype(str)
            beliefs["vector_json"] = beliefs["vector_json"].fillna("[]").astype(str)
            beliefs["vector_dim"] = pd.to_numeric(beliefs["vector_dim"], errors="coerce").fillna(0).astype(int)
            for _, row in beliefs.iterrows():
                belief_id = str(row.get("belief_id") or "")
                if not belief_id:
                    continue
                first_origin = str(row.get("first_origin_message_id") or "").strip()
                last_origin = str(row.get("last_origin_message_id") or "").strip()
                citations = [
                    belief_id,
                    *([f"origin:{first_origin}"] if first_origin else []),
                    *([f"origin:{last_origin}"] if last_origin and last_origin != first_origin else []),
                ]
                scope_type = str(row.get("scope_type") or "")
                scope_key = str(row.get("scope_key") or "")
                rows.append(
                    {
                        "tenant_id": str(row.get("tenant_id") or "default"),
                        "doc_id": belief_id,
                        "entity_type": "l0_abstract",
                        "entity_id": belief_id,
                        "source_tier": "L0",
                        "session_id": str(row.get("session_id") or ""),
                        "updated_at": _utc_timestamp(row.get("updated_at")),
                        "text": str(row.get("summary") or ""),
                        "path": f"memory/l0/{scope_type}_{_safe_path_fragment(scope_key)}.md",
                        "start_line": 1,
                        "end_line": 1,
                        "snippet": _trim_text(row.get("summary") or "", 700),
                        "citation": belief_id,
                        "citations_json": _json_citations(citations),
                        "channel": "",
                        "chat_type": "",
                        "account_id": "",
                        "group_id": str(row.get("group_id") or ""),
                        "topic_id": str(row.get("topic_id") or ""),
                        "topic_path": str(row.get("topic_id") or ""),
                        "message_thread_id": "",
                        "sender_id": "",
                        "origin_message_id": first_origin,
                        "projection_kind": str(row.get("projection_kind") or ""),
                        "projection_scope": str(row.get("projection_scope") or ""),
                        "vector_ref": str(row.get("vector_ref") or ""),
                        "vector_dim": int(row.get("vector_dim") or 0),
                        "vector_json": str(row.get("vector_json") or "[]"),
                    }
                )

        if not self._state.session_rollups_df.empty:
            rollups = self._state.session_rollups_df.copy().reset_index(drop=True)
            rollups["tenant_id"] = rollups["tenant_id"].fillna("default").astype(str)
            rollups["rollup_id"] = rollups["rollup_id"].fillna("").astype(str)
            rollups["session_id"] = rollups["session_id"].fillna("").astype(str)
            rollups["window_kind"] = rollups["window_kind"].fillna("").astype(str)
            rollups["window_key"] = rollups["window_key"].fillna("").astype(str)
            rollups["summary"] = rollups["summary"].fillna("").astype(str)
            rollups["updated_at"] = pd.to_datetime(rollups["updated_at"], utc=True, errors="coerce")
            rollups["vector_ref"] = rollups["vector_ref"].fillna("").astype(str)
            rollups["vector_json"] = rollups["vector_json"].fillna("[]").astype(str)
            rollups["vector_dim"] = pd.to_numeric(rollups["vector_dim"], errors="coerce").fillna(0).astype(int)
            projection_rows = self._messages_for_query_locked(
                tenant_id="*",
                session_id=None,
                row_mode="projection",
                include_deleted=False,
            ).reset_index(drop=True)
            for _, row in rollups.iterrows():
                primary = str(row.get("rollup_id") or "")
                if not primary:
                    primary = (
                        "rollup:"
                        f"{str(row.get('tenant_id') or 'default')}:"
                        f"{str(row.get('session_id') or '')}:"
                        f"{str(row.get('window_kind') or '')}:"
                        f"{str(row.get('window_key') or '')}"
                    )
                supporting = projection_rows[
                    (projection_rows["tenant_id"].astype(str) == str(row.get("tenant_id") or "default"))
                    & (projection_rows["session_id"].astype(str) == str(row.get("session_id") or ""))
                ]
                bucket_start = pd.to_datetime(row.get("bucket_start"), utc=True, errors="coerce")
                bucket_end = pd.to_datetime(row.get("bucket_end"), utc=True, errors="coerce")
                if pd.notna(bucket_start):
                    supporting = supporting[supporting["ts"] >= bucket_start]
                if pd.notna(bucket_end):
                    supporting = supporting[supporting["ts"] < bucket_end]
                citations = [primary, *_origin_bounds(supporting)]
                rows.append(
                    {
                        "tenant_id": str(row.get("tenant_id") or "default"),
                        "doc_id": primary,
                        "entity_type": "session_rollup",
                        "entity_id": primary,
                        "source_tier": "L1",
                        "session_id": str(row.get("session_id") or ""),
                        "updated_at": _utc_timestamp(row.get("updated_at")),
                        "text": str(row.get("summary") or ""),
                        "path": (
                            "memory/rollups/"
                            f"{_safe_path_fragment(row.get('session_id') or '')}/"
                            f"{_safe_path_fragment(row.get('window_kind') or '')}/"
                            f"{_safe_path_fragment(row.get('window_key') or '')}.md"
                        ),
                        "start_line": 1,
                        "end_line": 1,
                        "snippet": _trim_text(row.get("summary") or "", 700),
                        "citation": primary,
                        "citations_json": _json_citations(citations),
                        "channel": "",
                        "chat_type": "",
                        "account_id": "",
                        "group_id": "",
                        "topic_id": "",
                        "topic_path": "",
                        "message_thread_id": "",
                        "sender_id": "",
                        "origin_message_id": (
                            citations[1].split(":", 1)[1] if len(citations) > 1 else ""
                        ),
                        "projection_kind": "",
                        "projection_scope": str(row.get("session_id") or ""),
                        "vector_ref": str(row.get("vector_ref") or ""),
                        "vector_dim": int(row.get("vector_dim") or 0),
                        "vector_json": str(row.get("vector_json") or "[]"),
                    }
                )

        topic_tenants = sorted(
            {
                str(item)
                for item in self._state.topics_df.get("tenant_id", pd.Series(dtype="string")).fillna("default").astype(str).tolist()
                if str(item)
            }
        )
        for tenant_id in topic_tenants:
            topic_rows = self._topic_rows_for_query_locked(
                tenant_id=tenant_id,
                session_id=None,
                topic_id=None,
            )
            if topic_rows.empty:
                continue
            topic_state = self._state.topics_df.copy().reset_index(drop=True)
            topic_state["tenant_id"] = topic_state["tenant_id"].fillna("default").astype(str)
            topic_state["topic_id"] = topic_state["topic_id"].fillna("default").astype(str)
            topic_state["canonical_topic_id"] = topic_state["canonical_topic_id"].fillna(topic_state["topic_id"]).astype(str)
            topic_aliases: Dict[str, set[str]] = {}
            scoped_topic_state = topic_state[topic_state["tenant_id"].astype(str) == str(tenant_id)]
            for _, topic_state_row in scoped_topic_state.iterrows():
                canonical_id = str(topic_state_row["canonical_topic_id"] or topic_state_row["topic_id"])
                topic_aliases.setdefault(canonical_id, set()).add(str(topic_state_row["topic_id"]))
            tenant_raw_rows = raw_rows[raw_rows["tenant_id"].astype(str) == str(tenant_id)].copy()
            if "source_topic_id" not in tenant_raw_rows.columns:
                tenant_raw_rows["source_topic_id"] = tenant_raw_rows["topic_id"]
            tenant_raw_rows["source_topic_id"] = tenant_raw_rows["source_topic_id"].fillna(
                tenant_raw_rows["topic_id"]
            ).astype(str)
            for _, row in topic_rows.iterrows():
                canonical_topic_id = str(row.get("canonical_topic_id") or row.get("topic_id") or "default")
                aliases = topic_aliases.get(canonical_topic_id, {canonical_topic_id})
                supporting = tenant_raw_rows[
                    tenant_raw_rows["source_topic_id"].astype(str).isin(sorted(aliases))
                    | (tenant_raw_rows["topic_id"].astype(str) == canonical_topic_id)
                ]
                citations = [f"topic:{canonical_topic_id}", *_origin_bounds(supporting)]
                rows.append(
                    {
                        "tenant_id": tenant_id,
                        "doc_id": f"topic:{canonical_topic_id}",
                        "entity_type": "topic",
                        "entity_id": canonical_topic_id,
                        "source_tier": "L2",
                        "session_id": f"topic:{canonical_topic_id}",
                        "updated_at": _utc_timestamp(row.get("updated_at")),
                        "text": str(row.get("summary") or row.get("vector_text") or ""),
                        "path": f"memory/topics/{_safe_path_fragment(canonical_topic_id)}.md",
                        "start_line": 1,
                        "end_line": 1,
                        "snippet": _trim_text(row.get("summary") or row.get("vector_text") or "", 700),
                        "citation": f"topic:{canonical_topic_id}",
                        "citations_json": _json_citations(citations),
                        "channel": "",
                        "chat_type": "",
                        "account_id": "",
                        "group_id": "",
                        "topic_id": canonical_topic_id,
                        "topic_path": str(row.get("topic_path") or canonical_topic_id),
                        "message_thread_id": "",
                        "sender_id": "",
                        "origin_message_id": (
                            citations[1].split(":", 1)[1] if len(citations) > 1 else ""
                        ),
                        "projection_kind": "",
                        "projection_scope": "",
                        "vector_ref": str(row.get("vector_ref") or ""),
                        "vector_dim": int(row.get("vector_dim") or 0),
                        "vector_json": str(row.get("vector_json") or "[]"),
                    }
                )

        if not self._state.capsules_df.empty:
            capsules = self._state.capsules_df.copy().reset_index(drop=True)
            capsules["tenant_id"] = capsules["tenant_id"].fillna("default").astype(str)
            capsules["capsule_id"] = capsules["capsule_id"].fillna("").astype(str)
            capsules["summary"] = capsules["summary"].fillna("").astype(str)
            capsules["updated_at"] = pd.to_datetime(capsules["updated_at"], utc=True, errors="coerce")
            capsules["vector_ref"] = capsules["vector_ref"].fillna("").astype(str)
            capsules["vector_json"] = capsules["vector_json"].fillna("[]").astype(str)
            capsules["vector_dim"] = pd.to_numeric(capsules["vector_dim"], errors="coerce").fillna(0).astype(int)
            for _, row in capsules.iterrows():
                capsule_id = str(row.get("capsule_id") or "")
                if not capsule_id:
                    continue
                first_origin = str(row.get("first_origin_message_id") or "").strip()
                last_origin = str(row.get("last_origin_message_id") or "").strip()
                citations = [
                    f"capsule:{capsule_id}",
                    *( [f"origin:{first_origin}"] if first_origin else [] ),
                    *( [f"origin:{last_origin}"] if last_origin and last_origin != first_origin else [] ),
                ]
                rows.append(
                    {
                        "tenant_id": str(row.get("tenant_id") or "default"),
                        "doc_id": f"capsule:{capsule_id}",
                        "entity_type": "capsule",
                        "entity_id": capsule_id,
                        "source_tier": "L2",
                        "session_id": str(row.get("session_id") or ""),
                        "updated_at": _utc_timestamp(row.get("updated_at")),
                        "text": str(row.get("summary") or ""),
                        "path": (
                            "memory/capsules/"
                            f"{_safe_path_fragment(row.get('topic_id') or 'default')}/"
                            f"{int(row.get('capsule_ordinal') or 0):04d}.md"
                        ),
                        "start_line": 1,
                        "end_line": 1,
                        "snippet": _trim_text(row.get("summary") or "", 700),
                        "citation": f"capsule:{capsule_id}",
                        "citations_json": _json_citations(citations),
                        "channel": "",
                        "chat_type": "",
                        "account_id": "",
                        "group_id": "",
                        "topic_id": str(row.get("topic_id") or "default"),
                        "topic_path": str(row.get("topic_path") or row.get("topic_id") or "default"),
                        "message_thread_id": "",
                        "sender_id": "",
                        "origin_message_id": first_origin,
                        "projection_kind": "",
                        "projection_scope": "",
                        "vector_ref": str(row.get("vector_ref") or ""),
                        "vector_dim": int(row.get("vector_dim") or 0),
                        "vector_json": str(row.get("vector_json") or "[]"),
                    }
                )

        if not rows:
            return pd.DataFrame(columns=SEARCH_DOC_COLUMNS)
        frame = pd.DataFrame(rows, columns=SEARCH_DOC_COLUMNS)
        frame["tenant_id"] = frame["tenant_id"].fillna("default").astype(str)
        frame["doc_id"] = frame["doc_id"].fillna("").astype(str)
        frame["entity_type"] = frame["entity_type"].fillna("").astype(str)
        frame["entity_id"] = frame["entity_id"].fillna("").astype(str)
        frame["source_tier"] = frame["source_tier"].fillna("L0").astype(str)
        frame["session_id"] = frame["session_id"].fillna("").astype(str)
        frame["updated_at"] = pd.to_datetime(frame["updated_at"], utc=True, errors="coerce")
        frame["text"] = frame["text"].fillna("").astype(str)
        frame["path"] = frame["path"].fillna("").astype(str)
        frame["start_line"] = pd.to_numeric(frame["start_line"], errors="coerce").fillna(1).astype(int)
        frame["end_line"] = pd.to_numeric(frame["end_line"], errors="coerce").fillna(1).astype(int)
        frame["snippet"] = frame["snippet"].fillna("").astype(str)
        frame["citation"] = frame["citation"].fillna("").astype(str)
        frame["citations_json"] = frame["citations_json"].fillna("[]").astype(str)
        frame["channel"] = frame["channel"].fillna("").astype(str)
        frame["chat_type"] = frame["chat_type"].fillna("").astype(str)
        frame["account_id"] = frame["account_id"].fillna("").astype(str)
        frame["group_id"] = frame["group_id"].fillna("").astype(str)
        frame["topic_id"] = frame["topic_id"].fillna("").astype(str)
        frame["topic_path"] = frame["topic_path"].fillna("").astype(str)
        frame["message_thread_id"] = frame["message_thread_id"].fillna("").astype(str)
        frame["sender_id"] = frame["sender_id"].fillna("").astype(str)
        frame["origin_message_id"] = frame["origin_message_id"].fillna("").astype(str)
        frame["projection_kind"] = frame["projection_kind"].fillna("").astype(str)
        frame["projection_scope"] = frame["projection_scope"].fillna("").astype(str)
        frame["vector_ref"] = frame["vector_ref"].fillna("").astype(str)
        frame["vector_dim"] = pd.to_numeric(frame["vector_dim"], errors="coerce").fillna(0).astype(int)
        frame["vector_json"] = frame["vector_json"].fillna("[]").astype(str)
        frame = frame.sort_values(
            ["tenant_id", "doc_id", "updated_at"],
            ascending=[True, True, True],
            kind="stable",
        ).drop_duplicates(
            subset=["tenant_id", "doc_id"],
            keep="last",
        )
        return frame[SEARCH_DOC_COLUMNS].reset_index(drop=True)

    def _rebuild_search_indexes_locked(self, *, vector_dim: int) -> int:
        search_docs = self._materialize_search_docs_locked()
        self._state.search_docs_df = search_docs
        self._state.lexical_index_df = materialize_lexical_index(search_docs)
        self._state.vector_index_df = materialize_vector_index(search_docs, dim=vector_dim)
        self._invalidate_search_docs_index_locked()
        self._invalidate_lexical_index_locked()
        self._invalidate_vector_index_locked()
        return int(search_docs.shape[0])

    async def rebuild_search_indexes(self, *, vector_dim: int = DEFAULT_TOPIC_VECTOR_DIM) -> int:
        async with self._lock:
            return self._rebuild_search_indexes_locked(vector_dim=vector_dim)

    def _build_search_docs_index_locked(self) -> pd.DataFrame:
        df = self._state.search_docs_df
        if df.empty:
            empty = df.copy()
            self._search_docs_indexed_df = empty.set_index(SEARCH_DOC_MULTIINDEX_LEVELS, drop=False)
            self._search_docs_index_dirty = False
            return self._search_docs_indexed_df
        indexed = df.copy()
        indexed["tenant_id"] = indexed["tenant_id"].fillna("default").astype(str)
        indexed["doc_id"] = indexed["doc_id"].fillna("").astype(str)
        indexed["entity_type"] = indexed["entity_type"].fillna("").astype(str)
        indexed["entity_id"] = indexed["entity_id"].fillna("").astype(str)
        indexed["source_tier"] = indexed["source_tier"].fillna("L0").astype(str)
        indexed["session_id"] = indexed["session_id"].fillna("").astype(str)
        indexed["text"] = indexed["text"].fillna("").astype(str)
        indexed["path"] = indexed["path"].fillna("").astype(str)
        indexed["start_line"] = pd.to_numeric(indexed["start_line"], errors="coerce").fillna(1).astype(int)
        indexed["end_line"] = pd.to_numeric(indexed["end_line"], errors="coerce").fillna(1).astype(int)
        indexed["snippet"] = indexed["snippet"].fillna("").astype(str)
        indexed["citation"] = indexed["citation"].fillna("").astype(str)
        indexed["citations_json"] = indexed["citations_json"].fillna("[]").astype(str)
        indexed["channel"] = indexed["channel"].fillna("").astype(str)
        indexed["chat_type"] = indexed["chat_type"].fillna("").astype(str)
        indexed["account_id"] = indexed["account_id"].fillna("").astype(str)
        indexed["group_id"] = indexed["group_id"].fillna("").astype(str)
        indexed["topic_id"] = indexed["topic_id"].fillna("").astype(str)
        indexed["topic_path"] = indexed["topic_path"].fillna("").astype(str)
        indexed["message_thread_id"] = indexed["message_thread_id"].fillna("").astype(str)
        indexed["sender_id"] = indexed["sender_id"].fillna("").astype(str)
        indexed["origin_message_id"] = indexed["origin_message_id"].fillna("").astype(str)
        indexed["projection_kind"] = indexed["projection_kind"].fillna("").astype(str)
        indexed["projection_scope"] = indexed["projection_scope"].fillna("").astype(str)
        indexed["vector_ref"] = indexed["vector_ref"].fillna("").astype(str)
        indexed["vector_dim"] = pd.to_numeric(indexed["vector_dim"], errors="coerce").fillna(0).astype(int)
        indexed["vector_json"] = indexed["vector_json"].fillna("[]").astype(str)
        indexed["updated_at"] = pd.to_datetime(indexed["updated_at"], utc=True, errors="coerce")
        indexed = indexed.set_index(SEARCH_DOC_MULTIINDEX_LEVELS, drop=False).sort_index(kind="stable")
        self._search_docs_indexed_df = indexed
        self._search_docs_index_dirty = False
        return indexed

    def _build_lexical_index_locked(self) -> pd.DataFrame:
        df = self._state.lexical_index_df
        if df.empty:
            empty = df.copy()
            self._lexical_index_indexed_df = empty.set_index(LEXICAL_INDEX_MULTIINDEX_LEVELS, drop=False)
            self._lexical_index_dirty = False
            return self._lexical_index_indexed_df
        indexed = df.copy()
        indexed["tenant_id"] = indexed["tenant_id"].fillna("default").astype(str)
        indexed["doc_id"] = indexed["doc_id"].fillna("").astype(str)
        indexed["token"] = indexed["token"].fillna("").astype(str)
        indexed["term_freq"] = pd.to_numeric(indexed["term_freq"], errors="coerce").fillna(0).astype(int)
        indexed["doc_len"] = pd.to_numeric(indexed["doc_len"], errors="coerce").fillna(0).astype(int)
        indexed["updated_at"] = pd.to_datetime(indexed["updated_at"], utc=True, errors="coerce")
        indexed = indexed.set_index(LEXICAL_INDEX_MULTIINDEX_LEVELS, drop=False).sort_index(kind="stable")
        self._lexical_index_indexed_df = indexed
        self._lexical_index_dirty = False
        return indexed

    def _build_vector_index_locked(self) -> pd.DataFrame:
        df = self._state.vector_index_df
        if df.empty:
            empty = df.copy()
            self._vector_indexed_df = empty.set_index(VECTOR_INDEX_MULTIINDEX_LEVELS, drop=False)
            self._vector_index_dirty = False
            return self._vector_indexed_df
        indexed = df.copy()
        indexed["tenant_id"] = indexed["tenant_id"].fillna("default").astype(str)
        indexed["doc_id"] = indexed["doc_id"].fillna("").astype(str)
        indexed["vector_dim"] = pd.to_numeric(indexed["vector_dim"], errors="coerce").fillna(0).astype(int)
        indexed["vector_json"] = indexed["vector_json"].fillna("[]").astype(str)
        indexed["vector_norm"] = pd.to_numeric(indexed["vector_norm"], errors="coerce").fillna(0.0).astype(float)
        indexed["updated_at"] = pd.to_datetime(indexed["updated_at"], utc=True, errors="coerce")
        indexed = indexed.set_index(VECTOR_INDEX_MULTIINDEX_LEVELS, drop=False).sort_index(kind="stable")
        self._vector_indexed_df = indexed
        self._vector_index_dirty = False
        return indexed

    async def search_index_entries(
        self,
        *,
        tenant_id: str,
        doc_ids: Sequence[str],
    ) -> Tuple[List[LexicalPosting], List[VectorEntry]]:
        scoped_doc_ids = sorted({str(item) for item in doc_ids if str(item)})
        if not scoped_doc_ids:
            return [], []
        async with self._lock:
            if self._lexical_index_dirty or self._lexical_index_indexed_df is None:
                lexical_indexed = self._build_lexical_index_locked()
            else:
                lexical_indexed = self._lexical_index_indexed_df
            if self._vector_index_dirty or self._vector_indexed_df is None:
                vector_indexed = self._build_vector_index_locked()
            else:
                vector_indexed = self._vector_indexed_df

            lexical_scoped = lexical_indexed[
                (lexical_indexed["tenant_id"].astype(str) == str(tenant_id))
                & (lexical_indexed["doc_id"].astype(str).isin(scoped_doc_ids))
            ]
            vector_scoped = vector_indexed[
                (vector_indexed["tenant_id"].astype(str) == str(tenant_id))
                & (vector_indexed["doc_id"].astype(str).isin(scoped_doc_ids))
            ]

            lexical_entries = [
                LexicalPosting(
                    doc_id=str(row.get("doc_id") or ""),
                    token=str(row.get("token") or ""),
                    term_freq=(
                        0 if pd.isna(row.get("term_freq")) else int(row.get("term_freq"))
                    ),
                    doc_len=0 if pd.isna(row.get("doc_len")) else int(row.get("doc_len")),
                )
                for _, row in lexical_scoped.iterrows()
            ]
            vector_entries = [
                VectorEntry(
                    doc_id=str(row.get("doc_id") or ""),
                    vector=parse_vector_json(row.get("vector_json")),
                    norm=(
                        0.0 if pd.isna(row.get("vector_norm")) else float(row.get("vector_norm"))
                    ),
                )
                for _, row in vector_scoped.iterrows()
            ]
            return lexical_entries, vector_entries

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

    def _compact_messages_for_storage(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=MESSAGES_COLUMNS)
        compacted = df.reset_index(drop=True).copy()
        for col in MESSAGES_COLUMNS:
            if col not in compacted.columns:
                compacted[col] = None
        compacted["tenant_id"] = compacted["tenant_id"].fillna("default").astype(str)
        compacted["origin_message_id"] = compacted["origin_message_id"].fillna(
            compacted["message_id"]
        ).astype(str)
        compacted["projection_kind"] = compacted["projection_kind"].fillna("").astype(str)
        compacted["projection_scope"] = compacted["projection_scope"].fillna("").astype(str)
        compacted["ts"] = pd.to_datetime(compacted["ts"], utc=True, errors="coerce")
        compacted["updated_at"] = pd.to_datetime(compacted["updated_at"], utc=True, errors="coerce")
        compacted = compacted.sort_values(["ts", "updated_at"], kind="stable")
        compacted = compacted.drop_duplicates(subset=MESSAGE_IDENTITY_COLUMNS, keep="last")
        compacted = compacted.drop_duplicates(subset=MESSAGES_COLUMNS, keep="last")
        return compacted[MESSAGES_COLUMNS].reset_index(drop=True)

    def _message_channel_file(self, row: pd.Series) -> MessageChannelFile:
        return MessageChannelFile.from_record(row)

    def _messages_for_channel_file_locked(self, channel_file: MessageChannelFile) -> pd.DataFrame:
        scoped = self._messages_for_query_locked(
            tenant_id=channel_file.tenant_id,
            session_id=channel_file.scope_value if channel_file.scope_kind == "session" else None,
            channel=channel_file.channel,
            chat_type=channel_file.chat_type,
            group_id=channel_file.scope_value if channel_file.scope_kind == "group" else None,
            message_thread_id=channel_file.scope_value if channel_file.scope_kind == "thread" else None,
        )
        if scoped.empty:
            return scoped
        if channel_file.scope_kind == "native":
            return scoped[
                scoped["native_channel_id"].fillna("").astype(str) == channel_file.scope_value
            ]
        return scoped

    def _visible_message_rows_locked(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.reset_index(drop=True).copy()
        out["origin_message_id"] = out["origin_message_id"].fillna(out["message_id"]).astype(str)
        out["projection_kind"] = out["projection_kind"].fillna("").astype(str)
        out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
        out["updated_at"] = pd.to_datetime(out["updated_at"], utc=True, errors="coerce")
        out["_is_projection"] = ~out["projection_kind"].astype(str).isin(RAW_COMPAT_PROJECTION_KINDS)
        out = out.sort_values(["ts", "updated_at", "_is_projection"], kind="stable")
        out = out.drop_duplicates(subset=["origin_message_id"], keep="last")
        return self._chronological_messages(out.drop(columns=["_is_projection"]))

    def _default_projection_scope_from_payload(self, payload: Dict[str, object]) -> str:
        native_channel_id = str(payload.get("native_channel_id") or "").strip()
        group_id = str(payload.get("group_id") or "").strip()
        message_thread_id = str(payload.get("message_thread_id") or "").strip()
        session_id = str(payload.get("session_id") or "default").strip() or "default"
        if native_channel_id:
            return f"native:{native_channel_id}"
        if group_id:
            return f"group:{group_id}"
        if message_thread_id:
            return f"thread:{message_thread_id}"
        return f"session:{session_id}"

    def _message_view_rows_locked(
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
        effective_row_mode: Literal["projection", "raw", "all"] = (
            "all" if row_mode == "projection" else row_mode
        )
        scoped = self._messages_for_query_locked(
            tenant_id=tenant_id,
            session_id=session_id,
            channel=channel,
            chat_type=chat_type,
            group_id=group_id,
            topic_id=topic_id,
            message_thread_id=message_thread_id,
            row_mode=effective_row_mode,
            include_deleted=include_deleted,
        )
        if session_id is not None:
            if self._messages_index_dirty or self._messages_indexed_df is None:
                indexed = self._build_messages_index_locked()
            else:
                indexed = self._messages_indexed_df
            native = indexed[indexed["tenant_id"].astype(str) == str(tenant_id)]
            native = native[native["native_session_id"].fillna("").astype(str) == str(session_id)]
            if channel is not None:
                native = native[native["channel"].astype(str) == str(channel)]
            if chat_type is not None:
                native = native[native["chat_type"].astype(str) == str(chat_type)]
            if group_id is not None:
                native = native[_group_identity_mask(native, group_id)]
            if topic_id is not None:
                canonical_ids = self._canonical_topic_ids_locked(tenant_id, [str(topic_id)])
                if canonical_ids:
                    native = native[native["topic_id"].astype(str).isin(canonical_ids)]
                else:
                    native = native[native["topic_id"].astype(str) == str(topic_id)]
            if message_thread_id is not None:
                native = native[native["message_thread_id"].astype(str) == str(message_thread_id)]
            if not include_deleted:
                native = native[native["message_state"].astype(str) != MESSAGE_STATE_DELETED]
            if not native.empty:
                scoped = pd.concat(
                    [scoped.reset_index(drop=True), native.reset_index(drop=True)],
                    ignore_index=True,
                )
                scoped = scoped.drop_duplicates(subset=["message_id"], keep="last")
        if scoped.empty:
            return scoped
        if row_mode == "projection":
            return self._visible_message_rows_locked(scoped)
        return self._chronological_messages(
            self._filter_message_rows(scoped, row_mode=row_mode, include_deleted=include_deleted)
        )

    def _message_block_lines(self, row: pd.Series) -> List[str]:
        role = str(row["role"])
        ts_text = pd.to_datetime(row["ts"], utc=True).isoformat()
        lines = [
            f"## {ts_text} [{role}]",
            f"- message_id: {str(row['message_id'])}",
            f"- origin_message_id: {str(row.get('origin_message_id') or row['message_id'])}",
            f"- session_id: {str(row['session_id'])}",
        ]
        projection_kind = str(row.get("projection_kind") or "raw")
        projection_scope = str(row.get("projection_scope") or "")
        if projection_kind != "raw" or projection_scope:
            lines.append(
                f"- projection: {projection_kind} | scope: {projection_scope or '(blank)'}"
            )
        channel = str(row.get("channel") or "")
        chat_type = str(row.get("chat_type") or "")
        if channel or chat_type:
            lines.append(f"- channel: {channel or '(blank)'}")
            lines.append(f"- chat_type: {chat_type or '(blank)'}")
        topic_id = str(row.get("topic_id") or "")
        if topic_id and topic_id != "default":
            lines.append(f"- topic_id: {topic_id}")
        sender_id = str(row.get("sender_id") or "")
        if sender_id:
            lines.append(f"- sender_id: {sender_id}")
        if str(row.get("message_state") or "") == MESSAGE_STATE_DELETED:
            lines.append("- tombstone: deleted")
        lines.append("")
        lines.extend(str(row.get("content") or "").splitlines() or [""])
        lines.append("")
        return lines

    def _message_layout_entries(self, df: pd.DataFrame) -> List[Dict[str, object]]:
        offsets: Dict[str, int] = {}
        entries: List[Dict[str, object]] = []
        for _, row in self._chronological_messages(df).iterrows():
            path = self._message_channel_file(row).virtual_path
            block_lines = self._message_block_lines(row)
            start_line = offsets.get(path, 0) + 1
            end_line = start_line + len(block_lines) - 1
            offsets[path] = end_line
            entries.append(
                {
                    "row": row,
                    "path": path,
                    "start_line": start_line,
                    "end_line": end_line,
                }
            )
        return entries

    def _legacy_session_virtual_path(self, rel_path: str) -> Optional[Tuple[str, str]]:
        normalized = rel_path.replace("\\", "/").lstrip("/")
        if not normalized.startswith("memory/") or not normalized.endswith(".md"):
            return None
        parts = normalized.split("/")
        if len(parts) == 2:
            return ("default", parts[-1].replace(".md", ""))
        if len(parts) == 3:
            return (parts[1], parts[2].replace(".md", ""))
        return None

    @property
    def state(self) -> DataFramesState:
        return self._state

    def _rebuild_projection_state_locked(self, *, vector_dim: int) -> int:
        self._state.projections_df = materialize_projection_state(
            self._state.messages_df,
            vector_dim=vector_dim,
        )
        return int(self._state.projections_df.shape[0])

    def _rebuild_belief_state_locked(self, *, vector_dim: int) -> int:
        self._state.beliefs_df = materialize_l0_beliefs(
            self._state.messages_df,
            vector_dim=vector_dim,
        )
        return int(self._state.beliefs_df.shape[0])

    def _refresh_first_class_state_locked(self, *, vector_dim: int) -> None:
        self._rebuild_projection_state_locked(vector_dim=vector_dim)
        self._rebuild_belief_state_locked(vector_dim=vector_dim)

    def _rebuild_embedding_index_metadata_locked(self) -> int:
        self._state.embedding_index_metadata_df = materialize_embedding_index_metadata(
            self._state.messages_df,
            self._state.session_rollups_df,
            self._state.topics_df,
            self._state.capsules_df,
            self._state.beliefs_df,
            self._state.projections_df,
        )
        return int(self._state.embedding_index_metadata_df.shape[0])

    async def rebuild_embedding_index_metadata(self) -> int:
        async with self._lock:
            return self._rebuild_embedding_index_metadata_locked()

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
                "created_at": pd.Timestamp.now(tz="UTC"),
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

    def _ensure_projection_sessions_locked(
        self,
        *,
        tenant_id: str,
        session_ids: Sequence[str],
        created_at: object,
    ) -> None:
        missing_session_ids = [
            str(session_id)
            for session_id in sorted({str(item) for item in session_ids if str(item)})
            if not self._session_exists_locked(str(tenant_id), str(session_id))
        ]
        if not missing_session_ids:
            return
        created_at_ts = _utc_timestamp(created_at)
        row_df = pd.DataFrame(
            [
                {
                    "tenant_id": str(tenant_id),
                    "session_id": session_id,
                    "parent_session_id": "",
                    "origin": "projection",
                    "created_at": created_at_ts,
                }
                for session_id in missing_session_ids
            ],
            columns=SESSIONS_COLUMNS,
        )
        if self._state.sessions_df.empty:
            self._state.sessions_df = row_df
        else:
            self._state.sessions_df = pd.concat(
                [self._state.sessions_df, row_df],
                ignore_index=True,
            )
        self._invalidate_sessions_index_locked()

    async def add_message(self, payload: Dict[str, object]) -> None:
        tenant_id = str(payload.get("tenant_id") or "default")
        session_id = str(payload.get("session_id") or "default")
        await self.ensure_session(tenant_id=tenant_id, session_id=session_id)
        async with self._lock:
            ts = pd.to_datetime(payload.get("ts"), utc=True, errors="coerce")
            if pd.isna(ts):
                ts = pd.Timestamp.now(tz="UTC")
            is_deleted = bool(payload.get("is_deleted") or payload.get("deleted_at"))
            message_id = str(payload.get("message_id") or "")
            origin_message_id = str(payload.get("origin_message_id") or message_id)
            projection_kind = str(payload.get("projection_kind") or "raw")
            projection_scope = str(payload.get("projection_scope") or "").strip()
            if not projection_scope:
                projection_scope = self._default_projection_scope_from_payload(payload)
            raw_topic_confidence = payload.get("topic_confidence")
            row = {
                "message_id": message_id,
                "origin_message_id": origin_message_id,
                "tenant_id": tenant_id,
                "session_id": session_id,
                "role": str(payload.get("role") or "user"),
                "content": str(payload.get("content") or ""),
                "ts": ts,
                "channel": str(payload.get("channel") or ""),
                "chat_type": str(payload.get("chat_type") or ""),
                "account_id": str(payload.get("account_id") or ""),
                "account_key": str(payload.get("account_key") or ""),
                "from_id": str(payload.get("from_id") or ""),
                "from_user_key": str(payload.get("from_user_key") or ""),
                "to_id": str(payload.get("to_id") or ""),
                "to_user_key": str(payload.get("to_user_key") or ""),
                "projection_target_user_key": str(payload.get("projection_target_user_key") or ""),
                "sender_id": str(payload.get("sender_id") or ""),
                "sender_user_key": str(payload.get("sender_user_key") or ""),
                "sender_name": str(payload.get("sender_name") or ""),
                "sender_username": str(payload.get("sender_username") or ""),
                "sender_e164": str(payload.get("sender_e164") or ""),
                "group_id": str(payload.get("group_id") or ""),
                "group_chat_key": str(payload.get("group_chat_key") or ""),
                "group_subject": str(payload.get("group_subject") or ""),
                "group_channel": str(payload.get("group_channel") or ""),
                "group_space": str(payload.get("group_space") or ""),
                "native_channel_id": str(payload.get("native_channel_id") or ""),
                "message_thread_id": str(payload.get("message_thread_id") or ""),
                "thread_parent_id": str(payload.get("thread_parent_id") or ""),
                "reply_to_id": str(payload.get("reply_to_id") or ""),
                "topic_id": str(payload.get("topic_id") or "default"),
                "source_topic_id": str(payload.get("source_topic_id") or payload.get("topic_id") or "default"),
                "topic_parent_id": str(payload.get("topic_parent_id") or ""),
                "topic_path": str(payload.get("topic_path") or payload.get("topic_id") or "default"),
                "source_topic_path": str(
                    payload.get("source_topic_path")
                    or payload.get("topic_path")
                    or payload.get("topic_id")
                    or "default"
                ),
                "topic_confidence": float(raw_topic_confidence) if raw_topic_confidence is not None else 1.0,
                "topic_source": str(payload.get("topic_source") or "explicit"),
                "embedding_ref": str(payload.get("embedding_ref") or ""),
                "capsule_level": str(payload.get("capsule_level") or "L0"),
                "idempotency_key": str(payload.get("idempotency_key") or ""),
                "projection_kind": projection_kind,
                "projection_scope": projection_scope,
                "visibility": str(payload.get("visibility") or ("raw" if projection_kind in RAW_COMPAT_PROJECTION_KINDS else "")),
                "platform": str(payload.get("platform") or ""),
                "platform_message_id": str(payload.get("platform_message_id") or ""),
                "native_session_id": str(payload.get("native_session_id") or ""),
                "message_state": MESSAGE_STATE_DELETED if is_deleted else str(payload.get("message_state") or MESSAGE_STATE_ACTIVE),
                "updated_at": pd.to_datetime(payload.get("updated_at"), utc=True, errors="coerce")
                if payload.get("updated_at") is not None
                else ts,
                "deleted_at": pd.to_datetime(payload.get("deleted_at"), utc=True, errors="coerce")
                if payload.get("deleted_at") is not None
                else ts if is_deleted else pd.NaT,
            }
            row_df = pd.DataFrame([{**row, "is_deleted": is_deleted}], columns=[*MESSAGES_COLUMNS, "is_deleted"])
            base_messages = self._state.messages_df.reset_index(drop=True).copy()
            if "is_deleted" not in base_messages.columns:
                base_messages["is_deleted"] = (
                    base_messages.get("message_state", pd.Series([""] * len(base_messages)))
                    .astype(str)
                    .eq(MESSAGE_STATE_DELETED)
                )
            if not base_messages.empty:
                existing_mask = (
                    (base_messages["tenant_id"].fillna("default").astype(str) == tenant_id)
                    & (
                        base_messages["origin_message_id"]
                        .fillna(base_messages["message_id"])
                        .astype(str)
                        == origin_message_id
                    )
                    & (base_messages["projection_kind"].fillna("raw").astype(str) == projection_kind)
                    & (base_messages["projection_scope"].fillna("").astype(str) == projection_scope)
                )
                if existing_mask.any():
                    base_messages = base_messages.loc[~existing_mask].reset_index(drop=True)
            merged = (
                row_df
                if base_messages.empty
                else pd.concat([base_messages, row_df], ignore_index=True)
            )
            self._state.messages_df = merged
            self._invalidate_messages_index_locked()

    async def apply_message_bundle(
        self,
        bundle: Dict[str, object],
        *,
        vector_dim: int = DEFAULT_TOPIC_VECTOR_DIM,
    ) -> MessageUpsertResult:
        tenant_id = str(bundle.get("tenant_id") or "default")
        projections = list(bundle.get("projections") or [])
        for row in projections:
            session_id = str(row.get("session_id") or "")
            if session_id:
                await self.ensure_session(tenant_id=tenant_id, session_id=session_id)
        async with self._lock:
            origin_message_id = str(bundle.get("origin_message_id") or "")
            existing = self._state.messages_df[
                (self._state.messages_df["tenant_id"].astype(str) == tenant_id)
                & (self._state.messages_df["origin_message_id"].astype(str) == origin_message_id)
            ]
            replaced_existing = not existing.empty
            preserved_projection_messages = pd.DataFrame(columns=MESSAGES_COLUMNS)
            if replaced_existing:
                raw_row_df = pd.DataFrame([dict(bundle["raw_message"])], columns=MESSAGES_COLUMNS)
                canonical_projection_df = pd.DataFrame(projections, columns=MESSAGES_COLUMNS)
                preserved_projection_messages = _preserved_projection_messages(
                    existing,
                    raw_row_df,
                    canonical_projection_df,
                )
                for column in ["ts", "updated_at", "deleted_at"]:
                    preserved_projection_messages[column] = preserved_projection_messages[column].apply(
                        _iso_timestamp_or_none
                    )
            if replaced_existing:
                self._state.messages_df = self._state.messages_df[
                    ~(
                        (self._state.messages_df["tenant_id"].astype(str) == tenant_id)
                        & (self._state.messages_df["origin_message_id"].astype(str) == origin_message_id)
                    )
                ]
            rows = [dict(bundle["raw_message"]), *[dict(item) for item in projections]]
            if not preserved_projection_messages.empty:
                rows.extend(preserved_projection_messages.to_dict("records"))
            row_df = pd.DataFrame(rows, columns=MESSAGES_COLUMNS)
            if self._state.messages_df.empty:
                self._state.messages_df = row_df
            else:
                self._state.messages_df = pd.concat(
                    [self._state.messages_df, row_df],
                    ignore_index=True,
                )
            projection_session_ids = [str(item.get("session_id") or "") for item in projections]
            if not preserved_projection_messages.empty:
                projection_session_ids.extend(
                    preserved_projection_messages["session_id"].astype(str).tolist()
                )
            self._ensure_projection_sessions_locked(
                tenant_id=tenant_id,
                session_ids=projection_session_ids,
                created_at=bundle["raw_message"].get("ts"),
            )
            self._invalidate_messages_index_locked()
            self._refresh_first_class_state_locked(vector_dim=vector_dim)
            self._rebuild_embedding_index_metadata_locked()
            affected_sessions = sorted(
                {
                    str(row.get("session_id") or "")
                    for row in row_df.to_dict("records")
                    if str(row.get("projection_kind") or "") != RAW_PROJECTION_KIND
                    and str(row.get("session_id") or "")
                }
            )
            return MessageUpsertResult(
                origin_message_id=origin_message_id,
                affected_sessions=affected_sessions,
                affected_projections=int(
                    row_df[row_df["projection_kind"].astype(str) != RAW_PROJECTION_KIND].shape[0]
                ),
                replaced_existing=replaced_existing,
            )

    def _resolve_origin_message_id_locked(
        self,
        *,
        tenant_id: str,
        origin_message_id: Optional[str] = None,
        platform: Optional[str] = None,
        account_id: Optional[str] = None,
        platform_message_id: Optional[str] = None,
    ) -> Optional[str]:
        explicit = str(origin_message_id or "").strip()
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
        resolved_platform = None
        if platform is not None and str(platform).strip():
            resolved_platform = normalize_platform(str(platform))
        df = self._state.messages_df
        mask = (
            (df["tenant_id"].astype(str) == str(tenant_id))
            & (df["projection_kind"].astype(str) == RAW_PROJECTION_KIND)
            & (df["platform_message_id"].astype(str) == platform_message_id_text)
        )
        if resolved_platform is not None:
            mask &= df["platform"].astype(str) == str(resolved_platform)
        if account_id is not None and str(account_id).strip():
            account_id_text = str(account_id).strip()
            account_candidates = {account_id_text}
            if resolved_platform is not None:
                account_candidates.add(
                    normalize_identity(str(resolved_platform), account_id_text, "account")
                )
            if "account_key" not in df.columns:
                df = df.copy()
                df["account_key"] = ""
            mask &= (
                df["account_id"].astype(str).isin(account_candidates)
                | df["account_key"].astype(str).isin(account_candidates)
            )
        matches = df[mask]["origin_message_id"].astype(str).dropna().unique().tolist()
        if not matches:
            return None
        if len(matches) > 1:
            raise ValueError(
                f"platform_message_id resolved to multiple origin_message_id values: {platform_message_id_text}"
            )
        return str(matches[0])

    async def resolve_origin_message_id(
        self,
        *,
        tenant_id: str,
        origin_message_id: Optional[str] = None,
        platform: Optional[str] = None,
        account_id: Optional[str] = None,
        platform_message_id: Optional[str] = None,
    ) -> Optional[str]:
        async with self._lock:
            return self._resolve_origin_message_id_locked(
                tenant_id=tenant_id,
                origin_message_id=origin_message_id,
                platform=platform,
                account_id=account_id,
                platform_message_id=platform_message_id,
            )

    async def resolve_message_topic(
        self,
        *,
        tenant_id: str,
        reference_message_id: Optional[str],
        platform: Optional[str] = None,
        account_id: Optional[str] = None,
    ) -> Optional[Dict[str, str]]:
        reference_text = str(reference_message_id or "").strip()
        if not reference_text:
            return None
        async with self._lock:
            resolved_origin = self._resolve_origin_message_id_locked(
                tenant_id=tenant_id,
                origin_message_id=reference_text,
            )
            if resolved_origin is None:
                resolved_origin = self._resolve_origin_message_id_locked(
                    tenant_id=tenant_id,
                    platform=platform,
                    account_id=account_id,
                    platform_message_id=reference_text,
                )
            if resolved_origin is None:
                return None
            df = self._state.messages_df
            scoped = df[
                (df["tenant_id"].astype(str) == str(tenant_id))
                & (df["projection_kind"].astype(str) == RAW_PROJECTION_KIND)
                & (df["origin_message_id"].astype(str) == str(resolved_origin))
                & (df["message_state"].astype(str) != MESSAGE_STATE_DELETED)
            ].copy()
            if scoped.empty:
                return None
            scoped["updated_at"] = pd.to_datetime(scoped["updated_at"], utc=True, errors="coerce")
            scoped["ts"] = pd.to_datetime(scoped["ts"], utc=True, errors="coerce")
            scoped = scoped.sort_values(
                ["updated_at", "ts", "message_id"],
                ascending=[False, False, False],
                kind="stable",
            )
            row = scoped.iloc[0]
            return {
                "origin_message_id": str(resolved_origin),
                "topic_id": str(row.get("source_topic_id") or row.get("topic_id") or ""),
                "topic_path": str(row.get("source_topic_path") or row.get("topic_path") or ""),
                "topic_source": str(row.get("topic_source") or ""),
            }

    async def infer_projection_target_user_key(
        self,
        *,
        tenant_id: str,
        platform: Optional[str] = None,
        account_id: Optional[str] = None,
        chat_type: Optional[str] = None,
        session_id: Optional[str] = None,
        group_id: Optional[str] = None,
        reply_to_id: Optional[str] = None,
        thread_parent_id: Optional[str] = None,
        message_thread_id: Optional[str] = None,
    ) -> Optional[str]:
        resolved_platform = normalize_platform(str(platform)) if str(platform or "").strip() else None
        resolved_chat_type = str(chat_type or "").strip().lower()
        async with self._lock:
            df = self._state.messages_df
            if df.empty:
                return None
            scoped = df[
                (df["tenant_id"].astype(str) == str(tenant_id))
                & (df["projection_kind"].astype(str) == RAW_PROJECTION_KIND)
                & (df["message_state"].astype(str) != MESSAGE_STATE_DELETED)
            ].copy()
            if scoped.empty:
                return None
            if resolved_platform is not None:
                scoped = scoped[scoped["platform"].astype(str) == str(resolved_platform)]
                if scoped.empty:
                    return None
            account_text = str(account_id or "").strip()
            if account_text:
                account_candidates = {account_text}
                if resolved_platform is not None:
                    try:
                        account_candidates.add(
                            normalize_identity(str(resolved_platform), account_text, "account")
                        )
                    except ValueError:
                        pass
                scoped = scoped[
                    scoped["account_id"].astype(str).isin(account_candidates)
                    | scoped["account_key"].astype(str).isin(account_candidates)
                ]
                if scoped.empty:
                    return None
            group_text = str(group_id or "").strip()
            if group_text:
                group_candidates = {group_text}
                if resolved_platform is not None:
                    try:
                        group_candidates.add(
                            normalize_identity(str(resolved_platform), group_text, "chat")
                        )
                    except ValueError:
                        pass
                scoped = scoped[
                    scoped["group_id"].astype(str).isin(group_candidates)
                    | scoped["group_chat_key"].astype(str).isin(group_candidates)
                ]
                if scoped.empty:
                    return None
            if resolved_chat_type:
                scoped = scoped[scoped["chat_type"].astype(str) == resolved_chat_type]
                if scoped.empty:
                    return None

            reference_frames: List[pd.DataFrame] = []
            for reference_id in [reply_to_id, thread_parent_id]:
                reference_text = str(reference_id or "").strip()
                if not reference_text:
                    continue
                matched = scoped[
                    (scoped["origin_message_id"].astype(str) == reference_text)
                    | (scoped["platform_message_id"].astype(str) == reference_text)
                ]
                if not matched.empty:
                    reference_frames.append(matched)

            message_thread_text = str(message_thread_id or "").strip()
            if message_thread_text:
                matched = scoped[scoped["message_thread_id"].astype(str) == message_thread_text]
                if not matched.empty:
                    reference_frames.append(matched)

            session_text = str(session_id or "").strip()
            if session_text and resolved_chat_type == "direct":
                matched = scoped[scoped["native_session_id"].astype(str) == session_text]
                if not matched.empty:
                    reference_frames.append(matched)

            for frame in reference_frames:
                user_key = _infer_projection_target_user_key(frame)
                if user_key:
                    return user_key
            return None

    async def edit_message(
        self,
        *,
        tenant_id: str,
        origin_message_id: str,
        content: str,
        edited_at: datetime,
        vector_dim: int = DEFAULT_TOPIC_VECTOR_DIM,
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
            updated_embedding_ref = deterministic_embedding_ref("raw_message", content)
            self._state.messages_df.loc[mask, "content"] = str(content)
            self._state.messages_df.loc[mask, "embedding_ref"] = updated_embedding_ref
            self._state.messages_df.loc[mask, "updated_at"] = edited_at_text
            self._state.messages_df.loc[mask, "message_state"] = "edited"
            self._state.messages_df.loc[mask, "deleted_at"] = None
            self._invalidate_messages_index_locked()
            self._refresh_first_class_state_locked(vector_dim=vector_dim)
            self._rebuild_embedding_index_metadata_locked()
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
        vector_dim: int = DEFAULT_TOPIC_VECTOR_DIM,
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
            deleted_embedding_ref = deterministic_embedding_ref("raw_message", "")
            self._state.messages_df.loc[mask, "content"] = ""
            self._state.messages_df.loc[mask, "embedding_ref"] = deleted_embedding_ref
            self._state.messages_df.loc[mask, "message_state"] = MESSAGE_STATE_DELETED
            deleted_at_text = pd.to_datetime(deleted_at, utc=True).isoformat()
            self._state.messages_df.loc[mask, "updated_at"] = deleted_at_text
            self._state.messages_df.loc[mask, "deleted_at"] = deleted_at_text
            self._invalidate_messages_index_locked()
            self._refresh_first_class_state_locked(vector_dim=vector_dim)
            self._rebuild_embedding_index_metadata_locked()
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
            self._rebuild_embedding_index_metadata_locked()
            return int(self._state.session_rollups_df.shape[0])

    async def rebuild_all_topics(self, *, vector_dim: int = DEFAULT_TOPIC_VECTOR_DIM) -> int:
        async with self._lock:
            self._state.topics_df = materialize_topic_lifecycle(
                self._state.messages_df,
                vector_dim=vector_dim,
            )
            self._apply_materialized_topics_to_messages_locked()
            self._refresh_first_class_state_locked(vector_dim=vector_dim)
            self._rebuild_embedding_index_metadata_locked()
            return int(self._state.topics_df.shape[0])

    async def rebuild_all_capsules(self, *, vector_dim: int = DEFAULT_TOPIC_VECTOR_DIM) -> int:
        async with self._lock:
            self._state.capsules_df = materialize_capsule_lifecycle(
                self._state.messages_df,
                topics_frame=self._state.topics_df,
                vector_dim=vector_dim,
            )
            self._invalidate_capsules_index_locked()
            self._rebuild_embedding_index_metadata_locked()
            return int(self._state.capsules_df.shape[0])

    async def rebuild_storage_from_authoritative_raw(
        self,
        *,
        vector_dim: int = DEFAULT_TOPIC_VECTOR_DIM,
    ) -> StorageRebuildResult:
        async with self._lock:
            rebuilt = rebuild_materialized_storage_from_raw(
                self._state.messages_df,
                self._state.sessions_df,
                vector_dim=vector_dim,
            )
            self._state.messages_df = rebuilt["messages"]
            self._state.projections_df = rebuilt["projections"]
            self._state.beliefs_df = rebuilt["beliefs"]
            self._state.sessions_df = rebuilt["sessions"]
            self._state.session_rollups_df = rebuilt["session_rollups"]
            self._state.topics_df = rebuilt["topics"]
            self._apply_materialized_topics_to_messages_locked()
            self._state.capsules_df = rebuilt["capsules"]
            self._refresh_first_class_state_locked(vector_dim=vector_dim)
            self._state.embedding_index_metadata_df = rebuilt["embedding_index_metadata"]
            self._rebuild_embedding_index_metadata_locked()
            self._rebuild_search_indexes_locked(vector_dim=vector_dim)
            self._invalidate_all_indexes_locked()
            return StorageRebuildResult(
                raw_message_count=int(authoritative_raw_messages(self._state.messages_df).shape[0]),
                projection_message_count=int(rebuilt["projection_messages"].shape[0]),
                projection_state_count=int(self._state.projections_df.shape[0]),
                session_count=int(self._state.sessions_df.shape[0]),
                session_rollup_count=int(self._state.session_rollups_df.shape[0]),
                topic_count=int(self._state.topics_df.shape[0]),
                capsule_count=int(self._state.capsules_df.shape[0]),
                belief_count=int(self._state.beliefs_df.shape[0]),
                embedding_metadata_count=int(self._state.embedding_index_metadata_df.shape[0]),
            )

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
                self._rebuild_embedding_index_metadata_locked()
                return 0
            self._state.session_rollups_df = pd.concat(
                [self._state.session_rollups_df, materialized[SESSION_ROLLUPS_COLUMNS]],
                ignore_index=True,
            )
            self._invalidate_session_rollups_index_locked()
            self._rebuild_embedding_index_metadata_locked()
            return int(materialized.shape[0])

    async def refresh_capsules(
        self,
        tenant_id: str,
        session_id: str,
        *,
        vector_dim: int = DEFAULT_TOPIC_VECTOR_DIM,
    ) -> int:
        async with self._lock:
            impacted_topic_ids = self._topic_ids_for_session_locked(tenant_id, str(session_id), include_deleted=True)
            if impacted_topic_ids:
                self._state.capsules_df = self._state.capsules_df[
                    ~(
                        (self._state.capsules_df["tenant_id"].astype(str) == str(tenant_id))
                        & (self._state.capsules_df["topic_id"].astype(str).isin(impacted_topic_ids))
                    )
                ]
            self._invalidate_capsules_index_locked()
            if not impacted_topic_ids:
                self._rebuild_embedding_index_metadata_locked()
                return 0
            tenant_messages = self._messages_for_query_locked(
                tenant_id=tenant_id,
                session_id=None,
                row_mode="raw",
                include_deleted=True,
            )
            materialized = materialize_capsule_lifecycle(
                tenant_messages,
                topics_frame=self._state.topics_df[
                    self._state.topics_df["tenant_id"].astype(str) == str(tenant_id)
                ],
                vector_dim=vector_dim,
            )
            if materialized.empty:
                self._rebuild_embedding_index_metadata_locked()
                return 0
            materialized = materialized[
                materialized["topic_id"].astype(str).isin(impacted_topic_ids)
            ].reset_index(drop=True)
            if materialized.empty:
                self._rebuild_embedding_index_metadata_locked()
                return 0
            self._state.capsules_df = pd.concat(
                [self._state.capsules_df, materialized[CAPSULES_COLUMNS]],
                ignore_index=True,
            )
            self._invalidate_capsules_index_locked()
            self._rebuild_embedding_index_metadata_locked()
            return int(materialized.shape[0])

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
                "created_at": pd.Timestamp.now(tz="UTC"),
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
            self._refresh_first_class_state_locked(vector_dim=DEFAULT_TOPIC_VECTOR_DIM)
            self._rebuild_embedding_index_metadata_locked()

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

    def _normalize_semantic_jobs_locked(self) -> pd.DataFrame:
        df = self._state.semantic_jobs_df
        if df.empty:
            self._state.semantic_jobs_df = pd.DataFrame(columns=SEMANTIC_JOBS_COLUMNS)
            return self._state.semantic_jobs_df
        normalized = df.copy().reset_index(drop=True)
        for column in SEMANTIC_JOBS_COLUMNS:
            if column not in normalized.columns:
                normalized[column] = None
        normalized = normalized[SEMANTIC_JOBS_COLUMNS]
        for column in [
            "job_id",
            "tenant_id",
            "status",
            "impacted_sessions_json",
            "claimed_sessions_json",
            "cause",
            "last_error",
            "lease_owner",
        ]:
            normalized[column] = normalized[column].fillna("").astype(str)
        for column in ["latest_wal_seq", "claimed_wal_seq", "attempt_count"]:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce").fillna(0).astype(int)
        for column in [
            "enqueued_at",
            "started_at",
            "updated_at",
            "lease_expires_at",
            "available_at",
        ]:
            normalized[column] = (
                pd.to_datetime(normalized[column], utc=True, errors="coerce")
                .astype("datetime64[ns, UTC]")
            )
        self._state.semantic_jobs_df = normalized
        return self._state.semantic_jobs_df

    def _semantic_job_stats_locked(self, jobs: pd.DataFrame) -> SemanticJobStats:
        if jobs.empty:
            return SemanticJobStats(pending=0, running=0, total=0, max_wal_seq=0)
        pending = int((jobs["status"].astype(str) == SEMANTIC_JOB_STATUS_PENDING).sum())
        running = int((jobs["status"].astype(str) == SEMANTIC_JOB_STATUS_RUNNING).sum())
        max_wal_seq = int(pd.to_numeric(jobs["latest_wal_seq"], errors="coerce").fillna(0).max())
        return SemanticJobStats(
            pending=pending,
            running=running,
            total=int(jobs.shape[0]),
            max_wal_seq=max_wal_seq,
        )

    async def clear_semantic_jobs(self) -> None:
        async with self._lock:
            self._state.semantic_jobs_df = pd.DataFrame(columns=SEMANTIC_JOBS_COLUMNS)

    async def recover_semantic_jobs_for_startup(self) -> SemanticJobStats:
        async with self._lock:
            jobs = self._normalize_semantic_jobs_locked()
            if jobs.empty:
                return SemanticJobStats(pending=0, running=0, total=0, max_wal_seq=0)
            now = pd.Timestamp.now(tz="UTC")
            for row_id in jobs.index:
                status = str(jobs.at[row_id, "status"] or SEMANTIC_JOB_STATUS_PENDING)
                impacted_sessions = _json_string_list(jobs.at[row_id, "impacted_sessions_json"])
                claimed_sessions = _json_string_list(jobs.at[row_id, "claimed_sessions_json"])
                merged_sessions = sorted({*impacted_sessions, *claimed_sessions})
                self._state.semantic_jobs_df.at[row_id, "impacted_sessions_json"] = json.dumps(
                    merged_sessions,
                    separators=(",", ":"),
                )
                self._state.semantic_jobs_df.at[row_id, "claimed_sessions_json"] = "[]"
                self._state.semantic_jobs_df.at[row_id, "lease_owner"] = ""
                self._state.semantic_jobs_df.at[row_id, "lease_expires_at"] = pd.NaT
                if status != SEMANTIC_JOB_STATUS_PENDING:
                    self._state.semantic_jobs_df.at[row_id, "status"] = SEMANTIC_JOB_STATUS_PENDING
                    self._state.semantic_jobs_df.at[row_id, "available_at"] = now
                    self._state.semantic_jobs_df.at[row_id, "updated_at"] = now
                elif pd.isna(jobs.at[row_id, "available_at"]):
                    self._state.semantic_jobs_df.at[row_id, "available_at"] = now
            jobs = self._normalize_semantic_jobs_locked()
            return self._semantic_job_stats_locked(jobs)

    async def enqueue_semantic_refresh(
        self,
        *,
        tenant_id: str,
        wal_seq: int,
        session_ids: Sequence[str],
        cause: str,
    ) -> SemanticJobStats:
        normalized_sessions = sorted({str(item) for item in session_ids if str(item)})
        now = pd.Timestamp.now(tz="UTC")
        job_id = f"semantic:{tenant_id}"
        async with self._lock:
            jobs = self._normalize_semantic_jobs_locked()
            if jobs.empty:
                row_id = None
            else:
                matches = jobs.index[jobs["job_id"].astype(str) == job_id]
                row_id = int(matches[-1]) if len(matches) else None
            if row_id is None:
                row = {
                    "job_id": job_id,
                    "tenant_id": str(tenant_id),
                    "status": SEMANTIC_JOB_STATUS_PENDING,
                    "latest_wal_seq": int(wal_seq),
                    "claimed_wal_seq": 0,
                    "impacted_sessions_json": json.dumps(normalized_sessions, separators=(",", ":")),
                    "claimed_sessions_json": "[]",
                    "cause": str(cause or ""),
                    "attempt_count": 0,
                    "last_error": "",
                    "enqueued_at": now,
                    "started_at": pd.NaT,
                    "updated_at": now,
                    "lease_owner": "",
                    "lease_expires_at": pd.NaT,
                    "available_at": now,
                }
                if jobs.empty:
                    self._state.semantic_jobs_df = pd.DataFrame([row], columns=SEMANTIC_JOBS_COLUMNS)
                else:
                    self._state.semantic_jobs_df = pd.concat(
                        [jobs, pd.DataFrame([row], columns=SEMANTIC_JOBS_COLUMNS)],
                        ignore_index=True,
                    )
            else:
                existing_sessions = _json_string_list(jobs.at[row_id, "impacted_sessions_json"])
                merged_sessions = sorted({*existing_sessions, *normalized_sessions})
                latest_wal_seq = max(int(jobs.at[row_id, "latest_wal_seq"] or 0), int(wal_seq))
                status = str(jobs.at[row_id, "status"] or SEMANTIC_JOB_STATUS_PENDING)
                if status != SEMANTIC_JOB_STATUS_RUNNING:
                    self._state.semantic_jobs_df.at[row_id, "status"] = SEMANTIC_JOB_STATUS_PENDING
                    self._state.semantic_jobs_df.at[row_id, "available_at"] = now
                    self._state.semantic_jobs_df.at[row_id, "lease_owner"] = ""
                    self._state.semantic_jobs_df.at[row_id, "lease_expires_at"] = pd.NaT
                self._state.semantic_jobs_df.at[row_id, "latest_wal_seq"] = latest_wal_seq
                self._state.semantic_jobs_df.at[row_id, "impacted_sessions_json"] = json.dumps(
                    merged_sessions,
                    separators=(",", ":"),
                )
                self._state.semantic_jobs_df.at[row_id, "cause"] = str(cause or "")
                self._state.semantic_jobs_df.at[row_id, "last_error"] = ""
                self._state.semantic_jobs_df.at[row_id, "updated_at"] = now
            jobs = self._normalize_semantic_jobs_locked()
            return self._semantic_job_stats_locked(jobs)

    async def semantic_job_stats(self) -> SemanticJobStats:
        async with self._lock:
            jobs = self._normalize_semantic_jobs_locked()
            return self._semantic_job_stats_locked(jobs)

    async def has_pending_semantic_refresh(
        self,
        *,
        tenant_id: str,
        session_id: Optional[str] = None,
    ) -> bool:
        async with self._lock:
            jobs = self._normalize_semantic_jobs_locked()
            if jobs.empty:
                return False
            jobs = jobs[
                jobs["status"].astype(str).isin(
                    [SEMANTIC_JOB_STATUS_PENDING, SEMANTIC_JOB_STATUS_RUNNING]
                )
            ]
            if jobs.empty:
                return False
            tenant_jobs = jobs[
                (jobs["tenant_id"].astype(str) == str(tenant_id))
                | (jobs["tenant_id"].astype(str) == "*")
            ]
            if tenant_jobs.empty:
                return False
            if session_id is None:
                return True
            session_text = str(session_id).strip()
            if not session_text:
                return True
            session_candidates = {
                session_text,
                *self._resolve_session_ids_locked(str(tenant_id), session_text, include_mirrors=True),
            }
            for _, row in tenant_jobs.iterrows():
                impacted_sessions = {
                    *_json_string_list(row.get("impacted_sessions_json")),
                    *_json_string_list(row.get("claimed_sessions_json")),
                }
                if not impacted_sessions or impacted_sessions.intersection(session_candidates):
                    return True
            return False

    async def claim_next_semantic_job(
        self,
        *,
        worker_id: str,
        lease_seconds: float,
    ) -> Optional[SemanticJobClaim]:
        async with self._lock:
            jobs = self._normalize_semantic_jobs_locked()
            if jobs.empty:
                return None
            now = pd.Timestamp.now(tz="UTC")
            available_at = pd.to_datetime(jobs["available_at"], utc=True, errors="coerce")
            lease_expires = pd.to_datetime(jobs["lease_expires_at"], utc=True, errors="coerce")
            pending_mask = (
                (jobs["status"].astype(str) == SEMANTIC_JOB_STATUS_PENDING)
                & (available_at.isna() | (available_at <= now))
            )
            expired_running_mask = (
                (jobs["status"].astype(str) == SEMANTIC_JOB_STATUS_RUNNING)
                & lease_expires.notna()
                & (lease_expires <= now)
            )
            candidates = jobs[pending_mask | expired_running_mask].copy()
            if candidates.empty:
                return None
            candidates["sort_seq"] = pd.to_numeric(
                candidates["latest_wal_seq"], errors="coerce"
            ).fillna(0).astype(int)
            candidates["sort_time"] = pd.to_datetime(
                candidates["updated_at"], utc=True, errors="coerce"
            ).fillna(now)
            row_id = int(
                candidates.sort_values(
                    ["sort_seq", "sort_time", "tenant_id"],
                    ascending=[True, True, True],
                    kind="stable",
                ).index[0]
            )
            claimed_wal_seq = int(jobs.at[row_id, "latest_wal_seq"] or 0)
            impacted_sessions = _json_string_list(jobs.at[row_id, "impacted_sessions_json"])
            self._state.semantic_jobs_df.at[row_id, "status"] = SEMANTIC_JOB_STATUS_RUNNING
            self._state.semantic_jobs_df.at[row_id, "claimed_wal_seq"] = claimed_wal_seq
            self._state.semantic_jobs_df.at[row_id, "claimed_sessions_json"] = json.dumps(
                impacted_sessions,
                separators=(",", ":"),
            )
            self._state.semantic_jobs_df.at[row_id, "attempt_count"] = (
                int(jobs.at[row_id, "attempt_count"] or 0) + 1
            )
            self._state.semantic_jobs_df.at[row_id, "started_at"] = now
            self._state.semantic_jobs_df.at[row_id, "updated_at"] = now
            self._state.semantic_jobs_df.at[row_id, "lease_owner"] = str(worker_id)
            self._state.semantic_jobs_df.at[row_id, "lease_expires_at"] = now + pd.Timedelta(
                seconds=max(1.0, float(lease_seconds))
            )
            self._state.semantic_jobs_df.at[row_id, "last_error"] = ""
            return SemanticJobClaim(
                job_id=str(jobs.at[row_id, "job_id"] or ""),
                tenant_id=str(jobs.at[row_id, "tenant_id"] or "default"),
                claimed_wal_seq=claimed_wal_seq,
                latest_wal_seq=claimed_wal_seq,
                impacted_sessions=impacted_sessions,
                attempt_count=int(self._state.semantic_jobs_df.at[row_id, "attempt_count"] or 0),
                cause=str(jobs.at[row_id, "cause"] or ""),
            )

    async def complete_semantic_job(
        self,
        *,
        job_id: str,
        worker_id: str,
        claimed_wal_seq: int,
        retry_delay_seconds: float,
        error: Optional[str] = None,
    ) -> SemanticJobStats:
        async with self._lock:
            def _stats(frame: pd.DataFrame) -> SemanticJobStats:
                if frame.empty:
                    return SemanticJobStats(pending=0, running=0, total=0, max_wal_seq=0)
                pending = int((frame["status"].astype(str) == SEMANTIC_JOB_STATUS_PENDING).sum())
                running = int((frame["status"].astype(str) == SEMANTIC_JOB_STATUS_RUNNING).sum())
                max_wal_seq = int(
                    pd.to_numeric(frame["latest_wal_seq"], errors="coerce").fillna(0).max()
                )
                return SemanticJobStats(
                    pending=pending,
                    running=running,
                    total=int(frame.shape[0]),
                    max_wal_seq=max_wal_seq,
                )

            jobs = self._normalize_semantic_jobs_locked()
            if jobs.empty:
                return _stats(jobs)
            matches = jobs.index[jobs["job_id"].astype(str) == str(job_id)]
            if not len(matches):
                return _stats(jobs)
            row_id = int(matches[-1])
            now = pd.Timestamp.now(tz="UTC")
            owner = str(jobs.at[row_id, "lease_owner"] or "")
            if owner and owner != str(worker_id):
                return _stats(jobs)
            latest_wal_seq = int(jobs.at[row_id, "latest_wal_seq"] or 0)
            if error:
                self._state.semantic_jobs_df.at[row_id, "status"] = SEMANTIC_JOB_STATUS_PENDING
                self._state.semantic_jobs_df.at[row_id, "last_error"] = str(error)[:1000]
                self._state.semantic_jobs_df.at[row_id, "updated_at"] = now
                self._state.semantic_jobs_df.at[row_id, "lease_owner"] = ""
                self._state.semantic_jobs_df.at[row_id, "lease_expires_at"] = pd.NaT
                self._state.semantic_jobs_df.at[row_id, "available_at"] = now + pd.Timedelta(
                    seconds=max(0.0, float(retry_delay_seconds))
                )
                self._state.semantic_jobs_df.at[row_id, "claimed_sessions_json"] = "[]"
            elif latest_wal_seq > int(claimed_wal_seq):
                self._state.semantic_jobs_df.at[row_id, "status"] = SEMANTIC_JOB_STATUS_PENDING
                self._state.semantic_jobs_df.at[row_id, "updated_at"] = now
                self._state.semantic_jobs_df.at[row_id, "lease_owner"] = ""
                self._state.semantic_jobs_df.at[row_id, "lease_expires_at"] = pd.NaT
                self._state.semantic_jobs_df.at[row_id, "available_at"] = now
                self._state.semantic_jobs_df.at[row_id, "claimed_sessions_json"] = "[]"
                self._state.semantic_jobs_df.at[row_id, "last_error"] = ""
            else:
                self._state.semantic_jobs_df = jobs[
                    jobs["job_id"].astype(str) != str(job_id)
                ].reset_index(drop=True)
            jobs = self._normalize_semantic_jobs_locked()
            return _stats(jobs)

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
            now = pd.Timestamp.now(tz="UTC")
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
        projection_kind: Optional[str] = None,
        projection_scope: Optional[str] = None,
        row_mode: Literal["projection", "raw", "all"] = "projection",
    ) -> List[Dict[str, object]]:
        async with self._lock:
            df = self._message_view_rows_locked(
                tenant_id=tenant_id,
                session_id=session_id,
                channel=channel,
                chat_type=chat_type,
                group_id=group_id,
                topic_id=topic_id,
                message_thread_id=message_thread_id,
                row_mode=row_mode,
            )
            if projection_kind is not None:
                df = df[df["projection_kind"].fillna("").astype(str) == str(projection_kind)]
            if projection_scope is not None:
                df = df[df["projection_scope"].fillna("").astype(str) == str(projection_scope)]
            if df.empty:
                return []
            docs: List[Dict[str, object]] = []
            for entry in self._message_layout_entries(df):
                row = entry["row"]
                tid = str(row["tenant_id"])
                sid = str(row["session_id"])
                path = str(entry["path"])
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
                        "line_no": int(entry["start_line"]),
                    }
                )
            return docs

    async def retrieval_documents(
        self,
        *,
        tenant_id: str,
        session_id: Optional[str],
        channel: Optional[str] = None,
        chat_type: Optional[str] = None,
        group_id: Optional[str] = None,
        topic_id: Optional[str] = None,
        message_thread_id: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        async with self._lock:
            projection_rows = self._messages_for_query_locked(
                tenant_id=tenant_id,
                session_id=session_id,
                channel=channel,
                chat_type=chat_type,
                group_id=group_id,
                topic_id=topic_id,
                message_thread_id=message_thread_id,
                row_mode="projection",
                include_deleted=False,
            )
            raw_rows = self._raw_messages_for_query_locked(
                tenant_id=tenant_id,
                session_id=session_id,
                channel=channel,
                chat_type=chat_type,
                group_id=group_id,
                topic_id=topic_id,
                message_thread_id=message_thread_id,
                include_deleted=False,
            )
            rollup_rows = self._session_rollup_rows_for_scope_locked(
                tenant_id=tenant_id,
                session_id=session_id,
                channel=channel,
                chat_type=chat_type,
                group_id=group_id,
                topic_id=topic_id,
                message_thread_id=message_thread_id,
            )
            topic_rows = self._topic_rows_for_query_locked(
                tenant_id=tenant_id,
                session_id=session_id,
                topic_id=topic_id,
            )
            capsule_rows = self._capsule_rows_for_scope_locked(
                tenant_id=tenant_id,
                session_id=session_id,
                topic_id=topic_id,
            )
            if (
                projection_rows.empty
                and raw_rows.empty
                and rollup_rows.empty
                and topic_rows.empty
                and capsule_rows.empty
            ):
                return []

            if self._search_docs_index_dirty or self._search_docs_indexed_df is None:
                search_docs_indexed = self._build_search_docs_index_locked()
            else:
                search_docs_indexed = self._search_docs_indexed_df

            docs: List[Dict[str, object]] = []
            projection_rows = projection_rows.reset_index(drop=True)
            raw_rows = self._chronological_messages(raw_rows)
            rollup_rows = rollup_rows.reset_index(drop=True)
            topic_rows = topic_rows.reset_index(drop=True)
            capsule_rows = capsule_rows.reset_index(drop=True)

            def _persisted_doc(doc_id: str) -> Optional[Dict[str, object]]:
                if search_docs_indexed is None or search_docs_indexed.empty:
                    return None
                try:
                    hit = search_docs_indexed.loc[(str(tenant_id), str(doc_id))]
                except KeyError:
                    return None
                row = hit if isinstance(hit, pd.Series) else hit.iloc[-1]
                try:
                    citations = json.loads(str(row.get("citations_json") or "[]"))
                except json.JSONDecodeError:
                    citations = []
                return {
                    "doc_id": str(row.get("doc_id") or doc_id),
                    "text": str(row.get("text") or ""),
                    "path": str(row.get("path") or ""),
                    "start_line": int(row.get("start_line") or 1),
                    "end_line": int(row.get("end_line") or row.get("start_line") or 1),
                    "snippet": str(row.get("snippet") or "")[:700],
                    "source_tier": str(row.get("source_tier") or "L0"),
                    "entity_type": str(row.get("entity_type") or "raw_message"),
                    "entity_id": str(row.get("entity_id") or row.get("doc_id") or doc_id),
                    "citation": str(row.get("citation") or "") or None,
                    "citations": [str(item) for item in citations if str(item)],
                    "channel": str(row.get("channel") or "") or None,
                    "chat_type": str(row.get("chat_type") or "") or None,
                    "account_id": str(row.get("account_id") or "") or None,
                    "group_id": str(row.get("group_id") or "") or None,
                    "topic_id": str(row.get("topic_id") or "") or None,
                    "topic_path": str(row.get("topic_path") or "") or None,
                    "message_thread_id": str(row.get("message_thread_id") or "") or None,
                    "sender_id": str(row.get("sender_id") or "") or None,
                    "origin_message_id": str(row.get("origin_message_id") or "") or None,
                    "projection_kind": str(row.get("projection_kind") or "") or None,
                    "projection_scope": str(row.get("projection_scope") or "") or None,
                }
            for idx, (_, row) in enumerate(raw_rows.iterrows(), start=1):
                origin_id = str(row.get("origin_message_id") or row["message_id"])
                primary = f"raw:{origin_id}"
                persisted = _persisted_doc(primary)
                if persisted is not None:
                    docs.append(persisted)
                    continue
                citations = [f"origin:{origin_id}"]
                docs.append(
                    {
                        "doc_id": primary,
                        "text": str(row.get("content") or ""),
                        "path": f"memory/raw/{_safe_path_fragment(tenant_id)}/{origin_id}.md",
                        "start_line": idx,
                        "end_line": idx,
                        "snippet": _trim_text(row.get("content") or "", 700),
                        "source_tier": "L0",
                        "entity_type": "raw_message",
                        "entity_id": origin_id,
                        "citation": citations[0],
                        "citations": citations,
                        "channel": str(row.get("channel") or "") or None,
                        "chat_type": str(row.get("chat_type") or "") or None,
                        "account_id": str(row.get("account_id") or "") or None,
                        "group_id": str(row.get("group_id") or "") or None,
                        "topic_id": str(row.get("topic_id") or "default"),
                        "topic_path": str(row.get("topic_path") or row.get("topic_id") or "default"),
                        "message_thread_id": str(row.get("message_thread_id") or "") or None,
                        "sender_id": str(row.get("sender_id") or "") or None,
                        "origin_message_id": origin_id,
                        "projection_kind": str(row.get("projection_kind") or "") or None,
                        "projection_scope": str(row.get("projection_scope") or "") or None,
                    }
                )

            topic_aliases: Dict[str, set[str]] = {}
            if not self._state.topics_df.empty:
                topic_state = self._state.topics_df.copy().reset_index(drop=True)
                topic_state["tenant_id"] = topic_state["tenant_id"].fillna("default").astype(str)
                topic_state["topic_id"] = topic_state["topic_id"].fillna("default").astype(str)
                topic_state["canonical_topic_id"] = topic_state["canonical_topic_id"].fillna(
                    topic_state["topic_id"]
                ).astype(str)
                scoped_topics = topic_state[topic_state["tenant_id"].astype(str) == str(tenant_id)]
                for _, row in scoped_topics.iterrows():
                    canonical_id = str(row["canonical_topic_id"] or row["topic_id"])
                    topic_aliases.setdefault(canonical_id, set()).add(str(row["topic_id"]))
            if "source_topic_id" not in raw_rows.columns:
                raw_rows = raw_rows.copy()
                raw_rows["source_topic_id"] = raw_rows["topic_id"]

            def _origin_bounds(frame: pd.DataFrame) -> List[str]:
                if frame.empty:
                    return []
                ordered = self._chronological_messages(frame)
                first_origin = str(ordered.iloc[0].get("origin_message_id") or ordered.iloc[0]["message_id"])
                last_origin = str(ordered.iloc[-1].get("origin_message_id") or ordered.iloc[-1]["message_id"])
                citations = [f"origin:{first_origin}"]
                if last_origin and last_origin != first_origin:
                    citations.append(f"origin:{last_origin}")
                return citations

            def _scope_id() -> str:
                if session_id:
                    return f"session_{_safe_path_fragment(session_id)}"
                if topic_id:
                    return f"topic_{_safe_path_fragment(topic_id)}"
                if group_id:
                    return f"group_{_safe_path_fragment(group_id)}"
                return f"tenant_{_safe_path_fragment(tenant_id)}"

            if not raw_rows.empty:
                l0_primary = f"l0:{tenant_id}:{_scope_id()}"
                persisted = _persisted_doc(l0_primary)
                if persisted is not None:
                    docs.append(persisted)
                else:
                    l0_header = (
                        f"l0 scope={_scope_id()} raw_messages={int(raw_rows.shape[0])} "
                        f"rollups={int(rollup_rows.shape[0])} topics={int(topic_rows.shape[0])} "
                        f"capsules={int(capsule_rows.shape[0])}"
                    )
                    l0_sections: List[str] = [l0_header]
                    lifetime_rollups = rollup_rows[rollup_rows["window_kind"].astype(str) == "lifetime"]
                    if not lifetime_rollups.empty:
                        latest_lifetimes = lifetime_rollups.sort_values(
                            ["source_last_ts", "session_id"],
                            ascending=[False, True],
                            kind="stable",
                        ).head(3)
                        for _, row in latest_lifetimes.iterrows():
                            l0_sections.append(
                                "session "
                                f"{str(row.get('session_id') or '')} "
                                f"{_trim_text(row.get('summary') or '', 900)}"
                            )
                    if not topic_rows.empty:
                        for _, row in topic_rows.head(3).iterrows():
                            l0_sections.append(
                                "topic "
                                f"{str(row.get('canonical_topic_id') or row.get('topic_id') or 'default')} "
                                f"{_trim_text(row.get('summary') or '', 700)}"
                            )
                    if not capsule_rows.empty:
                        recent_capsules = capsule_rows.sort_values(
                            ["updated_at", "capsule_ordinal"],
                            ascending=[False, False],
                            kind="stable",
                        ).head(2)
                        for _, row in recent_capsules.iterrows():
                            l0_sections.append(
                                "capsule "
                                f"{str(row.get('capsule_id') or '')} "
                                f"{_trim_text(row.get('summary') or '', 700)}"
                            )
                    raw_tail = raw_rows.tail(4)
                    for _, row in raw_tail.iterrows():
                        l0_sections.append(_trim_text(row.get("content") or "", 350))
                    l0_text = _trim_text("\n".join(part for part in l0_sections if part), RETRIEVAL_ABSTRACT_MAX_CHARS)
                    l0_citations = _dedupe_citations([l0_primary, *_origin_bounds(raw_rows)], limit=3)
                    docs.append(
                        {
                            "doc_id": l0_primary,
                            "text": l0_text,
                            "path": f"memory/l0/{_scope_id()}.md",
                            "start_line": 1,
                            "end_line": 1,
                            "snippet": _trim_text(l0_text, 700),
                            "source_tier": "L0",
                            "entity_type": "l0_abstract",
                            "entity_id": l0_primary,
                            "citation": l0_primary,
                            "citations": l0_citations,
                            "channel": None,
                            "chat_type": None,
                            "account_id": None,
                            "group_id": None,
                            "topic_id": None,
                            "topic_path": None,
                            "message_thread_id": None,
                            "sender_id": None,
                            "origin_message_id": (
                                l0_citations[1].split(":", 1)[1] if len(l0_citations) > 1 else None
                            ),
                            "projection_kind": None,
                            "projection_scope": None,
                        }
                    )

            if not rollup_rows.empty:
                ordered_rollups = rollup_rows.sort_values(
                    ["source_last_ts", "window_kind", "window_key"],
                    ascending=[False, True, True],
                    kind="stable",
                )
                projection_rows = self._chronological_messages(projection_rows)
                for _, row in ordered_rollups.iterrows():
                    session_rollup_id = str(row.get("rollup_id") or "")
                    session_key = str(row.get("session_id") or "")
                    bucket_start = pd.to_datetime(row.get("bucket_start"), utc=True, errors="coerce")
                    bucket_end = pd.to_datetime(row.get("bucket_end"), utc=True, errors="coerce")
                    supporting = projection_rows[projection_rows["session_id"].astype(str) == session_key]
                    if pd.notna(bucket_start):
                        supporting = supporting[supporting["ts"] >= bucket_start]
                    if pd.notna(bucket_end):
                        supporting = supporting[supporting["ts"] < bucket_end]
                    primary = session_rollup_id or (
                        "rollup:"
                        f"{tenant_id}:{session_key}:{str(row.get('window_kind') or '')}:{str(row.get('window_key') or '')}"
                    )
                    persisted = _persisted_doc(primary)
                    if persisted is not None:
                        docs.append(persisted)
                        continue
                    citations = _dedupe_citations([primary, *_origin_bounds(supporting)], limit=3)
                    docs.append(
                        {
                            "doc_id": primary,
                            "text": str(row.get("summary") or ""),
                            "path": (
                                "memory/rollups/"
                                f"{_safe_path_fragment(session_key)}/"
                                f"{_safe_path_fragment(row.get('window_kind') or '')}/"
                                f"{_safe_path_fragment(row.get('window_key') or '')}.md"
                            ),
                            "start_line": 1,
                            "end_line": 1,
                            "snippet": _trim_text(row.get("summary") or "", 700),
                            "source_tier": "L1",
                            "entity_type": "session_rollup",
                            "entity_id": primary,
                            "citation": primary,
                            "citations": citations,
                            "channel": None,
                            "chat_type": None,
                            "account_id": None,
                            "group_id": None,
                            "topic_id": None,
                            "topic_path": None,
                            "message_thread_id": None,
                            "sender_id": None,
                            "origin_message_id": (
                                citations[1].split(":", 1)[1] if len(citations) > 1 else None
                            ),
                            "projection_kind": None,
                            "projection_scope": session_key or None,
                        }
                    )

            if not topic_rows.empty:
                ordered_topics = topic_rows.sort_values(
                    ["updated_at", "canonical_topic_id"],
                    ascending=[False, True],
                    kind="stable",
                )
                for _, row in ordered_topics.iterrows():
                    canonical_topic_id = str(row.get("canonical_topic_id") or row.get("topic_id") or "default")
                    primary = f"topic:{canonical_topic_id}"
                    persisted = _persisted_doc(primary)
                    if persisted is not None:
                        docs.append(persisted)
                        continue
                    aliases = topic_aliases.get(canonical_topic_id, {canonical_topic_id})
                    supporting = raw_rows[
                        raw_rows["source_topic_id"].astype(str).isin(sorted(aliases))
                        | (raw_rows["topic_id"].astype(str) == canonical_topic_id)
                    ]
                    citations = _dedupe_citations([primary, *_origin_bounds(supporting)], limit=3)
                    docs.append(
                        {
                            "doc_id": primary,
                            "text": str(row.get("summary") or row.get("vector_text") or ""),
                            "path": f"memory/topics/{_safe_path_fragment(canonical_topic_id)}.md",
                            "start_line": 1,
                            "end_line": 1,
                            "snippet": _trim_text(row.get("summary") or row.get("vector_text") or "", 700),
                            "source_tier": "L2",
                            "entity_type": "topic",
                            "entity_id": canonical_topic_id,
                            "citation": primary,
                            "citations": citations,
                            "channel": None,
                            "chat_type": None,
                            "account_id": None,
                            "group_id": None,
                            "topic_id": canonical_topic_id,
                            "topic_path": str(row.get("topic_path") or canonical_topic_id),
                            "message_thread_id": None,
                            "sender_id": None,
                            "origin_message_id": (
                                citations[1].split(":", 1)[1] if len(citations) > 1 else None
                            ),
                            "projection_kind": None,
                            "projection_scope": None,
                        }
                    )

            if not capsule_rows.empty:
                ordered_capsules = capsule_rows.sort_values(
                    ["updated_at", "capsule_ordinal"],
                    ascending=[False, False],
                    kind="stable",
                )
                for _, row in ordered_capsules.iterrows():
                    capsule_id = str(row.get("capsule_id") or "")
                    first_origin = str(row.get("first_origin_message_id") or "").strip()
                    last_origin = str(row.get("last_origin_message_id") or "").strip()
                    primary = f"capsule:{capsule_id}"
                    persisted = _persisted_doc(primary)
                    if persisted is not None:
                        docs.append(persisted)
                        continue
                    citations = _dedupe_citations(
                        [
                            primary,
                            f"origin:{first_origin}" if first_origin else "",
                            f"origin:{last_origin}" if last_origin else "",
                        ],
                        limit=3,
                    )
                    docs.append(
                        {
                            "doc_id": primary,
                            "text": str(row.get("summary") or ""),
                            "path": (
                                "memory/capsules/"
                                f"{_safe_path_fragment(row.get('topic_id') or 'default')}/"
                                f"{int(row.get('capsule_ordinal') or 0):04d}.md"
                            ),
                            "start_line": 1,
                            "end_line": 1,
                            "snippet": _trim_text(row.get("summary") or "", 700),
                            "source_tier": "L2",
                            "entity_type": "capsule",
                            "entity_id": capsule_id,
                            "citation": primary,
                            "citations": citations,
                            "channel": None,
                            "chat_type": None,
                            "account_id": None,
                            "group_id": None,
                            "topic_id": str(row.get("topic_id") or "default"),
                            "topic_path": str(row.get("topic_path") or row.get("topic_id") or "default"),
                            "message_thread_id": None,
                            "sender_id": None,
                            "origin_message_id": first_origin or None,
                            "projection_kind": None,
                            "projection_scope": None,
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

    async def list_projection_state(
        self,
        *,
        tenant_id: str,
        session_id: Optional[str] = None,
        projection_kind: Optional[str] = None,
        origin_message_id: Optional[str] = None,
        group_id: Optional[str] = None,
        include_deleted: bool = False,
    ) -> List[Dict[str, object]]:
        def _json_contains(text: object, needle: str) -> bool:
            try:
                return str(needle) in {str(item) for item in json.loads(str(text or "[]"))}
            except Exception:
                return False

        async with self._lock:
            df = self._state.projections_df.copy().reset_index(drop=True)
            if df.empty:
                return []
            df["tenant_id"] = df["tenant_id"].fillna("default").astype(str)
            df["session_id"] = df["session_id"].fillna("").astype(str)
            df["projection_kind"] = df["projection_kind"].fillna("").astype(str)
            df["group_id"] = df["group_id"].fillna("").astype(str)
            df["group_chat_key"] = df["group_chat_key"].fillna("").astype(str)
            df["native_session_id"] = df["native_session_id"].fillna("").astype(str)
            df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True, errors="coerce")
            scoped = df[df["tenant_id"].astype(str) == str(tenant_id)].copy()
            if session_id is not None:
                resolved = self._resolve_session_ids_locked(str(tenant_id), str(session_id), include_mirrors=True)
                session_candidates = sorted({str(session_id), *resolved})
                scoped = scoped[
                    scoped["session_id"].astype(str).isin(session_candidates)
                    | scoped["native_session_id"].astype(str).isin(session_candidates)
                    | scoped["native_session_ids_json"].apply(lambda value: any(_json_contains(value, item) for item in session_candidates))
                ]
            if projection_kind is not None:
                scoped = scoped[scoped["projection_kind"].astype(str) == str(projection_kind)]
            if group_id is not None:
                scoped = scoped[_group_identity_mask(scoped, group_id)]
            if origin_message_id is not None:
                scoped = scoped[
                    scoped["origin_message_ids_json"].apply(lambda value: _json_contains(value, str(origin_message_id)))
                ]
            if not include_deleted:
                scoped = scoped[scoped["active_message_count"].astype(int) > 0]
            scoped = scoped.sort_values(
                ["updated_at", "projection_kind", "session_id"],
                ascending=[False, True, True],
                kind="stable",
            )
            out: List[Dict[str, object]] = []
            for _, row in scoped.iterrows():
                out.append(
                    {
                        "projection_id": str(row.get("projection_id") or ""),
                        "tenant_id": str(row.get("tenant_id") or "default"),
                        "session_id": str(row.get("session_id") or ""),
                        "projection_kind": str(row.get("projection_kind") or ""),
                        "projection_scope": str(row.get("projection_scope") or ""),
                        "visibility": str(row.get("visibility") or ""),
                        "chat_type": str(row.get("chat_type") or ""),
                        "native_session_id": str(row.get("native_session_id") or ""),
                        "native_session_ids": json.loads(str(row.get("native_session_ids_json") or "[]")),
                        "paired_projection_ids": json.loads(str(row.get("paired_projection_ids_json") or "[]")),
                        "paired_session_ids": json.loads(str(row.get("paired_session_ids_json") or "[]")),
                        "paired_projection_scopes": json.loads(str(row.get("paired_projection_scopes_json") or "[]")),
                        "account_id": str(row.get("account_id") or ""),
                        "account_key": str(row.get("account_key") or ""),
                        "group_id": str(row.get("group_id") or ""),
                        "group_chat_key": str(row.get("group_chat_key") or ""),
                        "sender_id": str(row.get("sender_id") or ""),
                        "sender_user_key": str(row.get("sender_user_key") or ""),
                        "topic_ids": json.loads(str(row.get("topic_ids_json") or "[]")),
                        "origin_message_count": int(row.get("origin_message_count") or 0),
                        "active_message_count": int(row.get("active_message_count") or 0),
                        "deleted_message_count": int(row.get("deleted_message_count") or 0),
                        "first_origin_message_id": str(row.get("first_origin_message_id") or ""),
                        "last_origin_message_id": str(row.get("last_origin_message_id") or ""),
                        "origin_message_ids": json.loads(str(row.get("origin_message_ids_json") or "[]")),
                        "source_first_ts": pd.to_datetime(row.get("source_first_ts"), utc=True).isoformat(),
                        "source_last_ts": pd.to_datetime(row.get("source_last_ts"), utc=True).isoformat(),
                        "summary": str(row.get("summary") or ""),
                        "vector_ref": str(row.get("vector_ref") or ""),
                        "updated_at": pd.to_datetime(row.get("updated_at"), utc=True).isoformat(),
                    }
                )
            return out

    async def list_belief_state(
        self,
        *,
        tenant_id: str,
        scope_type: Optional[str] = None,
        session_id: Optional[str] = None,
        topic_id: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        async with self._lock:
            df = self._state.beliefs_df.copy().reset_index(drop=True)
            if df.empty:
                return []
            df["tenant_id"] = df["tenant_id"].fillna("default").astype(str)
            df["scope_type"] = df["scope_type"].fillna("").astype(str)
            df["session_id"] = df["session_id"].fillna("").astype(str)
            df["topic_id"] = df["topic_id"].fillna("").astype(str)
            df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True, errors="coerce")
            scoped = df[df["tenant_id"].astype(str) == str(tenant_id)].copy()
            if scope_type is not None:
                scoped = scoped[scoped["scope_type"].astype(str) == str(scope_type)]
            if session_id is not None:
                resolved = self._resolve_session_ids_locked(str(tenant_id), str(session_id), include_mirrors=True)
                session_candidates = sorted({str(session_id), *resolved})
                scoped = scoped[scoped["session_id"].astype(str).isin(session_candidates)]
            if topic_id is not None:
                canonical_ids = self._canonical_topic_ids_locked(str(tenant_id), [str(topic_id)])
                topic_candidates = sorted({str(topic_id), *canonical_ids})
                scoped = scoped[scoped["topic_id"].astype(str).isin(topic_candidates)]
            scoped = scoped.sort_values(
                ["updated_at", "scope_type", "scope_key"],
                ascending=[False, True, True],
                kind="stable",
            )
            out: List[Dict[str, object]] = []
            for _, row in scoped.iterrows():
                out.append(
                    {
                        "belief_id": str(row.get("belief_id") or ""),
                        "tenant_id": str(row.get("tenant_id") or "default"),
                        "scope_type": str(row.get("scope_type") or ""),
                        "scope_key": str(row.get("scope_key") or ""),
                        "session_id": str(row.get("session_id") or ""),
                        "topic_id": str(row.get("topic_id") or ""),
                        "group_id": str(row.get("group_id") or ""),
                        "projection_kind": str(row.get("projection_kind") or ""),
                        "projection_scope": str(row.get("projection_scope") or ""),
                        "first_ts": pd.to_datetime(row.get("first_ts"), utc=True).isoformat(),
                        "last_ts": pd.to_datetime(row.get("last_ts"), utc=True).isoformat(),
                        "raw_message_count": int(row.get("raw_message_count") or 0),
                        "first_origin_message_id": str(row.get("first_origin_message_id") or ""),
                        "last_origin_message_id": str(row.get("last_origin_message_id") or ""),
                        "source_message_ids": json.loads(str(row.get("source_message_ids_json") or "[]")),
                        "source_session_ids": json.loads(str(row.get("source_session_ids_json") or "[]")),
                        "topic_ids": json.loads(str(row.get("topic_ids_json") or "[]")),
                        "summary": str(row.get("summary") or ""),
                        "vector_ref": str(row.get("vector_ref") or ""),
                        "updated_at": pd.to_datetime(row.get("updated_at"), utc=True).isoformat(),
                    }
                )
            return out

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
            if df.empty:
                return []
            df = df.reset_index(drop=True).sort_values(
                ["updated_at", "capsule_ordinal"],
                kind="stable",
                ascending=[False, False],
            )
            out: List[Dict[str, object]] = []
            for _, row in df.iterrows():
                out.append(
                    {
                        "capsule_id": str(row["capsule_id"]),
                        "topic_id": str(row.get("topic_id") or "default"),
                        "topic_path": str(row.get("topic_path") or row.get("topic_id") or "default"),
                        "capsule_ordinal": int(row.get("capsule_ordinal") or 0),
                        "capsule_state": str(row.get("capsule_state") or ""),
                        "summary": str(row.get("summary") or ""),
                        "level": str(row.get("level") or "L2"),
                        "score": float(row.get("score") or 0.0),
                        "source_message_count": int(row.get("source_message_count") or 0),
                        "source_body_char_count": int(row.get("source_body_char_count") or 0),
                        "threshold_body_char_count": int(row.get("threshold_body_char_count") or 0),
                        "source_session_ids": json.loads(str(row.get("source_session_ids_json") or "[]")),
                        "source_topic_ids": json.loads(str(row.get("source_topic_ids_json") or "[]")),
                        "active_message_count": int(row.get("active_message_count") or 0),
                        "edited_message_count": int(row.get("edited_message_count") or 0),
                        "topic_message_count": int(row.get("topic_message_count") or 0),
                        "topic_body_char_count": int(row.get("topic_body_char_count") or 0),
                        "opened_at": (
                            pd.to_datetime(row.get("opened_at"), utc=True).isoformat()
                            if pd.notna(row.get("opened_at"))
                            else None
                        ),
                        "sealed_at": (
                            pd.to_datetime(row.get("sealed_at"), utc=True).isoformat()
                            if pd.notna(row.get("sealed_at"))
                            else None
                        ),
                        "prev_capsule_id": str(row.get("prev_capsule_id") or ""),
                        "next_capsule_id": str(row.get("next_capsule_id") or ""),
                        "back_link_ids": json.loads(str(row.get("back_link_ids_json") or "[]")),
                        "forward_link_ids": json.loads(str(row.get("forward_link_ids_json") or "[]")),
                        "vector_ref": str(row.get("vector_ref") or ""),
                        "pointer": json.loads(str(row.get("pointer_json") or "{}")),
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

    async def topic_runtime_rows(self) -> List[Dict[str, object]]:
        async with self._lock:
            topics = self._state.topics_df
            if topics.empty:
                return []
            scoped = topics.copy().reset_index(drop=True)
            scoped["tenant_id"] = scoped["tenant_id"].fillna("default").astype(str)
            scoped["topic_id"] = scoped["topic_id"].fillna("default").astype(str)
            scoped["canonical_topic_id"] = scoped["canonical_topic_id"].fillna(scoped["topic_id"]).astype(str)
            scoped["topic_path"] = scoped["topic_path"].fillna(scoped["canonical_topic_id"]).astype(str)
            scoped["status"] = scoped["status"].fillna("").astype(str)
            scoped["summary"] = scoped["summary"].fillna("").astype(str)
            scoped["vector_text"] = scoped["vector_text"].fillna(scoped["summary"]).astype(str)
            scoped["message_count"] = pd.to_numeric(
                scoped["message_count"], errors="coerce"
            ).fillna(0).astype(int)
            scoped["updated_at"] = pd.to_datetime(scoped["updated_at"], utc=True, errors="coerce")
            scoped = scoped[scoped["status"].astype(str) != "compacted"].copy()
            if scoped.empty:
                return []
            scoped["_canonical_priority"] = (
                scoped["topic_id"].astype(str) != scoped["canonical_topic_id"].astype(str)
            ).astype(int)
            scoped = scoped.sort_values(
                ["tenant_id", "canonical_topic_id", "_canonical_priority", "updated_at", "topic_id"],
                ascending=[True, True, True, False, True],
                kind="stable",
            )
            scoped = scoped.groupby(
                ["tenant_id", "canonical_topic_id"], sort=True, as_index=False
            ).head(1)
            rows: List[Dict[str, object]] = []
            for _, row in scoped.iterrows():
                rows.append(
                    {
                        "tenant_id": str(row.get("tenant_id") or "default"),
                        "topic_id": str(row.get("canonical_topic_id") or row.get("topic_id") or "default"),
                        "topic_path": str(row.get("topic_path") or row.get("canonical_topic_id") or row.get("topic_id") or "default"),
                        "vector_text": str(row.get("vector_text") or row.get("summary") or ""),
                        "message_count": int(row.get("message_count") or 0),
                    }
                )
            return rows

    async def needs_materialized_rebuild(self) -> bool:
        async with self._lock:
            raw_messages = authoritative_raw_messages(self._state.messages_df)
            if raw_messages.empty:
                return False
            return bool(
                self._state.topics_df.empty
                or self._state.capsules_df.empty
                or self._state.projections_df.empty
                or self._state.session_rollups_df.empty
                or self._state.search_docs_df.empty
                or self._state.lexical_index_df.empty
                or self._state.vector_index_df.empty
            )

    async def virtual_memory_file(self, rel_path: str) -> Tuple[str, str]:
        normalized = rel_path.replace("\\", "/").lstrip("/")
        channel_file = MessageChannelFile.from_virtual_path(normalized)
        async with self._lock:
            if channel_file is not None:
                if channel_file.scope_kind == "session":
                    df = self._message_view_rows_locked(
                        tenant_id=channel_file.tenant_id,
                        session_id=channel_file.scope_value,
                        channel=channel_file.channel,
                        chat_type=channel_file.chat_type,
                        row_mode="projection",
                        include_deleted=False,
                    )
                else:
                    df = self._visible_message_rows_locked(
                        self._messages_for_channel_file_locked(channel_file)
                    )
                canonical = channel_file.virtual_path
            else:
                legacy = self._legacy_session_virtual_path(normalized)
                if legacy is None:
                    raise FileNotFoundError(f"unsupported memory path: {rel_path}")
                tenant_id, session_id = legacy
                df = self._message_view_rows_locked(
                    tenant_id=tenant_id,
                    session_id=session_id,
                    row_mode="projection",
                    include_deleted=False,
                )
                canonical = normalized
            if df.empty:
                raise FileNotFoundError(f"no memory found for path={rel_path}")
            lines: List[str] = []
            for _, row in df.iterrows():
                lines.extend(self._message_block_lines(row))
        return "\n".join(lines).rstrip() + "\n", canonical

    async def save_parquet(self, parquet_dir: Path) -> None:
        async with self._lock:
            state = self._state
            await asyncio.to_thread(self._save_parquet_sync, parquet_dir, state)

    def _save_parquet_sync(self, parquet_dir: Path, state: DataFramesState) -> None:
        parquet_dir.mkdir(parents=True, exist_ok=True)
        timestamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d-%H%M%S")
        tmp_parquet = parquet_dir.parent / f"{parquet_dir.name}.tmp-save-{timestamp}"
        if tmp_parquet.exists():
            shutil.rmtree(tmp_parquet)
        tmp_parquet.mkdir(parents=True, exist_ok=True)

        def _write_partitioned(df: pd.DataFrame, name: str) -> None:
            if df.empty:
                target = tmp_parquet / name / "dt=empty"
                target.mkdir(parents=True, exist_ok=True)
                df.head(0).to_parquet(target / f"part-{timestamp}.parquet", index=False)
                return
            write_df = df.copy()
            if "ts" in write_df.columns:
                write_df["dt"] = pd.to_datetime(write_df["ts"], utc=True).dt.strftime("%Y-%m-%d")
            elif "updated_at" in write_df.columns:
                write_df["dt"] = pd.to_datetime(write_df["updated_at"], utc=True).dt.strftime("%Y-%m-%d")
            elif "created_at" in write_df.columns:
                write_df["dt"] = pd.to_datetime(write_df["created_at"], utc=True).dt.strftime("%Y-%m-%d")
            else:
                write_df["dt"] = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
            for dt, part in write_df.groupby("dt"):
                target = tmp_parquet / name / f"dt={dt}"
                target.mkdir(parents=True, exist_ok=True)
                part.drop(columns=["dt"]).to_parquet(target / f"part-{timestamp}.parquet", index=False)

        messages_dir = tmp_parquet / "messages"
        messages_dir.mkdir(parents=True, exist_ok=True)
        messages_df = self._compact_messages_for_storage(state.messages_df)
        if not messages_df.empty:
            write_df = messages_df.sort_values("ts", kind="stable").copy()
            write_df["_channel_file_path"] = [
                str(self._message_channel_file(row).storage_relative_path)
                for _, row in write_df.iterrows()
            ]
            for rel_path, part in write_df.groupby("_channel_file_path", sort=True):
                target = messages_dir / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                part.drop(columns=["_channel_file_path"]).to_parquet(target, index=False)
        _write_partitioned(state.capsules_df, "capsules")
        _write_partitioned(state.beliefs_df, "beliefs")
        _write_partitioned(state.projections_df, "projections")
        _write_partitioned(state.session_rollups_df, "session_rollups")
        _write_partitioned(state.topics_df, "topics")
        _write_partitioned(state.search_docs_df, "search_docs")
        _write_partitioned(state.lexical_index_df, "lexical_index")
        _write_partitioned(state.vector_index_df, "vector_index")
        _write_partitioned(state.embedding_index_metadata_df, "embedding_index_metadata")
        _write_partitioned(state.cache_index_df, "cache_index")
        _write_partitioned(state.sessions_df, "sessions")
        _write_partitioned(state.snapshots_df, "snapshots")
        _write_partitioned(state.semantic_jobs_df, "semantic_jobs")
        if parquet_dir.exists():
            shutil.rmtree(parquet_dir)
        tmp_parquet.rename(parquet_dir)

    async def load_parquet(self, parquet_dir: Path) -> None:
        async with self._lock:
            await asyncio.to_thread(self._load_parquet_sync, parquet_dir)

    def _load_parquet_sync(self, parquet_dir: Path) -> None:
        def _read_all(name: str, columns: List[str]) -> pd.DataFrame:
            base = parquet_dir / name
            if not base.exists():
                return pd.DataFrame(columns=columns)
            files = table_parquet_files(base)
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
            beliefs_df=_read_all("beliefs", BELIEFS_COLUMNS),
            projections_df=_read_all("projections", PROJECTIONS_COLUMNS),
            session_rollups_df=_read_all("session_rollups", SESSION_ROLLUPS_COLUMNS),
            topics_df=_read_all("topics", TOPICS_COLUMNS),
            search_docs_df=_read_all("search_docs", SEARCH_DOC_COLUMNS),
            lexical_index_df=_read_all("lexical_index", LEXICAL_INDEX_COLUMNS),
            vector_index_df=_read_all("vector_index", VECTOR_INDEX_COLUMNS),
            embedding_index_metadata_df=_read_all(
                "embedding_index_metadata",
                EMBEDDING_INDEX_METADATA_COLUMNS,
            ),
            cache_index_df=_read_all("cache_index", CACHE_INDEX_COLUMNS),
            sessions_df=_read_all("sessions", SESSIONS_COLUMNS),
            snapshots_df=_read_all("snapshots", SNAPSHOTS_COLUMNS),
            semantic_jobs_df=_read_all("semantic_jobs", SEMANTIC_JOBS_COLUMNS),
        )
        self._state.messages_df["is_deleted"] = (
            self._state.messages_df["message_state"].fillna("").astype(str) == MESSAGE_STATE_DELETED
        )
        self._invalidate_all_indexes_locked()
