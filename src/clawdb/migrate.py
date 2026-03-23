from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import pandas as pd

from .capsules import CAPSULES_COLUMNS
from .dataframes import (
    CACHE_INDEX_COLUMNS,
    EMBEDDING_INDEX_METADATA_COLUMNS,
    MESSAGES_COLUMNS,
    SESSION_ROLLUPS_COLUMNS,
    SESSIONS_COLUMNS,
    SNAPSHOTS_COLUMNS,
    rebuild_materialized_storage_from_raw,
)
from .lineage import (
    MESSAGE_STATE_ACTIVE,
    PLATFORM_IDENTITY_COLUMNS,
    RAW_PROJECTION_KIND,
    materialize_message_bundle,
    normalize_platform_identities,
)
from .metadata import DataFrameMetadataStore
from .topics import TOPICS_COLUMNS


CURRENT_SCHEMA_VERSION = 9
SCHEMA_VERSION_SLOT = "schema_version"


@dataclass(frozen=True)
class TableMigrationPlan:
    table: str
    file_count: int
    row_count: int
    missing_columns: List[str]
    needs_rewrite: bool


@dataclass(frozen=True)
class SchemaMigrationPlan:
    source_version: int
    target_version: int
    needs_migration: bool
    needs_metadata_update: bool
    tables: List[TableMigrationPlan]
    reason: str


@dataclass(frozen=True)
class SchemaMigrationResult:
    plan: SchemaMigrationPlan
    applied: bool
    backup_dir: Optional[str]
    report_path: Optional[str]


TABLE_COLUMNS: Mapping[str, List[str]] = {
    "messages": MESSAGES_COLUMNS,
    "capsules": CAPSULES_COLUMNS,
    "session_rollups": SESSION_ROLLUPS_COLUMNS,
    "topics": TOPICS_COLUMNS,
    "embedding_index_metadata": EMBEDDING_INDEX_METADATA_COLUMNS,
    "cache_index": CACHE_INDEX_COLUMNS,
    "sessions": SESSIONS_COLUMNS,
    "snapshots": SNAPSHOTS_COLUMNS,
}


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _to_datetime_utc(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    return parsed.fillna(pd.Timestamp.now(tz="UTC"))


def _to_float(series: pd.Series, default: float) -> pd.Series:
    parsed = pd.to_numeric(series, errors="coerce")
    return parsed.fillna(default).astype(float)


def _to_int(series: pd.Series, default: int = 0) -> pd.Series:
    parsed = pd.to_numeric(series, errors="coerce")
    return parsed.fillna(default).astype(int)


def _fill_string(series: pd.Series, default: str = "") -> pd.Series:
    return series.fillna(default).astype(str)


def _table_files(table_dir: Path) -> List[Path]:
    if not table_dir.exists():
        return []
    partitioned = sorted(table_dir.glob("dt=*/part-*.parquet"))
    flat = sorted(table_dir.glob("*.parquet"))
    files = partitioned + flat
    uniq: Dict[str, Path] = {}
    for file_path in files:
        uniq[str(file_path.resolve())] = file_path
    return list(uniq.values())


def _read_table(parquet_dir: Path, table: str) -> Tuple[pd.DataFrame, List[Path]]:
    table_dir = parquet_dir / table
    files = _table_files(table_dir)
    if not files:
        return pd.DataFrame(), []
    parts: List[pd.DataFrame] = []
    for file_path in files:
        try:
            parts.append(pd.read_parquet(file_path))
        except Exception:
            continue
    if not parts:
        return pd.DataFrame(), files
    non_empty = [part for part in parts if not part.empty]
    if not non_empty:
        return parts[0].iloc[0:0].copy(), files
    return pd.concat(non_empty, ignore_index=True), files


def _ensure_columns(
    frame: pd.DataFrame,
    columns: List[str],
    defaults: Mapping[str, object | Callable[[pd.DataFrame], object]],
) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        if col not in out.columns:
            default = defaults.get(col, "")
            out[col] = default(out) if callable(default) else default
    return out


def _normalize_messages(frame: pd.DataFrame) -> pd.DataFrame:
    defaults: Dict[str, object | Callable[[pd.DataFrame], object]] = {
        "message_id": "",
        "origin_message_id": lambda df: df.get("message_id", pd.Series([""] * len(df))),
        "tenant_id": "default",
        "session_id": "default",
        "role": "user",
        "content": "",
        "ts": pd.Timestamp.now(tz="UTC"),
        "channel": "",
        "chat_type": "",
        "account_id": "",
        "account_key": "",
        "from_id": "",
        "from_user_key": "",
        "to_id": "",
        "to_user_key": "",
        "sender_id": "",
        "sender_user_key": "",
        "sender_name": "",
        "sender_username": "",
        "sender_e164": "",
        "group_id": "",
        "group_chat_key": "",
        "group_subject": "",
        "group_channel": "",
        "group_space": "",
        "native_channel_id": "",
        "message_thread_id": "",
        "thread_parent_id": "",
        "reply_to_id": "",
        "topic_id": "default",
        "topic_parent_id": "",
        "topic_path": lambda df: df.get("topic_id", pd.Series(["default"] * len(df))),
        "topic_confidence": 1.0,
        "topic_source": "explicit",
        "embedding_ref": "",
        "capsule_level": "L0",
        "idempotency_key": "",
        "projection_kind": "",
        "projection_scope": "",
        "visibility": "",
        "platform": "",
        "platform_message_id": "",
        "native_session_id": "",
        "message_state": MESSAGE_STATE_ACTIVE,
        "updated_at": lambda df: df.get("ts", pd.Series([pd.Timestamp.now(tz="UTC")] * len(df))),
        "deleted_at": None,
    }
    out = _ensure_columns(frame, MESSAGES_COLUMNS, defaults)

    out["message_id"] = _fill_string(out["message_id"])
    out["origin_message_id"] = _fill_string(out["origin_message_id"])
    out["tenant_id"] = _fill_string(out["tenant_id"], "default")
    out["session_id"] = _fill_string(out["session_id"], "default")
    out["role"] = _fill_string(out["role"], "user")
    out["content"] = _fill_string(out["content"])
    out["ts"] = _to_datetime_utc(out["ts"])
    out["channel"] = _fill_string(out["channel"])
    out["chat_type"] = _fill_string(out["chat_type"])
    out["account_id"] = _fill_string(out["account_id"])
    out["account_key"] = _fill_string(out["account_key"])
    out["from_id"] = _fill_string(out["from_id"])
    out["from_user_key"] = _fill_string(out["from_user_key"])
    out["to_id"] = _fill_string(out["to_id"])
    out["to_user_key"] = _fill_string(out["to_user_key"])
    out["sender_id"] = _fill_string(out["sender_id"])
    out["sender_user_key"] = _fill_string(out["sender_user_key"])
    out["sender_name"] = _fill_string(out["sender_name"])
    out["sender_username"] = _fill_string(out["sender_username"])
    out["sender_e164"] = _fill_string(out["sender_e164"])
    out["group_id"] = _fill_string(out["group_id"])
    out["group_chat_key"] = _fill_string(out["group_chat_key"])
    out["group_subject"] = _fill_string(out["group_subject"])
    out["group_channel"] = _fill_string(out["group_channel"])
    out["group_space"] = _fill_string(out["group_space"])
    out["native_channel_id"] = _fill_string(out["native_channel_id"])
    out["message_thread_id"] = _fill_string(out["message_thread_id"])
    out["thread_parent_id"] = _fill_string(out["thread_parent_id"])
    out["reply_to_id"] = _fill_string(out["reply_to_id"])
    out["topic_id"] = _fill_string(out["topic_id"], "default")
    out["topic_parent_id"] = _fill_string(out["topic_parent_id"])
    out["topic_path"] = _fill_string(out["topic_path"])
    out.loc[out["topic_path"] == "", "topic_path"] = out.loc[out["topic_path"] == "", "topic_id"]
    out["topic_source"] = _fill_string(out["topic_source"], "explicit")
    out["topic_confidence"] = _to_float(out["topic_confidence"], 1.0)
    out["embedding_ref"] = _fill_string(out["embedding_ref"])
    out["capsule_level"] = _fill_string(out["capsule_level"], "L0")
    out["idempotency_key"] = _fill_string(out["idempotency_key"])
    out["projection_kind"] = _fill_string(out["projection_kind"])
    out["projection_scope"] = _fill_string(out["projection_scope"])
    out["visibility"] = _fill_string(out["visibility"])
    out["platform"] = _fill_string(out["platform"])
    out["platform_message_id"] = _fill_string(out["platform_message_id"])
    out["native_session_id"] = _fill_string(out["native_session_id"])
    out["message_state"] = _fill_string(out["message_state"], MESSAGE_STATE_ACTIVE)
    out["updated_at"] = _to_datetime_utc(out["updated_at"])
    out["deleted_at"] = pd.to_datetime(out["deleted_at"], utc=True, errors="coerce")

    if not out.empty:
        identity_rows = [normalize_platform_identities(row) for row in out.to_dict("records")]
        identity_frame = pd.DataFrame(identity_rows)
        for column in PLATFORM_IDENTITY_COLUMNS:
            out[column] = _fill_string(identity_frame.get(column, pd.Series([""] * len(out))))

    legacy_rows = "projection_kind" not in frame.columns or "origin_message_id" not in frame.columns
    if legacy_rows:
        expanded: List[Dict[str, object]] = []
        for row in out.to_dict("records"):
            bundle = materialize_message_bundle(row)
            expanded.append(dict(bundle["raw_message"]))
            expanded.extend([dict(item) for item in bundle["projections"]])
        if not expanded:
            return out.iloc[0:0].copy()[MESSAGES_COLUMNS]
        expanded_frame = pd.DataFrame(expanded, columns=MESSAGES_COLUMNS)
        expanded_frame["ts"] = _to_datetime_utc(expanded_frame["ts"])
        expanded_frame["updated_at"] = _to_datetime_utc(expanded_frame["updated_at"])
        expanded_frame["deleted_at"] = pd.to_datetime(
            expanded_frame["deleted_at"], utc=True, errors="coerce"
        )
        return expanded_frame[MESSAGES_COLUMNS]

    has_raw_rows = (out["projection_kind"].astype(str) == RAW_PROJECTION_KIND).any()
    if not has_raw_rows and not out.empty:
        raw_rows: List[Dict[str, object]] = []
        for _, row in out.iterrows():
            if str(row.get("projection_kind") or "") == RAW_PROJECTION_KIND:
                continue
            bundle = materialize_message_bundle(row.to_dict())
            raw_rows.append(dict(bundle["raw_message"]))
        if raw_rows:
            out = pd.concat([out, pd.DataFrame(raw_rows, columns=MESSAGES_COLUMNS)], ignore_index=True)
            out = out.drop_duplicates(subset=["message_id"], keep="first")
    return out[MESSAGES_COLUMNS]


def _normalize_capsules(frame: pd.DataFrame) -> pd.DataFrame:
    defaults: Dict[str, object] = {
        "capsule_id": "",
        "tenant_id": "default",
        "session_id": "topic:default",
        "topic_id": "default",
        "topic_path": "default",
        "capsule_ordinal": 0,
        "capsule_state": "open",
        "summary": "",
        "level": "L2",
        "score": 0.0,
        "source_message_count": 0,
        "source_body_char_count": 0,
        "threshold_body_char_count": 100000,
        "first_origin_message_id": "",
        "last_origin_message_id": "",
        "source_message_ids_json": "[]",
        "source_first_ts": pd.Timestamp.now(tz="UTC"),
        "source_last_ts": pd.Timestamp.now(tz="UTC"),
        "prev_capsule_id": "",
        "next_capsule_id": "",
        "back_link_ids_json": "[]",
        "forward_link_ids_json": "[]",
        "pointer_json": "{}",
        "vector_text": "",
        "vector_ref": "",
        "vector_dim": 64,
        "vector_json": "[]",
        "source_hash": "",
        "updated_at": pd.Timestamp.now(tz="UTC"),
    }
    out = _ensure_columns(frame, CAPSULES_COLUMNS, defaults)
    out["capsule_id"] = _fill_string(out["capsule_id"])
    out["tenant_id"] = _fill_string(out["tenant_id"], "default")
    out["session_id"] = _fill_string(out["session_id"], "topic:default")
    out["topic_id"] = _fill_string(out["topic_id"], "default")
    out["topic_path"] = _fill_string(out["topic_path"], "default")
    out["capsule_ordinal"] = _to_int(out["capsule_ordinal"], 0)
    out["capsule_state"] = _fill_string(out["capsule_state"], "open")
    out["summary"] = _fill_string(out["summary"])
    out["level"] = _fill_string(out["level"], "L2")
    out["score"] = _to_float(out["score"], 0.0)
    out["source_message_count"] = _to_int(out["source_message_count"], 0)
    out["source_body_char_count"] = _to_int(out["source_body_char_count"], 0)
    out["threshold_body_char_count"] = _to_int(out["threshold_body_char_count"], 100000)
    out["first_origin_message_id"] = _fill_string(out["first_origin_message_id"])
    out["last_origin_message_id"] = _fill_string(out["last_origin_message_id"])
    out["source_message_ids_json"] = _fill_string(out["source_message_ids_json"], "[]")
    out["source_first_ts"] = _to_datetime_utc(out["source_first_ts"])
    out["source_last_ts"] = _to_datetime_utc(out["source_last_ts"])
    out["prev_capsule_id"] = _fill_string(out["prev_capsule_id"])
    out["next_capsule_id"] = _fill_string(out["next_capsule_id"])
    out["back_link_ids_json"] = _fill_string(out["back_link_ids_json"], "[]")
    out["forward_link_ids_json"] = _fill_string(out["forward_link_ids_json"], "[]")
    out["pointer_json"] = _fill_string(out["pointer_json"], "{}")
    out["vector_text"] = _fill_string(out["vector_text"])
    out["vector_ref"] = _fill_string(out["vector_ref"])
    out["vector_dim"] = _to_int(out["vector_dim"], 64)
    out["vector_json"] = _fill_string(out["vector_json"], "[]")
    out["source_hash"] = _fill_string(out["source_hash"])
    out["updated_at"] = _to_datetime_utc(out["updated_at"])
    return out[CAPSULES_COLUMNS]


def _normalize_session_rollups(frame: pd.DataFrame) -> pd.DataFrame:
    defaults: Dict[str, object] = {
        "rollup_id": "",
        "tenant_id": "default",
        "session_id": "default",
        "window_kind": "daily",
        "window_key": "",
        "bucket_start": pd.Timestamp.now(tz="UTC"),
        "bucket_end": pd.Timestamp.now(tz="UTC"),
        "source_first_ts": pd.Timestamp.now(tz="UTC"),
        "source_last_ts": pd.Timestamp.now(tz="UTC"),
        "message_count": 0,
        "content_char_count": 0,
        "summary": "",
        "vector_text": "",
        "vector_ref": "",
        "vector_dim": 64,
        "vector_json": "[]",
        "updated_at": pd.Timestamp.now(tz="UTC"),
    }
    out = _ensure_columns(frame, SESSION_ROLLUPS_COLUMNS, defaults)
    out["rollup_id"] = _fill_string(out["rollup_id"])
    out["tenant_id"] = _fill_string(out["tenant_id"], "default")
    out["session_id"] = _fill_string(out["session_id"], "default")
    out["window_kind"] = _fill_string(out["window_kind"], "daily")
    out["window_key"] = _fill_string(out["window_key"])
    out["bucket_start"] = _to_datetime_utc(out["bucket_start"])
    out["bucket_end"] = _to_datetime_utc(out["bucket_end"])
    out["source_first_ts"] = _to_datetime_utc(out["source_first_ts"])
    out["source_last_ts"] = _to_datetime_utc(out["source_last_ts"])
    out["message_count"] = _to_int(out["message_count"], 0)
    out["content_char_count"] = _to_int(out["content_char_count"], 0)
    out["summary"] = _fill_string(out["summary"])
    out["vector_text"] = _fill_string(out["vector_text"])
    out["vector_ref"] = _fill_string(out["vector_ref"])
    out["vector_dim"] = _to_int(out["vector_dim"], 64)
    out["vector_json"] = _fill_string(out["vector_json"], "[]")
    out["updated_at"] = _to_datetime_utc(out["updated_at"])
    return out[SESSION_ROLLUPS_COLUMNS]


def _normalize_topics(frame: pd.DataFrame) -> pd.DataFrame:
    defaults: Dict[str, object] = {
        "topic_id": "default",
        "tenant_id": "default",
        "canonical_topic_id": "default",
        "topic_parent_id": "",
        "topic_path": "default",
        "source_topic_id": "default",
        "status": "active",
        "historical_message_count": 0,
        "message_count": 0,
        "deleted_message_count": 0,
        "content_char_count": 0,
        "keywords_json": "[]",
        "merged_topic_ids_json": "[]",
        "split_topic_ids_json": "[]",
        "drift_score": 0.0,
        "drift_corrected_at": pd.NaT,
        "first_ts": pd.Timestamp.now(tz="UTC"),
        "last_ts": pd.Timestamp.now(tz="UTC"),
        "summary": "",
        "vector_text": "",
        "vector_ref": "",
        "vector_dim": 64,
        "vector_json": "[]",
        "updated_at": pd.Timestamp.now(tz="UTC"),
    }
    out = _ensure_columns(frame, TOPICS_COLUMNS, defaults)
    out["topic_id"] = _fill_string(out["topic_id"], "default")
    out["tenant_id"] = _fill_string(out["tenant_id"], "default")
    out["canonical_topic_id"] = _fill_string(out["canonical_topic_id"], "default")
    out["topic_parent_id"] = _fill_string(out["topic_parent_id"])
    out["topic_path"] = _fill_string(out["topic_path"], "default")
    out["source_topic_id"] = _fill_string(out["source_topic_id"], "default")
    out["status"] = _fill_string(out["status"], "active")
    out["historical_message_count"] = _to_int(out["historical_message_count"], 0)
    out["message_count"] = _to_int(out["message_count"], 0)
    out["deleted_message_count"] = _to_int(out["deleted_message_count"], 0)
    out["content_char_count"] = _to_int(out["content_char_count"], 0)
    out["keywords_json"] = _fill_string(out["keywords_json"], "[]")
    out["merged_topic_ids_json"] = _fill_string(out["merged_topic_ids_json"], "[]")
    out["split_topic_ids_json"] = _fill_string(out["split_topic_ids_json"], "[]")
    out["drift_score"] = _to_float(out["drift_score"], 0.0)
    out["drift_corrected_at"] = pd.to_datetime(out["drift_corrected_at"], utc=True, errors="coerce")
    out["first_ts"] = _to_datetime_utc(out["first_ts"])
    out["last_ts"] = _to_datetime_utc(out["last_ts"])
    out["summary"] = _fill_string(out["summary"])
    out["vector_text"] = _fill_string(out["vector_text"])
    out["vector_ref"] = _fill_string(out["vector_ref"])
    out["vector_dim"] = _to_int(out["vector_dim"], 64)
    out["vector_json"] = _fill_string(out["vector_json"], "[]")
    out["updated_at"] = _to_datetime_utc(out["updated_at"])
    return out[TOPICS_COLUMNS]


def _normalize_embedding_index_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    defaults: Dict[str, object] = {
        "tenant_id": "default",
        "entity_type": "",
        "entity_id": "",
        "embedding_ref": "",
        "embedding_provider": "",
        "embedding_model": "",
        "source_hash": "",
        "reembed_policy": "",
        "updated_at": pd.Timestamp.now(tz="UTC"),
    }
    out = _ensure_columns(frame, EMBEDDING_INDEX_METADATA_COLUMNS, defaults)
    out["tenant_id"] = _fill_string(out["tenant_id"], "default")
    out["entity_type"] = _fill_string(out["entity_type"])
    out["entity_id"] = _fill_string(out["entity_id"])
    out["embedding_ref"] = _fill_string(out["embedding_ref"])
    out["embedding_provider"] = _fill_string(out["embedding_provider"])
    out["embedding_model"] = _fill_string(out["embedding_model"])
    out["source_hash"] = _fill_string(out["source_hash"])
    out["reembed_policy"] = _fill_string(out["reembed_policy"])
    out["updated_at"] = _to_datetime_utc(out["updated_at"])
    return out[EMBEDDING_INDEX_METADATA_COLUMNS]


def _normalize_cache_index(frame: pd.DataFrame) -> pd.DataFrame:
    defaults: Dict[str, object] = {
        "key": "",
        "tenant_id": "default",
        "session_id": "_",
        "query_type": "memory_search",
        "capsule_level": "mixed",
        "entity_type": "search",
        "entity_id": "",
        "last_access": pd.Timestamp.now(tz="UTC"),
        "hit_count": 0,
        "miss_count": 0,
    }
    out = _ensure_columns(frame, CACHE_INDEX_COLUMNS, defaults)
    out["key"] = _fill_string(out["key"])
    out["tenant_id"] = _fill_string(out["tenant_id"], "default")
    out["session_id"] = _fill_string(out["session_id"], "_")
    out["query_type"] = _fill_string(out["query_type"], "memory_search")
    out["capsule_level"] = _fill_string(out["capsule_level"], "mixed")
    out["entity_type"] = _fill_string(out["entity_type"], "search")
    out["entity_id"] = _fill_string(out["entity_id"])
    out["last_access"] = _to_datetime_utc(out["last_access"])
    out["hit_count"] = _to_int(out["hit_count"], 0)
    out["miss_count"] = _to_int(out["miss_count"], 0)
    return out[CACHE_INDEX_COLUMNS]


def _normalize_sessions(frame: pd.DataFrame) -> pd.DataFrame:
    defaults: Dict[str, object] = {
        "tenant_id": "default",
        "session_id": "default",
        "parent_session_id": "",
        "origin": "normal",
        "created_at": pd.Timestamp.now(tz="UTC"),
    }
    out = _ensure_columns(frame, SESSIONS_COLUMNS, defaults)
    out["tenant_id"] = _fill_string(out["tenant_id"], "default")
    out["session_id"] = _fill_string(out["session_id"], "default")
    out["parent_session_id"] = _fill_string(out["parent_session_id"])
    out["origin"] = _fill_string(out["origin"], "normal")
    out["created_at"] = _to_datetime_utc(out["created_at"])
    return out[SESSIONS_COLUMNS]


def _normalize_snapshots(frame: pd.DataFrame) -> pd.DataFrame:
    defaults: Dict[str, object] = {
        "snapshot_id": "",
        "tenant_id": "default",
        "session_id": "default",
        "wal_seq": 0,
        "note": "",
        "created_at": pd.Timestamp.now(tz="UTC"),
    }
    out = _ensure_columns(frame, SNAPSHOTS_COLUMNS, defaults)
    out["snapshot_id"] = _fill_string(out["snapshot_id"])
    out["tenant_id"] = _fill_string(out["tenant_id"], "default")
    out["session_id"] = _fill_string(out["session_id"], "default")
    out["wal_seq"] = _to_int(out["wal_seq"], 0)
    out["note"] = _fill_string(out["note"])
    out["created_at"] = _to_datetime_utc(out["created_at"])
    return out[SNAPSHOTS_COLUMNS]


NORMALIZERS: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "messages": _normalize_messages,
    "capsules": _normalize_capsules,
    "session_rollups": _normalize_session_rollups,
    "topics": _normalize_topics,
    "embedding_index_metadata": _normalize_embedding_index_metadata,
    "cache_index": _normalize_cache_index,
    "sessions": _normalize_sessions,
    "snapshots": _normalize_snapshots,
}


def _infer_schema_version(messages_frame: pd.DataFrame) -> int:
    if messages_frame.empty:
        return CURRENT_SCHEMA_VERSION
    cols = set(messages_frame.columns)
    identity_cols = set(PLATFORM_IDENTITY_COLUMNS)
    lineage_cols = {
        "origin_message_id",
        "projection_kind",
        "projection_scope",
        "platform_message_id",
        "native_session_id",
        "message_state",
    }
    im_cols = {
        "channel",
        "chat_type",
        "account_id",
        "group_id",
        "message_thread_id",
        "topic_path",
        "topic_source",
        "topic_confidence",
    }
    topic_cols = {"topic_parent_id", "topic_path", "topic_source", "topic_confidence"}
    if lineage_cols.union(identity_cols).issubset(cols):
        return CURRENT_SCHEMA_VERSION
    if lineage_cols.issubset(cols):
        return 8
    if im_cols.issubset(cols):
        return 3
    if topic_cols.issubset(cols):
        return 2
    return 1


def _partition_value(frame: pd.DataFrame, table: str) -> pd.Series:
    if table == "messages":
        return pd.to_datetime(frame["ts"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
    if table in {"capsules", "session_rollups", "topics", "embedding_index_metadata"}:
        return pd.to_datetime(frame["updated_at"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
    if table in {"sessions", "snapshots"}:
        ts_col = "created_at"
        return pd.to_datetime(frame[ts_col], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
    return pd.Series(["1970-01-01"] * len(frame))


def _write_table(frame: pd.DataFrame, table: str, parquet_dir: Path, timestamp: str) -> None:
    base = parquet_dir / table
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        target = base / "dt=empty"
        target.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=TABLE_COLUMNS[table]).to_parquet(
            target / f"part-{timestamp}.parquet", index=False
        )
        return
    write_df = frame.copy()
    write_df["dt"] = _partition_value(write_df, table).fillna("unknown")
    for dt, part in write_df.groupby("dt", sort=True):
        partition = str(dt) if dt else "unknown"
        target = base / f"dt={partition}"
        target.mkdir(parents=True, exist_ok=True)
        part.drop(columns=["dt"]).to_parquet(target / f"part-{timestamp}.parquet", index=False)


def _resolve_backup_dir(data_root: Path, explicit_backup_dir: Optional[Path]) -> Path:
    if explicit_backup_dir is not None:
        return explicit_backup_dir
    return data_root / "backups" / "schema-migrations"


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, "true" if default else "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _rollup_vector_dim_from_env() -> int:
    raw = os.getenv("CLAWDB_TOPIC_GEP_DIM", "64").strip()
    try:
        return max(8, int(raw))
    except Exception:
        return 64


def _build_plan_sync(
    parquet_dir: Path,
    metadata_parquet_path: Path,
    target_version: int,
) -> SchemaMigrationPlan:
    tables: List[TableMigrationPlan] = []
    frames: Dict[str, pd.DataFrame] = {}
    for table, columns in TABLE_COLUMNS.items():
        frame, files = _read_table(parquet_dir, table)
        frames[table] = frame
        existing_cols = set(frame.columns)
        missing = [col for col in columns if col not in existing_cols]
        tables.append(
            TableMigrationPlan(
                table=table,
                file_count=len(files),
                row_count=int(frame.shape[0]),
                missing_columns=missing,
                needs_rewrite=bool(missing),
            )
        )
    source_from_metadata = _read_schema_version_sync(metadata_parquet_path)
    source_version = (
        source_from_metadata
        if source_from_metadata is not None
        else _infer_schema_version(frames.get("messages", pd.DataFrame()))
    )
    has_table_rewrite = any(item.needs_rewrite for item in tables)
    needs_metadata_update = source_version != target_version
    needs_migration = has_table_rewrite or needs_metadata_update
    reason = "already up to date"
    if has_table_rewrite:
        reason = "missing columns detected in one or more parquet tables"
    elif needs_metadata_update:
        reason = "schema metadata version differs from target"
    return SchemaMigrationPlan(
        source_version=source_version,
        target_version=target_version,
        needs_migration=needs_migration,
        needs_metadata_update=needs_metadata_update,
        tables=tables,
        reason=reason,
    )


def _read_schema_version_sync(metadata_parquet_path: Path) -> Optional[int]:
    if not metadata_parquet_path.exists():
        return None
    try:
        df = pd.read_parquet(metadata_parquet_path)
    except Exception:
        return None
    for col in ["slot", "last_seq", "updated_at"]:
        if col not in df.columns:
            df[col] = None
    subset = df[df["slot"].astype(str) == SCHEMA_VERSION_SLOT]
    if subset.empty:
        return None
    if "updated_at" in subset.columns:
        subset = subset.sort_values("updated_at", kind="stable")
    row = subset.iloc[-1]
    try:
        return int(row["last_seq"])
    except Exception:
        return None


def _backup_sync(
    *,
    data_root: Path,
    parquet_dir: Path,
    metadata_parquet_path: Path,
    backup_root: Path,
    timestamp: str,
) -> Path:
    backup_dir = backup_root / f"schema-backup-{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    if parquet_dir.exists():
        shutil.copytree(parquet_dir, backup_dir / "parquet", dirs_exist_ok=True)
    checkpoints_dir = data_root / "checkpoints"
    if checkpoints_dir.exists():
        for name in ["latest.json", "metadata.parquet"]:
            src = checkpoints_dir / name
            if src.exists():
                dst = backup_dir / "checkpoints" / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
    if metadata_parquet_path.exists():
        dst = backup_dir / "metadata.parquet"
        shutil.copy2(metadata_parquet_path, dst)
    return backup_dir


async def build_schema_migration_plan(
    *,
    parquet_dir: Path,
    metadata_parquet_path: Path,
    target_version: int = CURRENT_SCHEMA_VERSION,
) -> SchemaMigrationPlan:
    return await asyncio.to_thread(
        _build_plan_sync,
        parquet_dir,
        metadata_parquet_path,
        target_version,
    )


async def migrate_schema(
    *,
    data_root: Path,
    parquet_dir: Path,
    metadata_parquet_path: Path,
    target_version: int = CURRENT_SCHEMA_VERSION,
    dry_run: bool = False,
    force: bool = False,
    backup: bool = True,
    backup_dir: Optional[Path] = None,
) -> SchemaMigrationResult:
    plan = await build_schema_migration_plan(
        parquet_dir=parquet_dir,
        metadata_parquet_path=metadata_parquet_path,
        target_version=target_version,
    )
    if not force and not plan.needs_migration:
        return SchemaMigrationResult(plan=plan, applied=False, backup_dir=None, report_path=None)
    if dry_run:
        return SchemaMigrationResult(plan=plan, applied=False, backup_dir=None, report_path=None)

    timestamp = _iso_now()
    backup_path: Optional[Path] = None
    if backup:
        resolved_backup_root = _resolve_backup_dir(data_root, backup_dir)
        backup_path = await asyncio.to_thread(
            _backup_sync,
            data_root=data_root,
            parquet_dir=parquet_dir,
            metadata_parquet_path=metadata_parquet_path,
            backup_root=resolved_backup_root,
            timestamp=timestamp,
        )

    normalized_tables: Dict[str, pd.DataFrame] = {}
    for table in TABLE_COLUMNS:
        frame, _ = await asyncio.to_thread(_read_table, parquet_dir, table)
        normalizer = NORMALIZERS[table]
        normalized_tables[table] = await asyncio.to_thread(normalizer, frame)
    rebuilt_storage = await asyncio.to_thread(
        rebuild_materialized_storage_from_raw,
        normalized_tables["messages"],
        normalized_tables["sessions"],
        vector_dim=_rollup_vector_dim_from_env(),
    )
    normalized_tables["messages"] = rebuilt_storage["messages"]
    normalized_tables["sessions"] = rebuilt_storage["sessions"]
    normalized_tables["session_rollups"] = rebuilt_storage["session_rollups"]
    normalized_tables["topics"] = rebuilt_storage["topics"]
    normalized_tables["capsules"] = rebuilt_storage["capsules"]
    normalized_tables["embedding_index_metadata"] = rebuilt_storage["embedding_index_metadata"]

    tmp_parquet = parquet_dir.parent / f"{parquet_dir.name}.tmp-schema-{timestamp}"
    if tmp_parquet.exists():
        await asyncio.to_thread(shutil.rmtree, tmp_parquet)
    for table, frame in normalized_tables.items():
        await asyncio.to_thread(_write_table, frame, table, tmp_parquet, timestamp)

    if parquet_dir.exists():
        await asyncio.to_thread(shutil.rmtree, parquet_dir)
    await asyncio.to_thread(tmp_parquet.rename, parquet_dir)

    metadata = DataFrameMetadataStore(metadata_parquet_path)
    await metadata.save_checkpoint(target_version, slot=SCHEMA_VERSION_SLOT)

    report_payload = {
        "timestamp": timestamp,
        "source_version": plan.source_version,
        "target_version": target_version,
        "reason": plan.reason,
        "tables": [asdict(item) for item in plan.tables],
        "backup_dir": str(backup_path) if backup_path else None,
    }
    report_dir = data_root / "checkpoints"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"schema-migration-report-{timestamp}.json"
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    applied_plan = SchemaMigrationPlan(
        source_version=plan.source_version,
        target_version=target_version,
        needs_migration=False,
        needs_metadata_update=False,
        tables=[
            TableMigrationPlan(
                table=item.table,
                file_count=item.file_count,
                row_count=item.row_count,
                missing_columns=item.missing_columns,
                needs_rewrite=False,
            )
            for item in plan.tables
        ],
        reason="migration applied",
    )
    return SchemaMigrationResult(
        plan=applied_plan,
        applied=True,
        backup_dir=str(backup_path) if backup_path else None,
        report_path=str(report_path),
    )


async def auto_migrate_if_needed(data_root: Path, parquet_dir: Path, metadata_parquet_path: Path) -> None:
    if not _bool_env("CLAWDB_SCHEMA_AUTO_MIGRATE", True):
        return
    target_version = int(os.getenv("CLAWDB_SCHEMA_VERSION", str(CURRENT_SCHEMA_VERSION)))
    should_backup = _bool_env("CLAWDB_SCHEMA_MIGRATE_BACKUP", True)
    raw_backup_dir = os.getenv("CLAWDB_SCHEMA_MIGRATE_BACKUP_DIR", "").strip()
    backup_dir = Path(raw_backup_dir).expanduser().resolve() if raw_backup_dir else None
    result = await migrate_schema(
        data_root=data_root,
        parquet_dir=parquet_dir,
        metadata_parquet_path=metadata_parquet_path,
        target_version=target_version,
        dry_run=False,
        force=False,
        backup=should_backup,
        backup_dir=backup_dir,
    )
    if result.applied:
        print(
            json.dumps(
                {
                    "event": "schema.migration.applied",
                    "target_version": target_version,
                    "backup_dir": result.backup_dir,
                    "report_path": result.report_path,
                }
            )
        )


def _plan_to_dict(plan: SchemaMigrationPlan) -> Dict[str, object]:
    return {
        "source_version": plan.source_version,
        "target_version": plan.target_version,
        "needs_migration": plan.needs_migration,
        "needs_metadata_update": plan.needs_metadata_update,
        "reason": plan.reason,
        "tables": [asdict(item) for item in plan.tables],
    }


async def _run_cli_async(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root).expanduser().resolve()
    parquet_dir = (
        Path(args.parquet_dir).expanduser().resolve()
        if args.parquet_dir
        else data_root / "parquet"
    )
    metadata_parquet_path = (
        Path(args.metadata_parquet).expanduser().resolve()
        if args.metadata_parquet
        else data_root / "checkpoints" / "metadata.parquet"
    )
    backup_dir = Path(args.backup_dir).expanduser().resolve() if args.backup_dir else None
    result = await migrate_schema(
        data_root=data_root,
        parquet_dir=parquet_dir,
        metadata_parquet_path=metadata_parquet_path,
        target_version=int(args.target_version),
        dry_run=bool(args.dry_run),
        force=bool(args.force),
        backup=bool(args.backup),
        backup_dir=backup_dir,
    )
    payload = {
        "applied": result.applied,
        "backup_dir": result.backup_dir,
        "report_path": result.report_path,
        "plan": _plan_to_dict(result.plan),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Applied: {payload['applied']}")
        print(f"Source version: {result.plan.source_version}")
        print(f"Target version: {result.plan.target_version}")
        print(f"Needs migration: {result.plan.needs_migration}")
        print(f"Reason: {result.plan.reason}")
        if result.backup_dir:
            print(f"Backup: {result.backup_dir}")
        if result.report_path:
            print(f"Report: {result.report_path}")
        for table in result.plan.tables:
            missing = ",".join(table.missing_columns) if table.missing_columns else "-"
            print(
                f"- {table.table}: rows={table.row_count} files={table.file_count} "
                f"needs_rewrite={table.needs_rewrite} missing={missing}"
            )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m clawdb.migrate",
        description="Migrate clawdb parquet schema forward without losing history.",
    )
    parser.add_argument("--data-root", default="data", help="ClawDB data root (default: data)")
    parser.add_argument("--parquet-dir", default="", help="Override parquet directory path")
    parser.add_argument(
        "--metadata-parquet",
        default="",
        help="Override metadata parquet path (default: <data-root>/checkpoints/metadata.parquet)",
    )
    parser.add_argument(
        "--target-version",
        type=int,
        default=CURRENT_SCHEMA_VERSION,
        help=f"Target schema version (default: {CURRENT_SCHEMA_VERSION})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print migration plan")
    parser.add_argument("--force", action="store_true", help="Rewrite parquet even when plan is clean")
    parser.add_argument(
        "--no-backup",
        dest="backup",
        action="store_false",
        help="Disable pre-migration backup snapshot",
    )
    parser.add_argument(
        "--backup-dir",
        default="",
        help="Backup output directory (default: <data-root>/backups/schema-migrations)",
    )
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    parser.set_defaults(backup=True)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_run_cli_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
