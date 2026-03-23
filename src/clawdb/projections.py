from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from .lineage import RAW_PROJECTION_KIND
from .topics import DEFAULT_TOPIC_VECTOR_DIM, _vectorize


PROJECTION_SUMMARY_MAX_CHARS = 4000
PROJECTION_SNIPPET_LIMIT = 3

PROJECTIONS_COLUMNS = [
    "projection_id",
    "tenant_id",
    "session_id",
    "projection_kind",
    "projection_scope",
    "visibility",
    "chat_type",
    "native_session_id",
    "native_session_ids_json",
    "paired_projection_ids_json",
    "paired_session_ids_json",
    "paired_projection_scopes_json",
    "account_id",
    "account_key",
    "group_id",
    "group_chat_key",
    "sender_id",
    "sender_user_key",
    "topic_ids_json",
    "origin_message_count",
    "active_message_count",
    "deleted_message_count",
    "first_origin_message_id",
    "last_origin_message_id",
    "origin_message_ids_json",
    "source_first_ts",
    "source_last_ts",
    "summary",
    "vector_text",
    "vector_ref",
    "vector_dim",
    "vector_json",
    "source_hash",
    "updated_at",
]


def _utc_timestamp(value: object) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return pd.Timestamp.now(tz="UTC")
    return ts


def _normalize_text(value: object) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n")


def _safe_path_fragment(value: object) -> str:
    raw = str(value or "").strip() or "_"
    out: List[str] = []
    for char in raw:
        if char.isalnum() or char in {"-", "_", "."}:
            out.append(char)
        else:
            out.append("_")
    return "".join(out)


def _ordered_unique(values: Sequence[object]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _json_list(values: Sequence[object]) -> str:
    return json.dumps(_ordered_unique(values), separators=(",", ":"))


def _serialize_vector(text: str, dim: int) -> str:
    vec = [round(float(item), 8) for item in _vectorize(text, max(8, int(dim)))]
    return json.dumps(vec, separators=(",", ":"))


def _projection_id(tenant_id: str, projection_kind: str, session_id: str) -> str:
    return f"projection:{tenant_id}:{projection_kind}:{_safe_path_fragment(session_id)}"


def _source_hash(rows: Sequence[pd.Series]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row.get("origin_message_id") or row.get("message_id") or "").encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(str(row.get("projection_kind") or "").encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(_utc_timestamp(row.get("ts")).isoformat().encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(_normalize_text(row.get("content")).encode("utf-8"))
        digest.update(b"\x1e")
    return digest.hexdigest()


def _render_projection_summary(
    *,
    projection_id: str,
    projection_kind: str,
    projection_scope: str,
    visibility: str,
    message_count: int,
    active_message_count: int,
    deleted_message_count: int,
    native_session_ids: Sequence[str],
    paired_session_ids: Sequence[str],
    topic_ids: Sequence[str],
    source_rows: Sequence[pd.Series],
) -> str:
    header = (
        f"projection:{projection_id} "
        f"kind={projection_kind} "
        f"scope={projection_scope or '-'} "
        f"visibility={visibility or '-'} "
        f"messages={int(message_count)} "
        f"active={int(active_message_count)} "
        f"deleted={int(deleted_message_count)} "
        f"native_sessions={','.join(_ordered_unique(native_session_ids)[:4]) or '-'} "
        f"paired_sessions={','.join(_ordered_unique(paired_session_ids)[:4]) or '-'} "
        f"topics={','.join(_ordered_unique(topic_ids)[:4]) or '-'}"
    )
    snippets: List[str] = []
    for row in list(source_rows)[-PROJECTION_SNIPPET_LIMIT:]:
        snippet = _normalize_text(row.get("content")).strip()
        if snippet:
            snippets.append(snippet[:240])
    if not snippets:
        return header[:PROJECTION_SUMMARY_MAX_CHARS]
    return (header + "\n" + "\n".join(f"- {item}" for item in snippets))[:PROJECTION_SUMMARY_MAX_CHARS]


def materialize_projection_state(
    messages_frame: pd.DataFrame,
    *,
    vector_dim: int = DEFAULT_TOPIC_VECTOR_DIM,
) -> pd.DataFrame:
    if messages_frame.empty:
        return pd.DataFrame(columns=PROJECTIONS_COLUMNS)
    resolved_dim = max(8, int(vector_dim))
    scoped = messages_frame.copy().reset_index(drop=True)
    scoped["tenant_id"] = scoped["tenant_id"].fillna("default").astype(str)
    scoped["message_id"] = scoped["message_id"].fillna("").astype(str)
    scoped["origin_message_id"] = scoped["origin_message_id"].fillna(scoped["message_id"]).astype(str)
    scoped["session_id"] = scoped["session_id"].fillna("").astype(str)
    scoped["projection_kind"] = scoped["projection_kind"].fillna("").astype(str)
    scoped["projection_scope"] = scoped["projection_scope"].fillna("").astype(str)
    scoped["visibility"] = scoped["visibility"].fillna("").astype(str)
    scoped["native_session_id"] = scoped["native_session_id"].fillna("").astype(str)
    scoped["chat_type"] = scoped["chat_type"].fillna("").astype(str)
    scoped["topic_id"] = scoped["topic_id"].fillna("default").astype(str)
    scoped["message_state"] = scoped["message_state"].fillna("active").astype(str)
    scoped["content"] = scoped["content"].fillna("").astype(str)
    scoped["ts"] = pd.to_datetime(scoped["ts"], utc=True, errors="coerce")
    scoped = scoped[
        (scoped["projection_kind"].astype(str) != RAW_PROJECTION_KIND)
        & (scoped["session_id"].astype(str) != "")
        & (scoped["projection_scope"].astype(str) != "")
        & scoped["ts"].notna()
    ].copy()
    if scoped.empty:
        return pd.DataFrame(columns=PROJECTIONS_COLUMNS)

    raw_rows = messages_frame.copy().reset_index(drop=True)
    raw_rows["tenant_id"] = raw_rows["tenant_id"].fillna("default").astype(str)
    raw_rows["message_id"] = raw_rows["message_id"].fillna("").astype(str)
    raw_rows["origin_message_id"] = raw_rows["origin_message_id"].fillna(raw_rows["message_id"]).astype(str)
    raw_rows["projection_kind"] = raw_rows["projection_kind"].fillna("").astype(str)
    raw_rows["native_session_id"] = raw_rows["native_session_id"].fillna("").astype(str)
    raw_rows = raw_rows[raw_rows["projection_kind"].astype(str) == RAW_PROJECTION_KIND].copy()
    raw_lookup: Dict[Tuple[str, str], str] = {}
    for _, row in raw_rows.iterrows():
        raw_lookup[(str(row["tenant_id"]), str(row["origin_message_id"]))] = str(row.get("native_session_id") or "")

    projection_key_by_origin: Dict[Tuple[str, str], List[Tuple[str, str, str, str]]] = {}
    for _, row in scoped.iterrows():
        key = (
            str(row["tenant_id"]),
            str(row["session_id"]),
            str(row["projection_kind"]),
            str(row["projection_scope"]),
        )
        projection_key_by_origin.setdefault(
            (str(row["tenant_id"]), str(row["origin_message_id"])),
            [],
        ).append(key)

    rows: List[Dict[str, object]] = []
    materialized_at = pd.Timestamp.utcnow()
    for (tenant_id, session_id, projection_kind, projection_scope), group in scoped.groupby(
        ["tenant_id", "session_id", "projection_kind", "projection_scope"],
        sort=True,
    ):
        ordered = group.sort_values(["ts", "origin_message_id"], kind="stable").reset_index(drop=True)
        origin_ids = _ordered_unique(ordered["origin_message_id"].astype(str).tolist())
        native_session_ids = _ordered_unique(
            [
                str(row.get("native_session_id") or raw_lookup.get((str(tenant_id), str(row.get("origin_message_id") or "")), ""))
                for _, row in ordered.iterrows()
            ]
        )
        projection_id = _projection_id(str(tenant_id), str(projection_kind), str(session_id))
        paired_keys: List[Tuple[str, str, str, str]] = []
        for origin_id in origin_ids:
            paired_keys.extend(projection_key_by_origin.get((str(tenant_id), str(origin_id)), []))
        paired_keys = [
            item
            for item in paired_keys
            if item != (str(tenant_id), str(session_id), str(projection_kind), str(projection_scope))
        ]
        paired_projection_ids = [
            _projection_id(item[0], item[2], item[1])
            for item in paired_keys
        ]
        paired_session_ids = [item[1] for item in paired_keys]
        paired_projection_scopes = [item[3] for item in paired_keys]
        source_records = list(ordered.to_dict("records"))
        source_hash = _source_hash(source_records)
        active_message_count = int((ordered["message_state"].astype(str) != "deleted").sum())
        deleted_message_count = int((ordered["message_state"].astype(str) == "deleted").sum())
        summary = _render_projection_summary(
            projection_id=projection_id,
            projection_kind=str(projection_kind),
            projection_scope=str(projection_scope),
            visibility=str(ordered.iloc[0].get("visibility") or ""),
            message_count=int(ordered.shape[0]),
            active_message_count=active_message_count,
            deleted_message_count=deleted_message_count,
            native_session_ids=native_session_ids,
            paired_session_ids=paired_session_ids,
            topic_ids=ordered["topic_id"].astype(str).tolist(),
            source_rows=source_records,
        )
        rows.append(
            {
                "projection_id": projection_id,
                "tenant_id": str(tenant_id),
                "session_id": str(session_id),
                "projection_kind": str(projection_kind),
                "projection_scope": str(projection_scope),
                "visibility": str(ordered.iloc[0].get("visibility") or ""),
                "chat_type": str(ordered.iloc[0].get("chat_type") or ""),
                "native_session_id": native_session_ids[0] if native_session_ids else "",
                "native_session_ids_json": _json_list(native_session_ids),
                "paired_projection_ids_json": _json_list(paired_projection_ids),
                "paired_session_ids_json": _json_list(paired_session_ids),
                "paired_projection_scopes_json": _json_list(paired_projection_scopes),
                "account_id": str(ordered.iloc[0].get("account_id") or ""),
                "account_key": str(ordered.iloc[0].get("account_key") or ""),
                "group_id": str(ordered.iloc[0].get("group_id") or ""),
                "group_chat_key": str(ordered.iloc[0].get("group_chat_key") or ""),
                "sender_id": str(ordered.iloc[0].get("sender_id") or ""),
                "sender_user_key": str(ordered.iloc[0].get("sender_user_key") or ""),
                "topic_ids_json": _json_list(ordered["topic_id"].astype(str).tolist()),
                "origin_message_count": len(origin_ids),
                "active_message_count": active_message_count,
                "deleted_message_count": deleted_message_count,
                "first_origin_message_id": str(origin_ids[0] if origin_ids else ""),
                "last_origin_message_id": str(origin_ids[-1] if origin_ids else ""),
                "origin_message_ids_json": _json_list(origin_ids),
                "source_first_ts": _utc_timestamp(ordered.iloc[0]["ts"]),
                "source_last_ts": _utc_timestamp(ordered.iloc[-1]["ts"]),
                "summary": summary,
                "vector_text": summary,
                "vector_ref": f"projection:{projection_id}:{source_hash[:12]}",
                "vector_dim": resolved_dim,
                "vector_json": _serialize_vector(summary, resolved_dim),
                "source_hash": source_hash,
                "updated_at": materialized_at,
            }
        )

    if not rows:
        return pd.DataFrame(columns=PROJECTIONS_COLUMNS)
    frame = pd.DataFrame(rows, columns=PROJECTIONS_COLUMNS)
    frame["projection_id"] = frame["projection_id"].fillna("").astype(str)
    frame["tenant_id"] = frame["tenant_id"].fillna("default").astype(str)
    frame["session_id"] = frame["session_id"].fillna("").astype(str)
    frame["projection_kind"] = frame["projection_kind"].fillna("").astype(str)
    frame["projection_scope"] = frame["projection_scope"].fillna("").astype(str)
    frame["visibility"] = frame["visibility"].fillna("").astype(str)
    frame["chat_type"] = frame["chat_type"].fillna("").astype(str)
    frame["native_session_id"] = frame["native_session_id"].fillna("").astype(str)
    frame["native_session_ids_json"] = frame["native_session_ids_json"].fillna("[]").astype(str)
    frame["paired_projection_ids_json"] = frame["paired_projection_ids_json"].fillna("[]").astype(str)
    frame["paired_session_ids_json"] = frame["paired_session_ids_json"].fillna("[]").astype(str)
    frame["paired_projection_scopes_json"] = frame["paired_projection_scopes_json"].fillna("[]").astype(str)
    frame["account_id"] = frame["account_id"].fillna("").astype(str)
    frame["account_key"] = frame["account_key"].fillna("").astype(str)
    frame["group_id"] = frame["group_id"].fillna("").astype(str)
    frame["group_chat_key"] = frame["group_chat_key"].fillna("").astype(str)
    frame["sender_id"] = frame["sender_id"].fillna("").astype(str)
    frame["sender_user_key"] = frame["sender_user_key"].fillna("").astype(str)
    frame["topic_ids_json"] = frame["topic_ids_json"].fillna("[]").astype(str)
    frame["origin_message_count"] = pd.to_numeric(frame["origin_message_count"], errors="coerce").fillna(0).astype(int)
    frame["active_message_count"] = pd.to_numeric(frame["active_message_count"], errors="coerce").fillna(0).astype(int)
    frame["deleted_message_count"] = pd.to_numeric(frame["deleted_message_count"], errors="coerce").fillna(0).astype(int)
    frame["first_origin_message_id"] = frame["first_origin_message_id"].fillna("").astype(str)
    frame["last_origin_message_id"] = frame["last_origin_message_id"].fillna("").astype(str)
    frame["origin_message_ids_json"] = frame["origin_message_ids_json"].fillna("[]").astype(str)
    frame["source_first_ts"] = pd.to_datetime(frame["source_first_ts"], utc=True, errors="coerce")
    frame["source_last_ts"] = pd.to_datetime(frame["source_last_ts"], utc=True, errors="coerce")
    frame["summary"] = frame["summary"].fillna("").astype(str)
    frame["vector_text"] = frame["vector_text"].fillna("").astype(str)
    frame["vector_ref"] = frame["vector_ref"].fillna("").astype(str)
    frame["vector_dim"] = pd.to_numeric(frame["vector_dim"], errors="coerce").fillna(0).astype(int)
    frame["vector_json"] = frame["vector_json"].fillna("[]").astype(str)
    frame["source_hash"] = frame["source_hash"].fillna("").astype(str)
    frame["updated_at"] = pd.to_datetime(frame["updated_at"], utc=True, errors="coerce")
    frame = frame.sort_values(
        ["tenant_id", "projection_kind", "session_id", "updated_at"],
        ascending=[True, True, True, True],
        kind="stable",
    ).drop_duplicates(
        subset=["tenant_id", "projection_id"],
        keep="last",
    )
    return frame[PROJECTIONS_COLUMNS].reset_index(drop=True)
