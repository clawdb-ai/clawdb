from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Sequence

import pandas as pd

from .lineage import MESSAGE_STATE_DELETED, RAW_PROJECTION_KIND
from .topics import DEFAULT_TOPIC_VECTOR_DIM, _vectorize


BELIEF_SUMMARY_MAX_CHARS = 4000
BELIEF_SNIPPET_LIMIT = 4

BELIEFS_COLUMNS = [
    "belief_id",
    "tenant_id",
    "scope_type",
    "scope_key",
    "session_id",
    "topic_id",
    "group_id",
    "projection_kind",
    "projection_scope",
    "first_ts",
    "last_ts",
    "raw_message_count",
    "first_origin_message_id",
    "last_origin_message_id",
    "source_message_ids_json",
    "source_session_ids_json",
    "topic_ids_json",
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


def _belief_id(tenant_id: str, scope_type: str, scope_key: str) -> str:
    return f"l0:{tenant_id}:{scope_type}_{_safe_path_fragment(scope_key)}"


def _source_hash(rows: Sequence[pd.Series]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row.get("origin_message_id") or row.get("message_id") or "").encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(_utc_timestamp(row.get("ts")).isoformat().encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(_normalize_text(row.get("content")).encode("utf-8"))
        digest.update(b"\x1e")
    return digest.hexdigest()


def _render_belief_summary(
    *,
    belief_id: str,
    scope_type: str,
    scope_key: str,
    raw_message_count: int,
    first_origin_message_id: str,
    last_origin_message_id: str,
    first_ts: pd.Timestamp,
    last_ts: pd.Timestamp,
    topic_ids: Sequence[str],
    source_rows: Sequence[pd.Series],
    projection_kind: str,
    projection_scope: str,
) -> str:
    header = (
        f"belief:{belief_id} "
        f"scope={scope_type}:{scope_key} "
        f"messages={int(raw_message_count)} "
        f"range={_utc_timestamp(first_ts).isoformat()}..{_utc_timestamp(last_ts).isoformat()} "
        f"topics={','.join(_ordered_unique(topic_ids)[:4]) or '-'} "
        f"first={first_origin_message_id or '-'} "
        f"last={last_origin_message_id or '-'}"
    )
    if projection_kind:
        header += f" projection={projection_kind}"
    if projection_scope:
        header += f" projection_scope={projection_scope}"
    snippets: List[str] = []
    for row in list(source_rows)[-BELIEF_SNIPPET_LIMIT:]:
        snippet = _normalize_text(row.get("content")).strip()
        if snippet:
            snippets.append(snippet[:240])
    if not snippets:
        return header[:BELIEF_SUMMARY_MAX_CHARS]
    return (header + "\n" + "\n".join(f"- {item}" for item in snippets))[:BELIEF_SUMMARY_MAX_CHARS]


def _raw_rows(messages_frame: pd.DataFrame) -> pd.DataFrame:
    if messages_frame.empty:
        return pd.DataFrame()
    scoped = messages_frame.copy().reset_index(drop=True)
    scoped["tenant_id"] = scoped["tenant_id"].fillna("default").astype(str)
    scoped["message_id"] = scoped["message_id"].fillna("").astype(str)
    scoped["origin_message_id"] = scoped["origin_message_id"].fillna(scoped["message_id"]).astype(str)
    scoped["session_id"] = scoped["session_id"].fillna("").astype(str)
    scoped["native_session_id"] = scoped["native_session_id"].fillna("").astype(str)
    scoped["topic_id"] = scoped["topic_id"].fillna("default").astype(str)
    scoped["projection_kind"] = scoped["projection_kind"].fillna("").astype(str)
    scoped["message_state"] = scoped["message_state"].fillna("active").astype(str)
    scoped["content"] = scoped["content"].fillna("").astype(str)
    scoped["ts"] = pd.to_datetime(scoped["ts"], utc=True, errors="coerce")
    scoped = scoped[
        (scoped["projection_kind"].astype(str) == RAW_PROJECTION_KIND)
        & (scoped["message_state"].astype(str) != MESSAGE_STATE_DELETED)
        & scoped["ts"].notna()
    ].copy()
    return scoped


def _projection_rows(messages_frame: pd.DataFrame) -> pd.DataFrame:
    if messages_frame.empty:
        return pd.DataFrame()
    scoped = messages_frame.copy().reset_index(drop=True)
    scoped["tenant_id"] = scoped["tenant_id"].fillna("default").astype(str)
    scoped["message_id"] = scoped["message_id"].fillna("").astype(str)
    scoped["origin_message_id"] = scoped["origin_message_id"].fillna(scoped["message_id"]).astype(str)
    scoped["session_id"] = scoped["session_id"].fillna("").astype(str)
    scoped["projection_kind"] = scoped["projection_kind"].fillna("").astype(str)
    scoped["projection_scope"] = scoped["projection_scope"].fillna("").astype(str)
    scoped["message_state"] = scoped["message_state"].fillna("active").astype(str)
    scoped["ts"] = pd.to_datetime(scoped["ts"], utc=True, errors="coerce")
    scoped = scoped[
        (scoped["projection_kind"].astype(str) != RAW_PROJECTION_KIND)
        & (scoped["message_state"].astype(str) != MESSAGE_STATE_DELETED)
        & (scoped["session_id"].astype(str) != "")
        & scoped["ts"].notna()
    ].copy()
    return scoped


def materialize_l0_beliefs(
    messages_frame: pd.DataFrame,
    *,
    vector_dim: int = DEFAULT_TOPIC_VECTOR_DIM,
) -> pd.DataFrame:
    raw_rows = _raw_rows(messages_frame)
    if raw_rows.empty:
        return pd.DataFrame(columns=BELIEFS_COLUMNS)

    resolved_dim = max(8, int(vector_dim))
    projection_rows = _projection_rows(messages_frame)
    rows: List[Dict[str, object]] = []
    materialized_at = pd.Timestamp.utcnow()

    for tenant_id, group in raw_rows.groupby("tenant_id", sort=True):
        ordered = group.sort_values(["ts", "origin_message_id"], kind="stable").reset_index(drop=True)
        origin_ids = ordered["origin_message_id"].astype(str).tolist()
        topic_ids = ordered["topic_id"].astype(str).tolist()
        session_ids = [
            str(row.get("native_session_id") or row.get("session_id") or "")
            for _, row in ordered.iterrows()
        ]
        source_records = list(ordered.to_dict("records"))
        source_hash = _source_hash(source_records)
        belief_id = _belief_id(str(tenant_id), "tenant", str(tenant_id))
        summary = _render_belief_summary(
            belief_id=belief_id,
            scope_type="tenant",
            scope_key=str(tenant_id),
            raw_message_count=int(ordered.shape[0]),
            first_origin_message_id=str(origin_ids[0] if origin_ids else ""),
            last_origin_message_id=str(origin_ids[-1] if origin_ids else ""),
            first_ts=_utc_timestamp(ordered.iloc[0]["ts"]),
            last_ts=_utc_timestamp(ordered.iloc[-1]["ts"]),
            topic_ids=topic_ids,
            source_rows=source_records,
            projection_kind="",
            projection_scope="",
        )
        rows.append(
            {
                "belief_id": belief_id,
                "tenant_id": str(tenant_id),
                "scope_type": "tenant",
                "scope_key": str(tenant_id),
                "session_id": "",
                "topic_id": "",
                "group_id": "",
                "projection_kind": "",
                "projection_scope": "",
                "first_ts": _utc_timestamp(ordered.iloc[0]["ts"]),
                "last_ts": _utc_timestamp(ordered.iloc[-1]["ts"]),
                "raw_message_count": int(ordered.shape[0]),
                "first_origin_message_id": str(origin_ids[0] if origin_ids else ""),
                "last_origin_message_id": str(origin_ids[-1] if origin_ids else ""),
                "source_message_ids_json": _json_list(origin_ids),
                "source_session_ids_json": _json_list(session_ids),
                "topic_ids_json": _json_list(topic_ids),
                "summary": summary,
                "vector_text": summary,
                "vector_ref": f"belief:{belief_id}:{source_hash[:12]}",
                "vector_dim": resolved_dim,
                "vector_json": _serialize_vector(summary, resolved_dim),
                "source_hash": source_hash,
                "updated_at": materialized_at,
            }
        )

    if not projection_rows.empty:
        for (tenant_id, session_id), group in projection_rows.groupby(["tenant_id", "session_id"], sort=True):
            ordered_projection = group.sort_values(["ts", "origin_message_id"], kind="stable").reset_index(drop=True)
            origin_ids = _ordered_unique(ordered_projection["origin_message_id"].astype(str).tolist())
            supporting = raw_rows[
                (raw_rows["tenant_id"].astype(str) == str(tenant_id))
                & (raw_rows["origin_message_id"].astype(str).isin(origin_ids))
            ].copy()
            if supporting.empty:
                continue
            supporting = supporting.sort_values(["ts", "origin_message_id"], kind="stable").reset_index(drop=True)
            topic_ids = supporting["topic_id"].astype(str).tolist()
            native_session_ids = [
                str(item)
                for item in supporting["native_session_id"].fillna(supporting["session_id"]).astype(str).tolist()
            ]
            projection_kind = str(ordered_projection.iloc[0].get("projection_kind") or "")
            projection_scope = str(ordered_projection.iloc[0].get("projection_scope") or "")
            group_id = str(ordered_projection.iloc[0].get("group_id") or "")
            belief_id = _belief_id(str(tenant_id), "session", str(session_id))
            supporting_records = list(supporting.to_dict("records"))
            source_hash = _source_hash(supporting_records)
            summary = _render_belief_summary(
                belief_id=belief_id,
                scope_type="session",
                scope_key=str(session_id),
                raw_message_count=int(supporting.shape[0]),
                first_origin_message_id=str(origin_ids[0] if origin_ids else ""),
                last_origin_message_id=str(origin_ids[-1] if origin_ids else ""),
                first_ts=_utc_timestamp(supporting.iloc[0]["ts"]),
                last_ts=_utc_timestamp(supporting.iloc[-1]["ts"]),
                topic_ids=topic_ids,
                source_rows=supporting_records,
                projection_kind=projection_kind,
                projection_scope=projection_scope,
            )
            rows.append(
                {
                    "belief_id": belief_id,
                    "tenant_id": str(tenant_id),
                    "scope_type": "session",
                    "scope_key": str(session_id),
                    "session_id": str(session_id),
                    "topic_id": "",
                    "group_id": group_id,
                    "projection_kind": projection_kind,
                    "projection_scope": projection_scope,
                    "first_ts": _utc_timestamp(supporting.iloc[0]["ts"]),
                    "last_ts": _utc_timestamp(supporting.iloc[-1]["ts"]),
                    "raw_message_count": int(supporting.shape[0]),
                    "first_origin_message_id": str(origin_ids[0] if origin_ids else ""),
                    "last_origin_message_id": str(origin_ids[-1] if origin_ids else ""),
                    "source_message_ids_json": _json_list(origin_ids),
                    "source_session_ids_json": _json_list(native_session_ids),
                    "topic_ids_json": _json_list(topic_ids),
                    "summary": summary,
                    "vector_text": summary,
                    "vector_ref": f"belief:{belief_id}:{source_hash[:12]}",
                    "vector_dim": resolved_dim,
                    "vector_json": _serialize_vector(summary, resolved_dim),
                    "source_hash": source_hash,
                    "updated_at": materialized_at,
                }
            )

    for (tenant_id, topic_id), group in raw_rows.groupby(["tenant_id", "topic_id"], sort=True):
        ordered = group.sort_values(["ts", "origin_message_id"], kind="stable").reset_index(drop=True)
        origin_ids = ordered["origin_message_id"].astype(str).tolist()
        session_ids = ordered["native_session_id"].fillna(ordered["session_id"]).astype(str).tolist()
        belief_id = _belief_id(str(tenant_id), "topic", str(topic_id))
        supporting_records = list(ordered.to_dict("records"))
        source_hash = _source_hash(supporting_records)
        summary = _render_belief_summary(
            belief_id=belief_id,
            scope_type="topic",
            scope_key=str(topic_id),
            raw_message_count=int(ordered.shape[0]),
            first_origin_message_id=str(origin_ids[0] if origin_ids else ""),
            last_origin_message_id=str(origin_ids[-1] if origin_ids else ""),
            first_ts=_utc_timestamp(ordered.iloc[0]["ts"]),
            last_ts=_utc_timestamp(ordered.iloc[-1]["ts"]),
            topic_ids=[str(topic_id)],
            source_rows=supporting_records,
            projection_kind="",
            projection_scope="",
        )
        rows.append(
            {
                "belief_id": belief_id,
                "tenant_id": str(tenant_id),
                "scope_type": "topic",
                "scope_key": str(topic_id),
                "session_id": "",
                "topic_id": str(topic_id),
                "group_id": "",
                "projection_kind": "",
                "projection_scope": "",
                "first_ts": _utc_timestamp(ordered.iloc[0]["ts"]),
                "last_ts": _utc_timestamp(ordered.iloc[-1]["ts"]),
                "raw_message_count": int(ordered.shape[0]),
                "first_origin_message_id": str(origin_ids[0] if origin_ids else ""),
                "last_origin_message_id": str(origin_ids[-1] if origin_ids else ""),
                "source_message_ids_json": _json_list(origin_ids),
                "source_session_ids_json": _json_list(session_ids),
                "topic_ids_json": _json_list([topic_id]),
                "summary": summary,
                "vector_text": summary,
                "vector_ref": f"belief:{belief_id}:{source_hash[:12]}",
                "vector_dim": resolved_dim,
                "vector_json": _serialize_vector(summary, resolved_dim),
                "source_hash": source_hash,
                "updated_at": materialized_at,
            }
        )

    if not rows:
        return pd.DataFrame(columns=BELIEFS_COLUMNS)
    frame = pd.DataFrame(rows, columns=BELIEFS_COLUMNS)
    frame["tenant_id"] = frame["tenant_id"].fillna("default").astype(str)
    frame["belief_id"] = frame["belief_id"].fillna("").astype(str)
    frame["scope_type"] = frame["scope_type"].fillna("").astype(str)
    frame["scope_key"] = frame["scope_key"].fillna("").astype(str)
    frame["session_id"] = frame["session_id"].fillna("").astype(str)
    frame["topic_id"] = frame["topic_id"].fillna("").astype(str)
    frame["group_id"] = frame["group_id"].fillna("").astype(str)
    frame["projection_kind"] = frame["projection_kind"].fillna("").astype(str)
    frame["projection_scope"] = frame["projection_scope"].fillna("").astype(str)
    frame["first_ts"] = pd.to_datetime(frame["first_ts"], utc=True, errors="coerce")
    frame["last_ts"] = pd.to_datetime(frame["last_ts"], utc=True, errors="coerce")
    frame["raw_message_count"] = pd.to_numeric(frame["raw_message_count"], errors="coerce").fillna(0).astype(int)
    frame["first_origin_message_id"] = frame["first_origin_message_id"].fillna("").astype(str)
    frame["last_origin_message_id"] = frame["last_origin_message_id"].fillna("").astype(str)
    frame["source_message_ids_json"] = frame["source_message_ids_json"].fillna("[]").astype(str)
    frame["source_session_ids_json"] = frame["source_session_ids_json"].fillna("[]").astype(str)
    frame["topic_ids_json"] = frame["topic_ids_json"].fillna("[]").astype(str)
    frame["summary"] = frame["summary"].fillna("").astype(str)
    frame["vector_text"] = frame["vector_text"].fillna("").astype(str)
    frame["vector_ref"] = frame["vector_ref"].fillna("").astype(str)
    frame["vector_dim"] = pd.to_numeric(frame["vector_dim"], errors="coerce").fillna(0).astype(int)
    frame["vector_json"] = frame["vector_json"].fillna("[]").astype(str)
    frame["source_hash"] = frame["source_hash"].fillna("").astype(str)
    frame["updated_at"] = pd.to_datetime(frame["updated_at"], utc=True, errors="coerce")
    frame = frame.sort_values(
        ["tenant_id", "scope_type", "scope_key", "updated_at"],
        ascending=[True, True, True, True],
        kind="stable",
    ).drop_duplicates(
        subset=["tenant_id", "belief_id"],
        keep="last",
    )
    return frame[BELIEFS_COLUMNS].reset_index(drop=True)
