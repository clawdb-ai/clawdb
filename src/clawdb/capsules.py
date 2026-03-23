from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from .lineage import MESSAGE_STATE_DELETED, RAW_PROJECTION_KIND
from .topics import DEFAULT_TOPIC_VECTOR_DIM, _vectorize


CAPSULE_THRESHOLD_CHARS = 100_000
CAPSULE_STATE_OPEN = "open"
CAPSULE_STATE_SEALED = "sealed"
CAPSULE_SUMMARY_MAX_CHARS = 4000
CAPSULE_SNIPPET_LIMIT = 4

CAPSULES_COLUMNS = [
    "capsule_id",
    "tenant_id",
    "session_id",
    "topic_id",
    "topic_path",
    "capsule_ordinal",
    "capsule_state",
    "summary",
    "level",
    "score",
    "source_message_count",
    "source_body_char_count",
    "threshold_body_char_count",
    "first_origin_message_id",
    "last_origin_message_id",
    "source_message_ids_json",
    "source_session_ids_json",
    "source_topic_ids_json",
    "active_message_count",
    "edited_message_count",
    "topic_message_count",
    "topic_body_char_count",
    "source_first_ts",
    "source_last_ts",
    "opened_at",
    "sealed_at",
    "prev_capsule_id",
    "next_capsule_id",
    "back_link_ids_json",
    "forward_link_ids_json",
    "pointer_json",
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


def _normalize_body_text(value: object) -> str:
    text = str(value or "")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def capsule_body_char_count(value: object) -> int:
    return len(_normalize_body_text(value))


def _json_list(values: Sequence[object]) -> str:
    seen: set[str] = set()
    ordered: List[str] = []
    for item in values:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return json.dumps(ordered, separators=(",", ":"))


def _serialize_vector(text: str, dim: int) -> str:
    vec = [round(float(item), 8) for item in _vectorize(text, max(8, int(dim)))]
    return json.dumps(vec, separators=(",", ":"))


def _topic_lookup(topics_frame: pd.DataFrame) -> Tuple[Dict[Tuple[str, str], str], Dict[Tuple[str, str], str]]:
    canonical_lookup: Dict[Tuple[str, str], str] = {}
    topic_path_lookup: Dict[Tuple[str, str], str] = {}
    if topics_frame.empty:
        return canonical_lookup, topic_path_lookup
    scoped = topics_frame.copy().reset_index(drop=True)
    scoped["tenant_id"] = scoped["tenant_id"].fillna("default").astype(str)
    scoped["topic_id"] = scoped["topic_id"].fillna("default").astype(str)
    scoped["canonical_topic_id"] = scoped["canonical_topic_id"].fillna(scoped["topic_id"]).astype(str)
    scoped["topic_path"] = scoped["topic_path"].fillna(scoped["canonical_topic_id"]).astype(str)
    for _, row in scoped.iterrows():
        tenant_id = str(row["tenant_id"])
        topic_id = str(row["topic_id"])
        canonical_topic_id = str(row["canonical_topic_id"] or topic_id)
        canonical_lookup[(tenant_id, topic_id)] = canonical_topic_id
    for _, row in scoped.iterrows():
        tenant_id = str(row["tenant_id"])
        topic_id = str(row["topic_id"])
        canonical_topic_id = str(row["canonical_topic_id"] or topic_id)
        topic_path = str(row["topic_path"] or canonical_topic_id)
        key = (tenant_id, canonical_topic_id)
        if key not in topic_path_lookup or topic_id == canonical_topic_id:
            topic_path_lookup[key] = topic_path
    return canonical_lookup, topic_path_lookup


def _source_hash(rows: Sequence[pd.Series]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row.get("origin_message_id") or row.get("message_id") or "").encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(_utc_timestamp(row.get("ts")).isoformat().encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(_normalize_body_text(row.get("content")).encode("utf-8"))
        digest.update(b"\x1e")
    return digest.hexdigest()


def _render_capsule_summary(
    *,
    capsule_id: str,
    topic_id: str,
    topic_path: str,
    capsule_ordinal: int,
    capsule_state: str,
    source_message_count: int,
    source_body_char_count: int,
    threshold_body_char_count: int,
    first_origin_message_id: str,
    last_origin_message_id: str,
    source_first_ts: pd.Timestamp,
    source_last_ts: pd.Timestamp,
    source_rows: Sequence[pd.Series],
) -> str:
    header = (
        f"capsule:{capsule_id} "
        f"topic={topic_id} "
        f"path={topic_path or topic_id} "
        f"ordinal={int(capsule_ordinal)} "
        f"state={capsule_state} "
        f"messages={int(source_message_count)} "
        f"chars={int(source_body_char_count)}/{int(threshold_body_char_count)} "
        f"range={_utc_timestamp(source_first_ts).isoformat()}..{_utc_timestamp(source_last_ts).isoformat()} "
        f"first={first_origin_message_id or '-'} "
        f"last={last_origin_message_id or '-'}"
    )
    snippets: List[str] = []
    for row in list(source_rows)[-CAPSULE_SNIPPET_LIMIT:]:
        snippet = _normalize_body_text(row.get("content")).strip()
        if not snippet:
            continue
        snippets.append(snippet[:240])
    if not snippets:
        return header[:CAPSULE_SUMMARY_MAX_CHARS]
    return (header + "\n" + "\n".join(f"- {snippet}" for snippet in snippets))[:CAPSULE_SUMMARY_MAX_CHARS]


def materialize_capsule_lifecycle(
    messages_frame: pd.DataFrame,
    *,
    topics_frame: pd.DataFrame,
    vector_dim: int = DEFAULT_TOPIC_VECTOR_DIM,
    threshold_chars: int = CAPSULE_THRESHOLD_CHARS,
) -> pd.DataFrame:
    if messages_frame.empty:
        return pd.DataFrame(columns=CAPSULES_COLUMNS)
    resolved_dim = max(8, int(vector_dim))
    resolved_threshold = max(1, int(threshold_chars))
    scoped = messages_frame.copy().reset_index(drop=True)
    if "projection_kind" not in scoped.columns or "message_state" not in scoped.columns:
        return pd.DataFrame(columns=CAPSULES_COLUMNS)
    scoped = scoped[scoped["projection_kind"].astype(str) == RAW_PROJECTION_KIND]
    if scoped.empty:
        return pd.DataFrame(columns=CAPSULES_COLUMNS)
    scoped["tenant_id"] = scoped["tenant_id"].fillna("default").astype(str)
    scoped["topic_id"] = scoped["topic_id"].fillna("default").astype(str)
    scoped["origin_message_id"] = scoped["origin_message_id"].fillna(scoped["message_id"]).astype(str)
    scoped["session_id"] = scoped["session_id"].fillna("").astype(str)
    scoped["native_session_id"] = scoped["native_session_id"].fillna(scoped["session_id"]).astype(str)
    if "source_topic_id" not in scoped.columns:
        scoped["source_topic_id"] = scoped["topic_id"]
    scoped["source_topic_id"] = scoped["source_topic_id"].fillna(scoped["topic_id"]).astype(str)
    scoped["message_state"] = scoped["message_state"].fillna("active").astype(str)
    scoped["ts"] = pd.to_datetime(scoped["ts"], utc=True, errors="coerce")
    scoped["updated_at"] = pd.to_datetime(scoped["updated_at"], utc=True, errors="coerce")
    scoped = scoped[scoped["ts"].notna()].copy()
    if scoped.empty:
        return pd.DataFrame(columns=CAPSULES_COLUMNS)
    canonical_lookup, topic_path_lookup = _topic_lookup(topics_frame)
    scoped["canonical_topic_id"] = scoped.apply(
        lambda row: canonical_lookup.get(
            (str(row["tenant_id"]), str(row["topic_id"])),
            str(row["topic_id"]),
        ),
        axis=1,
    )
    scoped["body_text"] = scoped["content"].apply(_normalize_body_text)
    scoped["body_char_count"] = scoped["body_text"].apply(capsule_body_char_count)
    live_rows = scoped[
        (scoped["message_state"].astype(str) != MESSAGE_STATE_DELETED) & (scoped["body_char_count"].astype(int) > 0)
    ].copy()
    if live_rows.empty:
        return pd.DataFrame(columns=CAPSULES_COLUMNS)

    rows: List[Dict[str, object]] = []
    for (tenant_id, canonical_topic_id), group in live_rows.groupby(["tenant_id", "canonical_topic_id"], sort=True):
        ordered = group.sort_values(["ts", "origin_message_id"], kind="stable").reset_index(drop=True)
        topic_path = topic_path_lookup.get((str(tenant_id), str(canonical_topic_id)), str(canonical_topic_id))
        topic_message_count = int(ordered.shape[0])
        topic_body_char_count = int(ordered["body_char_count"].astype(int).sum())
        capsule_sources: List[pd.Series] = []
        capsule_body_char_count_total = 0
        capsule_ordinal = 0

        def _emit_capsule(capsule_rows: Sequence[pd.Series], capsule_state: str) -> None:
            nonlocal capsule_ordinal
            if not capsule_rows:
                return
            capsule_ordinal += 1
            source_rows = list(capsule_rows)
            source_body_char_count = sum(int(item.get("body_char_count") or 0) for item in source_rows)
            first_row = source_rows[0]
            last_row = source_rows[-1]
            capsule_id = f"capsule:{tenant_id}:{canonical_topic_id}:{capsule_ordinal:04d}"
            source_hash = _source_hash(source_rows)
            source_session_ids = [
                item.get("native_session_id") or item.get("session_id") or ""
                for item in source_rows
            ]
            source_topic_ids = [
                item.get("source_topic_id") or item.get("topic_id") or ""
                for item in source_rows
            ]
            active_message_count = len(source_rows)
            edited_message_count = sum(
                1 for item in source_rows if str(item.get("message_state") or "") == "edited"
            )
            opened_at = _utc_timestamp(first_row.get("ts"))
            sealed_at = _utc_timestamp(last_row.get("ts")) if capsule_state == CAPSULE_STATE_SEALED else pd.NaT
            summary = _render_capsule_summary(
                capsule_id=capsule_id,
                topic_id=str(canonical_topic_id),
                topic_path=str(topic_path),
                capsule_ordinal=capsule_ordinal,
                capsule_state=capsule_state,
                source_message_count=len(source_rows),
                source_body_char_count=source_body_char_count,
                threshold_body_char_count=resolved_threshold,
                first_origin_message_id=str(first_row.get("origin_message_id") or ""),
                last_origin_message_id=str(last_row.get("origin_message_id") or ""),
                source_first_ts=_utc_timestamp(first_row.get("ts")),
                source_last_ts=_utc_timestamp(last_row.get("ts")),
                source_rows=source_rows,
            )
            rows.append(
                {
                    "capsule_id": capsule_id,
                    "tenant_id": str(tenant_id),
                    "session_id": f"topic:{canonical_topic_id}",
                    "topic_id": str(canonical_topic_id),
                    "topic_path": str(topic_path),
                    "capsule_ordinal": capsule_ordinal,
                    "capsule_state": capsule_state,
                    "summary": summary,
                    "level": "L2",
                    "score": min(1.0, float(source_body_char_count) / float(resolved_threshold)),
                    "source_message_count": len(source_rows),
                    "source_body_char_count": source_body_char_count,
                    "threshold_body_char_count": resolved_threshold,
                    "first_origin_message_id": str(first_row.get("origin_message_id") or ""),
                    "last_origin_message_id": str(last_row.get("origin_message_id") or ""),
                    "source_message_ids_json": _json_list(
                        [item.get("origin_message_id") or item.get("message_id") or "" for item in source_rows]
                    ),
                    "source_session_ids_json": _json_list(source_session_ids),
                    "source_topic_ids_json": _json_list(source_topic_ids),
                    "active_message_count": active_message_count,
                    "edited_message_count": edited_message_count,
                    "topic_message_count": topic_message_count,
                    "topic_body_char_count": topic_body_char_count,
                    "source_first_ts": _utc_timestamp(first_row.get("ts")),
                    "source_last_ts": _utc_timestamp(last_row.get("ts")),
                    "opened_at": opened_at,
                    "sealed_at": sealed_at,
                    "prev_capsule_id": "",
                    "next_capsule_id": "",
                    "back_link_ids_json": "[]",
                    "forward_link_ids_json": "[]",
                    "pointer_json": "{}",
                    "vector_text": summary,
                    "vector_ref": (
                        f"capsule:{tenant_id}:{canonical_topic_id}:{capsule_ordinal:04d}:{source_hash[:12]}"
                    ),
                    "vector_dim": resolved_dim,
                    "vector_json": _serialize_vector(summary, resolved_dim),
                    "source_hash": source_hash,
                    "updated_at": max(
                        _utc_timestamp(item.get("updated_at") if pd.notna(item.get("updated_at")) else item.get("ts"))
                        for item in source_rows
                    ),
                }
            )

        for _, row in ordered.iterrows():
            capsule_sources.append(row)
            capsule_body_char_count_total += int(row.get("body_char_count") or 0)
            if capsule_body_char_count_total >= resolved_threshold:
                _emit_capsule(capsule_sources, CAPSULE_STATE_SEALED)
                capsule_sources = []
                capsule_body_char_count_total = 0
        if capsule_sources:
            _emit_capsule(capsule_sources, CAPSULE_STATE_OPEN)

    if not rows:
        return pd.DataFrame(columns=CAPSULES_COLUMNS)

    frame = pd.DataFrame(rows, columns=CAPSULES_COLUMNS)
    for (tenant_id, topic_id), topic_rows in frame.groupby(["tenant_id", "topic_id"], sort=True):
        ordered_idx = topic_rows.sort_values("capsule_ordinal", kind="stable").index.tolist()
        capsule_ids = frame.loc[ordered_idx, "capsule_id"].astype(str).tolist()
        for pos, row_idx in enumerate(ordered_idx):
            prev_capsule_id = capsule_ids[pos - 1] if pos > 0 else ""
            next_capsule_id = capsule_ids[pos + 1] if pos + 1 < len(capsule_ids) else ""
            back_links = capsule_ids[:pos]
            forward_links = capsule_ids[pos + 1 :]
            frame.at[row_idx, "prev_capsule_id"] = prev_capsule_id
            frame.at[row_idx, "next_capsule_id"] = next_capsule_id
            frame.at[row_idx, "back_link_ids_json"] = _json_list(back_links)
            frame.at[row_idx, "forward_link_ids_json"] = _json_list(forward_links)
            frame.at[row_idx, "pointer_json"] = json.dumps(
                {
                    "tenant_id": str(tenant_id),
                    "topic_id": str(topic_id),
                    "topic_path": str(frame.at[row_idx, "topic_path"] or topic_id),
                    "capsule_id": str(frame.at[row_idx, "capsule_id"]),
                    "capsule_ordinal": int(frame.at[row_idx, "capsule_ordinal"]),
                    "capsule_state": str(frame.at[row_idx, "capsule_state"]),
                    "active_message_count": int(frame.at[row_idx, "active_message_count"]),
                    "edited_message_count": int(frame.at[row_idx, "edited_message_count"]),
                    "topic_message_count": int(frame.at[row_idx, "topic_message_count"]),
                    "topic_body_char_count": int(frame.at[row_idx, "topic_body_char_count"]),
                    "source_session_ids": json.loads(str(frame.at[row_idx, "source_session_ids_json"]) or "[]"),
                    "source_topic_ids": json.loads(str(frame.at[row_idx, "source_topic_ids_json"]) or "[]"),
                    "opened_at": (
                        _utc_timestamp(frame.at[row_idx, "opened_at"]).isoformat()
                        if pd.notna(frame.at[row_idx, "opened_at"])
                        else None
                    ),
                    "sealed_at": (
                        _utc_timestamp(frame.at[row_idx, "sealed_at"]).isoformat()
                        if pd.notna(frame.at[row_idx, "sealed_at"])
                        else None
                    ),
                    "prev_capsule_id": prev_capsule_id or None,
                    "next_capsule_id": next_capsule_id or None,
                    "back_links": back_links,
                    "forward_links": forward_links,
                },
                separators=(",", ":"),
            )
    return frame[CAPSULES_COLUMNS].sort_values(
        ["tenant_id", "topic_id", "capsule_ordinal"],
        kind="stable",
    ).reset_index(drop=True)
