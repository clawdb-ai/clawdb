from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import List, Sequence

import pandas as pd

from .embeddings import normalize_embedding_text
from .topics import _vectorize


SEARCH_DOC_COLUMNS = [
    "tenant_id",
    "doc_id",
    "entity_type",
    "entity_id",
    "source_tier",
    "session_id",
    "updated_at",
    "text",
    "path",
    "start_line",
    "end_line",
    "snippet",
    "citation",
    "citations_json",
    "channel",
    "chat_type",
    "account_id",
    "group_id",
    "topic_id",
    "topic_path",
    "message_thread_id",
    "sender_id",
    "origin_message_id",
    "projection_kind",
    "projection_scope",
    "vector_ref",
    "vector_dim",
    "vector_json",
]

LEXICAL_INDEX_COLUMNS = [
    "tenant_id",
    "doc_id",
    "token",
    "term_freq",
    "doc_len",
    "updated_at",
]

VECTOR_INDEX_COLUMNS = [
    "tenant_id",
    "doc_id",
    "vector_dim",
    "vector_json",
    "vector_norm",
    "updated_at",
]


@dataclass(frozen=True)
class LexicalPosting:
    doc_id: str
    token: str
    term_freq: int
    doc_len: int


@dataclass(frozen=True)
class VectorEntry:
    doc_id: str
    vector: List[float]
    norm: float


def tokenize_lexical(text: str) -> List[str]:
    normalized = normalize_embedding_text(text).lower()
    return [token for token in re.findall(r"\w+", normalized) if token]


def serialize_vector_json(values: Sequence[float]) -> str:
    return json.dumps([float(value) for value in values], ensure_ascii=False, separators=(",", ":"))


def parse_vector_json(value: object) -> List[float]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: List[float] = []
        for item in value:
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                out.append(0.0)
        return out
    raw = str(value).strip()
    if not raw:
        return []
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(decoded, list):
        return []
    out = []
    for item in decoded:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def materialize_lexical_index(search_docs_frame: pd.DataFrame) -> pd.DataFrame:
    if search_docs_frame.empty:
        return pd.DataFrame(columns=LEXICAL_INDEX_COLUMNS)
    rows = []
    scoped = search_docs_frame.copy().reset_index(drop=True)
    scoped["tenant_id"] = scoped["tenant_id"].fillna("default").astype(str)
    scoped["doc_id"] = scoped["doc_id"].fillna("").astype(str)
    scoped["text"] = scoped["text"].fillna("").astype(str)
    scoped["updated_at"] = pd.to_datetime(scoped["updated_at"], utc=True, errors="coerce")
    for _, row in scoped.iterrows():
        doc_id = str(row.get("doc_id") or "")
        if not doc_id:
            continue
        tokens = tokenize_lexical(str(row.get("text") or ""))
        if not tokens:
            continue
        counts = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        updated_at = pd.to_datetime(row.get("updated_at"), utc=True, errors="coerce")
        if pd.isna(updated_at):
            updated_at = pd.Timestamp.now(tz="UTC")
        for token, term_freq in counts.items():
            rows.append(
                {
                    "tenant_id": str(row.get("tenant_id") or "default"),
                    "doc_id": doc_id,
                    "token": token,
                    "term_freq": int(term_freq),
                    "doc_len": int(len(tokens)),
                    "updated_at": updated_at,
                }
            )
    if not rows:
        return pd.DataFrame(columns=LEXICAL_INDEX_COLUMNS)
    frame = pd.DataFrame(rows, columns=LEXICAL_INDEX_COLUMNS)
    frame["tenant_id"] = frame["tenant_id"].fillna("default").astype(str)
    frame["doc_id"] = frame["doc_id"].fillna("").astype(str)
    frame["token"] = frame["token"].fillna("").astype(str)
    frame["term_freq"] = pd.to_numeric(frame["term_freq"], errors="coerce").fillna(0).astype(int)
    frame["doc_len"] = pd.to_numeric(frame["doc_len"], errors="coerce").fillna(0).astype(int)
    frame["updated_at"] = pd.to_datetime(frame["updated_at"], utc=True, errors="coerce")
    return frame.sort_values(
        ["tenant_id", "token", "doc_id"],
        ascending=[True, True, True],
        kind="stable",
    ).reset_index(drop=True)


def materialize_vector_index(search_docs_frame: pd.DataFrame, *, dim: int) -> pd.DataFrame:
    if search_docs_frame.empty:
        return pd.DataFrame(columns=VECTOR_INDEX_COLUMNS)
    rows = []
    scoped = search_docs_frame.copy().reset_index(drop=True)
    scoped["tenant_id"] = scoped["tenant_id"].fillna("default").astype(str)
    scoped["doc_id"] = scoped["doc_id"].fillna("").astype(str)
    scoped["text"] = scoped["text"].fillna("").astype(str)
    scoped["updated_at"] = pd.to_datetime(scoped["updated_at"], utc=True, errors="coerce")
    resolved_dim = max(8, int(dim))
    for _, row in scoped.iterrows():
        doc_id = str(row.get("doc_id") or "")
        if not doc_id:
            continue
        provided_vector = parse_vector_json(row.get("vector_json"))
        if provided_vector:
            vector = [float(value) for value in provided_vector]
            vector_dim = max(8, int(row.get("vector_dim") or len(vector) or resolved_dim))
        else:
            vector = [float(value) for value in _vectorize(str(row.get("text") or ""), resolved_dim)]
            vector_dim = resolved_dim
        updated_at = pd.to_datetime(row.get("updated_at"), utc=True, errors="coerce")
        if pd.isna(updated_at):
            updated_at = pd.Timestamp.now(tz="UTC")
        rows.append(
            {
                "tenant_id": str(row.get("tenant_id") or "default"),
                "doc_id": doc_id,
                "vector_dim": int(vector_dim),
                "vector_json": serialize_vector_json(vector),
                "vector_norm": float(math.sqrt(sum(value * value for value in vector))),
                "updated_at": updated_at,
            }
        )
    if not rows:
        return pd.DataFrame(columns=VECTOR_INDEX_COLUMNS)
    frame = pd.DataFrame(rows, columns=VECTOR_INDEX_COLUMNS)
    frame["tenant_id"] = frame["tenant_id"].fillna("default").astype(str)
    frame["doc_id"] = frame["doc_id"].fillna("").astype(str)
    frame["vector_dim"] = pd.to_numeric(frame["vector_dim"], errors="coerce").fillna(resolved_dim).astype(int)
    frame["vector_json"] = frame["vector_json"].fillna("[]").astype(str)
    frame["vector_norm"] = pd.to_numeric(frame["vector_norm"], errors="coerce").fillna(0.0).astype(float)
    frame["updated_at"] = pd.to_datetime(frame["updated_at"], utc=True, errors="coerce")
    return frame.sort_values(
        ["tenant_id", "doc_id"],
        ascending=[True, True],
        kind="stable",
    ).reset_index(drop=True)
