from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .lineage import MESSAGE_STATE_DELETED, RAW_PROJECTION_KIND
from .textsize import utf8_text_size


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
TOPIC_ID_SUFFIX_RE = re.compile(r"(?P<prefix>[A-Za-z0-9_-]+)-(?P<ordinal>\d+)$")
TOPIC_SHARD_ID_RE = re.compile(r"^(?P<canonical>.+)::shard:(?P<ordinal>\d+)$")
TOPIC_STATUS_ACTIVE = "active"
TOPIC_STATUS_MERGED = "merged"
TOPIC_STATUS_SPLIT = "split"
TOPIC_STATUS_COMPACTED = "compacted"

DEFAULT_TOPIC_VECTOR_DIM = 64

TOPICS_COLUMNS = [
    "topic_id",
    "tenant_id",
    "canonical_topic_id",
    "topic_parent_id",
    "topic_path",
    "source_topic_id",
    "status",
    "historical_message_count",
    "message_count",
    "deleted_message_count",
    "content_char_count",
    "keywords_json",
    "merged_topic_ids_json",
    "split_topic_ids_json",
    "drift_score",
    "drift_corrected_at",
    "first_ts",
    "last_ts",
    "summary",
    "vector_text",
    "vector_ref",
    "vector_dim",
    "vector_json",
    "updated_at",
]

TOPIC_SUMMARY_MAX_CHARS = 2400
TOPIC_KEYWORD_LIMIT = 6
TOPIC_MERGE_MAX_LIVE_MESSAGES = 2
TOPIC_MERGE_MIN_SCORE = 0.82
TOPIC_SHARD_THRESHOLD_CHARS = 100_000
TOPIC_DRIFT_THRESHOLD = 0.6
TOPIC_REPARENT_MIN_SCORE = 0.56

TOPIC_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def _tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def _hash_to_index_sign(token: str, dim: int) -> Tuple[int, float]:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    idx = int.from_bytes(digest[:4], "big") % dim
    sign = 1.0 if (digest[4] & 1) == 0 else -1.0
    return idx, sign


def _vectorize(text: str, dim: int) -> List[float]:
    vec = [0.0] * dim
    tokens = _tokenize(text)
    if not tokens:
        return vec
    for token in tokens:
        idx, sign = _hash_to_index_sign(token, dim)
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 0.0:
        return vec
    return [v / norm for v in vec]


def _utc_timestamp(value: object) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return pd.Timestamp.now(tz="UTC")
    return ts


def _serialize_vector(text: str, dim: int) -> str:
    vec = [round(float(item), 8) for item in _vectorize(text, max(8, int(dim)))]
    return json.dumps(vec, separators=(",", ":"))


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
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


def _mean_vector(vectors: Sequence[Sequence[float]], dim: int) -> List[float]:
    resolved_dim = max(8, int(dim))
    if not vectors:
        return [0.0] * resolved_dim
    mean = [0.0] * resolved_dim
    for vec in vectors:
        for idx in range(min(resolved_dim, len(vec))):
            mean[idx] += float(vec[idx])
    scale = 1.0 / float(len(vectors))
    return [value * scale for value in mean]


def _keyword_overlap(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = {item for item in left if item}
    right_set = {item for item in right if item}
    if not left_set or not right_set:
        return 0.0
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


def _mode_string(values: Sequence[str], default: str = "") -> str:
    filtered = [str(item).strip() for item in values if str(item).strip()]
    if not filtered:
        return default
    counts = Counter(filtered)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ranked[0][0]


def _top_keywords(texts: Sequence[str], *, limit: int = TOPIC_KEYWORD_LIMIT) -> List[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        for token in _tokenize(text):
            if len(token) <= 2 or token in TOPIC_STOPWORDS:
                continue
            counts[token] += 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _ in ranked[: max(1, int(limit))]]


def _topic_leaf_name(topic_id: str, path_hint: str) -> str:
    if path_hint and "/" in path_hint:
        leaf = path_hint.rsplit("/", 1)[-1].strip()
        if leaf:
            return leaf
    return str(topic_id).strip() or "default"


def _path_parent_hint(path_hint: str) -> str:
    normalized = str(path_hint).strip().strip("/")
    if "/" not in normalized:
        return ""
    return normalized.rsplit("/", 1)[0]


def _json_list(values: Sequence[str]) -> str:
    items = sorted({str(item) for item in values if str(item)})
    return json.dumps(items, separators=(",", ":"))


def _topic_shard_id(topic_id: str, ordinal: int) -> str:
    return f"{str(topic_id)}::shard:{int(ordinal):04d}"


def _render_topic_text(
    *,
    topic_id: str,
    status: str,
    canonical_topic_id: str,
    parent_id: str,
    message_count: int,
    historical_message_count: int,
    drift_score: float,
    keywords: Sequence[str],
    merged_topic_ids: Sequence[str],
    split_topic_ids: Sequence[str],
    anchor_messages: Sequence["TopicMessage"],
) -> str:
    summary = (
        f"topic:{topic_id} "
        f"status={status} "
        f"canonical={canonical_topic_id} "
        f"parent={parent_id or '-'} "
        f"messages={message_count}/{historical_message_count} "
        f"drift={drift_score:.3f} "
        f"keywords={','.join(keywords) or '-'} "
        f"merged={','.join(sorted({item for item in merged_topic_ids if item})) or '-'} "
        f"split={','.join(sorted({item for item in split_topic_ids if item})) or '-'}"
    )
    if not anchor_messages:
        return summary[:TOPIC_SUMMARY_MAX_CHARS]
    snippets = []
    for message in list(anchor_messages)[-4:]:
        content = message.content.strip()
        if not content:
            continue
        snippets.append(content[:240])
    if not snippets:
        return summary[:TOPIC_SUMMARY_MAX_CHARS]
    text = f"{summary}\n" + "\n".join(f"- {snippet}" for snippet in snippets)
    return text[:TOPIC_SUMMARY_MAX_CHARS]


@dataclass(frozen=True)
class TopicMessage:
    message_id: str
    ts: pd.Timestamp
    content: str
    vector: List[float]


@dataclass(frozen=True)
class TopicSlice:
    tenant_id: str
    topic_id: str
    parent_hint: str
    path_hint: str
    leaf_name: str
    messages: List[TopicMessage]
    message_count: int
    historical_message_count: int
    deleted_message_count: int
    content_char_count: int
    keywords: List[str]
    centroid: List[float]
    recent_centroid: List[float]
    drift_score: float
    first_ts: pd.Timestamp
    last_ts: pd.Timestamp


@dataclass(frozen=True)
class TopicRowPlan:
    tenant_id: str
    topic_id: str
    source_topic_id: str
    canonical_topic_id: str
    status: str
    preferred_parent_id: str
    leaf_name: str
    slice: TopicSlice
    merged_topic_ids: List[str]
    split_topic_ids: List[str]


def _build_topic_messages(frame: pd.DataFrame, vector_dim: int) -> List[TopicMessage]:
    if frame.empty:
        return []
    ordered = frame.sort_values("ts", kind="stable").reset_index(drop=True)
    out: List[TopicMessage] = []
    for _, row in ordered.iterrows():
        content = str(row.get("content") or "")
        out.append(
            TopicMessage(
                message_id=str(row.get("origin_message_id") or row.get("message_id") or ""),
                ts=_utc_timestamp(row.get("ts")),
                content=content,
                vector=_vectorize(content, vector_dim),
            )
        )
    return out


def _build_topic_slice(
    *,
    tenant_id: str,
    topic_id: str,
    parent_hint: str,
    path_hint: str,
    messages: Sequence[TopicMessage],
    historical_message_count: int,
    deleted_message_count: int,
    first_ts: pd.Timestamp,
    last_ts: pd.Timestamp,
    vector_dim: int,
) -> TopicSlice:
    ordered = sorted(messages, key=lambda item: (item.ts, item.message_id))
    recent_window = max(1, math.ceil(len(ordered) / 3)) if ordered else 0
    centroid = _mean_vector([message.vector for message in ordered], vector_dim)
    recent_messages = ordered[-recent_window:] if recent_window else []
    history_messages = ordered[:-recent_window] if recent_window else ordered
    recent_centroid = _mean_vector([message.vector for message in recent_messages], vector_dim)
    drift_score = 0.0
    if len(ordered) >= 2:
        history_centroid = _mean_vector([message.vector for message in history_messages], vector_dim)
        drift_score = max(0.0, 1.0 - _cosine_similarity(history_centroid, recent_centroid))
    texts = [message.content for message in ordered if message.content.strip()]
    return TopicSlice(
        tenant_id=str(tenant_id),
        topic_id=str(topic_id),
        parent_hint=str(parent_hint or ""),
        path_hint=str(path_hint or topic_id),
        leaf_name=_topic_leaf_name(str(topic_id), str(path_hint or topic_id)),
        messages=list(ordered),
        message_count=len(ordered),
        historical_message_count=int(historical_message_count),
        deleted_message_count=int(deleted_message_count),
        content_char_count=sum(utf8_text_size(message.content) for message in ordered),
        keywords=_top_keywords(texts),
        centroid=centroid,
        recent_centroid=recent_centroid,
        drift_score=drift_score,
        first_ts=_utc_timestamp(first_ts),
        last_ts=_utc_timestamp(last_ts),
    )


def _merge_score(left: TopicSlice, right: TopicSlice) -> float:
    cosine = _cosine_similarity(left.centroid, right.centroid)
    overlap = _keyword_overlap(left.keywords, right.keywords)
    return (0.8 * cosine) + (0.2 * overlap)


def _sequential_topic_shards(
    messages: Sequence[TopicMessage],
    *,
    threshold_chars: int,
) -> List[List[TopicMessage]]:
    ordered = sorted(messages, key=lambda item: (item.ts, item.message_id))
    if not ordered:
        return []
    resolved_threshold = max(1, int(threshold_chars))
    total_chars = sum(utf8_text_size(message.content) for message in ordered)
    shards: List[List[TopicMessage]] = []
    current: List[TopicMessage] = []
    current_chars = 0
    for message in ordered:
        message_chars = utf8_text_size(message.content)
        current.append(message)
        current_chars += message_chars
        if current_chars >= resolved_threshold:
            shards.append(list(current))
            current = []
            current_chars = 0
    if current:
        shards.append(list(current))
    return [shard for shard in shards if shard]


def materialize_topic_message_routes(
    messages_frame: pd.DataFrame,
    *,
    shard_threshold_chars: int = TOPIC_SHARD_THRESHOLD_CHARS,
) -> Dict[Tuple[str, str], Dict[str, str]]:
    if messages_frame.empty:
        return {}
    scoped = messages_frame.copy().reset_index(drop=True)
    if "projection_kind" not in scoped.columns or "message_state" not in scoped.columns:
        return {}
    scoped = scoped[scoped["projection_kind"].astype(str) == RAW_PROJECTION_KIND].copy()
    if scoped.empty:
        return {}
    scoped["tenant_id"] = scoped["tenant_id"].fillna("default").astype(str)
    scoped["topic_id"] = scoped["topic_id"].fillna("default").astype(str)
    if "source_topic_id" not in scoped.columns:
        scoped["source_topic_id"] = scoped["topic_id"]
    scoped["source_topic_id"] = scoped["source_topic_id"].fillna(scoped["topic_id"]).astype(str)
    if "source_topic_path" not in scoped.columns:
        scoped["source_topic_path"] = scoped.get("topic_path", scoped["source_topic_id"])
    scoped["source_topic_path"] = scoped["source_topic_path"].fillna(
        scoped.get("topic_path", scoped["source_topic_id"])
    ).astype(str)
    scoped["message_state"] = scoped["message_state"].fillna("active").astype(str)
    scoped["origin_message_id"] = scoped["origin_message_id"].fillna(scoped["message_id"]).astype(str)
    scoped["ts"] = pd.to_datetime(scoped["ts"], utc=True, errors="coerce")
    scoped = scoped[scoped["ts"].notna()].copy()
    if scoped.empty:
        return {}

    routes: Dict[Tuple[str, str], Dict[str, str]] = {}
    resolved_threshold = max(1, int(shard_threshold_chars))
    for (tenant_id, source_topic_id), group in scoped.groupby(["tenant_id", "source_topic_id"], sort=True):
        ordered = group.sort_values(["ts", "origin_message_id", "message_id"], kind="stable").reset_index(drop=True)
        live_rows = ordered[ordered["message_state"].astype(str) != MESSAGE_STATE_DELETED].copy()
        if live_rows.empty:
            continue
        path_hint = _mode_string(
            live_rows["source_topic_path"].astype(str).tolist()
            or ordered["source_topic_path"].astype(str).tolist(),
            default=str(source_topic_id),
        )
        shard_ordinal = 1
        shard_chars = 0
        for _, row in live_rows.iterrows():
            shard_id = _topic_shard_id(str(source_topic_id), shard_ordinal)
            shard_path = f"{path_hint.rstrip('/')}/shard-{shard_ordinal:04d}"
            origin_id = str(row.get("origin_message_id") or row.get("message_id") or "")
            if origin_id:
                routes[(str(tenant_id), origin_id)] = {
                    "topic_id": shard_id,
                    "topic_path": shard_path,
                    "topic_parent_id": str(source_topic_id),
                }
            shard_chars += utf8_text_size(row.get("content") or "")
            if shard_chars >= resolved_threshold:
                shard_ordinal += 1
                shard_chars = 0
    return routes


def _resolve_parent_ids(plans: Sequence[TopicRowPlan]) -> Dict[str, str]:
    candidates = {
        plan.topic_id: plan
        for plan in plans
        if plan.status != TOPIC_STATUS_COMPACTED and plan.slice.message_count > 0
    }
    parent_ids: Dict[str, str] = {}
    sorted_plans = sorted(
        plans,
        key=lambda item: (
            item.status == TOPIC_STATUS_MERGED,
            -item.slice.message_count,
            item.topic_id,
        ),
    )
    for plan in sorted_plans:
        preferred = str(plan.preferred_parent_id or "")
        if preferred and preferred != plan.topic_id and preferred in candidates:
            parent_ids[plan.topic_id] = preferred
            continue
        if plan.status == TOPIC_STATUS_COMPACTED or plan.slice.message_count <= 0:
            parent_ids[plan.topic_id] = ""
            continue
        best_parent = ""
        best_score = 0.0
        for candidate_id, candidate in candidates.items():
            if candidate_id == plan.topic_id:
                continue
            if candidate.status == TOPIC_STATUS_MERGED:
                continue
            if candidate.slice.message_count < plan.slice.message_count:
                continue
            score = (
                0.7 * _cosine_similarity(plan.slice.centroid, candidate.slice.centroid)
                + 0.3 * _keyword_overlap(plan.slice.keywords, candidate.slice.keywords)
            )
            if score > best_score:
                best_score = score
                best_parent = candidate_id
        parent_ids[plan.topic_id] = best_parent if best_score >= TOPIC_REPARENT_MIN_SCORE else ""
    return parent_ids


def _resolve_paths(plans: Sequence[TopicRowPlan], parent_ids: Dict[str, str]) -> Dict[str, str]:
    plan_map = {plan.topic_id: plan for plan in plans}
    resolved: Dict[str, str] = {}

    def _path_for(topic_id: str, stack: Optional[set[str]] = None) -> str:
        if topic_id in resolved:
            return resolved[topic_id]
        active_stack = stack or set()
        if topic_id in active_stack:
            resolved[topic_id] = topic_id
            return topic_id
        active_stack.add(topic_id)
        plan = plan_map[topic_id]
        parent_id = str(parent_ids.get(topic_id) or "")
        leaf = str(plan.leaf_name or topic_id)
        if not parent_id:
            path = str(plan.slice.path_hint or topic_id)
            resolved[topic_id] = path or topic_id
            return resolved[topic_id]
        parent_path = _path_for(parent_id, active_stack)
        resolved[topic_id] = f"{parent_path.rstrip('/')}/{leaf}"
        return resolved[topic_id]

    for plan in plans:
        _path_for(plan.topic_id)
    return resolved


def materialize_topic_lifecycle(
    messages_frame: pd.DataFrame,
    *,
    vector_dim: int = DEFAULT_TOPIC_VECTOR_DIM,
    shard_threshold_chars: int = TOPIC_SHARD_THRESHOLD_CHARS,
) -> pd.DataFrame:
    if messages_frame.empty:
        return pd.DataFrame(columns=TOPICS_COLUMNS)
    resolved_dim = max(8, int(vector_dim))
    scoped = messages_frame.copy().reset_index(drop=True)
    if "projection_kind" not in scoped.columns or "message_state" not in scoped.columns:
        return pd.DataFrame(columns=TOPICS_COLUMNS)
    scoped = scoped[scoped["projection_kind"].astype(str) == RAW_PROJECTION_KIND]
    if scoped.empty:
        return pd.DataFrame(columns=TOPICS_COLUMNS)
    scoped["tenant_id"] = scoped["tenant_id"].fillna("default").astype(str)
    scoped["topic_id"] = scoped["topic_id"].fillna("default").astype(str)
    if "source_topic_id" not in scoped.columns:
        scoped["source_topic_id"] = scoped["topic_id"]
    scoped["source_topic_id"] = scoped["source_topic_id"].fillna(scoped["topic_id"]).astype(str)
    scoped["topic_parent_id"] = scoped["topic_parent_id"].fillna("").astype(str)
    scoped["topic_path"] = scoped["topic_path"].fillna(scoped["topic_id"]).astype(str)
    if "source_topic_path" not in scoped.columns:
        scoped["source_topic_path"] = scoped["topic_path"]
    scoped["source_topic_path"] = scoped["source_topic_path"].fillna(scoped["topic_path"]).astype(str)
    scoped["message_state"] = scoped["message_state"].fillna("active").astype(str)
    scoped["content"] = scoped["content"].fillna("").astype(str)
    scoped["ts"] = pd.to_datetime(scoped["ts"], utc=True, errors="coerce")
    scoped = scoped[scoped["ts"].notna()]
    if scoped.empty:
        return pd.DataFrame(columns=TOPICS_COLUMNS)

    seeds: Dict[Tuple[str, str], TopicSlice] = {}
    for (tenant_id, source_topic_id), group in scoped.groupby(["tenant_id", "source_topic_id"], sort=True):
        ordered = group.sort_values("ts", kind="stable").reset_index(drop=True)
        live_rows = ordered[ordered["message_state"].astype(str) != MESSAGE_STATE_DELETED].copy()
        parent_hint = _mode_string(
            live_rows["topic_parent_id"].astype(str).tolist() or ordered["topic_parent_id"].astype(str).tolist()
        )
        path_hint = _mode_string(
            live_rows["source_topic_path"].astype(str).tolist()
            or ordered["source_topic_path"].astype(str).tolist(),
            default=str(source_topic_id),
        )
        if not parent_hint:
            parent_hint = _path_parent_hint(path_hint)
        seeds[(str(tenant_id), str(source_topic_id))] = _build_topic_slice(
            tenant_id=str(tenant_id),
            topic_id=str(source_topic_id),
            parent_hint=parent_hint,
            path_hint=path_hint,
            messages=_build_topic_messages(live_rows, resolved_dim),
            historical_message_count=int(ordered.shape[0]),
            deleted_message_count=int((ordered["message_state"].astype(str) == MESSAGE_STATE_DELETED).sum()),
            first_ts=_utc_timestamp(ordered["ts"].min()),
            last_ts=_utc_timestamp(ordered["ts"].max()),
            vector_dim=resolved_dim,
        )

    resolved_shard_threshold = max(1, int(shard_threshold_chars))
    plans: List[TopicRowPlan] = []
    for (tenant_id, topic_id), seed in sorted(seeds.items()):
        if seed.message_count == 0:
            plans.append(
                TopicRowPlan(
                    tenant_id=tenant_id,
                    topic_id=topic_id,
                    source_topic_id=topic_id,
                    canonical_topic_id=topic_id,
                    status=TOPIC_STATUS_COMPACTED,
                    preferred_parent_id=seed.parent_hint,
                    leaf_name=seed.leaf_name,
                    slice=seed,
                    merged_topic_ids=[],
                    split_topic_ids=[],
                )
            )
            continue

        shard_groups = _sequential_topic_shards(
            seed.messages,
            threshold_chars=resolved_shard_threshold,
        )
        shard_ids = [_topic_shard_id(topic_id, idx) for idx in range(1, len(shard_groups) + 1)]
        plans.append(
            TopicRowPlan(
                tenant_id=tenant_id,
                topic_id=topic_id,
                source_topic_id=topic_id,
                canonical_topic_id=topic_id,
                status=TOPIC_STATUS_SPLIT,
                preferred_parent_id=seed.parent_hint,
                leaf_name=seed.leaf_name,
                slice=seed,
                merged_topic_ids=[],
                split_topic_ids=shard_ids,
            )
        )
        for shard_idx, shard_messages in enumerate(shard_groups, start=1):
            shard_id = shard_ids[shard_idx - 1]
            shard_path_hint = f"{seed.path_hint.rstrip('/')}/shard-{shard_idx:04d}"
            shard_slice = _build_topic_slice(
                tenant_id=tenant_id,
                topic_id=shard_id,
                parent_hint=topic_id,
                path_hint=shard_path_hint,
                messages=shard_messages,
                historical_message_count=len(shard_messages),
                deleted_message_count=0,
                first_ts=min(message.ts for message in shard_messages),
                last_ts=max(message.ts for message in shard_messages),
                vector_dim=resolved_dim,
            )
            plans.append(
                TopicRowPlan(
                    tenant_id=tenant_id,
                    topic_id=shard_id,
                    source_topic_id=topic_id,
                    canonical_topic_id=topic_id,
                    status=TOPIC_STATUS_ACTIVE,
                    preferred_parent_id=topic_id,
                    leaf_name=shard_slice.leaf_name,
                    slice=shard_slice,
                    merged_topic_ids=[],
                    split_topic_ids=[],
                )
            )

    canonical_topic_ids = {plan.topic_id for plan in plans}
    parent_ids: Dict[str, str] = {}
    for plan in plans:
        preferred_parent = str(plan.preferred_parent_id or "")
        if preferred_parent and preferred_parent != plan.topic_id and preferred_parent in canonical_topic_ids:
            parent_ids[plan.topic_id] = preferred_parent
            continue
        parent_ids[plan.topic_id] = ""
    paths = _resolve_paths(plans, parent_ids)
    materialized_at = pd.Timestamp.now(tz="UTC")
    rows: List[Dict[str, object]] = []
    for plan in sorted(plans, key=lambda item: (item.tenant_id, item.topic_id)):
        drift_corrected_at = (
            materialized_at
            if plan.slice.message_count > 0 and plan.slice.drift_score >= TOPIC_DRIFT_THRESHOLD
            else pd.NaT
        )
        recent_window = max(1, math.ceil(len(plan.slice.messages) / 3)) if plan.slice.messages else 0
        anchor_messages = (
            plan.slice.messages[-recent_window:]
            if plan.slice.message_count > 0 and plan.slice.drift_score >= TOPIC_DRIFT_THRESHOLD
            else plan.slice.messages
        )
        topic_text = _render_topic_text(
            topic_id=plan.topic_id,
            status=plan.status,
            canonical_topic_id=plan.canonical_topic_id,
            parent_id=str(parent_ids.get(plan.topic_id) or ""),
            message_count=plan.slice.message_count,
            historical_message_count=plan.slice.historical_message_count,
            drift_score=plan.slice.drift_score,
            keywords=plan.slice.keywords,
            merged_topic_ids=plan.merged_topic_ids,
            split_topic_ids=plan.split_topic_ids,
            anchor_messages=anchor_messages,
        )
        rows.append(
            {
                "topic_id": plan.topic_id,
                "tenant_id": plan.tenant_id,
                "canonical_topic_id": plan.canonical_topic_id,
                "topic_parent_id": str(parent_ids.get(plan.topic_id) or ""),
                "topic_path": str(paths.get(plan.topic_id) or plan.topic_id),
                "source_topic_id": plan.source_topic_id,
                "status": plan.status,
                "historical_message_count": int(plan.slice.historical_message_count),
                "message_count": int(plan.slice.message_count),
                "deleted_message_count": int(plan.slice.deleted_message_count),
                "content_char_count": int(plan.slice.content_char_count),
                "keywords_json": _json_list(plan.slice.keywords),
                "merged_topic_ids_json": _json_list(plan.merged_topic_ids),
                "split_topic_ids_json": _json_list(plan.split_topic_ids),
                "drift_score": float(plan.slice.drift_score),
                "drift_corrected_at": drift_corrected_at,
                "first_ts": plan.slice.first_ts,
                "last_ts": plan.slice.last_ts,
                "summary": topic_text,
                "vector_text": topic_text,
                "vector_ref": f"topic:{hashlib.sha256(topic_text.encode('utf-8')).hexdigest()}",
                "vector_dim": resolved_dim,
                "vector_json": _serialize_vector(topic_text, resolved_dim),
                "updated_at": materialized_at,
            }
        )
    if not rows:
        return pd.DataFrame(columns=TOPICS_COLUMNS)
    frame = pd.DataFrame(rows, columns=TOPICS_COLUMNS)
    for col in ["drift_corrected_at", "first_ts", "last_ts", "updated_at"]:
        frame[col] = pd.to_datetime(frame[col], utc=True, errors="coerce")
    for col in [
        "historical_message_count",
        "message_count",
        "deleted_message_count",
        "content_char_count",
        "vector_dim",
    ]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0).astype(int)
    frame["drift_score"] = pd.to_numeric(frame["drift_score"], errors="coerce").fillna(0.0).astype(float)
    return frame[TOPICS_COLUMNS]


@dataclass
class TopicStats:
    topic_id: str
    count: int
    mean: List[float]


class GaussianEwensTopicModel:
    """
    Online Gaussian-Ewens-process style topic assignment over dense vectors.
    """

    def __init__(
        self,
        *,
        dim: int = 64,
        concentration: float = 0.8,
        sigma2: float = 0.7,
        prior_sigma2: float = 1.2,
        min_dot_product: float = 0.45,
        topic_prefix: str = "geptopic",
    ) -> None:
        self.dim = max(8, int(dim))
        self.concentration = max(1e-6, float(concentration))
        self.sigma2 = max(1e-6, float(sigma2))
        self.prior_sigma2 = max(1e-6, float(prior_sigma2))
        self.min_dot_product = max(-1.0, min(1.0, float(min_dot_product)))
        self.topic_prefix = topic_prefix
        self._stats: Dict[str, TopicStats] = {}
        self._total = 0
        self._next = 1

    def _observe_existing_topic_id(self, topic_id: str) -> None:
        match = TOPIC_ID_SUFFIX_RE.search(str(topic_id or "").strip())
        if match is None:
            return
        if str(match.group("prefix") or "").strip() != self.topic_prefix:
            return
        try:
            ordinal = int(match.group("ordinal"))
        except Exception:
            return
        if ordinal >= self._next:
            self._next = ordinal + 1

    def _squared_dist(self, a: List[float], b: List[float]) -> float:
        dims = max(len(a), len(b))
        total = 0.0
        for idx in range(dims):
            left = float(a[idx]) if idx < len(a) else 0.0
            right = float(b[idx]) if idx < len(b) else 0.0
            total += (left - right) * (left - right)
        return total

    def _log_gaussian(self, x: List[float], mean: List[float], sigma2: float) -> float:
        dist2 = self._squared_dist(x, mean)
        return -0.5 * (dist2 / sigma2)

    def _new_topic_id(self) -> str:
        topic_id = f"{self.topic_prefix}-{self._next:06d}"
        self._next += 1
        return topic_id

    def _normalize_vector(self, vector: Sequence[float]) -> List[float]:
        raw = [float(item) for item in vector]
        if not raw:
            return [0.0] * self.dim
        norm = math.sqrt(sum(item * item for item in raw))
        if norm <= 0.0:
            return [0.0] * len(raw)
        return [item / norm for item in raw]

    def _mean_size(self, topic_id: str, vector_length: int) -> List[float]:
        stats = self._stats.get(topic_id)
        if stats is None:
            return [0.0] * max(1, int(vector_length))
        if len(stats.mean) >= vector_length:
            return list(stats.mean)
        return list(stats.mean) + ([0.0] * (vector_length - len(stats.mean)))

    def propose_topic(self, text: str) -> str:
        return self.propose_topic_vector(_vectorize(text, self.dim))

    def propose_topic_vector(self, vector: Sequence[float]) -> str:
        x = self._normalize_vector(vector)
        if not self._stats:
            return self._new_topic_id()

        total_plus_theta = float(self._total) + self.concentration
        best_topic = ""
        best_score = float("-inf")
        best_similarity = float("-inf")

        for topic_id, stats in self._stats.items():
            similarity = _cosine_similarity(x, self._mean_size(topic_id, len(x)))
            best_similarity = max(best_similarity, similarity)
            if similarity < self.min_dot_product:
                continue
            prior = math.log(max(1e-12, float(stats.count) / total_plus_theta))
            ll = self._log_gaussian(x, self._mean_size(topic_id, len(x)), self.sigma2)
            score = prior + ll
            if score > best_score:
                best_score = score
                best_topic = topic_id

        if not best_topic and best_similarity < self.min_dot_product:
            return self._new_topic_id()

        new_prior = math.log(max(1e-12, self.concentration / total_plus_theta))
        zero_mean = [0.0] * len(x)
        new_ll = self._log_gaussian(x, zero_mean, self.prior_sigma2)
        new_score = new_prior + new_ll

        if not best_topic or new_score > best_score:
            return self._new_topic_id()
        return best_topic

    def observe(self, topic_id: str, text: str) -> None:
        self.observe_vector(topic_id, _vectorize(text, self.dim))

    def observe_vector(self, topic_id: str, vector: Sequence[float]) -> None:
        self._observe_existing_topic_id(topic_id)
        x = self._normalize_vector(vector)
        stats = self._stats.get(topic_id)
        if stats is None:
            self._stats[topic_id] = TopicStats(topic_id=topic_id, count=1, mean=x)
            self._total += 1
            return

        n = stats.count
        new_n = n + 1
        dims = max(len(stats.mean), len(x))
        current_mean = list(stats.mean) + ([0.0] * (dims - len(stats.mean)))
        observed = list(x) + ([0.0] * (dims - len(x)))
        stats.mean = [((m * n) + xv) / new_n for m, xv in zip(current_mean, observed)]
        stats.count = new_n
        self._total += 1

    def observe_replay(self, topic_id: str, text: str) -> None:
        # replay should restore model state; behavior identical to observe
        self.observe(topic_id, text)

    def observe_replay_vector(self, topic_id: str, vector: Sequence[float]) -> None:
        self.observe_vector(topic_id, vector)

    def hydrate_topic_vector(self, topic_id: str, count: int, mean: Sequence[float]) -> None:
        resolved_count = max(0, int(count))
        if resolved_count <= 0:
            return
        self._observe_existing_topic_id(topic_id)
        normalized_mean = self._normalize_vector(mean)
        self._stats[str(topic_id)] = TopicStats(
            topic_id=str(topic_id),
            count=resolved_count,
            mean=normalized_mean,
        )
        self._total += resolved_count
