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


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
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
TOPIC_SPLIT_MIN_MESSAGES = 4
TOPIC_SPLIT_SEED_MAX_SIMILARITY = 0.18
TOPIC_SPLIT_CLUSTER_MAX_SIMILARITY = 0.45
TOPIC_SPLIT_MIN_MARGIN = 0.15
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
        content_char_count=sum(len(message.content) for message in ordered),
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


def _propose_split(messages: Sequence[TopicMessage], vector_dim: int) -> List[List[TopicMessage]]:
    ordered = list(messages)
    if len(ordered) < TOPIC_SPLIT_MIN_MESSAGES:
        return []
    seed_pair: Optional[Tuple[int, int]] = None
    seed_score = 1.0
    for left_idx in range(len(ordered)):
        for right_idx in range(left_idx + 1, len(ordered)):
            score = _cosine_similarity(ordered[left_idx].vector, ordered[right_idx].vector)
            if score < seed_score:
                seed_score = score
                seed_pair = (left_idx, right_idx)
    if seed_pair is None or seed_score > TOPIC_SPLIT_SEED_MAX_SIMILARITY:
        return []
    seed_left = ordered[seed_pair[0]]
    seed_right = ordered[seed_pair[1]]
    left_cluster: List[TopicMessage] = []
    right_cluster: List[TopicMessage] = []
    for message in ordered:
        left_score = _cosine_similarity(message.vector, seed_left.vector)
        right_score = _cosine_similarity(message.vector, seed_right.vector)
        if left_score >= right_score:
            left_cluster.append(message)
        else:
            right_cluster.append(message)
    if min(len(left_cluster), len(right_cluster)) < 2:
        return []
    left_centroid = _mean_vector([message.vector for message in left_cluster], vector_dim)
    right_centroid = _mean_vector([message.vector for message in right_cluster], vector_dim)
    if _cosine_similarity(left_centroid, right_centroid) > TOPIC_SPLIT_CLUSTER_MAX_SIMILARITY:
        return []
    margins: List[float] = []
    for cluster, own_centroid, other_centroid in (
        (left_cluster, left_centroid, right_centroid),
        (right_cluster, right_centroid, left_centroid),
    ):
        for message in cluster:
            margins.append(
                _cosine_similarity(message.vector, own_centroid)
                - _cosine_similarity(message.vector, other_centroid)
            )
    avg_margin = (sum(margins) / len(margins)) if margins else 0.0
    if avg_margin < TOPIC_SPLIT_MIN_MARGIN:
        return []
    return [sorted(left_cluster, key=lambda item: (item.ts, item.message_id)), sorted(right_cluster, key=lambda item: (item.ts, item.message_id))]


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

    canonical_map: Dict[Tuple[str, str], str] = {}
    merged_groups: Dict[Tuple[str, str], List[str]] = {}
    for tenant_id in sorted({tenant for tenant, _ in seeds.keys()}):
        tenant_seeds = {
            topic_id: seed
            for (seed_tenant, topic_id), seed in seeds.items()
            if seed_tenant == tenant_id and seed.message_count > 0
        }
        seen_topics: set[str] = set()
        for topic_id, seed in sorted(tenant_seeds.items()):
            if topic_id in seen_topics:
                continue
            group = [topic_id]
            seen_topics.add(topic_id)
            for other_id, other in sorted(tenant_seeds.items()):
                if other_id in seen_topics or other_id == topic_id:
                    continue
                if seed.parent_hint != other.parent_hint:
                    continue
                if max(seed.message_count, other.message_count) > TOPIC_MERGE_MAX_LIVE_MESSAGES:
                    continue
                if _merge_score(seed, other) < TOPIC_MERGE_MIN_SCORE:
                    continue
                group.append(other_id)
                seen_topics.add(other_id)
            canonical = sorted(
                group,
                key=lambda item: (
                    -tenant_seeds[item].message_count,
                    -tenant_seeds[item].historical_message_count,
                    tenant_seeds[item].first_ts,
                    item,
                ),
            )[0]
            merged_groups[(tenant_id, canonical)] = sorted(group)
            for member in group:
                canonical_map[(tenant_id, member)] = canonical
        for topic_id in sorted(
            {item[1] for item in seeds.keys() if item[0] == tenant_id} - set(tenant_seeds.keys())
        ):
            canonical_map[(tenant_id, topic_id)] = topic_id
            merged_groups[(tenant_id, topic_id)] = [topic_id]
        for topic_id in tenant_seeds:
            canonical_map.setdefault((tenant_id, topic_id), topic_id)
            merged_groups.setdefault((tenant_id, canonical_map[(tenant_id, topic_id)]), [canonical_map[(tenant_id, topic_id)]])

    group_slices: Dict[Tuple[str, str], TopicSlice] = {}
    split_children: Dict[Tuple[str, str], List[TopicSlice]] = {}
    for (tenant_id, canonical_topic_id), members in merged_groups.items():
        member_slices = [seeds[(tenant_id, member)] for member in members if (tenant_id, member) in seeds]
        if not member_slices:
            continue
        merged_messages = [
            message
            for seed in member_slices
            for message in seed.messages
        ]
        merged_messages = sorted(merged_messages, key=lambda item: (item.ts, item.message_id))
        parent_hint = _mode_string([seed.parent_hint for seed in member_slices])
        canonical_seed = seeds[(tenant_id, canonical_topic_id)]
        path_hint = canonical_seed.path_hint
        group_slice = _build_topic_slice(
            tenant_id=tenant_id,
            topic_id=canonical_topic_id,
            parent_hint=parent_hint,
            path_hint=path_hint,
            messages=merged_messages,
            historical_message_count=sum(seed.historical_message_count for seed in member_slices),
            deleted_message_count=sum(seed.deleted_message_count for seed in member_slices),
            first_ts=min(seed.first_ts for seed in member_slices),
            last_ts=max(seed.last_ts for seed in member_slices),
            vector_dim=resolved_dim,
        )
        group_slices[(tenant_id, canonical_topic_id)] = group_slice
        clusters = _propose_split(group_slice.messages, resolved_dim)
        if not clusters:
            continue
        children: List[TopicSlice] = []
        for idx, cluster in enumerate(clusters, start=1):
            child_id = f"{canonical_topic_id}::split:{idx:02d}"
            child_slice = _build_topic_slice(
                tenant_id=tenant_id,
                topic_id=child_id,
                parent_hint=canonical_topic_id,
                path_hint=f"{group_slice.path_hint.rstrip('/')}/split-{idx:02d}",
                messages=cluster,
                historical_message_count=len(cluster),
                deleted_message_count=0,
                first_ts=min(message.ts for message in cluster),
                last_ts=max(message.ts for message in cluster),
                vector_dim=resolved_dim,
            )
            children.append(child_slice)
        split_children[(tenant_id, canonical_topic_id)] = children

    plans: List[TopicRowPlan] = []
    for (tenant_id, topic_id), seed in sorted(seeds.items()):
        canonical_topic_id = canonical_map.get((tenant_id, topic_id), topic_id)
        merged_ids = merged_groups.get((tenant_id, canonical_topic_id), [canonical_topic_id])
        group_slice = group_slices.get((tenant_id, canonical_topic_id), seed)
        children = split_children.get((tenant_id, canonical_topic_id), [])
        if seed.message_count == 0:
            status = TOPIC_STATUS_COMPACTED
            active_slice = seed
        elif topic_id != canonical_topic_id:
            status = TOPIC_STATUS_MERGED
            active_slice = seed
        elif children:
            status = TOPIC_STATUS_SPLIT
            active_slice = group_slice
        else:
            status = TOPIC_STATUS_ACTIVE
            active_slice = group_slice
        plans.append(
            TopicRowPlan(
                tenant_id=tenant_id,
                topic_id=topic_id,
                source_topic_id=topic_id,
                canonical_topic_id=canonical_topic_id,
                status=status,
                preferred_parent_id=active_slice.parent_hint,
                leaf_name=seed.leaf_name,
                slice=active_slice,
                merged_topic_ids=list(merged_ids if topic_id == canonical_topic_id else merged_ids),
                split_topic_ids=[child.topic_id for child in children] if topic_id == canonical_topic_id else [],
            )
        )
        if topic_id != canonical_topic_id:
            continue
        for child in children:
            plans.append(
                TopicRowPlan(
                    tenant_id=tenant_id,
                    topic_id=child.topic_id,
                    source_topic_id=topic_id,
                    canonical_topic_id=child.topic_id,
                    status=TOPIC_STATUS_ACTIVE,
                    preferred_parent_id=topic_id,
                    leaf_name=child.leaf_name,
                    slice=child,
                    merged_topic_ids=[],
                    split_topic_ids=[],
                )
            )

    parent_ids = _resolve_parent_ids(plans)
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
    Online Gaussian-Ewens-process style topic assignment:
    - Ewens prior for existing vs new topic probabilities.
    - Isotropic Gaussian likelihood over hashed text vectors.
    """

    def __init__(
        self,
        *,
        dim: int = 64,
        concentration: float = 0.8,
        sigma2: float = 0.7,
        prior_sigma2: float = 1.2,
        topic_prefix: str = "geptopic",
    ) -> None:
        self.dim = max(8, int(dim))
        self.concentration = max(1e-6, float(concentration))
        self.sigma2 = max(1e-6, float(sigma2))
        self.prior_sigma2 = max(1e-6, float(prior_sigma2))
        self.topic_prefix = topic_prefix
        self._stats: Dict[str, TopicStats] = {}
        self._total = 0
        self._next = 1

    def _squared_dist(self, a: List[float], b: List[float]) -> float:
        return sum((x - y) * (x - y) for x, y in zip(a, b))

    def _log_gaussian(self, x: List[float], mean: List[float], sigma2: float) -> float:
        dist2 = self._squared_dist(x, mean)
        return -0.5 * (dist2 / sigma2)

    def _new_topic_id(self) -> str:
        topic_id = f"{self.topic_prefix}-{self._next:06d}"
        self._next += 1
        return topic_id

    def propose_topic(self, text: str) -> str:
        x = _vectorize(text, self.dim)
        if not self._stats:
            return self._new_topic_id()

        total_plus_theta = float(self._total) + self.concentration
        best_topic = ""
        best_score = float("-inf")

        for topic_id, stats in self._stats.items():
            prior = math.log(max(1e-12, float(stats.count) / total_plus_theta))
            ll = self._log_gaussian(x, stats.mean, self.sigma2)
            score = prior + ll
            if score > best_score:
                best_score = score
                best_topic = topic_id

        new_prior = math.log(max(1e-12, self.concentration / total_plus_theta))
        zero_mean = [0.0] * self.dim
        new_ll = self._log_gaussian(x, zero_mean, self.prior_sigma2)
        new_score = new_prior + new_ll

        if new_score > best_score:
            return self._new_topic_id()
        return best_topic

    def observe(self, topic_id: str, text: str) -> None:
        x = _vectorize(text, self.dim)
        stats = self._stats.get(topic_id)
        if stats is None:
            self._stats[topic_id] = TopicStats(topic_id=topic_id, count=1, mean=x)
            self._total += 1
            return

        n = stats.count
        new_n = n + 1
        stats.mean = [((m * n) + xv) / new_n for m, xv in zip(stats.mean, x)]
        stats.count = new_n
        self._total += 1

    def observe_replay(self, topic_id: str, text: str) -> None:
        # replay should restore model state; behavior identical to observe
        self.observe(topic_id, text)
