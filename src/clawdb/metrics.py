from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from statistics import median
from time import time
from time import monotonic
from typing import Deque, Dict, Mapping, Sequence, Tuple

from prometheus_client import Counter, Gauge, Histogram


DEFAULT_HIT_AT_TARGETS: Dict[int, float] = {1: 0.50, 3: 0.75, 5: 0.85}
DEFAULT_NDCG_AT_TARGETS: Dict[int, float] = {3: 0.70, 5: 0.80}
DEFAULT_COLD_LATENCY_P95_MS_TARGET = 250.0
DEFAULT_WARM_LATENCY_P95_MS_TARGET = 50.0
DEFAULT_WORKING_SET_BYTES_TARGET = 512 * 1024 * 1024
DEFAULT_REBUILD_TIME_MS_TARGET = 5_000.0


CACHE_HITS = Counter("memory_cache_hits_total", "Total cache hits")
CACHE_MISSES = Counter("memory_cache_misses_total", "Total cache misses")
CACHE_EVICTIONS = Counter("memory_cache_evictions_total", "Total cache evictions")
CACHE_LOOKUP_LATENCY_MS = Histogram(
    "memory_cache_lookup_latency_ms",
    "Cache lookup latency in milliseconds",
    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100),
)
CACHE_LOOKUPS_BY_DIMENSION = Counter(
    "memory_cache_lookups_by_dimension_total",
    "Cache lookup outcomes by tenant/session/query dimensions",
    labelnames=("tenant_id", "session_id", "query_type", "capsule_level", "outcome"),
)
ACCEPTANCE_RUNS = Counter(
    "memory_acceptance_runs_total",
    "Acceptance benchmark runs by final status",
    labelnames=("status",),
)
ACCEPTANCE_LAST_RUN_UNIX = Gauge(
    "memory_acceptance_last_run_unix_seconds",
    "Unix timestamp for the most recent acceptance benchmark run",
)
ACCEPTANCE_CASE_COUNT = Gauge(
    "memory_acceptance_case_count",
    "Number of judged search cases in the most recent acceptance benchmark run",
)
ACCEPTANCE_LATENCY_SAMPLE_COUNT = Gauge(
    "memory_acceptance_latency_sample_count",
    "Latency sample counts captured in the most recent acceptance benchmark run",
    labelnames=("temperature",),
)
ACCEPTANCE_CHECK_ACTUAL = Gauge(
    "memory_acceptance_check_actual",
    "Actual values from the most recent acceptance benchmark run",
    labelnames=("name", "comparator", "unit"),
)
ACCEPTANCE_CHECK_TARGET = Gauge(
    "memory_acceptance_check_target",
    "Target values from the most recent acceptance benchmark run",
    labelnames=("name", "comparator", "unit"),
)
ACCEPTANCE_CHECK_PASSED = Gauge(
    "memory_acceptance_check_passed",
    "Pass state for each acceptance check in the most recent benchmark run",
    labelnames=("name",),
)


@dataclass
class CacheTelemetry:
    hits_total: int = 0
    misses_total: int = 0
    evictions_total: int = 0

    def __post_init__(self) -> None:
        self._events: Deque[Tuple[float, bool]] = deque(maxlen=100_000)
        self._latency_ms: Deque[float] = deque(maxlen=100_000)

    def observe_lookup(
        self,
        hit: bool,
        latency_ms: float,
        *,
        tenant_id: str,
        session_id: str,
        query_type: str,
        capsule_level: str,
    ) -> None:
        now = monotonic()
        self._events.append((now, hit))
        self._latency_ms.append(latency_ms)
        CACHE_LOOKUP_LATENCY_MS.observe(latency_ms)
        CACHE_LOOKUPS_BY_DIMENSION.labels(
            tenant_id=tenant_id,
            session_id=session_id,
            query_type=query_type,
            capsule_level=capsule_level,
            outcome="hit" if hit else "miss",
        ).inc()
        if hit:
            self.hits_total += 1
            CACHE_HITS.inc()
        else:
            self.misses_total += 1
            CACHE_MISSES.inc()

    def observe_eviction(self) -> None:
        self.evictions_total += 1
        CACHE_EVICTIONS.inc()

    def hit_ratio(self, window_seconds: float) -> float:
        if window_seconds <= 0:
            return 0.0
        cutoff = monotonic() - window_seconds
        hits = 0
        misses = 0
        for ts, hit in reversed(self._events):
            if ts < cutoff:
                break
            if hit:
                hits += 1
            else:
                misses += 1
        total = hits + misses
        return float(hits / total) if total else 0.0

    def p50_lookup_latency_ms(self) -> float:
        if not self._latency_ms:
            return 0.0
        return float(median(self._latency_ms))


def aggregate_ranked_relevance(
    ranked_match_keys: Sequence[Sequence[str]],
    judgments: Mapping[str, float],
) -> list[float]:
    """
    Resolve a ranked result list into graded relevance values.

    Each judgment may satisfy at most one result so mirrored citations across tiers
    do not inflate Hit@k or NDCG.
    """

    remaining = {
        str(match_key): float(relevance)
        for match_key, relevance in judgments.items()
        if float(relevance) > 0.0
    }
    resolved: list[float] = []
    for match_keys in ranked_match_keys:
        best_key = ""
        best_relevance = 0.0
        for raw_key in match_keys:
            key = str(raw_key)
            relevance = float(remaining.get(key, 0.0))
            if relevance > best_relevance:
                best_key = key
                best_relevance = relevance
        resolved.append(best_relevance)
        if best_key:
            remaining.pop(best_key, None)
    return resolved


def hit_at_k(ranked_relevance: Sequence[float], k: int) -> float:
    k = max(1, int(k))
    return 1.0 if any(float(value) > 0.0 for value in ranked_relevance[:k]) else 0.0


def ndcg_at_k(ranked_relevance: Sequence[float], ideal_relevance: Sequence[float], k: int) -> float:
    k = max(1, int(k))
    actual = _dcg(ranked_relevance[:k])
    ideal = _dcg(sorted((float(value) for value in ideal_relevance), reverse=True)[:k])
    if ideal <= 0.0:
        return 0.0
    return float(actual / ideal)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    bounded = min(100.0, max(0.0, float(q)))
    ordered = sorted(float(value) for value in values)
    rank = max(0, min(len(ordered) - 1, math.ceil((bounded / 100.0) * len(ordered)) - 1))
    return float(ordered[rank])


def record_acceptance_benchmark(
    *,
    passed: bool,
    case_count: int,
    cold_latency_sample_count: int,
    warm_latency_sample_count: int,
    checks: Sequence[Mapping[str, object]],
) -> None:
    ACCEPTANCE_RUNS.labels(status="passed" if passed else "failed").inc()
    ACCEPTANCE_LAST_RUN_UNIX.set(time())
    ACCEPTANCE_CASE_COUNT.set(max(0, int(case_count)))
    ACCEPTANCE_LATENCY_SAMPLE_COUNT.labels(temperature="cold").set(
        max(0, int(cold_latency_sample_count))
    )
    ACCEPTANCE_LATENCY_SAMPLE_COUNT.labels(temperature="warm").set(
        max(0, int(warm_latency_sample_count))
    )
    for check in checks:
        name = str(check.get("name") or "")
        comparator = str(check.get("comparator") or "")
        unit = str(check.get("unit") or "")
        actual = float(check.get("actual") or 0.0)
        target = float(check.get("target") or 0.0)
        passed_value = 1.0 if bool(check.get("passed")) else 0.0
        ACCEPTANCE_CHECK_ACTUAL.labels(name=name, comparator=comparator, unit=unit).set(actual)
        ACCEPTANCE_CHECK_TARGET.labels(name=name, comparator=comparator, unit=unit).set(target)
        ACCEPTANCE_CHECK_PASSED.labels(name=name).set(passed_value)


def _dcg(values: Sequence[float]) -> float:
    score = 0.0
    for idx, value in enumerate(values):
        relevance = float(value)
        if relevance <= 0.0:
            continue
        score += (2.0**relevance - 1.0) / math.log2(idx + 2.0)
    return float(score)
