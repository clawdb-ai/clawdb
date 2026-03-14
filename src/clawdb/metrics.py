from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import median
from time import monotonic
from typing import Deque, Tuple

from prometheus_client import Counter, Histogram


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
