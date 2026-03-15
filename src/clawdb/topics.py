from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


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
