from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence, Tuple

from .topics import _vectorize

RetrievalMode = Literal["hybrid", "lexical", "vector"]

RETRIEVAL_MODE_WEIGHTS: Dict[str, Tuple[float, float]] = {
    "hybrid": (0.30, 0.70),
    "lexical": (1.00, 0.00),
    "vector": (0.00, 1.00),
}


@dataclass(frozen=True)
class RetrievalDoc:
    doc_id: str
    text: str


@dataclass(frozen=True)
class RetrievalScore:
    doc_id: str
    score: float
    score_lexical: float
    score_vector: float


def resolve_retrieval_weights(mode: RetrievalMode) -> Tuple[float, float]:
    return RETRIEVAL_MODE_WEIGHTS.get(str(mode), RETRIEVAL_MODE_WEIGHTS["hybrid"])


def _tokenize(text: str) -> List[str]:
    return [token for token in text.lower().split() if token]


class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: List[RetrievalDoc] = []
        self._tf: Dict[str, Dict[str, int]] = {}
        self._df: Dict[str, int] = {}
        self._avg_len = 0.0
        self._doc_len: Dict[str, int] = {}

    def build(self, docs: Sequence[RetrievalDoc]) -> None:
        self._docs = list(docs)
        self._tf.clear()
        self._df.clear()
        self._doc_len.clear()
        total_len = 0
        for doc in self._docs:
            tokens = _tokenize(doc.text)
            total_len += len(tokens)
            self._doc_len[doc.doc_id] = len(tokens)
            tf: Dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self._tf[doc.doc_id] = tf
            for token in tf:
                self._df[token] = self._df.get(token, 0) + 1
        self._avg_len = (total_len / len(self._docs)) if self._docs else 0.0

    def search(self, query: str, top_k: int | None = None) -> List[Tuple[str, float]]:
        if not self._docs:
            return []
        query_tokens = _tokenize(query)
        n_docs = max(1, len(self._docs))
        out: List[Tuple[str, float]] = []
        for doc in self._docs:
            doc_tf = self._tf.get(doc.doc_id, {})
            doc_len = max(1, self._doc_len.get(doc.doc_id, 0))
            score = 0.0
            for token in query_tokens:
                tf = doc_tf.get(token, 0)
                if tf == 0:
                    continue
                df = self._df.get(token, 0)
                idf = math.log(1 + ((n_docs - df + 0.5) / (df + 0.5)))
                denom = tf + self.k1 * (1 - self.b + self.b * (doc_len / max(1e-6, self._avg_len)))
                score += idf * ((tf * (self.k1 + 1)) / max(1e-6, denom))
            if score > 0.0:
                out.append((doc.doc_id, float(score)))
        out.sort(key=lambda item: (-item[1], item[0]))
        if top_k is None:
            return out
        return out[: max(1, top_k)]


class HNSWIndex:
    """
    HNSW-style interface with deterministic in-process cosine fallback.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self._vectors: Dict[str, List[float]] = {}

    def build(self, docs: Sequence[RetrievalDoc]) -> None:
        self._vectors = {doc.doc_id: _vectorize(doc.text, self.dim) for doc in docs}

    def search(self, query: str, top_k: int | None = None) -> List[Tuple[str, float]]:
        query_vector = _vectorize(query, self.dim)
        out: List[Tuple[str, float]] = []
        for doc_id, vector in self._vectors.items():
            dot = sum(left * right for left, right in zip(query_vector, vector))
            query_norm = math.sqrt(sum(value * value for value in query_vector))
            vector_norm = math.sqrt(sum(value * value for value in vector))
            if query_norm <= 0.0 or vector_norm <= 0.0:
                continue
            similarity = dot / (query_norm * vector_norm)
            out.append((doc_id, float(max(0.0, similarity))))
        out.sort(key=lambda item: (-item[1], item[0]))
        if top_k is None:
            return out
        return out[: max(1, top_k)]


class HybridRetrievalEngine:
    def __init__(self, dim: int = 64) -> None:
        self.bm25 = BM25Index()
        self.hnsw = HNSWIndex(dim=dim)

    def search(
        self,
        query: str,
        docs: Sequence[RetrievalDoc],
        top_k: int,
        retrieval_mode: RetrievalMode = "hybrid",
    ) -> List[RetrievalScore]:
        if not docs:
            return []
        self.bm25.build(docs)
        self.hnsw.build(docs)
        lexical_weight, vector_weight = resolve_retrieval_weights(retrieval_mode)
        bm25_res = self.bm25.search(query, top_k=len(docs))
        vector_res = self.hnsw.search(query, top_k=len(docs))
        bm25_map = dict(bm25_res)
        vector_map = dict(vector_res)
        max_bm25 = max((score for _, score in bm25_res), default=0.0)
        ranked: List[RetrievalScore] = []
        for doc in docs:
            lexical_raw = float(bm25_map.get(doc.doc_id, 0.0))
            lexical_score = (lexical_raw / max_bm25) if max_bm25 > 0.0 else 0.0
            vector_score = float(vector_map.get(doc.doc_id, 0.0))
            score = (lexical_weight * lexical_score) + (vector_weight * vector_score)
            if score <= 0.0:
                continue
            ranked.append(
                RetrievalScore(
                    doc_id=doc.doc_id,
                    score=float(score),
                    score_lexical=float(lexical_score),
                    score_vector=float(vector_score),
                )
            )
        ranked.sort(
            key=lambda item: (
                -item.score,
                -item.score_lexical,
                -item.score_vector,
                item.doc_id,
            )
        )
        return ranked[: max(1, top_k)]
