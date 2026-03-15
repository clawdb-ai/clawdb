from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .topics import _vectorize


@dataclass(frozen=True)
class RetrievalDoc:
    doc_id: str
    text: str


def _tokenize(text: str) -> List[str]:
    return [t for t in text.lower().split() if t]


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
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            self._tf[doc.doc_id] = tf
            for tok in tf:
                self._df[tok] = self._df.get(tok, 0) + 1
        self._avg_len = (total_len / len(self._docs)) if self._docs else 0.0

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        if not self._docs:
            return []
        q_tokens = _tokenize(query)
        n_docs = max(1, len(self._docs))
        out: List[Tuple[str, float]] = []
        for doc in self._docs:
            doc_tf = self._tf.get(doc.doc_id, {})
            dlen = max(1, self._doc_len.get(doc.doc_id, 0))
            score = 0.0
            for tok in q_tokens:
                tf = doc_tf.get(tok, 0)
                if tf == 0:
                    continue
                df = self._df.get(tok, 0)
                idf = math.log(1 + ((n_docs - df + 0.5) / (df + 0.5)))
                denom = tf + self.k1 * (1 - self.b + self.b * (dlen / max(1e-6, self._avg_len)))
                score += idf * ((tf * (self.k1 + 1)) / max(1e-6, denom))
            if score > 0:
                out.append((doc.doc_id, float(score)))
        out.sort(key=lambda it: it[1], reverse=True)
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

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        q = _vectorize(query, self.dim)
        out: List[Tuple[str, float]] = []
        for doc_id, vec in self._vectors.items():
            dot = sum(a * b for a, b in zip(q, vec))
            qn = math.sqrt(sum(a * a for a in q))
            vn = math.sqrt(sum(b * b for b in vec))
            if qn <= 0 or vn <= 0:
                continue
            sim = dot / (qn * vn)
            out.append((doc_id, float(sim)))
        out.sort(key=lambda it: it[1], reverse=True)
        return out[: max(1, top_k)]


class NTopKSearch:
    def gather(
        self,
        bm25_results: Sequence[Tuple[str, float]],
        vector_results: Sequence[Tuple[str, float]],
        n_each: int,
    ) -> List[str]:
        picks: List[str] = []
        seen = set()
        for doc_id, _ in list(bm25_results)[: max(1, n_each)]:
            if doc_id not in seen:
                picks.append(doc_id)
                seen.add(doc_id)
        for doc_id, _ in list(vector_results)[: max(1, n_each)]:
            if doc_id not in seen:
                picks.append(doc_id)
                seen.add(doc_id)
        return picks


class HybridFusion:
    def fuse(
        self,
        candidates: Iterable[str],
        bm25_results: Sequence[Tuple[str, float]],
        vector_results: Sequence[Tuple[str, float]],
    ) -> Dict[str, float]:
        bm25_pos = {doc_id: i for i, (doc_id, _) in enumerate(bm25_results)}
        vec_pos = {doc_id: i for i, (doc_id, _) in enumerate(vector_results)}
        scores: Dict[str, float] = {}
        for doc_id in candidates:
            s = 0.0
            if doc_id in bm25_pos:
                s += 1.0 / (1.0 + bm25_pos[doc_id])
            if doc_id in vec_pos:
                s += 1.0 / (1.0 + vec_pos[doc_id])
            scores[doc_id] = s
        return scores


class HybridRetrievalEngine:
    def __init__(self, dim: int = 64) -> None:
        self.bm25 = BM25Index()
        self.hnsw = HNSWIndex(dim=dim)
        self.n_top_k = NTopKSearch()
        self.fusion = HybridFusion()

    def search(
        self,
        query: str,
        docs: Sequence[RetrievalDoc],
        top_k: int,
    ) -> List[Tuple[str, float, float, float]]:
        if not docs:
            return []
        self.bm25.build(docs)
        self.hnsw.build(docs)
        n_each = max(1, top_k)
        bm25_res = self.bm25.search(query, top_k=n_each * 3)
        vec_res = self.hnsw.search(query, top_k=n_each * 3)
        candidates = self.n_top_k.gather(bm25_res, vec_res, n_each=n_each)
        fused = self.fusion.fuse(candidates, bm25_res, vec_res)
        bm25_map = dict(bm25_res)
        vec_map = dict(vec_res)
        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        out = []
        for doc_id, score in ranked[: max(1, top_k)]:
            out.append(
                (
                    doc_id,
                    float(score),
                    float(bm25_map.get(doc_id, 0.0)),
                    float(vec_map.get(doc_id, 0.0)),
                )
            )
        return out
