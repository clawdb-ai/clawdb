from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Mapping, Sequence, Tuple

import numpy as np

from .search_index import LexicalPosting, VectorEntry, tokenize_lexical
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
    return tokenize_lexical(text)


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

    def build_from_postings(
        self,
        docs: Sequence[RetrievalDoc],
        postings: Sequence[LexicalPosting],
    ) -> None:
        self._docs = list(docs)
        self._tf.clear()
        self._df.clear()
        self._doc_len.clear()
        doc_map = {doc.doc_id: doc for doc in self._docs}
        posting_docs = set()
        for posting in postings:
            doc_id = str(posting.doc_id)
            if doc_id not in doc_map:
                continue
            posting_docs.add(doc_id)
            tf = self._tf.setdefault(doc_id, {})
            token = str(posting.token)
            tf[token] = int(posting.term_freq)
            self._doc_len[doc_id] = int(posting.doc_len)
        for doc_id in posting_docs:
            for token in self._tf.get(doc_id, {}):
                self._df[token] = self._df.get(token, 0) + 1
        for doc in self._docs:
            if doc.doc_id in posting_docs:
                continue
            tokens = _tokenize(doc.text)
            self._doc_len[doc.doc_id] = len(tokens)
            tf: Dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self._tf[doc.doc_id] = tf
            for token in tf:
                self._df[token] = self._df.get(token, 0) + 1
        total_len = sum(max(0, int(self._doc_len.get(doc.doc_id, 0))) for doc in self._docs)
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
        self._vectors: Dict[str, np.ndarray] = {}
        self._norms: Dict[str, float] = {}

    def build(self, docs: Sequence[RetrievalDoc]) -> None:
        self._vectors = {}
        self._norms = {}
        for doc in docs:
            vector = np.asarray(_vectorize(doc.text, self.dim), dtype=float)
            self._vectors[doc.doc_id] = vector
            self._norms[doc.doc_id] = float(np.linalg.norm(vector))

    def build_from_vectors(
        self,
        docs: Sequence[RetrievalDoc],
        vectors: Sequence[VectorEntry],
    ) -> None:
        self._vectors = {}
        self._norms = {}
        provided = {
            str(entry.doc_id): (
                np.asarray(list(entry.vector), dtype=float),
                float(entry.norm),
            )
            for entry in vectors
        }
        for doc in docs:
            cached = provided.get(doc.doc_id)
            if cached is not None:
                vector, norm = cached
            else:
                vector = np.asarray(_vectorize(doc.text, self.dim), dtype=float)
                norm = float(np.linalg.norm(vector))
            self._vectors[doc.doc_id] = vector
            self._norms[doc.doc_id] = norm

    def search(self, query: str, top_k: int | None = None) -> List[Tuple[str, float]]:
        query_vector = np.asarray(_vectorize(query, self.dim), dtype=float)
        query_norm = float(np.linalg.norm(query_vector))
        if query_norm <= 0.0:
            return []
        out: List[Tuple[str, float]] = []
        for doc_id, vector in self._vectors.items():
            vector_norm = self._norms.get(doc_id, 0.0)
            if query_norm <= 0.0 or vector_norm <= 0.0:
                continue
            similarity = float(np.dot(query_vector, vector) / (query_norm * vector_norm))
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
        lexical_postings: Sequence[LexicalPosting] | None = None,
        vector_entries: Sequence[VectorEntry] | None = None,
    ) -> List[RetrievalScore]:
        if not docs:
            return []
        if lexical_postings is None:
            self.bm25.build(docs)
        else:
            self.bm25.build_from_postings(docs, lexical_postings)
        if vector_entries is None:
            self.hnsw.build(docs)
        else:
            self.hnsw.build_from_vectors(docs, vector_entries)
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
