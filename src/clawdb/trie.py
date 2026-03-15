from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class TrieNode:
    children: Dict[str, "TrieNode"] = field(default_factory=dict)
    topic_hits: Dict[str, int] = field(default_factory=dict)


class TopicTrie:
    def __init__(self) -> None:
        self._root = TrieNode()
        self._topic_counts: Dict[str, int] = {}

    @property
    def topic_count(self) -> int:
        return len(self._topic_counts)

    def insert(self, topic_id: str, text: str) -> None:
        tokens = [tok.strip().lower() for tok in text.split() if tok.strip()]
        if not tokens:
            return
        self._topic_counts[topic_id] = self._topic_counts.get(topic_id, 0) + 1
        for token in tokens:
            node = self._root
            for ch in token:
                node = node.children.setdefault(ch, TrieNode())
            node.topic_hits[topic_id] = node.topic_hits.get(topic_id, 0) + 1

    def detect_topic(self, text: str) -> Optional[str]:
        scored = self.rank_topics(text, top_k=1)
        if not scored:
            return None
        return scored[0][0]

    def rank_topics(self, text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        tokens = [tok.strip().lower() for tok in text.split() if tok.strip()]
        score: Dict[str, float] = {}
        for token in tokens:
            node = self._root
            for ch in token:
                nxt = node.children.get(ch)
                if nxt is None:
                    node = None  # type: ignore[assignment]
                    break
                node = nxt
            if node is None:
                continue
            for topic_id, hits in node.topic_hits.items():
                score[topic_id] = score.get(topic_id, 0.0) + float(hits)
        ranked = sorted(score.items(), key=lambda item: item[1], reverse=True)
        return ranked[: max(1, top_k)]
