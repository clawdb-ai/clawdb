from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FolderJudger:
    l0_cutoff: int = 12
    l1_cutoff: int = 40

    def judge(self, topic_message_count: int) -> str:
        if topic_message_count <= self.l0_cutoff:
            return "L0"
        if topic_message_count <= self.l1_cutoff:
            return "L1"
        return "L2"
