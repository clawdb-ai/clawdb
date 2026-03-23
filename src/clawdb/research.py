from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Mapping, Sequence, Tuple

from .models import (
    AcceptanceBenchmarkRequest,
    AcceptanceJudgment,
    AcceptanceSearchCase,
    AcceptanceTargets,
    MessageDeleteRequest,
    MessageEditRequest,
    MessageIn,
    SearchRequest,
)


_RESEARCH_T0 = datetime(2026, 3, 1, 9, 0, tzinfo=timezone.utc)


def _ts(offset_minutes: int) -> datetime:
    return _RESEARCH_T0 + timedelta(minutes=int(offset_minutes))


def _message(
    *,
    message_id: str,
    session_id: str,
    sender_id: str,
    content: str,
    topic_id: str,
    offset_minutes: int,
    chat_type: str = "direct",
    group_id: str | None = None,
    group_subject: str | None = None,
    message_thread_id: str | None = None,
    thread_parent_id: str | None = None,
    reply_to_id: str | None = None,
) -> MessageIn:
    direct_target = "ou_research_bot"
    return MessageIn(
        tenant_id="research-template",
        session_id=session_id,
        role="user",
        content=content,
        channel="feishu",
        platform="feishu",
        chat_type=chat_type,
        account_id="botacct",
        sender_id=sender_id,
        from_id=sender_id,
        to_id=direct_target,
        group_id=group_id,
        group_subject=group_subject,
        native_channel_id=group_id,
        origin_message_id=message_id,
        message_thread_id=message_thread_id,
        thread_parent_id=thread_parent_id,
        reply_to_id=reply_to_id,
        topic_id=topic_id,
        message_id=message_id,
        ts=_ts(offset_minutes),
    )


def _case(
    *,
    label: str,
    query: str,
    judgments: Sequence[Tuple[str, float]],
    session_id: str | None = None,
    group_id: str | None = None,
    topic_id: str | None = None,
    message_thread_id: str | None = None,
    chat_type: str | None = None,
    retrieval_mode: str = "hybrid",
    max_results: int = 6,
) -> AcceptanceSearchCase:
    return AcceptanceSearchCase(
        label=label,
        search=SearchRequest(
            query=query,
            tenant_id="research-template",
            session_id=session_id,
            group_id=group_id,
            topic_id=topic_id,
            message_thread_id=message_thread_id,
            chat_type=chat_type,
            retrieval_mode=retrieval_mode,
            max_results=max_results,
        ),
        judgments=[
            AcceptanceJudgment(match_key=f"origin:{origin_message_id}", relevance=relevance)
            for origin_message_id, relevance in judgments
        ],
    )


@dataclass(frozen=True)
class ResearchCorpusSpec:
    name: str
    version: str
    description: str
    messages: Tuple[MessageIn, ...]
    edits: Tuple[MessageEditRequest, ...]
    deletes: Tuple[MessageDeleteRequest, ...]
    cases: Tuple[AcceptanceSearchCase, ...]
    targets: AcceptanceTargets

    def build_acceptance_request(
        self,
        *,
        tenant_id: str,
        latency_repetitions: int,
        targets: AcceptanceTargets | None = None,
    ) -> AcceptanceBenchmarkRequest:
        effective_targets = (targets or self.targets).model_copy(deep=True)
        cases = tuple(
            case.model_copy(
                deep=True,
                update={
                    "search": case.search.model_copy(
                        deep=True,
                        update={"tenant_id": str(tenant_id or "research-benchmark")},
                    )
                },
            )
            for case in self.cases
        )
        return AcceptanceBenchmarkRequest(
            cases=list(cases),
            latency_repetitions=max(1, int(latency_repetitions)),
            targets=effective_targets,
        )

    def coverage(self) -> Dict[str, object]:
        messages = tuple(self.messages)
        cases = tuple(self.cases)
        sessions = {message.session_id for message in messages}
        topics = {str(message.topic_id or "default") for message in messages}
        groups = {str(message.group_id or "") for message in messages if message.group_id}
        threads = {
            str(message.message_thread_id or "")
            for message in messages
            if message.message_thread_id
        }
        chat_types = sorted({str(message.chat_type or "") for message in messages if message.chat_type})
        retrieval_modes = sorted(
            {str(case.search.retrieval_mode or "hybrid") for case in cases}
        )
        filter_dimensions = set()
        scoped_case_count = 0
        for case in cases:
            scoped = False
            search = case.search
            for field_name in (
                "session_id",
                "group_id",
                "topic_id",
                "message_thread_id",
                "chat_type",
            ):
                if getattr(search, field_name):
                    filter_dimensions.add(field_name)
                    scoped = True
            if scoped:
                scoped_case_count += 1
        return {
            "message_count": len(messages),
            "edit_count": len(self.edits),
            "delete_count": len(self.deletes),
            "session_count": len(sessions),
            "topic_count": len(topics),
            "group_count": len(groups),
            "thread_count": len(threads),
            "case_count": len(cases),
            "scoped_case_count": scoped_case_count,
            "chat_types": chat_types,
            "retrieval_modes": retrieval_modes,
            "filter_dimensions": sorted(filter_dimensions),
        }


LOCAL_RESEARCH_CORPUS = ResearchCorpusSpec(
    name="local-default",
    version="2026-03-23",
    description=(
        "Local judged corpus spanning retrieval, projections, edits, deletes, "
        "topic repair, capsules, pipeline telemetry, and presentation surfaces."
    ),
    messages=(
        _message(
            message_id="rc-ret-001",
            session_id="atlas-research",
            sender_id="ou_architect",
            content=(
                "atlas search harness persists bm25 postings hnsw vectors and "
                "hybrid fusion for judged memory search"
            ),
            topic_id="retrieval_harness",
            offset_minutes=0,
        ),
        _message(
            message_id="rc-ret-002",
            session_id="atlas-research",
            sender_id="ou_architect",
            content=(
                "acceptance thresholds track hit at one hit at three hit at five "
                "ndcg at three ndcg at five plus cold warm latency"
            ),
            topic_id="retrieval_harness",
            offset_minutes=2,
        ),
        _message(
            message_id="rc-ret-003",
            session_id="atlas-research",
            sender_id="ou_researcher",
            content=(
                "research corpus covers session rollups topic summaries capsule "
                "summaries and l0 abstracts rebuilt from authoritative raw messages"
            ),
            topic_id="retrieval_harness",
            offset_minutes=4,
        ),
        _message(
            message_id="rc-ret-004",
            session_id="atlas-research",
            sender_id="ou_researcher",
            content=(
                "judged corpus includes lexical hybrid and vector retrieval modes "
                "with scoped queries over session group topic and thread surfaces"
            ),
            topic_id="retrieval_harness",
            offset_minutes=6,
        ),
        _message(
            message_id="rc-id-001",
            session_id="launch-mirror",
            sender_id="ou_release_owner",
            content=(
                "launch room mirrors feishu oc_launch_room updates into dm "
                "projections for ou_release_owner and ou_support_owner"
            ),
            topic_id="identity_projections",
            offset_minutes=8,
            chat_type="group",
            group_id="oc_launch_room",
            group_subject="Launch Room",
        ),
        _message(
            message_id="rc-id-002",
            session_id="launch-mirror",
            sender_id="ou_support_owner",
            content=(
                "thread checklist keeps origin anchors projection scope private_dm "
                "public_group and raw_global aligned during fanout"
            ),
            topic_id="identity_projections",
            offset_minutes=9,
            chat_type="group",
            group_id="oc_launch_room",
            group_subject="Launch Room",
            message_thread_id="launch-thread-1",
            thread_parent_id="rc-id-001",
            reply_to_id="rc-id-001",
        ),
        _message(
            message_id="rc-id-003",
            session_id="launch-mirror",
            sender_id="ou_observer",
            content=(
                "release rumor says watchdog locks were skipped and invalidation "
                "failed before launch"
            ),
            topic_id="identity_projections",
            offset_minutes=10,
            chat_type="group",
            group_id="oc_launch_room",
            group_subject="Launch Room",
        ),
        _message(
            message_id="rc-ed-001",
            session_id="quality-repair",
            sender_id="ou_qa_lead",
            content="placeholder benchmark note claims fake pass values and stale thresholds",
            topic_id="evaluation_quality",
            offset_minutes=12,
        ),
        _message(
            message_id="rc-top-001",
            session_id="quality-repair",
            sender_id="ou_qa_lead",
            content=(
                "topic repair observes canonical topic drift correction split merge "
                "and reparent operations for atlas incidents"
            ),
            topic_id="topic_lifecycle",
            offset_minutes=14,
        ),
        _message(
            message_id="rc-cap-001",
            session_id="quality-repair",
            sender_id="ou_qa_lead",
            content=(
                "capsule lifecycle records threshold chars rollover backlinks "
                "forward links vector refresh and raw rebuild discipline"
            ),
            topic_id="capsule_lifecycle",
            offset_minutes=16,
        ),
        _message(
            message_id="rc-ops-001",
            session_id="ops-war-room",
            sender_id="ou_ops_lead",
            content=(
                "semantic queue backlog watchdog wakes stalled jobs and lock "
                "manager preserves raw first rebuild consistency"
            ),
            topic_id="pipeline_ops",
            offset_minutes=18,
            chat_type="group",
            group_id="oc_ops_room",
            group_subject="Ops War Room",
        ),
        _message(
            message_id="rc-ops-002",
            session_id="ops-war-room",
            sender_id="ou_ops_lead",
            content=(
                "cache report tracks hits misses evictions and lookup latency by "
                "tenant session query dimensions"
            ),
            topic_id="pipeline_ops",
            offset_minutes=19,
            chat_type="group",
            group_id="oc_ops_room",
            group_subject="Ops War Room",
            message_thread_id="ops-thread-1",
            thread_parent_id="rc-ops-001",
            reply_to_id="rc-ops-001",
        ),
        _message(
            message_id="rc-pres-001",
            session_id="ops-war-room",
            sender_id="ou_presenter",
            content=(
                "forum presentation linear presentation and capsule cards all "
                "cite origin anchors across tiers"
            ),
            topic_id="presentation_surface",
            offset_minutes=21,
        ),
    ),
    edits=(
        MessageEditRequest(
            tenant_id="research-template",
            origin_message_id="rc-ed-001",
            content=(
                "benchmark note now records real thresholds measured retrieval "
                "quality and rebuild time guardrails"
            ),
            ts=_ts(13),
        ),
    ),
    deletes=(
        MessageDeleteRequest(
            tenant_id="research-template",
            origin_message_id="rc-id-003",
            ts=_ts(11),
        ),
    ),
    cases=(
        _case(
            label="hybrid-indexes",
            query="bm25 postings hnsw vectors hybrid fusion",
            judgments=(("rc-ret-001", 1.0),),
            session_id="atlas-research",
            retrieval_mode="hybrid",
        ),
        _case(
            label="quality-thresholds",
            query="hit at one ndcg at five cold warm latency",
            judgments=(("rc-ret-002", 1.0),),
            session_id="atlas-research",
            retrieval_mode="hybrid",
        ),
        _case(
            label="surface-coverage",
            query="session rollups topic summaries capsule summaries l0 abstracts",
            judgments=(("rc-ret-003", 1.0), ("rc-ret-004", 0.5)),
            session_id="atlas-research",
            retrieval_mode="hybrid",
            max_results=8,
        ),
        _case(
            label="scoped-modes",
            query="lexical hybrid vector retrieval modes scoped queries session group topic thread",
            judgments=(("rc-ret-004", 1.0),),
            session_id="atlas-research",
            retrieval_mode="vector",
        ),
        _case(
            label="group-projections",
            query="dm projections ou_release_owner ou_support_owner public_group fanout",
            judgments=(("rc-id-001", 1.0), ("rc-id-002", 0.8)),
            session_id="launch-mirror",
            group_id="oc_launch_room",
            chat_type="group",
            retrieval_mode="hybrid",
            max_results=8,
        ),
        _case(
            label="thread-projection-scope",
            query="origin anchors projection scope private_dm public_group raw_global",
            judgments=(("rc-id-002", 1.0),),
            session_id="launch-mirror",
            group_id="oc_launch_room",
            message_thread_id="launch-thread-1",
            chat_type="group",
            retrieval_mode="lexical",
        ),
        _case(
            label="edited-thresholds",
            query="real thresholds measured retrieval quality rebuild time guardrails",
            judgments=(("rc-ed-001", 1.0),),
            session_id="quality-repair",
            retrieval_mode="hybrid",
        ),
        _case(
            label="topic-repair",
            query="canonical topic drift correction split merge reparent",
            judgments=(("rc-top-001", 1.0),),
            session_id="quality-repair",
            topic_id="topic_lifecycle",
            retrieval_mode="lexical",
        ),
        _case(
            label="capsule-lifecycle",
            query="threshold chars rollover backlinks forward links vector refresh",
            judgments=(("rc-cap-001", 1.0),),
            session_id="quality-repair",
            topic_id="capsule_lifecycle",
            retrieval_mode="vector",
        ),
        _case(
            label="watchdog-rebuild",
            query="watchdog stalled jobs raw first rebuild consistency",
            judgments=(("rc-ops-001", 1.0),),
            session_id="ops-war-room",
            group_id="oc_ops_room",
            chat_type="group",
            retrieval_mode="hybrid",
        ),
        _case(
            label="cache-telemetry",
            query="hits misses evictions lookup latency tenant session query dimensions",
            judgments=(("rc-ops-002", 1.0),),
            session_id="ops-war-room",
            group_id="oc_ops_room",
            message_thread_id="ops-thread-1",
            chat_type="group",
            retrieval_mode="lexical",
        ),
        _case(
            label="presentation-surface",
            query="forum presentation linear presentation capsule cards origin anchors tiers",
            judgments=(("rc-pres-001", 1.0),),
            session_id="ops-war-room",
            topic_id="presentation_surface",
            retrieval_mode="hybrid",
        ),
    ),
    targets=AcceptanceTargets(),
)


RESEARCH_CORPORA: Mapping[str, ResearchCorpusSpec] = {
    LOCAL_RESEARCH_CORPUS.name: LOCAL_RESEARCH_CORPUS,
}


def get_research_corpus(name: str) -> ResearchCorpusSpec:
    corpus_name = str(name or LOCAL_RESEARCH_CORPUS.name)
    try:
        return RESEARCH_CORPORA[corpus_name]
    except KeyError as exc:
        available = ", ".join(sorted(RESEARCH_CORPORA))
        raise ValueError(f"unknown research corpus {corpus_name!r}; available: {available}") from exc


def list_research_corpora() -> List[ResearchCorpusSpec]:
    return [RESEARCH_CORPORA[name] for name in sorted(RESEARCH_CORPORA)]
