from __future__ import annotations

import math

import pytest

from clawdb.metrics import aggregate_ranked_relevance, hit_at_k, ndcg_at_k, percentile
from clawdb.models import (
    AcceptanceBenchmarkRequest,
    AcceptanceJudgment,
    AcceptanceSearchCase,
    AcceptanceTargets,
    MessageIn,
    SearchRequest,
)


def test_acceptance_metric_math_deduplicates_repeated_citations():
    ranked_keys = [
        ["entity:raw_message:raw-1", "origin:raw-1"],
        ["origin:raw-1"],
        ["entity:raw_message:raw-2", "origin:raw-2"],
    ]
    judgments = {
        "origin:raw-1": 3.0,
        "origin:raw-2": 1.0,
    }

    ranked_relevance = aggregate_ranked_relevance(ranked_keys, judgments)

    assert ranked_relevance == [3.0, 0.0, 1.0]
    assert hit_at_k(ranked_relevance, 1) == pytest.approx(1.0)
    assert hit_at_k(ranked_relevance, 2) == pytest.approx(1.0)

    expected = (
        ((2.0**3.0 - 1.0) / math.log2(2.0))
        + ((2.0**1.0 - 1.0) / math.log2(4.0))
    ) / (
        ((2.0**3.0 - 1.0) / math.log2(2.0))
        + ((2.0**1.0 - 1.0) / math.log2(3.0))
    )
    assert ndcg_at_k(ranked_relevance, [3.0, 1.0], 3) == pytest.approx(expected)
    assert percentile([1.0, 2.0, 3.0, 4.0], 95.0) == pytest.approx(4.0)


async def _ingest_acceptance_message(service, *, message_id: str, content: str):
    return await service.ingest_message(
        MessageIn(
            tenant_id="default",
            session_id="acceptance-session",
            role="user",
            content=content,
            channel="feishu",
            platform="feishu",
            chat_type="direct",
            account_id="botacct",
            sender_id="ou_acceptance_user",
            from_id="ou_acceptance_user",
            to_id="ou_acceptance_user",
            message_id=message_id,
            topic_id="acceptance_topic",
        )
    )


@pytest.mark.asyncio
async def test_service_acceptance_benchmark_evaluates_targets(service):
    alpha = await _ingest_acceptance_message(
        service,
        message_id="acceptance-alpha",
        content="saffron kilobyte lantern acceptance alpha anchor",
    )
    beta = await _ingest_acceptance_message(
        service,
        message_id="acceptance-beta",
        content="zircon meadow envelope acceptance beta anchor",
    )

    report = await service.evaluate_acceptance(
        AcceptanceBenchmarkRequest(
            cases=[
                AcceptanceSearchCase(
                    label="alpha",
                    search=SearchRequest(
                        query="saffron kilobyte lantern",
                        tenant_id="default",
                        session_id="acceptance-session",
                        retrieval_mode="hybrid",
                        max_results=6,
                    ),
                    judgments=[AcceptanceJudgment(match_key=f"origin:{alpha.origin_message_id}")],
                ),
                AcceptanceSearchCase(
                    label="beta",
                    search=SearchRequest(
                        query="zircon meadow envelope",
                        tenant_id="default",
                        session_id="acceptance-session",
                        retrieval_mode="hybrid",
                        max_results=6,
                    ),
                    judgments=[AcceptanceJudgment(match_key=f"origin:{beta.origin_message_id}")],
                ),
            ],
            latency_repetitions=1,
            targets=AcceptanceTargets(
                hit_at={1: 1.0, 3: 1.0},
                ndcg_at={3: 1.0},
                cold_latency_p95_ms=5_000.0,
                warm_latency_p95_ms=5_000.0,
                max_working_set_bytes=256 * 1024 * 1024,
                max_rebuild_time_ms=5_000.0,
            ),
        )
    )

    assert report.passed is True
    assert report.case_count == 2
    assert report.hit_at[1] == pytest.approx(1.0)
    assert report.hit_at[3] == pytest.approx(1.0)
    assert report.ndcg_at[3] == pytest.approx(1.0)
    assert report.cold_latency_ms_p95 >= 0.0
    assert report.warm_latency_ms_p95 >= 0.0
    assert report.working_set_bytes > 0
    assert report.dataframe_bytes > 0
    assert report.rebuild_time_ms >= 0.0
    assert report.authoritative_raw_messages == 2
    assert report.rebuilt_topics >= 1
    assert report.rebuilt_capsules >= 1
    assert {check.name for check in report.checks} == {
        "hit@1",
        "hit@3",
        "ndcg@3",
        "cold-latency-p95",
        "warm-latency-p95",
        "working-set-memory",
        "rebuild-time",
    }
    assert all(check.passed for check in report.checks)
    assert report.cases[0].top_match_keys
    assert report.cases[0].matched_relevance
