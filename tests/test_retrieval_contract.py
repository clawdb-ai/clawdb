from __future__ import annotations

import pytest

from clawdb.embeddings import EmbeddingAuthContext
from clawdb.models import MessageIn, SearchRequest
from clawdb.retrieval import HybridRetrievalEngine, RetrievalDoc, resolve_retrieval_weights


async def _ingest_retrieval_message(service, *, content: str, topic_id: str = "retrieval_topic"):
    return await service.ingest_message(
        MessageIn(
            tenant_id="default",
            session_id="retrieval-session",
            role="user",
            content=content,
            channel="feishu",
            platform="feishu",
            chat_type="direct",
            account_id="botacct",
            sender_id="ou_user_retrieval",
            from_id="ou_user_retrieval",
            to_id="ou_user_retrieval",
            topic_id=topic_id,
            message_id="retrieval-request-1",
        )
    )


def test_retrieval_engine_honors_exact_weights_by_mode(monkeypatch):
    engine = HybridRetrievalEngine(dim=8)
    docs = [RetrievalDoc(doc_id="a", text="alpha"), RetrievalDoc(doc_id="b", text="beta")]

    monkeypatch.setattr(engine.bm25, "build", lambda docs: None)
    monkeypatch.setattr(engine.hnsw, "build", lambda docs: None)
    monkeypatch.setattr(engine.bm25, "search", lambda query, top_k=None: [("a", 3.0), ("b", 1.5)])
    monkeypatch.setattr(engine.hnsw, "search", lambda query, top_k=None: [("a", 0.2), ("b", 0.9)])

    hybrid = engine.search("contract", docs, top_k=2, retrieval_mode="hybrid")
    lexical = engine.search("contract", docs, top_k=2, retrieval_mode="lexical")
    vector = engine.search("contract", docs, top_k=2, retrieval_mode="vector")

    assert resolve_retrieval_weights("hybrid") == pytest.approx((0.30, 0.70))
    assert resolve_retrieval_weights("lexical") == pytest.approx((1.0, 0.0))
    assert resolve_retrieval_weights("vector") == pytest.approx((0.0, 1.0))

    assert [item.doc_id for item in hybrid] == ["b", "a"]
    assert hybrid[0].score == pytest.approx((0.30 * 0.5) + (0.70 * 0.9))
    assert hybrid[1].score == pytest.approx((0.30 * 1.0) + (0.70 * 0.2))

    assert [item.doc_id for item in lexical] == ["a", "b"]
    assert [item.score for item in lexical] == pytest.approx([1.0, 0.5])

    assert [item.doc_id for item in vector] == ["b", "a"]
    assert [item.score for item in vector] == pytest.approx([0.9, 0.2])


@pytest.mark.asyncio
async def test_search_returns_cross_tier_results_with_structured_citations(service):
    ack = await _ingest_retrieval_message(
        service,
        content="retrieval contract anchor hybrid capsule topic summary",
    )

    result = await service.search(
        SearchRequest(
            query="anchor hybrid",
            tenant_id="default",
            max_results=10,
            retrieval_mode="hybrid",
        )
    )

    entity_types = [item.entity_type for item in result.results]
    assert result.cache_hit is False
    assert entity_types[0] == "raw_message"
    assert {"l0_abstract", "session_rollup", "topic", "capsule", "raw_message"} <= set(entity_types)
    assert all(item.retrieval_mode == "hybrid" for item in result.results)

    by_type = {item.entity_type: item for item in result.results}
    assert by_type["raw_message"].citation == f"origin:{ack.origin_message_id}"
    assert by_type["raw_message"].citations == [f"origin:{ack.origin_message_id}"]

    assert by_type["l0_abstract"].citation.startswith("l0:default:")
    assert by_type["l0_abstract"].citations[1:] == [f"origin:{ack.origin_message_id}"]

    assert by_type["session_rollup"].citation.startswith("rollup:default:")
    assert by_type["session_rollup"].citations[1:] == [f"origin:{ack.origin_message_id}"]

    assert by_type["topic"].citation == "topic:retrieval_topic"
    assert by_type["topic"].citations[1:] == [f"origin:{ack.origin_message_id}"]

    assert by_type["capsule"].citation.startswith("capsule:")
    assert by_type["capsule"].citations[1:] == [f"origin:{ack.origin_message_id}"]


@pytest.mark.asyncio
async def test_search_rerank_is_optional(service, monkeypatch):
    await _ingest_retrieval_message(
        service,
        content="rerank contract anchor hybrid capsule topic summary",
    )

    ctx = EmbeddingAuthContext(provider="openai", api_key="test-key", model="test-embed")

    async def _fake_embed(_ctx, texts):
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(service, "_embed_texts_cached", _fake_embed)

    without_rerank = await service.search(
        SearchRequest(
            query="rerank anchor",
            tenant_id="default",
            session_id="retrieval-session",
            max_results=5,
            retrieval_mode="hybrid",
            rerank="off",
        ),
        embedding_ctx=ctx,
    )
    with_rerank = await service.search(
        SearchRequest(
            query="rerank anchor",
            tenant_id="default",
            session_id="retrieval-session",
            max_results=5,
            retrieval_mode="hybrid",
            rerank="auto",
        ),
        embedding_ctx=ctx,
    )

    assert all(item.reranked is False for item in without_rerank.results)
    assert all(float(item.score_semantic) == 0.0 for item in without_rerank.results)

    assert any(item.reranked is True for item in with_rerank.results)
    assert all(float(item.score_semantic) == pytest.approx(1.0) for item in with_rerank.results)
