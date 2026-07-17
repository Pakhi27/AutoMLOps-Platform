"""Tests for external web retrieval (Corrective RAG)."""
from app.services.web_retrieval_tools import ArxivSearchClient, EvidenceRouter, to_retrieved_doc_dicts


def test_arxiv_search_returns_docs():
    client = ArxivSearchClient()
    docs = client.search("gradient boosting tabular classification", max_results=1)
    assert isinstance(docs, list)


def test_evidence_router_weak_detection():
    assert EvidenceRouter.internal_retrieval_is_weak([]) is True
    assert EvidenceRouter.internal_retrieval_is_weak([{"score": 0.1}]) is True
    assert EvidenceRouter.internal_retrieval_is_weak([{"score": 0.8}]) is False


def test_to_retrieved_doc_dicts():
    from app.services.web_retrieval_tools import ExternalDoc

    docs = [
        ExternalDoc(source="http://x", title="T", content="body", score=0.5, chunk_type="web", tags=["web"])
    ]
    out = to_retrieved_doc_dicts(docs)
    assert out[0]["chunk_type"] == "web"
    assert out[0]["category"] == "external_evidence"
