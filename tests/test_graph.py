from __future__ import annotations

from research_agent.dependencies import ResearchDependencies
from research_agent.graph import build_research_graph
from research_core.schemas import SearchHit
from research_core.search import StaticSearchProvider


def test_graph_fans_out_research_and_builds_cited_report():
    provider = StaticSearchProvider(
        [
            SearchHit(
                title="BM25 overview",
                url="https://example.org/bm25",
                content=(
                    "BM25 is a probabilistic ranking function for information retrieval. "
                    "It combines term frequency and inverse document frequency with document length normalization."
                ),
                score=0.92,
                published_date="2025-01-10",
            ),
            SearchHit(
                title="Hybrid retrieval survey",
                url="https://example.org/hybrid",
                content=(
                    "Hybrid retrieval combines lexical search with dense vector retrieval. "
                    "The two methods can complement each other when retrieving relevant evidence."
                ),
                score=0.88,
                published_date="2025-02-01",
            ),
        ]
    )
    graph = build_research_graph(
        ResearchDependencies(
            search_provider=provider,
            llm=None,
            max_queries=4,
            max_results_per_query=8,
            max_evidence_sources=8,
            retrieval_mode="hybrid",
        )
    )

    result = graph.invoke(
        {"question": "How do BM25 ranking and hybrid retrieval compare?"}
    )
    report = result["report"]

    assert len(provider.calls) >= 2
    assert len(report["query_plan"]) >= 2
    assert {source["url"] for source in report["sources"]} == {
        "https://example.org/bm25",
        "https://example.org/hybrid",
    }
    assert "[S1]" in report["answer"]
    assert any("No LLM was configured" in item for item in report["limitations"])


def test_graph_records_search_failures_without_fabricating_sources():
    class FailingSearchProvider:
        def search(self, query: str, max_results: int = 8):
            raise TimeoutError("search endpoint unavailable")

    graph = build_research_graph(
        ResearchDependencies(
            search_provider=FailingSearchProvider(),
            llm=None,
            max_queries=3,
            max_results_per_query=4,
        )
    )

    result = graph.invoke({"question": "What is the evidence for method X?"})
    report = result["report"]

    assert report["sources"] == []
    assert "could not find usable retrieved evidence" in report["answer"].lower()
    assert any("research tasks failed" in item for item in report["limitations"])
    assert len(result["search_errors"]) >= 1


def test_graph_rejects_short_questions():
    class UnusedSearchProvider:
        def search(self, query: str, max_results: int = 8):
            raise AssertionError("Search should not run for invalid input.")

    graph = build_research_graph(
        ResearchDependencies(search_provider=UnusedSearchProvider(), llm=None)
    )

    try:
        graph.invoke({"question": "short"})
    except ValueError as exc:
        assert "at least 8 characters" in str(exc)
    else:
        raise AssertionError("Expected question validation to fail.")
