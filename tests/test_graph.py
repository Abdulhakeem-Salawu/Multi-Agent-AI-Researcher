from __future__ import annotations

import json

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
    assert report["report_review"]["status"] == "not_run"
    assert report["claim_verifications"]
    assert all(item["status"] == "uncertain" for item in report["claim_verifications"])


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
    assert report["report_review"]["status"] == "not_run"


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


class ScriptedLLM:
    """Deterministic LLM stub for exercising verification and gap-loop routing."""

    def __init__(self, keep_first_verdict_insufficient: bool = False):
        self.verification_calls = 0
        self.keep_first_verdict_insufficient = keep_first_verdict_insufficient

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        if "research planner" in system_prompt:
            return json.dumps(["BM25 ranking function"])
        if "evidence-grounded research analyst" in system_prompt:
            return "BM25 uses term frequency and inverse document frequency to rank documents. [S1]"
        if "atomic claim extractor" in system_prompt:
            return json.dumps(
                [
                    {
                        "claim": "BM25 uses term frequency and inverse document frequency to rank documents.",
                        "source_ids": ["S1"],
                    }
                ]
            )
        if "independent claim verifier" in system_prompt:
            self.verification_calls += 1
            insufficient = (
                self.keep_first_verdict_insufficient or self.verification_calls == 1
            )
            if insufficient:
                return json.dumps(
                    {
                        "status": "insufficient_evidence",
                        "confidence": 0.35,
                        "supporting_evidence": [],
                        "contradicting_evidence": [],
                        "reasoning": "The available passage is not sufficient for a confident decision.",
                        "research_needed": True,
                    }
                )
            return json.dumps(
                {
                    "status": "supported",
                    "confidence": 0.91,
                    "supporting_evidence": [
                        {
                            "source_id": "S1",
                            "quote": (
                                "BM25 uses term frequency and inverse document frequency to rank documents."
                            ),
                            "explanation": "The passage directly states the claim.",
                        }
                    ],
                    "contradicting_evidence": [],
                    "reasoning": "The supplied passage directly supports this proposition.",
                    "research_needed": False,
                }
            )
        if "knowledge-gap research planner" in system_prompt:
            claim_id = json.loads(user_prompt)["unresolved_claims"][0]["claim_id"]
            return json.dumps(
                [
                    {
                        "query": "BM25 term frequency inverse document frequency primary explanation",
                        "claim_ids": [claim_id],
                    }
                ]
            )
        if "independent research-report quality reviewer" in system_prompt:
            return json.dumps(
                {
                    "overall_score": 4.5,
                    "completeness_score": 4.5,
                    "analytical_depth_score": 4.0,
                    "citation_quality_score": 5.0,
                    "ready_for_publication": True,
                    "strengths": ["The report cites a directly relevant source."],
                    "issues": [],
                    "rationale": "The report is concise and its claim has a verified evidence span.",
                }
            )
        raise AssertionError(f"Unexpected model call: {system_prompt}")


def test_claim_verifier_triggers_one_followup_and_rechecks_claim():
    evidence_text = (
        "BM25 uses term frequency and inverse document frequency to rank documents. "
        "Its scoring function also normalizes for document length."
    )
    provider = StaticSearchProvider(
        [
            SearchHit(
                title="BM25 technical explanation",
                url="https://example.org/bm25",
                content=evidence_text,
                score=0.95,
                published_date="2025-03-14",
            )
        ]
    )
    llm = ScriptedLLM()
    graph = build_research_graph(
        ResearchDependencies(
            search_provider=provider,
            llm=llm,
            max_queries=1,
            max_results_per_query=5,
            max_evidence_sources=5,
            max_research_rounds=1,
        )
    )

    result = graph.invoke({"question": "How does BM25 rank documents?"})
    report = result["report"]

    assert len(provider.calls) == 2
    assert report["research_rounds"] == 1
    assert len(report["query_plan"]) == 2
    assert report["claim_verifications"][0]["status"] == "supported"
    assert report["claim_verifications"][0]["supporting_evidence"][0]["quote"] in evidence_text
    assert llm.verification_calls == 2
    assert report["report_review"]["status"] == "reviewed"
    assert report["report_review"]["ready_for_publication"] is True


def test_research_gap_loop_stops_at_configured_round_limit():
    provider = StaticSearchProvider(
        [
            SearchHit(
                title="BM25 overview",
                url="https://example.org/bm25",
                content=(
                    "BM25 uses term frequency and inverse document frequency to rank documents."
                ),
                score=0.9,
            )
        ]
    )
    llm = ScriptedLLM(keep_first_verdict_insufficient=True)
    graph = build_research_graph(
        ResearchDependencies(
            search_provider=provider,
            llm=llm,
            max_queries=1,
            max_research_rounds=1,
        )
    )

    result = graph.invoke({"question": "How does BM25 rank documents?"})
    report = result["report"]

    assert len(provider.calls) == 2
    assert report["research_rounds"] == 1
    assert report["claim_verifications"][0]["status"] == "insufficient_evidence"
    assert report["report_review"]["ready_for_publication"] is False


class InvalidQuoteLLM(ScriptedLLM):
    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        if "independent claim verifier" in system_prompt:
            self.verification_calls += 1
            return json.dumps(
                {
                    "status": "supported",
                    "confidence": 0.99,
                    "supporting_evidence": [
                        {
                            "source_id": "S1",
                            "quote": "This quotation was fabricated and is not in the source.",
                            "explanation": "This quote is intentionally invalid for the safety test.",
                        }
                    ],
                    "contradicting_evidence": [],
                    "reasoning": "The intentionally invalid quote must not be trusted.",
                    "research_needed": False,
                }
            )
        return super().complete(system_prompt, user_prompt, temperature)


def test_verifier_cannot_mark_claim_supported_with_fabricated_quote():
    provider = StaticSearchProvider(
        [
            SearchHit(
                title="BM25 overview",
                url="https://example.org/bm25",
                content="BM25 uses term frequency and inverse document frequency to rank documents.",
                score=0.9,
            )
        ]
    )
    llm = InvalidQuoteLLM()
    graph = build_research_graph(
        ResearchDependencies(
            search_provider=provider,
            llm=llm,
            max_queries=1,
            max_research_rounds=1,
        )
    )

    report = graph.invoke(
        {"question": "How does BM25 rank documents?"}
    )["report"]

    verification = report["claim_verifications"][0]
    assert verification["status"] == "uncertain"
    assert verification["supporting_evidence"] == []
    assert verification["span_warnings"]
    assert report["report_review"]["ready_for_publication"] is False
