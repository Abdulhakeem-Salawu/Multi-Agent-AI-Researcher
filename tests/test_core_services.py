from __future__ import annotations

from datetime import datetime, timezone

from research_core.audit import audit_claims
from research_core.retrieval import rank_hits
from research_core.schemas import ResearchReport, RetrievedHit
from research_core.storage import ResearchRunStore


def test_rank_hits_deduplicates_urls_and_uses_stable_query_order():
    common = {
        "title": "Hybrid retrieval",
        "url": "https://example.org/hybrid",
        "content": "Hybrid retrieval combines lexical and vector search for relevant evidence.",
        "score": 0.9,
        "published_date": "2025-01-01",
        "retrieved_at": datetime.now(timezone.utc),
    }
    late_duplicate = RetrievedHit(
        **common,
        query="secondary query",
        query_index=1,
        result_index=0,
    )
    primary = {**common, "title": "Primary title"}
    first_hit = RetrievedHit(
        **primary,
        query="primary query",
        query_index=0,
        result_index=0,
    )
    ranked = rank_hits(
        "How does hybrid retrieval combine lexical and vector search?",
        [late_duplicate, first_hit],
        top_k=5,
    )
    assert len(ranked) == 1
    assert ranked[0][0].title == "Primary title"


def test_claim_audit_labels_lexical_match_as_candidate_not_verified():
    answer = (
        "- BM25 ranks documents using term frequency and inverse document frequency [S1].\n"
        "- This claim has no attached citation and needs review."
    )
    checks = audit_claims(
        answer,
        {
            "S1": (
                "BM25 ranks documents using term frequency and inverse document frequency "
                "with document length normalization."
            )
        },
    )
    assert checks[0].status == "candidate_supported"
    assert "does not establish entailment" in checks[0].reason
    assert checks[1].status == "uncited"


def test_completed_report_store_round_trip(tmp_path):
    store = ResearchRunStore(tmp_path / "research.sqlite3")
    report = ResearchReport(
        question="How does BM25 rank documents?",
        answer="BM25 is a ranking method.",
        query_plan=["BM25 ranking"],
    )
    run_id = store.save(report)
    saved = store.get(run_id)

    assert saved is not None
    assert saved["question"] == report.question
    assert saved["run_id"] == run_id
    assert saved["sources"] == []
