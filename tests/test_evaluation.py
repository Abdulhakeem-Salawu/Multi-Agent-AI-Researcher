from evaluation.metrics import evaluate_report, report_metrics, retrieval_metrics


def test_retrieval_metrics_match_expected_ranked_relevance():
    result = retrieval_metrics(
        [
            "https://example.org/a",
            "https://example.org/x",
            "https://example.org/b",
        ],
        ["https://example.org/a", "https://example.org/b"],
        k=3,
    )

    assert result["precision_at_k"] == 2 / 3
    assert result["recall_at_k"] == 1.0
    assert result["mrr_at_k"] == 1.0
    assert result["ndcg_at_k"] > 0.9


def test_report_metrics_detect_invalid_citations_and_unresolved_claims():
    report = {
        "answer": "Claim one [S1]. Claim two [S9].",
        "sources": [{"id": "S1", "url": "https://example.org/a"}],
        "claims": [{"claim": "Claim one."}, {"claim": "Claim two."}],
        "claim_verifications": [
            {"status": "supported"},
            {"status": "uncertain"},
        ],
        "report_review": {
            "overall_score": 4.0,
            "ready_for_publication": False,
        },
    }

    result = report_metrics(report)

    assert result["citation_count"] == 2
    assert result["invalid_citation_count"] == 1
    assert result["citation_precision"] == 0.5
    assert result["verification_coverage"] == 1.0
    assert result["supported_claim_rate"] == 0.5
    assert result["unresolved_claim_count"] == 1
    assert result["publication_ready"] == 0


def test_evaluate_report_adds_human_labelled_retrieval_metrics():
    result = evaluate_report(
        {
            "answer": "The answer [S1].",
            "sources": [{"id": "S1", "url": "https://example.org/a"}],
        },
        relevant_source_urls=["https://example.org/a"],
    )

    assert result["retrieval"]["recall_at_k"] == 1.0
    assert result["report"]["citation_precision"] == 1.0
