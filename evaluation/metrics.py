from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from typing import Any

_CITATION_RE = re.compile(r"\[(S\d+)\]")
_SPACE_RE = re.compile(r"\s+")


def retrieval_metrics(
    retrieved_source_urls: Iterable[str],
    relevant_source_urls: Iterable[str],
    *,
    k: int = 5,
) -> dict[str, float]:
    if k < 1:
        raise ValueError("k must be at least 1.")

    retrieved = list(retrieved_source_urls)[:k]
    relevant = set(relevant_source_urls)
    hits = [1 if url in relevant else 0 for url in retrieved]
    hit_count = sum(hits)
    precision = hit_count / k
    recall = hit_count / len(relevant) if relevant else 0.0

    reciprocal_rank = 0.0
    for rank, is_relevant in enumerate(hits, start=1):
        if is_relevant:
            reciprocal_rank = 1.0 / rank
            break

    dcg = sum(
        relevance / math.log2(rank + 1)
        for rank, relevance in enumerate(hits, start=1)
    )
    ideal_hits = [1] * min(len(relevant), k)
    ideal_dcg = sum(
        relevance / math.log2(rank + 1)
        for rank, relevance in enumerate(ideal_hits, start=1)
    )
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0

    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "mrr_at_k": reciprocal_rank,
        "ndcg_at_k": ndcg,
    }


def report_metrics(report: Mapping[str, Any]) -> dict[str, float | int | None]:
    """Score structure and evidence bookkeeping, not whether the claims are objectively true."""
    sources = report.get("sources", []) or []
    valid_source_ids = {
        str(item.get("id"))
        for item in sources
        if isinstance(item, Mapping) and item.get("id") is not None
    }
    citations = _CITATION_RE.findall(str(report.get("answer", "")))
    valid_citations = sum(1 for source_id in citations if source_id in valid_source_ids)

    verifications = report.get("claim_verifications", []) or []
    verification_count = len(verifications)
    supported_count = sum(
        1 for item in verifications
        if isinstance(item, Mapping) and item.get("status") == "supported"
    )
    contradicted_count = sum(
        1 for item in verifications
        if isinstance(item, Mapping) and item.get("status") == "contradicted"
    )
    unresolved_count = verification_count - supported_count

    claims = report.get("claims", []) or []
    claim_count = len(claims)
    coverage_denominator = max(claim_count, verification_count)
    review = report.get("report_review") or {}
    overall_score = review.get("overall_score") if isinstance(review, Mapping) else None
    ready = review.get("ready_for_publication") if isinstance(review, Mapping) else None

    return {
        "source_count": len(sources),
        "citation_count": len(citations),
        "invalid_citation_count": len(citations) - valid_citations,
        "citation_precision": valid_citations / len(citations) if citations else 0.0,
        "claim_count": claim_count,
        "verification_count": verification_count,
        "verification_coverage": verification_count / coverage_denominator if coverage_denominator else 0.0,
        "supported_claim_rate": supported_count / verification_count if verification_count else 0.0,
        "contradicted_claim_count": contradicted_count,
        "unresolved_claim_count": unresolved_count,
        "report_review_score": overall_score,
        "publication_ready": int(bool(ready)),
    }


def evaluate_report(
    report: Mapping[str, Any],
    *,
    relevant_source_urls: Iterable[str] | None = None,
    retrieval_k: int = 5,
) -> dict[str, Any]:
    output: dict[str, Any] = {"report": report_metrics(report)}
    if relevant_source_urls is not None:
        source_urls = [
            str(item.get("url"))
            for item in (report.get("sources", []) or [])
            if isinstance(item, Mapping) and item.get("url")
        ]
        output["retrieval"] = retrieval_metrics(
            source_urls,
            relevant_source_urls,
            k=retrieval_k,
        )
    return output
