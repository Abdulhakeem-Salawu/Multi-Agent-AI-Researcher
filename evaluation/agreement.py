from __future__ import annotations

from itertools import combinations
from typing import Any

from .schemas import BenchmarkCase


def cohens_kappa(
    ratings_a: list[Any],
    ratings_b: list[Any],
    *,
    ordinal: bool = False,
    categories: list[Any] | None = None,
) -> float | None:
    """Cohen's kappa; quadratic weights are used for ordinal ratings when requested."""
    if len(ratings_a) != len(ratings_b):
        raise ValueError("Both annotators must have the same number of ratings.")
    if not ratings_a:
        return None

    all_categories = categories or sorted(set(ratings_a) | set(ratings_b), key=str)
    index = {category: position for position, category in enumerate(all_categories)}
    if any(value not in index for value in ratings_a + ratings_b):
        raise ValueError("Ratings contain a value outside the declared category set.")
    count = len(all_categories)
    if count < 2:
        return 1.0 if ratings_a == ratings_b else None

    if ordinal:
        denominator = (count - 1) ** 2
        observed = sum(
            1.0 - ((index[a] - index[b]) ** 2 / denominator)
            for a, b in zip(ratings_a, ratings_b)
        ) / len(ratings_a)
        marginal_a = [ratings_a.count(value) / len(ratings_a) for value in all_categories]
        marginal_b = [ratings_b.count(value) / len(ratings_b) for value in all_categories]
        expected = sum(
            marginal_a[i] * marginal_b[j] * (1.0 - ((i - j) ** 2 / denominator))
            for i in range(count)
            for j in range(count)
        )
    else:
        observed = sum(a == b for a, b in zip(ratings_a, ratings_b)) / len(ratings_a)
        marginal_a = [ratings_a.count(value) / len(ratings_a) for value in all_categories]
        marginal_b = [ratings_b.count(value) / len(ratings_b) for value in all_categories]
        expected = sum(a * b for a, b in zip(marginal_a, marginal_b))

    if expected == 1.0:
        return 1.0 if observed == 1.0 else None
    return (observed - expected) / (1.0 - expected)


def case_inter_annotator_agreement(case: BenchmarkCase) -> dict[str, Any]:
    if len(case.annotations) < 2:
        return {
            "case_id": case.case_id,
            "annotator_count": len(case.annotations),
            "pair_count": 0,
            "source_relevance_quadratic_weighted_kappa": None,
            "claim_status_kappa": None,
            "warnings": ["At least two independent annotations are required to measure agreement."],
        }

    source_scores: list[float] = []
    claim_scores: list[float] = []
    warnings: list[str] = []

    for left, right in combinations(case.annotations, 2):
        left_source = {item.source_id: item.grade for item in left.source_relevance}
        right_source = {item.source_id: item.grade for item in right.source_relevance}
        shared_sources = sorted(set(left_source) & set(right_source))
        if shared_sources:
            score = cohens_kappa(
                [left_source[key] for key in shared_sources],
                [right_source[key] for key in shared_sources],
                ordinal=True,
                categories=[0, 1, 2, 3],
            )
            if score is not None:
                source_scores.append(score)
        else:
            warnings.append(
                f"No overlapping source relevance judgments for {left.annotator_id} and {right.annotator_id}."
            )

        left_claims = {item.claim_id: item.status for item in left.claim_judgments}
        right_claims = {item.claim_id: item.status for item in right.claim_judgments}
        shared_claims = sorted(set(left_claims) & set(right_claims))
        if shared_claims:
            score = cohens_kappa(
                [left_claims[key] for key in shared_claims],
                [right_claims[key] for key in shared_claims],
                categories=["supported", "contradicted", "insufficient_evidence", "uncertain"],
            )
            if score is not None:
                claim_scores.append(score)
        else:
            warnings.append(
                f"No overlapping claim judgments for {left.annotator_id} and {right.annotator_id}."
            )

    return {
        "case_id": case.case_id,
        "annotator_count": len(case.annotations),
        "pair_count": len(list(combinations(case.annotations, 2))),
        "source_relevance_quadratic_weighted_kappa": (
            sum(source_scores) / len(source_scores) if source_scores else None
        ),
        "claim_status_kappa": sum(claim_scores) / len(claim_scores) if claim_scores else None,
        "warnings": warnings,
    }
