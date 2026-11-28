from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from evaluation.metrics import retrieval_metrics

from .models import ExternalBenchmarkItem

_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_PUNCTUATION_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_qa_answer(value: str) -> str:
    value = value.lower()
    value = _PUNCTUATION_RE.sub(" ", value)
    value = _ARTICLE_RE.sub(" ", value)
    return _WHITESPACE_RE.sub(" ", value).strip()


def score_frames_answer(prediction: str, reference: str) -> dict[str, float]:
    """Return SQuAD-style normalized exact match and token F1."""
    predicted = normalize_qa_answer(prediction).split()
    expected = normalize_qa_answer(reference).split()
    exact_match = float(predicted == expected)

    if not predicted or not expected:
        token_f1 = float(predicted == expected)
    else:
        overlap = sum((Counter(predicted) & Counter(expected)).values())
        if overlap == 0:
            token_f1 = 0.0
        else:
            precision = overlap / len(predicted)
            recall = overlap / len(expected)
            token_f1 = 2 * precision * recall / (precision + recall)
    return {"exact_match": exact_match, "token_f1": token_f1}


def score_frames_retrieval(
    item: ExternalBenchmarkItem,
    retrieved_article_urls: list[str],
    *,
    k: int = 10,
) -> dict[str, float]:
    if item.dataset_id != "frames":
        raise ValueError("score_frames_retrieval requires a FRAMES item.")
    relevant = (item.annotations or {}).get("relevant_article_urls", [])
    return retrieval_metrics(retrieved_article_urls, relevant, k=k)


def score_browsecomp_plus_retrieval(
    item: ExternalBenchmarkItem,
    retrieved_docids: list[str],
    *,
    k: int = 10,
) -> dict[str, dict[str, float]]:
    if item.dataset_id != "browsecomp-plus":
        raise ValueError("score_browsecomp_plus_retrieval requires a BrowseComp-Plus item.")
    annotations = item.annotations or {}
    return {
        "evidence": retrieval_metrics(retrieved_docids, annotations.get("evidence_docids", []), k=k),
        "gold": retrieval_metrics(retrieved_docids, annotations.get("gold_docids", []), k=k),
    }


def score_drb2_rubric_judgments(
    item: ExternalBenchmarkItem,
    judgments: dict[str, int],
) -> dict[str, Any]:
    """Summarise DRB II's native three-way labels without collapsing their meaning."""
    if item.dataset_id != "deepresearch-bench-ii":
        raise ValueError("score_drb2_rubric_judgments requires a DRB II item.")
    rubrics = (item.annotations or {}).get("rubrics", [])
    expected = {rubric["rubric_id"] for rubric in rubrics}
    actual = set(judgments)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise ValueError(
            f"DRB II rubric coverage mismatch; missing={sorted(missing)}, extra={sorted(extra)}."
        )
    invalid = {
        rubric_id: value
        for rubric_id, value in judgments.items()
        if type(value) is not int or value not in {-1, 0, 1}
    }
    if invalid:
        raise ValueError(f"DRB II rubric scores must be -1, 0, or 1: {invalid}.")

    counts: dict[str, Counter[int]] = defaultdict(Counter)
    for rubric in rubrics:
        counts[rubric["dimension"]][judgments[rubric["rubric_id"]]] += 1

    dimensions: dict[str, dict[str, float | int]] = {}
    total = Counter()
    for dimension, values in sorted(counts.items()):
        total.update(values)
        count = sum(values.values())
        dimensions[dimension] = {
            "rubric_count": count,
            "satisfied_count": values[1],
            "not_mentioned_count": values[0],
            "blocked_reference_count": values[-1],
            "satisfied_rate": values[1] / count if count else 0.0,
            "blocked_reference_rate": values[-1] / count if count else 0.0,
        }
    total_count = sum(total.values())
    return {
        "dataset_id": item.dataset_id,
        "task_id": item.task_id,
        "rubric_count": total_count,
        "satisfied_count": total[1],
        "not_mentioned_count": total[0],
        "blocked_reference_count": total[-1],
        "satisfied_rate": total[1] / total_count if total_count else 0.0,
        "blocked_reference_rate": total[-1] / total_count if total_count else 0.0,
        "dimensions": dimensions,
        "score_semantics": "1=satisfied, 0=not mentioned, -1=blocked reference",
    }
