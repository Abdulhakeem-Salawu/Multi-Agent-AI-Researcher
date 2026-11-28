from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .schemas import BenchmarkCase, RunAnnotation

RUBRIC_DIMENSIONS = {
    "completeness",
    "analytical_depth",
    "factual_support",
    "citation_quality",
    "uncertainty_calibration",
    "temporal_accuracy",
    "presentation",
}


def graded_retrieval_metrics(
    retrieved_urls: list[str],
    relevance_by_url: Mapping[str, int],
    *,
    k: int = 5,
    relevant_threshold: int = 2,
) -> dict[str, float | int]:
    if k < 1:
        raise ValueError("k must be at least 1.")
    ranked = retrieved_urls[:k]
    grades = [int(relevance_by_url.get(url, 0)) for url in ranked]
    relevant_total = sum(grade >= relevant_threshold for grade in relevance_by_url.values())
    relevant_retrieved = sum(grade >= relevant_threshold for grade in grades)
    precision = relevant_retrieved / k
    recall = relevant_retrieved / relevant_total if relevant_total else 0.0
    reciprocal_rank = next(
        (1.0 / rank for rank, grade in enumerate(grades, 1) if grade >= relevant_threshold),
        0.0,
    )
    dcg = sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(grades, 1)
    )
    ideal = sorted(relevance_by_url.values(), reverse=True)[:k]
    ideal_dcg = sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal, 1)
    )
    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "mrr_at_k": reciprocal_rank,
        "ndcg_at_k": dcg / ideal_dcg if ideal_dcg else 0.0,
        "relevant_sources_available": relevant_total,
        "relevant_sources_retrieved": relevant_retrieved,
    }


def _classification_metrics(expected: list[str], predicted: list[str]) -> dict[str, Any]:
    labels = ["supported", "contradicted", "insufficient_evidence", "uncertain"]
    confusion = {actual: {guess: 0 for guess in labels} for actual in labels}
    for actual, guess in zip(expected, predicted):
        if actual in confusion and guess in confusion[actual]:
            confusion[actual][guess] += 1

    per_class: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for label in labels:
        true_positive = confusion[label][label]
        false_positive = sum(confusion[actual][label] for actual in labels if actual != label)
        false_negative = sum(confusion[label][guess] for guess in labels if guess != label)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
        if true_positive + false_negative:
            f1_values.append(f1)

    count = len(expected)
    accuracy = sum(a == p for a, p in zip(expected, predicted)) / count if count else None
    return {
        "matched_claim_count": count,
        "status_accuracy_on_matched_claims": accuracy,
        "macro_f1_present_classes": sum(f1_values) / len(f1_values) if f1_values else None,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def score_report_against_case(
    report: Mapping[str, Any],
    case: BenchmarkCase,
    run_annotation: RunAnnotation,
    *,
    k: int = 5,
    benchmark_id: str | None = None,
    benchmark_version: str | None = None,
) -> dict[str, Any]:
    if case.adjudication is None:
        raise ValueError(f"Benchmark case {case.case_id} has no adjudicated gold labels.")
    if run_annotation.case_id != case.case_id:
        raise ValueError("Run annotation case_id does not match the benchmark case.")
    if benchmark_id is not None and run_annotation.benchmark_id != benchmark_id:
        raise ValueError("Run annotation benchmark_id does not match the dataset.")
    if benchmark_version is not None and run_annotation.benchmark_version != benchmark_version:
        raise ValueError("Run annotation benchmark_version does not match the dataset.")
    if run_annotation.adjudication is None:
        raise ValueError("Run annotation needs a human adjudication before report scoring.")
    annotator_ids = [item.annotator_id for item in run_annotation.annotations]
    if len(set(annotator_ids)) < 2:
        raise ValueError("Run scoring requires at least two independent human annotations.")
    if len(annotator_ids) != len(set(annotator_ids)):
        raise ValueError("Run annotation contains duplicate annotator IDs.")
    if not run_annotation.benchmark_id.strip() or not str(run_annotation.report_run_id).strip():
        raise ValueError("Run annotation is missing benchmark or report run identity.")
    if not run_annotation.git_commit.strip() or not run_annotation.model_name.strip():
        raise ValueError("Run annotation must identify the Git commit and model.")

    gold_ids = {item.claim_id for item in case.expected_claims}
    expected_status_ids = {item.claim_id for item in case.adjudication.claim_judgments}
    if expected_status_ids != gold_ids:
        raise ValueError("Benchmark adjudication must contain exactly one status for every expected claim.")

    alignment_items = run_annotation.adjudication.claim_alignments
    gold_alignment_ids: list[str] = []
    output_alignment_ids: list[str] = []
    for alignment in alignment_items:
        gold_id = alignment.benchmark_claim_id
        output_id = alignment.report_claim_id
        if alignment.relation in {"equivalent", "partial"}:
            if not gold_id or not output_id:
                raise ValueError("Equivalent/partial alignments require both gold and report claim IDs.")
            gold_alignment_ids.append(gold_id)
            output_alignment_ids.append(output_id)
        elif alignment.relation == "missing":
            if not gold_id or output_id:
                raise ValueError("Missing alignment requires a gold claim ID and no report claim ID.")
            gold_alignment_ids.append(gold_id)
        elif alignment.relation == "extra":
            if gold_id or not output_id:
                raise ValueError("Extra alignment requires a report claim ID and no gold claim ID.")
            output_alignment_ids.append(output_id)

    if len(gold_alignment_ids) != len(set(gold_alignment_ids)):
        raise ValueError("Each benchmark claim must appear in exactly one final claim alignment.")
    if len(output_alignment_ids) != len(set(output_alignment_ids)):
        raise ValueError("Each generated claim must appear in exactly one final claim alignment.")
    if set(gold_alignment_ids) != gold_ids:
        raise ValueError("Final claim alignments must account for every benchmark claim.")
    report_claim_ids = {
        str(item.get("claim_id"))
        for item in (report.get("claim_verifications") or [])
        if isinstance(item, Mapping) and item.get("claim_id")
    }
    if set(output_alignment_ids) != report_claim_ids:
        raise ValueError("Final claim alignments must account for every verified/generated report claim.")

    def validate_alignment_coverage(alignments: list[Any], annotator_id: str) -> None:
        gold_ids: list[str] = []
        output_ids: list[str] = []
        for item in alignments:
            if item.relation in {"equivalent", "partial"}:
                if not item.benchmark_claim_id or not item.report_claim_id:
                    raise ValueError(f"{annotator_id}: equivalent/partial alignment needs both claim IDs.")
                gold_ids.append(item.benchmark_claim_id)
                output_ids.append(item.report_claim_id)
            elif item.relation == "missing":
                if not item.benchmark_claim_id or item.report_claim_id:
                    raise ValueError(f"{annotator_id}: missing alignment needs only a gold claim ID.")
                gold_ids.append(item.benchmark_claim_id)
            elif item.relation == "extra":
                if item.benchmark_claim_id or not item.report_claim_id:
                    raise ValueError(f"{annotator_id}: extra alignment needs only a report claim ID.")
                output_ids.append(item.report_claim_id)
        if len(gold_ids) != len(set(gold_ids)) or set(gold_ids) != gold_ids_set:
            raise ValueError(f"{annotator_id}: annotations must account for every gold claim exactly once.")
        if len(output_ids) != len(set(output_ids)) or set(output_ids) != report_claim_ids:
            raise ValueError(f"{annotator_id}: annotations must account for every report claim exactly once.")

    gold_ids_set = {item.claim_id for item in case.expected_claims}
    for annotation in run_annotation.annotations:
        validate_alignment_coverage(annotation.claim_alignments, annotation.annotator_id)
        dimensions = [item.dimension for item in annotation.rubric_scores]
        if len(dimensions) != len(set(dimensions)) or set(dimensions) != RUBRIC_DIMENSIONS:
            raise ValueError(
                f"{annotation.annotator_id}: independent review must score all seven rubric dimensions exactly once."
            )

    rubric_scores = run_annotation.adjudication.rubric_scores
    rubric_dimensions = [item.dimension for item in rubric_scores]
    if len(rubric_dimensions) != len(set(rubric_dimensions)):
        raise ValueError("Final report rubric contains duplicate dimensions.")
    if set(rubric_dimensions) != RUBRIC_DIMENSIONS:
        raise ValueError("Final report rubric must score all seven rubric dimensions.")

    relevance = {
        str(source.url): next(
            (item.grade for item in case.adjudication.source_relevance if item.source_id == source.source_id),
            0,
        )
        for source in case.snapshot.sources
    }
    retrieved_urls = [
        str(source.get("url"))
        for source in (report.get("sources") or [])
        if isinstance(source, Mapping) and source.get("url")
    ]
    retrieval = graded_retrieval_metrics(retrieved_urls, relevance, k=k)

    gold_status = {item.claim_id: item.status for item in case.adjudication.claim_judgments}
    report_verifications = {
        str(item.get("claim_id")): str(item.get("status"))
        for item in (report.get("claim_verifications") or [])
        if isinstance(item, Mapping) and item.get("claim_id")
    }
    mapped_gold: set[str] = set()
    mapped_output: set[str] = set()
    expected_status: list[str] = []
    predicted_status: list[str] = []

    for alignment in alignment_items:
        if alignment.relation != "equivalent":
            continue
        gold_id = alignment.benchmark_claim_id
        output_id = alignment.report_claim_id
        if not gold_id or not output_id or gold_id not in gold_status or output_id not in report_verifications:
            continue
        if gold_id in mapped_gold or output_id in mapped_output:
            continue
        mapped_gold.add(gold_id)
        mapped_output.add(output_id)
        expected_status.append(gold_status[gold_id])
        predicted_status.append(report_verifications[output_id])

    gold_count = len(gold_status)
    output_count = len(report_verifications)
    equivalent_count = len(mapped_gold)
    classification = _classification_metrics(expected_status, predicted_status)

    # Compare verifier evidence-source citations against the gold evidence spans.
    report_source_urls = {
        str(item.get("id")): str(item.get("url"))
        for item in (report.get("sources") or [])
        if isinstance(item, Mapping) and item.get("id") and item.get("url")
    }
    gold_evidence_pairs: set[tuple[str, str]] = set()
    gold_quotes_by_pair: dict[tuple[str, str], set[str]] = {}
    adjudicated_claims = {
        item.claim_id: item
        for item in case.adjudication.claim_judgments
    }
    snapshot_urls = {item.source_id: str(item.url) for item in case.snapshot.sources}

    for gold_claim_id in mapped_gold:
        gold_judgment = adjudicated_claims[gold_claim_id]
        for span in gold_judgment.evidence_spans:
            source_url = snapshot_urls.get(span.source_id)
            if source_url:
                pair = (gold_claim_id, source_url)
                gold_evidence_pairs.add(pair)
                gold_quotes_by_pair.setdefault(pair, set()).add(" ".join(span.quote.split()).casefold())

    predicted_evidence_pairs: set[tuple[str, str]] = set()
    predicted_quotes_by_pair: dict[tuple[str, str], set[str]] = {}
    report_to_gold = {
        alignment.report_claim_id: alignment.benchmark_claim_id
        for alignment in alignment_items
        if alignment.relation == "equivalent"
        and alignment.report_claim_id
        and alignment.benchmark_claim_id
    }
    emitted_span_count = 0
    for verification in (report.get("claim_verifications") or []):
        if not isinstance(verification, Mapping):
            continue
        report_claim_id = str(verification.get("claim_id", ""))
        gold_claim_id = report_to_gold.get(report_claim_id)
        if not gold_claim_id:
            continue
        spans = list(verification.get("supporting_evidence") or []) + list(
            verification.get("contradicting_evidence") or []
        )
        for span in spans:
            if not isinstance(span, Mapping):
                continue
            emitted_span_count += 1
            source_url = report_source_urls.get(str(span.get("source_id", "")))
            quote = " ".join(str(span.get("quote", "")).split()).casefold()
            if source_url:
                pair = (gold_claim_id, source_url)
                predicted_evidence_pairs.add(pair)
                if quote:
                    predicted_quotes_by_pair.setdefault(pair, set()).add(quote)

    evidence_true_positive = len(gold_evidence_pairs & predicted_evidence_pairs)
    evidence_precision = (
        evidence_true_positive / len(predicted_evidence_pairs)
        if predicted_evidence_pairs else 0.0
    )
    evidence_recall = (
        evidence_true_positive / len(gold_evidence_pairs)
        if gold_evidence_pairs else 0.0
    )
    evidence_f1 = (
        2 * evidence_precision * evidence_recall / (evidence_precision + evidence_recall)
        if evidence_precision + evidence_recall else 0.0
    )
    exact_quote_matches = sum(
        1
        for pair, quotes in predicted_quotes_by_pair.items()
        if quotes & gold_quotes_by_pair.get(pair, set())
    )

    rubric_by_dimension: dict[str, list[int]] = {}
    for item in run_annotation.adjudication.rubric_scores:
        rubric_by_dimension.setdefault(item.dimension, []).append(item.score)
    rubric_means = {
        dimension: sum(scores) / len(scores)
        for dimension, scores in rubric_by_dimension.items()
        if scores
    }
    overall_rubric_mean = (
        sum(rubric_means.values()) / len(rubric_means) if rubric_means else None
    )

    return {
        "benchmark_id": run_annotation.benchmark_id,
        "benchmark_version": run_annotation.benchmark_version,
        "case_id": case.case_id,
        "report_run_id": run_annotation.report_run_id,
        "system": {
            "git_commit": run_annotation.git_commit,
            "model_name": run_annotation.model_name,
            "model_version": run_annotation.model_version,
            "prompt_versions": run_annotation.prompt_versions,
        },
        "retrieval": retrieval,
        "claim_evaluation": {
            "gold_claim_count": gold_count,
            "generated_claim_count": output_count,
            "equivalent_claim_matches": equivalent_count,
            "claim_coverage_recall": equivalent_count / gold_count if gold_count else 0.0,
            "claim_alignment_precision": equivalent_count / output_count if output_count else 0.0,
            "extra_generated_claim_count": sum(
                item.relation == "extra" for item in alignment_items
            ),
            "unresolved_generated_claim_count": sum(
                item.get("status") != "supported"
                for item in (report.get("claim_verifications") or [])
                if isinstance(item, Mapping)
            ),
            "contradicted_generated_claim_count": sum(
                item.get("status") == "contradicted"
                for item in (report.get("claim_verifications") or [])
                if isinstance(item, Mapping)
            ),
            "partial_claim_match_count": sum(
                item.relation == "partial" for item in alignment_items
            ),
            "missing_claim_count": sum(
                item.relation == "missing" for item in alignment_items
            ),
            **classification,
        },
        "evidence_grounding": {
            "gold_claim_source_pairs": len(gold_evidence_pairs),
            "predicted_claim_source_pairs": len(predicted_evidence_pairs),
            "emitted_evidence_spans": emitted_span_count,
            "claim_source_precision": evidence_precision,
            "claim_source_recall": evidence_recall,
            "claim_source_f1": evidence_f1,
            "exact_quote_match_pairs": exact_quote_matches,
            "exact_quote_match_rate": (
                exact_quote_matches / len(predicted_quotes_by_pair)
                if predicted_quotes_by_pair else 0.0
            ),
        },
        "human_report_rubric": {
            "score_scale": "0-4",
            "dimension_means": rubric_means,
            "mean_across_scored_dimensions": overall_rubric_mean,
            "dimensions_scored": len(rubric_means),
        },
        "interpretation": (
            "Claim alignment and report rubric scores come from the human-adjudicated run annotation. "
            "Status accuracy is calculated only on claims humans marked equivalent; claim coverage and "
            "alignment precision must be read alongside it. These scores are not objective truth guarantees."
        ),
    }
