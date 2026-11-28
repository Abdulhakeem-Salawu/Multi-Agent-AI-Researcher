from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .schemas import BenchmarkDataset


def load_dataset(path: str | Path) -> BenchmarkDataset:
    dataset_path = Path(path)
    try:
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
        return BenchmarkDataset.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"Could not load benchmark dataset {dataset_path}: {exc}") from exc


def validate_dataset(
    dataset: BenchmarkDataset,
    *,
    require_ready: bool = False,
    verify_snapshots: bool = False,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    root = Path(repo_root).resolve()

    case_ids = [case.case_id for case in dataset.cases]
    if len(case_ids) != len(set(case_ids)):
        errors.append("Dataset contains duplicate case_id values.")

    if dataset.status == "ready" or require_ready:
        if not dataset.cases:
            errors.append("A ready benchmark must contain at least one case.")
        if dataset.status != "ready":
            errors.append("Dataset status is 'draft'; promote it only after adjudication.")

    for case in dataset.cases:
        prefix = f"case {case.case_id}"
        snapshot = case.snapshot
        source_ids = [item.source_id for item in snapshot.sources]
        claim_ids = [item.claim_id for item in case.expected_claims]
        source_id_set = set(source_ids)
        claim_id_set = set(claim_ids)

        if len(source_ids) != len(source_id_set):
            errors.append(f"{prefix}: duplicate snapshot source_id values.")
        if len(claim_ids) != len(claim_id_set):
            errors.append(f"{prefix}: duplicate expected claim_id values.")
        if not snapshot.sources:
            warnings.append(f"{prefix}: no evidence snapshot sources have been added.")
        source_urls = [str(item.url) for item in snapshot.sources]
        if len(source_urls) != len(set(source_urls)):
            errors.append(f"{prefix}: duplicate source URLs should be represented by one canonical source record.")
        if (dataset.status == "ready" or require_ready) and not case.expected_claims:
            errors.append(f"{prefix}: ready cases require at least one atomic expected claim.")

        if dataset.status == "ready" or require_ready:
            if not snapshot.frozen or snapshot.frozen_at is None:
                errors.append(f"{prefix}: ready cases require a frozen snapshot and frozen_at timestamp.")
            for source in snapshot.sources:
                if source.redistribution in {"redistributable", "excerpt_only"}:
                    if not source.content_path or not source.sha256:
                        errors.append(
                            f"{prefix}: source {source.source_id} requires content_path and SHA-256 for its permitted snapshot."
                        )
                elif source.redistribution == "restricted":
                    if not source.controlled_snapshot_ref:
                        errors.append(
                            f"{prefix}: restricted source {source.source_id} requires a controlled_snapshot_ref."
                        )
                    if source.content_path or source.sha256:
                        errors.append(
                            f"{prefix}: restricted source {source.source_id} must not commit its full text or a repository snapshot hash."
                        )
                elif (
                    source.redistribution == "metadata_only"
                    and (source.content_path or source.sha256 or source.controlled_snapshot_ref)
                ):
                    errors.append(
                        f"{prefix}: metadata-only source {source.source_id} must not include a stored text snapshot."
                    )
                if not source.license_id or source.redistribution == "unknown":
                    errors.append(
                        f"{prefix}: source {source.source_id} requires license_id and explicit redistribution status."
                    )

        for annotation in case.annotations:
            source_judged = [item.source_id for item in annotation.source_relevance]
            claim_judged = [item.claim_id for item in annotation.claim_judgments]
            if len(source_judged) != len(set(source_judged)):
                errors.append(
                    f"{prefix}: annotator {annotation.annotator_id} has duplicate source relevance judgments."
                )
            if len(claim_judged) != len(set(claim_judged)):
                errors.append(
                    f"{prefix}: annotator {annotation.annotator_id} has duplicate claim judgments."
                )
            if set(source_judged) - source_id_set:
                errors.append(f"{prefix}: annotator {annotation.annotator_id} judged unknown sources.")
            if set(claim_judged) - claim_id_set:
                errors.append(f"{prefix}: annotator {annotation.annotator_id} judged unknown claims.")

        if dataset.status == "ready" or require_ready:
            annotator_ids = [item.annotator_id for item in case.annotations]
            if len(set(annotator_ids)) < 2:
                errors.append(f"{prefix}: ready cases require at least two independent annotators.")
            for annotation in case.annotations:
                if {item.source_id for item in annotation.source_relevance} != source_id_set:
                    errors.append(
                        f"{prefix}: annotator {annotation.annotator_id} did not grade every snapshot source."
                    )
                if {item.claim_id for item in annotation.claim_judgments} != claim_id_set:
                    errors.append(
                        f"{prefix}: annotator {annotation.annotator_id} did not label every expected claim."
                    )
            if case.adjudication is None:
                errors.append(f"{prefix}: ready cases require final adjudication.")
            else:
                judged_sources = [item.source_id for item in case.adjudication.source_relevance]
                judged_claims = [item.claim_id for item in case.adjudication.claim_judgments]
                if len(judged_sources) != len(set(judged_sources)):
                    errors.append(f"{prefix}: adjudication contains duplicate source judgments.")
                if len(judged_claims) != len(set(judged_claims)):
                    errors.append(f"{prefix}: adjudication contains duplicate claim judgments.")
                if set(judged_sources) != source_id_set:
                    errors.append(f"{prefix}: adjudication source relevance coverage is incomplete.")
                if set(judged_claims) != claim_id_set:
                    errors.append(f"{prefix}: adjudication claim coverage is incomplete.")
                for judgment in case.adjudication.claim_judgments:
                    if judgment.status in {"supported", "contradicted"} and not judgment.evidence_spans:
                        errors.append(
                            f"{prefix}: adjudicated {judgment.status} claim {judgment.claim_id} requires an evidence span."
                        )
                    for span in judgment.evidence_spans:
                        if span.source_id not in source_id_set:
                            errors.append(
                                f"{prefix}: claim {judgment.claim_id} references unknown source {span.source_id}."
                            )
                        if (
                            span.start_char is not None
                            and span.end_char is not None
                            and span.end_char - span.start_char != len(span.quote)
                        ):
                            errors.append(
                                f"{prefix}: evidence offsets have an inconsistent length for {judgment.claim_id}."
                            )

        if verify_snapshots:
            source_text: dict[str, str] = {}
            for source in snapshot.sources:
                if not source.content_path:
                    if source.redistribution == "restricted":
                        warnings.append(
                            f"{prefix}: restricted source {source.source_id} was not checked locally; verify controlled snapshot {source.controlled_snapshot_ref!r} out of band."
                        )
                    elif source.redistribution == "metadata_only":
                        warnings.append(
                            f"{prefix}: metadata-only source {source.source_id} has no local text to hash or quote-check."
                        )
                    else:
                        errors.append(f"{prefix}: source {source.source_id} has no content_path.")
                    continue
                snapshot_path = (root / source.content_path).resolve()
                if root not in snapshot_path.parents:
                    errors.append(
                        f"{prefix}: source {source.source_id} content_path escapes repository root."
                    )
                    continue
                if not snapshot_path.is_file():
                    errors.append(
                        f"{prefix}: snapshot file for {source.source_id} does not exist: {source.content_path}."
                    )
                    continue
                content_bytes = snapshot_path.read_bytes()
                digest = hashlib.sha256(content_bytes).hexdigest()
                if source.sha256 and digest.lower() != source.sha256.lower():
                    errors.append(f"{prefix}: SHA-256 mismatch for source {source.source_id}.")
                    continue
                try:
                    source_text[source.source_id] = content_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    errors.append(f"{prefix}: source snapshot {source.source_id} must be UTF-8 text.")

            all_judgments = [
                judgment
                for annotation in case.annotations
                for judgment in annotation.claim_judgments
            ]
            if case.adjudication:
                all_judgments.extend(case.adjudication.claim_judgments)
            source_by_id = {source.source_id: source for source in snapshot.sources}
            for judgment in all_judgments:
                for span in judgment.evidence_spans:
                    span_source = source_by_id.get(span.source_id)
                    if span_source is None:
                        errors.append(
                            f"{prefix}: evidence span for {judgment.claim_id} references unknown source {span.source_id}."
                        )
                        continue
                    if span_source.redistribution == "metadata_only":
                        errors.append(
                            f"{prefix}: metadata-only source {span.source_id} cannot be cited as a stored gold evidence span."
                        )
                        continue
                    text = source_text.get(span.source_id)
                    if text is None:
                        if span_source.redistribution == "restricted":
                            warnings.append(
                                f"{prefix}: evidence span for {judgment.claim_id} in restricted source {span.source_id} was not verified locally."
                            )
                        else:
                            errors.append(
                                f"{prefix}: evidence span for {judgment.claim_id} references unavailable source {span.source_id}."
                            )
                        continue
                    if span.quote not in text:
                        errors.append(
                            f"{prefix}: evidence quote for {judgment.claim_id} is not present in {span.source_id}."
                        )
                    if (span.start_char is None) != (span.end_char is None):
                        errors.append(
                            f"{prefix}: evidence offsets must provide both start_char and end_char for {judgment.claim_id}."
                        )
                    if span.start_char is not None and span.end_char is not None:
                        if span.end_char <= span.start_char:
                            errors.append(
                                f"{prefix}: evidence offsets are reversed for {judgment.claim_id}."
                            )
                        elif text[span.start_char : span.end_char] != span.quote:
                            errors.append(
                                f"{prefix}: evidence offsets do not match quote for {judgment.claim_id}."
                            )

    return {
        "benchmark_id": dataset.benchmark_id,
        "version": dataset.version,
        "status": dataset.status,
        "case_count": len(dataset.cases),
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def validate_dataset_file(
    path: str | Path,
    *,
    require_ready: bool = False,
    verify_snapshots: bool = False,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    return validate_dataset(
        load_dataset(path),
        require_ready=require_ready,
        verify_snapshots=verify_snapshots,
        repo_root=repo_root,
    )
