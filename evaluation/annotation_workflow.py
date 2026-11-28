from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .dataset import load_dataset, validate_dataset
from .schemas import CaseAdjudication, CaseAnnotation

FORM_VERSION = "1.0"
ANNOTATOR_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{2,80}$")


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _write_json(path: str | Path, payload: Any, *, overwrite: bool = False) -> Path:
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def _case_by_id(dataset: Any, case_id: str) -> Any:
    case = next((item for item in dataset.cases if item.case_id == case_id), None)
    if case is None:
        raise ValueError(f"Unknown case_id: {case_id}")
    return case


def _require_draft(dataset: Any) -> None:
    if dataset.status != "draft":
        raise ValueError("Released benchmark datasets are immutable; create a new draft version to change them.")


def _assert_frozen(case: Any) -> None:
    if not case.snapshot.frozen or case.snapshot.frozen_at is None:
        raise ValueError(
            f"Case {case.case_id} must have a frozen evidence snapshot before annotation."
        )
    if not case.snapshot.sources:
        raise ValueError(f"Case {case.case_id} has no evidence sources to annotate.")


def _case_fingerprint(case: Any) -> str:
    payload = {
        "case_id": case.case_id,
        "question": case.question,
        "scope": case.scope,
        "as_of": case.as_of.isoformat() if case.as_of else None,
        "snapshot": {
            "snapshot_id": case.snapshot.snapshot_id,
            "frozen": case.snapshot.frozen,
            "frozen_at": case.snapshot.frozen_at.isoformat() if case.snapshot.frozen_at else None,
            "capture_method": case.snapshot.capture_method,
            "sources": [source.model_dump(mode="json") for source in case.snapshot.sources],
        },
        "expected_claims": [claim.model_dump(mode="json") for claim in case.expected_claims],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _assert_form_case_fingerprint(form: dict[str, Any], case: Any) -> None:
    actual = form.get("case_fingerprint")
    if not isinstance(actual, str) or actual != _case_fingerprint(case):
        raise ValueError(
            f"Form for case {case.case_id} does not match the current frozen case/evidence version. "
            "Regenerate the form from the current dataset."
        )
    if form.get("question") != case.question or form.get("scope") != case.scope:
        raise ValueError("Form question or scope was changed after export.")

    form_claims = form.get("claim_judgments")
    if not isinstance(form_claims, list):
        raise ValueError("Form is missing its claim list.")
    actual_claims = [
        {
            "claim_id": item.get("claim_id"),
            "proposition": item.get("proposition"),
            "scope_notes": item.get("scope_notes", ""),
        }
        for item in form_claims
        if isinstance(item, dict)
    ]
    expected_claims = [
        {
            "claim_id": item.claim_id,
            "proposition": item.proposition,
            "scope_notes": item.scope_notes,
        }
        for item in case.expected_claims
    ]
    if actual_claims != expected_claims:
        raise ValueError("Form claims do not match the frozen case definition.")

    if form.get("form_type") == "research_bench.case_annotation":
        snapshot = form.get("snapshot")
        if not isinstance(snapshot, dict) or snapshot.get("snapshot_id") != case.snapshot.snapshot_id:
            raise ValueError("Form snapshot ID does not match the frozen case.")
        actual_sources = snapshot.get("sources")
        expected_sources = [
            {
                "source_id": source.source_id,
                "title": source.title,
                "url": str(source.url),
                "publisher": source.publisher,
                "source_type": source.source_type,
                "published_at": source.published_at.isoformat() if source.published_at else None,
                "retrieved_at": source.retrieved_at.isoformat(),
                "content_path": source.content_path,
                "sha256": source.sha256,
                "controlled_snapshot_ref": source.controlled_snapshot_ref,
                "license_id": source.license_id,
                "redistribution": source.redistribution,
            }
            for source in case.snapshot.sources
        ]
        if actual_sources != expected_sources:
            raise ValueError("Form source manifest does not match the frozen snapshot.")
    elif form.get("form_type") == "research_bench.case_adjudication":
        if form.get("snapshot_id") != case.snapshot.snapshot_id:
            raise ValueError("Adjudication form snapshot ID does not match the frozen case.")
        expected_annotations = [item.model_dump(mode="json") for item in case.annotations]
        if form.get("independent_annotations") != expected_annotations:
            raise ValueError("Adjudication form independent annotations do not match the imported submissions.")


def _validate_annotation(annotation: CaseAnnotation, case: Any) -> None:
    source_ids = {source.source_id for source in case.snapshot.sources}
    claim_ids = {claim.claim_id for claim in case.expected_claims}
    judged_source_ids = [item.source_id for item in annotation.source_relevance]
    judged_claim_ids = [item.claim_id for item in annotation.claim_judgments]

    if len(judged_source_ids) != len(set(judged_source_ids)):
        raise ValueError(f"{annotation.annotator_id}: duplicate source relevance judgments.")
    if len(judged_claim_ids) != len(set(judged_claim_ids)):
        raise ValueError(f"{annotation.annotator_id}: duplicate claim judgments.")
    if set(judged_source_ids) != source_ids:
        raise ValueError(f"{annotation.annotator_id}: grade every source exactly once.")
    if set(judged_claim_ids) != claim_ids:
        raise ValueError(f"{annotation.annotator_id}: label every expected claim exactly once.")

    source_by_id = {source.source_id: source for source in case.snapshot.sources}
    for judgment in annotation.claim_judgments:
        if judgment.status in {"supported", "contradicted"} and not judgment.evidence_spans:
            raise ValueError(
                f"{annotation.annotator_id}: {judgment.status} claim {judgment.claim_id} needs an evidence span."
            )
        for span in judgment.evidence_spans:
            source = source_by_id.get(span.source_id)
            if source is None:
                raise ValueError(
                    f"{annotation.annotator_id}: claim {judgment.claim_id} references unknown source {span.source_id}."
                )
            if source.redistribution == "metadata_only":
                raise ValueError(
                    f"{annotation.annotator_id}: metadata-only source {span.source_id} cannot provide a gold evidence span."
                )


def _annotation_form(dataset: Any, case: Any, annotator_id: str) -> dict[str, Any]:
    return {
        "form_schema_version": FORM_VERSION,
        "form_type": "research_bench.case_annotation",
        "case_fingerprint": _case_fingerprint(case),
        "benchmark_id": dataset.benchmark_id,
        "benchmark_version": dataset.version,
        "case_id": case.case_id,
        "question": case.question,
        "scope": case.scope,
        "as_of": case.as_of.isoformat() if case.as_of else None,
        "snapshot": {
            "snapshot_id": case.snapshot.snapshot_id,
            "frozen_at": case.snapshot.frozen_at.isoformat() if case.snapshot.frozen_at else None,
            "capture_method": case.snapshot.capture_method,
            "sources": [
                {
                    "source_id": source.source_id,
                    "title": source.title,
                    "url": str(source.url),
                    "publisher": source.publisher,
                    "source_type": source.source_type,
                    "published_at": source.published_at.isoformat() if source.published_at else None,
                    "retrieved_at": source.retrieved_at.isoformat(),
                    "content_path": source.content_path,
                    "sha256": source.sha256,
                    "controlled_snapshot_ref": source.controlled_snapshot_ref,
                    "license_id": source.license_id,
                    "redistribution": source.redistribution,
                }
                for source in case.snapshot.sources
            ],
        },
        "annotator_id": annotator_id,
        "annotated_at": None,
        "source_relevance_scale": {
            "0": "Irrelevant or misleading for this question.",
            "1": "Tangential background only.",
            "2": "Relevant context or partial support.",
            "3": "Directly answers or strongly substantiates an important part.",
        },
        "claim_status_choices": [
            "supported",
            "contradicted",
            "insufficient_evidence",
            "uncertain",
        ],
        "source_relevance": [
            {"source_id": source.source_id, "grade": None, "rationale": ""}
            for source in case.snapshot.sources
        ],
        "claim_judgments": [
            {
                "claim_id": claim.claim_id,
                "proposition": claim.proposition,
                "scope_notes": claim.scope_notes,
                "status": None,
                "evidence_spans": [],
                "rationale": "",
            }
            for claim in case.expected_claims
        ],
        "notes": "Annotate independently. Do not consult another annotator's labels. Use exact quotations from the frozen snapshot.",
    }


def export_annotation_forms(
    dataset_path: str | Path,
    output_dir: str | Path,
    annotator_ids: list[str],
    *,
    case_ids: list[str] | None = None,
    overwrite: bool = False,
) -> list[str]:
    if not annotator_ids or len(annotator_ids) != len(set(annotator_ids)):
        raise ValueError("Provide at least one unique annotator ID.")
    for annotator_id in annotator_ids:
        if not ANNOTATOR_ID_PATTERN.fullmatch(annotator_id):
            raise ValueError(
                f"Invalid annotator ID {annotator_id!r}; use 2-80 letters, digits, dots, underscores or hyphens."
            )

    dataset = load_dataset(dataset_path)
    _require_draft(dataset)
    selected = set(case_ids or [])
    unknown = selected - {case.case_id for case in dataset.cases}
    if unknown:
        raise ValueError(f"Unknown case IDs: {', '.join(sorted(unknown))}")

    output_paths: list[str] = []
    for case in dataset.cases:
        if selected and case.case_id not in selected:
            continue
        _assert_frozen(case)
        for annotator_id in annotator_ids:
            form_path = Path(output_dir) / case.case_id / f"{annotator_id}.json"
            form = _annotation_form(dataset, case, annotator_id)
            _write_json(form_path, form, overwrite=overwrite)
            output_paths.append(str(form_path))
    return output_paths


def import_annotation_forms(
    dataset_path: str | Path,
    annotation_paths: list[str | Path],
    output_path: str | Path,
    *,
    repo_root: str | Path = ".",
    overwrite: bool = False,
) -> dict[str, Any]:
    if not annotation_paths:
        raise ValueError("Provide at least one completed annotation form.")

    dataset = load_dataset(dataset_path)
    _require_draft(dataset)
    imported: list[dict[str, str]] = []
    pending: set[tuple[str, str]] = set()

    for path in annotation_paths:
        form = _read_json(path)
        if form.get("form_schema_version") != FORM_VERSION:
            raise ValueError(f"{path}: unsupported or missing form_schema_version.")
        if form.get("form_type") != "research_bench.case_annotation":
            raise ValueError(f"{path}: not a ResearchBench case-annotation form.")
        if form.get("benchmark_id") != dataset.benchmark_id:
            raise ValueError(f"{path}: benchmark_id does not match the dataset.")
        if form.get("benchmark_version") != dataset.version:
            raise ValueError(f"{path}: benchmark_version does not match the dataset.")

        case = _case_by_id(dataset, str(form.get("case_id", "")))
        _assert_frozen(case)
        _assert_form_case_fingerprint(form, case)
        annotator_id = str(form.get("annotator_id", ""))
        if not ANNOTATOR_ID_PATTERN.fullmatch(annotator_id):
            raise ValueError(f"{path}: invalid annotator_id.")
        key = (case.case_id, annotator_id)
        if key in pending or any(item.annotator_id == annotator_id for item in case.annotations):
            raise ValueError(
                f"{path}: annotator {annotator_id} already submitted a label set for {case.case_id}."
            )
        pending.add(key)

        try:
            annotation = CaseAnnotation.model_validate(
                {
                    "annotator_id": annotator_id,
                    "annotated_at": form.get("annotated_at"),
                    "source_relevance": form.get("source_relevance", []),
                    "claim_judgments": form.get("claim_judgments", []),
                    "notes": form.get("notes", ""),
                }
            )
        except ValidationError as exc:
            raise ValueError(f"{path}: invalid or incomplete annotation: {exc}") from exc
        _validate_annotation(annotation, case)
        case.annotations.append(annotation)
        imported.append({"case_id": case.case_id, "annotator_id": annotator_id})

    validation = validate_dataset(dataset, verify_snapshots=True, repo_root=repo_root)
    if not validation["valid"]:
        raise ValueError("Imported annotations failed dataset validation: " + "; ".join(validation["errors"]))

    _write_json(output_path, dataset.model_dump(mode="json"), overwrite=overwrite)
    return {
        "output": str(output_path),
        "imported_count": len(imported),
        "imported": imported,
        "dataset_status": dataset.status,
        "validation": validation,
    }


def export_adjudication_form(
    dataset_path: str | Path,
    case_id: str,
    adjudicator_id: str,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> str:
    if not ANNOTATOR_ID_PATTERN.fullmatch(adjudicator_id):
        raise ValueError("Invalid adjudicator ID.")
    dataset = load_dataset(dataset_path)
    _require_draft(dataset)
    case = _case_by_id(dataset, case_id)
    _assert_frozen(case)
    annotator_ids = [item.annotator_id for item in case.annotations]
    if len(set(annotator_ids)) < 2:
        raise ValueError("Adjudication form requires at least two independent annotations.")
    if adjudicator_id in annotator_ids:
        raise ValueError("The adjudicator must be different from the independent annotators.")
    for annotation in case.annotations:
        _validate_annotation(annotation, case)

    form = {
        "form_schema_version": FORM_VERSION,
        "form_type": "research_bench.case_adjudication",
        "case_fingerprint": _case_fingerprint(case),
        "benchmark_id": dataset.benchmark_id,
        "benchmark_version": dataset.version,
        "case_id": case.case_id,
        "question": case.question,
        "scope": case.scope,
        "snapshot_id": case.snapshot.snapshot_id,
        "adjudicator_id": adjudicator_id,
        "adjudicated_at": None,
        "independent_annotations": [item.model_dump(mode="json") for item in case.annotations],
        "source_relevance": [
            {"source_id": source.source_id, "grade": None, "rationale": ""}
            for source in case.snapshot.sources
        ],
        "claim_judgments": [
            {
                "claim_id": claim.claim_id,
                "proposition": claim.proposition,
                "scope_notes": claim.scope_notes,
                "status": None,
                "evidence_spans": [],
                "rationale": "",
            }
            for claim in case.expected_claims
        ],
        "rationale_for_disagreements": "",
        "notes": "Resolve disagreements against the frozen source snapshot. Record why the final label is preferred.",
    }
    _write_json(output_path, form, overwrite=overwrite)
    return str(output_path)


def import_adjudication_form(
    dataset_path: str | Path,
    form_path: str | Path,
    output_path: str | Path,
    *,
    repo_root: str | Path = ".",
    overwrite: bool = False,
) -> dict[str, Any]:
    dataset = load_dataset(dataset_path)
    _require_draft(dataset)
    form = _read_json(form_path)
    if form.get("form_schema_version") != FORM_VERSION:
        raise ValueError("Unsupported or missing form_schema_version.")
    if form.get("form_type") != "research_bench.case_adjudication":
        raise ValueError("Not a ResearchBench adjudication form.")
    if form.get("benchmark_id") != dataset.benchmark_id:
        raise ValueError("Adjudication benchmark_id does not match the dataset.")
    if form.get("benchmark_version") != dataset.version:
        raise ValueError("Adjudication benchmark_version does not match the dataset.")

    case = _case_by_id(dataset, str(form.get("case_id", "")))
    _assert_frozen(case)
    _assert_form_case_fingerprint(form, case)
    annotator_ids = {item.annotator_id for item in case.annotations}
    adjudicator_id = str(form.get("adjudicator_id", ""))
    if not ANNOTATOR_ID_PATTERN.fullmatch(adjudicator_id):
        raise ValueError("Invalid adjudicator_id.")
    if adjudicator_id in annotator_ids:
        raise ValueError("The adjudicator must be independent of the annotators.")
    if len(annotator_ids) < 2:
        raise ValueError("At least two independent annotations are required before adjudication.")
    if case.adjudication is not None:
        raise ValueError(f"Case {case.case_id} already has an adjudication.")

    try:
        adjudication = CaseAdjudication.model_validate(
            {
                "adjudicator_id": adjudicator_id,
                "adjudicated_at": form.get("adjudicated_at"),
                "source_relevance": form.get("source_relevance", []),
                "claim_judgments": form.get("claim_judgments", []),
                "rationale_for_disagreements": form.get("rationale_for_disagreements", ""),
            }
        )
    except ValidationError as exc:
        raise ValueError(f"Invalid or incomplete adjudication: {exc}") from exc

    source_ids = {source.source_id for source in case.snapshot.sources}
    claim_ids = {claim.claim_id for claim in case.expected_claims}
    if {item.source_id for item in adjudication.source_relevance} != source_ids:
        raise ValueError("Adjudication must grade every source exactly once.")
    if len(adjudication.source_relevance) != len(source_ids):
        raise ValueError("Adjudication contains duplicate source relevance judgments.")
    if {item.claim_id for item in adjudication.claim_judgments} != claim_ids:
        raise ValueError("Adjudication must label every expected claim exactly once.")
    if len(adjudication.claim_judgments) != len(claim_ids):
        raise ValueError("Adjudication contains duplicate claim judgments.")

    surrogate = CaseAnnotation(
        annotator_id=adjudicator_id,
        annotated_at=adjudication.adjudicated_at,
        source_relevance=adjudication.source_relevance,
        claim_judgments=adjudication.claim_judgments,
        notes="Adjudication validation",
    )
    _validate_annotation(surrogate, case)
    case.adjudication = adjudication

    validation = validate_dataset(dataset, verify_snapshots=True, repo_root=repo_root)
    if not validation["valid"]:
        raise ValueError("Adjudication failed dataset validation: " + "; ".join(validation["errors"]))
    _write_json(output_path, dataset.model_dump(mode="json"), overwrite=overwrite)
    return {
        "output": str(output_path),
        "case_id": case.case_id,
        "adjudicator_id": adjudicator_id,
        "validation": validation,
    }


def register_source_snapshot(
    dataset_path: str | Path,
    case_id: str,
    source_id: str,
    redistribution: str,
    license_id: str,
    output_path: str | Path,
    *,
    input_file: str | Path | None = None,
    controlled_ref: str | None = None,
    confirm_rights: bool = False,
    repo_root: str | Path = ".",
    snapshot_dir: str | Path = "evaluation/snapshots",
    overwrite_snapshot: bool = False,
    overwrite_dataset: bool = False,
) -> dict[str, Any]:
    dataset = load_dataset(dataset_path)
    _require_draft(dataset)
    output_target = Path(output_path)
    if output_target.exists() and not overwrite_dataset:
        raise FileExistsError(f"Refusing to overwrite existing file: {output_target}")
    case = _case_by_id(dataset, case_id)
    if case.snapshot.frozen:
        raise ValueError("Cannot change source registration after the case snapshot has been frozen.")
    source = next((item for item in case.snapshot.sources if item.source_id == source_id), None)
    if source is None:
        raise ValueError(f"Unknown source_id {source_id!r} in case {case_id}.")

    if not license_id.strip():
        raise ValueError("Provide a specific license identifier or documented rights basis.")
    allowed = {"redistributable", "excerpt_only", "metadata_only", "restricted"}
    if redistribution not in allowed:
        raise ValueError(f"redistribution must be one of: {', '.join(sorted(allowed))}.")

    root = Path(repo_root).resolve()
    content_path: str | None = None
    digest: str | None = None
    resolved_controlled_ref: str | None = None

    if redistribution in {"redistributable", "excerpt_only"}:
        if not confirm_rights:
            raise ValueError(
                "Pass --confirm-rights only after confirming that the supplied file may be stored under the declared license/permission."
            )
        if input_file is None:
            raise ValueError("redistributable and excerpt_only sources require --input-file.")
        source_file = Path(input_file).resolve()
        if not source_file.is_file():
            raise ValueError(f"Snapshot input file does not exist: {source_file}")
        content = source_file.read_bytes()
        if not content:
            raise ValueError("Snapshot input file is empty.")
        try:
            content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Source snapshots must be UTF-8 text.") from exc

        snapshot_root = (root / snapshot_dir).resolve()
        destination = (snapshot_root / case_id / f"{source_id}.txt").resolve()
        if root != snapshot_root and root not in snapshot_root.parents:
            raise ValueError("snapshot_dir must remain within repo_root.")
        if root not in destination.parents:
            raise ValueError("Snapshot destination escapes repo_root.")
        if destination.exists() and not overwrite_snapshot:
            raise FileExistsError(
                f"Snapshot already exists: {destination}. Use --overwrite-snapshot deliberately to replace it."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        content_path = destination.relative_to(root).as_posix()
    elif redistribution == "restricted":
        if input_file is not None:
            raise ValueError("Do not pass --input-file for restricted sources; full text must remain outside this repository.")
        if not controlled_ref or not controlled_ref.strip():
            raise ValueError("restricted sources require --controlled-ref.")
        resolved_controlled_ref = controlled_ref.strip()
    elif redistribution == "metadata_only":
        if input_file is not None or controlled_ref:
            raise ValueError("metadata_only sources must not include local text or controlled snapshot references.")

    previous_content_path = source.content_path
    source.license_id = license_id.strip()
    source.redistribution = redistribution
    source.content_path = content_path
    source.sha256 = digest
    source.controlled_snapshot_ref = resolved_controlled_ref
    source.retrieved_at = datetime.now(timezone.utc)

    _write_json(output_path, dataset.model_dump(mode="json"), overwrite=overwrite_dataset)

    if previous_content_path and previous_content_path != content_path:
        still_referenced = any(
            item.content_path == previous_content_path
            for dataset_case in dataset.cases
            for item in dataset_case.snapshot.sources
        )
        old_snapshot = (root / previous_content_path).resolve()
        if not still_referenced and root in old_snapshot.parents and old_snapshot.is_file():
            old_snapshot.unlink()

    return {
        "output": str(output_path),
        "case_id": case_id,
        "source_id": source_id,
        "redistribution": redistribution,
        "license_id": license_id.strip(),
        "content_path": content_path,
        "sha256": digest,
        "controlled_snapshot_ref": resolved_controlled_ref,
    }


def freeze_case_snapshot(
    dataset_path: str | Path,
    case_id: str,
    output_path: str | Path,
    *,
    repo_root: str | Path = ".",
    overwrite: bool = False,
) -> dict[str, Any]:
    dataset = load_dataset(dataset_path)
    _require_draft(dataset)
    case = _case_by_id(dataset, case_id)
    if case.snapshot.frozen:
        raise ValueError(f"Case {case_id} snapshot is already frozen.")
    if not case.snapshot.sources:
        raise ValueError("Cannot freeze a case without registered sources.")

    root = Path(repo_root).resolve()
    for source in case.snapshot.sources:
        if not source.license_id.strip() or source.redistribution in {"", "unknown"}:
            raise ValueError(
                f"Source {source.source_id} needs an explicit license_id and redistribution decision."
            )
        if source.redistribution in {"redistributable", "excerpt_only"}:
            if not source.content_path or not source.sha256:
                raise ValueError(
                    f"Source {source.source_id} requires a locally stored UTF-8 snapshot and SHA-256."
                )
            snapshot_path = (root / source.content_path).resolve()
            if root not in snapshot_path.parents or not snapshot_path.is_file():
                raise ValueError(f"Snapshot for {source.source_id} is missing or outside repo_root.")
            content = snapshot_path.read_bytes()
            if hashlib.sha256(content).hexdigest() != source.sha256.lower():
                raise ValueError(f"Snapshot SHA-256 mismatch for source {source.source_id}.")
            try:
                content.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"Snapshot for {source.source_id} is not valid UTF-8.") from exc
        elif source.redistribution == "restricted":
            if not source.controlled_snapshot_ref or source.content_path or source.sha256:
                raise ValueError(
                    f"Restricted source {source.source_id} requires a controlled reference and no local copy/hash."
                )
        elif source.redistribution == "metadata_only":
            if source.content_path or source.sha256 or source.controlled_snapshot_ref:
                raise ValueError(f"Metadata-only source {source.source_id} cannot carry a text snapshot.")
        else:
            raise ValueError(
                f"Unsupported redistribution state {source.redistribution!r} for {source.source_id}."
            )

    case.snapshot.frozen = True
    case.snapshot.frozen_at = datetime.now(timezone.utc)
    case.snapshot.capture_method = (
        "Rights-aware manual registration; local UTF-8 snapshots SHA-256 verified; "
        "restricted sources retained by controlled reference; metadata-only sources retain metadata only."
    )
    validation = validate_dataset(dataset, verify_snapshots=True, repo_root=root)
    if not validation["valid"]:
        raise ValueError("Snapshot validation failed: " + "; ".join(validation["errors"]))

    _write_json(output_path, dataset.model_dump(mode="json"), overwrite=overwrite)
    return {
        "output": str(output_path),
        "case_id": case_id,
        "frozen": True,
        "source_count": len(case.snapshot.sources),
        "validation": validation,
    }
