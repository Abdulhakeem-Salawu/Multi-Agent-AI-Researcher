from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from evaluation.annotation_workflow import (
    export_adjudication_form,
    export_annotation_forms,
    freeze_case_snapshot,
    import_adjudication_form,
    import_annotation_forms,
    register_source_snapshot,
)
from evaluation.dataset import load_dataset
from evaluation.schemas import (
    BenchmarkCase,
    BenchmarkDataset,
    DatasetProvenance,
    EvidenceSnapshot,
    GoldClaim,
    SnapshotSource,
)

NOW = datetime(2026, 11, 28, tzinfo=timezone.utc)
QUOTE = "This is the authoritative evidence passage for the synthetic candidate claim."


def _draft_dataset(tmp_path: Path) -> Path:
    dataset = BenchmarkDataset(
        benchmark_id="annotation-test",
        version="0.1.0-draft",
        status="draft",
        description="A synthetic dataset for annotation workflow tests.",
        provenance=DatasetProvenance(
            collection_method="Synthetic test fixture created by unit tests.",
            corpus_description="One synthetic test source; not research ground truth.",
            collected_at=NOW,
            dataset_license="CC0-1.0",
        ),
        cases=[
            BenchmarkCase(
                case_id="TEST-001",
                question="What does the synthetic source state?",
                scope="Use only the frozen test source.",
                snapshot=EvidenceSnapshot(
                    snapshot_id="test-snapshot-v0",
                    frozen=False,
                    capture_method="Source URL metadata only.",
                    sources=[
                        SnapshotSource(
                            source_id="TEST-SOURCE",
                            title="Synthetic evidence source",
                            url="https://example.org/source",
                            publisher="Example",
                            source_type="documentation",
                            retrieved_at=NOW,
                        )
                    ],
                ),
                expected_claims=[
                    GoldClaim(
                        claim_id="TEST-C1",
                        proposition="The synthetic source contains an authoritative evidence passage.",
                    )
                ],
            )
        ],
    )
    path = tmp_path / "benchmark.json"
    path.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")
    return path


def _register_and_freeze(tmp_path: Path) -> Path:
    dataset_path = _draft_dataset(tmp_path)
    source_file = tmp_path / "source.txt"
    source_file.write_text(QUOTE, encoding="utf-8")
    registered = tmp_path / "registered.json"
    register_source_snapshot(
        dataset_path,
        "TEST-001",
        "TEST-SOURCE",
        "redistributable",
        "CC0-1.0",
        registered,
        input_file=source_file,
        confirm_rights=True,
        repo_root=tmp_path,
    )
    frozen = tmp_path / "frozen.json"
    freeze_case_snapshot(registered, "TEST-001", frozen, repo_root=tmp_path)
    return frozen


def _complete_form(path: Path, annotator_id: str) -> Path:
    form = json.loads(path.read_text(encoding="utf-8"))
    form["annotated_at"] = NOW.isoformat()
    form["source_relevance"] = [
        {
            "source_id": "TEST-SOURCE",
            "grade": 3,
            "rationale": "The source directly addresses the question.",
        }
    ]
    form["claim_judgments"] = [
        {
            "claim_id": "TEST-C1",
            "status": "supported",
            "evidence_spans": [
                {
                    "source_id": "TEST-SOURCE",
                    "quote": QUOTE,
                    "start_char": 0,
                    "end_char": len(QUOTE),
                }
            ],
            "rationale": "The exact frozen passage directly supports the proposition.",
        }
    ]
    form["annotator_id"] = annotator_id
    path.write_text(json.dumps(form, indent=2), encoding="utf-8")
    return path


def test_snapshot_registration_requires_explicit_rights_confirmation(tmp_path):
    dataset_path = _draft_dataset(tmp_path)
    source_file = tmp_path / "source.txt"
    source_file.write_text(QUOTE, encoding="utf-8")

    with pytest.raises(ValueError, match="confirm-rights"):
        register_source_snapshot(
            dataset_path,
            "TEST-001",
            "TEST-SOURCE",
            "redistributable",
            "CC0-1.0",
            tmp_path / "out.json",
            input_file=source_file,
            repo_root=tmp_path,
        )


def test_register_and_freeze_records_hash_and_prevents_later_changes(tmp_path):
    frozen_path = _register_and_freeze(tmp_path)
    dataset = load_dataset(frozen_path)
    source = dataset.cases[0].snapshot.sources[0]

    assert dataset.cases[0].snapshot.frozen is True
    assert source.content_path == "evaluation/snapshots/TEST-001/TEST-SOURCE.txt"
    assert len(source.sha256 or "") == 64
    assert (tmp_path / source.content_path).read_text(encoding="utf-8") == QUOTE

    with pytest.raises(ValueError, match="already been frozen"):
        register_source_snapshot(
            frozen_path,
            "TEST-001",
            "TEST-SOURCE",
            "metadata_only",
            "CC0-1.0",
            tmp_path / "changed.json",
            repo_root=tmp_path,
        )


def test_annotation_forms_are_isolated_and_import_is_complete(tmp_path):
    frozen_path = _register_and_freeze(tmp_path)
    forms_dir = tmp_path / "forms"
    paths = export_annotation_forms(frozen_path, forms_dir, ["annotator_01", "annotator_02"])

    assert len(paths) == 2
    first = json.loads(Path(paths[0]).read_text(encoding="utf-8"))
    second = json.loads(Path(paths[1]).read_text(encoding="utf-8"))
    assert first["annotator_id"] != second["annotator_id"]
    assert first["source_relevance"][0]["grade"] is None
    assert first["claim_judgments"][0]["status"] is None
    assert "independent_annotations" not in first

    completed = [
        _complete_form(Path(paths[0]), "annotator_01"),
        _complete_form(Path(paths[1]), "annotator_02"),
    ]
    output = tmp_path / "annotated.json"
    result = import_annotation_forms(frozen_path, completed, output, repo_root=tmp_path)

    assert result["imported_count"] == 2
    stored = load_dataset(output)
    assert {item.annotator_id for item in stored.cases[0].annotations} == {
        "annotator_01",
        "annotator_02",
    }


def test_duplicate_annotator_submission_is_rejected(tmp_path):
    frozen_path = _register_and_freeze(tmp_path)
    paths = export_annotation_forms(frozen_path, tmp_path / "forms", ["annotator_01"])
    completed = _complete_form(Path(paths[0]), "annotator_01")
    first_output = tmp_path / "first.json"
    import_annotation_forms(frozen_path, [completed], first_output, repo_root=tmp_path)

    with pytest.raises(ValueError, match="already submitted"):
        import_annotation_forms(first_output, [completed], tmp_path / "second.json", repo_root=tmp_path)


def test_adjudication_requires_two_independent_annotators_and_imports(tmp_path):
    frozen_path = _register_and_freeze(tmp_path)
    paths = export_annotation_forms(frozen_path, tmp_path / "forms", ["annotator_01", "annotator_02"])
    completed = [
        _complete_form(Path(paths[0]), "annotator_01"),
        _complete_form(Path(paths[1]), "annotator_02"),
    ]
    annotated_path = tmp_path / "annotated.json"
    import_annotation_forms(frozen_path, completed, annotated_path, repo_root=tmp_path)

    form_path = tmp_path / "adjudication.json"
    export_adjudication_form(annotated_path, "TEST-001", "adjudicator_01", form_path)
    form = json.loads(form_path.read_text(encoding="utf-8"))
    form["adjudicated_at"] = NOW.isoformat()
    form["source_relevance"] = [
        {
            "source_id": "TEST-SOURCE",
            "grade": 3,
            "rationale": "The source directly answers the question.",
        }
    ]
    form["claim_judgments"] = [
        {
            "claim_id": "TEST-C1",
            "status": "supported",
            "evidence_spans": [
                {
                    "source_id": "TEST-SOURCE",
                    "quote": QUOTE,
                    "start_char": 0,
                    "end_char": len(QUOTE),
                }
            ],
            "rationale": "Both independent annotations agree with the frozen source passage.",
        }
    ]
    form["rationale_for_disagreements"] = "No disagreement remained after checking the frozen source."
    form_path.write_text(json.dumps(form, indent=2), encoding="utf-8")

    output = tmp_path / "adjudicated.json"
    result = import_adjudication_form(annotated_path, form_path, output, repo_root=tmp_path)

    assert result["case_id"] == "TEST-001"
    adjudicated = load_dataset(output).cases[0].adjudication
    assert adjudicated is not None
    assert adjudicated.adjudicator_id == "adjudicator_01"


def test_restricted_source_uses_external_reference_without_local_text(tmp_path):
    dataset_path = _draft_dataset(tmp_path)
    registered = tmp_path / "registered.json"
    register_source_snapshot(
        dataset_path,
        "TEST-001",
        "TEST-SOURCE",
        "restricted",
        "publisher-terms-reviewed",
        registered,
        controlled_ref="vault://research-bench/TEST-001/TEST-SOURCE",
        repo_root=tmp_path,
    )
    frozen = tmp_path / "frozen.json"
    freeze_case_snapshot(registered, "TEST-001", frozen, repo_root=tmp_path)

    source = load_dataset(frozen).cases[0].snapshot.sources[0]
    assert source.redistribution == "restricted"
    assert source.controlled_snapshot_ref.startswith("vault://")
    assert source.content_path is None
    assert source.sha256 is None
