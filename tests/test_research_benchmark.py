from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from evaluation.agreement import case_inter_annotator_agreement, cohens_kappa
from evaluation.dataset import load_dataset, validate_dataset
from evaluation.schemas import (
    BenchmarkCase,
    BenchmarkDataset,
    CaseAdjudication,
    CaseAnnotation,
    ClaimAlignment,
    ClaimJudgment,
    DatasetProvenance,
    EvidenceSnapshot,
    GoldClaim,
    GoldEvidenceSpan,
    RubricScore,
    RunAnnotation,
    RunOutputAdjudication,
    RunOutputAnnotation,
    SnapshotSource,
    SourceRelevanceJudgment,
)
from evaluation.scoring import score_report_against_case

NOW = datetime(2026, 11, 28, tzinfo=timezone.utc)
QUOTE = "BM25 ranks documents using term frequency and inverse document frequency."


def _rubric_scores():
    dimensions = [
        "completeness",
        "analytical_depth",
        "factual_support",
        "citation_quality",
        "uncertainty_calibration",
        "temporal_accuracy",
        "presentation",
    ]
    return [
        RubricScore(
            dimension=dimension,
            score=3,
            rationale=f"The report satisfies the {dimension.replace('_', ' ')} criterion.",
            evidence="Reviewed report section.",
        )
        for dimension in dimensions
    ]


def _alignment_records():
    return [
        ClaimAlignment(
            benchmark_claim_id="G1",
            report_claim_id="C100",
            relation="equivalent",
            rationale="Same proposition and same material scope.",
        ),
        ClaimAlignment(
            benchmark_claim_id=None,
            report_claim_id="C200",
            relation="extra",
            rationale="This additional generated proposition has no gold counterpart.",
        ),
    ]


def _ready_dataset(source_path: Path) -> BenchmarkDataset:
    content = QUOTE + " It also normalizes document length."
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(content, encoding="utf-8")
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    span = GoldEvidenceSpan(
        source_id="SRC1",
        quote=QUOTE,
        start_char=0,
        end_char=len(QUOTE),
    )
    relevance = SourceRelevanceJudgment(
        source_id="SRC1",
        grade=3,
        rationale="The passage directly explains the ranking factors asked about.",
    )
    claim_judgment = ClaimJudgment(
        claim_id="G1",
        status="supported",
        evidence_spans=[span],
        rationale="The frozen passage directly states both ranking factors.",
    )
    source = SnapshotSource(
        source_id="SRC1",
        title="BM25 technical description",
        url="https://example.org/bm25",
        publisher="Example Institute",
        source_type="documentation",
        retrieved_at=NOW,
        published_at=NOW,
        content_path="snapshots/bm25.txt",
        sha256=digest,
        license_id="CC-BY-4.0",
        redistribution="redistributable",
    )
    case = BenchmarkCase(
        case_id="BM25-001",
        question="How does BM25 rank documents?",
        scope="Explain the core lexical ranking factors.",
        as_of=NOW,
        topic_tags=["information-retrieval"],
        snapshot=EvidenceSnapshot(
            snapshot_id="bm25-frozen-01",
            frozen=True,
            frozen_at=NOW,
            capture_method="Saved UTF-8 source text",
            sources=[source],
        ),
        expected_claims=[
            GoldClaim(
                claim_id="G1",
                proposition="BM25 uses term frequency and inverse document frequency to rank documents.",
            )
        ],
        annotations=[
            CaseAnnotation(
                annotator_id="annotator_01",
                annotated_at=NOW,
                source_relevance=[relevance],
                claim_judgments=[claim_judgment],
            ),
            CaseAnnotation(
                annotator_id="annotator_02",
                annotated_at=NOW,
                source_relevance=[relevance],
                claim_judgments=[claim_judgment],
            ),
        ],
        adjudication=CaseAdjudication(
            adjudicator_id="adjudicator_01",
            adjudicated_at=NOW,
            source_relevance=[relevance],
            claim_judgments=[claim_judgment],
            rationale_for_disagreements="Both annotators agreed and the frozen passage directly supports the claim.",
        ),
    )
    return BenchmarkDataset(
        benchmark_id="research-bench-test",
        version="1.0.0",
        status="ready",
        description="Small synthetic fixture for benchmark validator and scorer tests.",
        provenance=DatasetProvenance(
            collection_method="Synthetic fixture created by unit tests.",
            corpus_description="One artificial BM25 technical passage; not research evidence.",
            collected_at=NOW,
            dataset_license="CC0-1.0",
            curation_notes="Synthetic test only.",
        ),
        cases=[case],
    )


def _run_annotation() -> RunAnnotation:
    return RunAnnotation(
        benchmark_id="research-bench-test",
        benchmark_version="1.0.0",
        case_id="BM25-001",
        report_run_id="run-123",
        git_commit="a" * 40,
        model_name="test-model",
        model_version="test-model-v1",
        prompt_versions={"research-writer": 2, "claim-verifier": 4},
        created_at=NOW,
        annotations=[
            RunOutputAnnotation(
                annotator_id="reviewer_01",
                annotated_at=NOW,
                claim_alignments=_alignment_records(),
                rubric_scores=_rubric_scores(),
            ),
            RunOutputAnnotation(
                annotator_id="reviewer_02",
                annotated_at=NOW,
                claim_alignments=_alignment_records(),
                rubric_scores=_rubric_scores(),
            ),
        ],
        adjudication=RunOutputAdjudication(
            adjudicator_id="review_adjudicator",
            adjudicated_at=NOW,
            claim_alignments=_alignment_records(),
            rubric_scores=_rubric_scores(),
            rationale_for_disagreements="The annotators agreed on alignment and rubric scores.",
        ),
    )


def test_draft_template_loads_and_warns_without_claiming_to_be_ready():
    path = Path("evaluation/datasets/benchmark_template.json")
    dataset = load_dataset(path)

    result = validate_dataset(dataset)

    assert result["valid"] is True
    assert result["status"] == "draft"
    assert result["case_count"] == 0
    assert validate_dataset(dataset, require_ready=True)["valid"] is False


def test_ready_dataset_requires_complete_annotations_and_valid_frozen_snapshot(tmp_path):
    dataset = _ready_dataset(tmp_path / "snapshots" / "bm25.txt")

    result = validate_dataset(
        dataset,
        require_ready=True,
        verify_snapshots=True,
        repo_root=tmp_path,
    )

    assert result["valid"] is True, result["errors"]


def test_snapshot_hash_and_evidence_quote_are_verified(tmp_path):
    dataset = _ready_dataset(tmp_path / "snapshots" / "bm25.txt")
    source = dataset.cases[0].snapshot.sources[0]
    source.sha256 = "0" * 64

    result = validate_dataset(dataset, verify_snapshots=True, repo_root=tmp_path)

    assert result["valid"] is False
    assert any("SHA-256 mismatch" in error for error in result["errors"])


def test_ready_dataset_supports_restricted_sources_without_committing_text(tmp_path):
    dataset = _ready_dataset(tmp_path / "snapshots" / "bm25.txt")
    source = dataset.cases[0].snapshot.sources[0]
    source.redistribution = "restricted"
    source.content_path = None
    source.sha256 = None
    source.controlled_snapshot_ref = "vault://research-bench/bm25/snapshot-01"

    result = validate_dataset(
        dataset,
        require_ready=True,
        verify_snapshots=True,
        repo_root=tmp_path,
    )

    assert result["valid"] is True, result["errors"]
    assert any("was not verified locally" in warning for warning in result["warnings"])


def test_ready_dataset_rejects_incomplete_independent_annotation(tmp_path):
    dataset = _ready_dataset(tmp_path / "snapshots" / "bm25.txt")
    dataset.cases[0].annotations[0].claim_judgments = []

    result = validate_dataset(dataset, require_ready=True, repo_root=tmp_path)

    assert result["valid"] is False
    assert any("did not label every expected claim" in error for error in result["errors"])


def test_inter_annotator_agreement_uses_fixed_ordinal_and_nominal_categories(tmp_path):
    dataset = _ready_dataset(tmp_path / "snapshots" / "bm25.txt")
    result = case_inter_annotator_agreement(dataset.cases[0])

    assert result["pair_count"] == 1
    assert result["source_relevance_quadratic_weighted_kappa"] == 1.0
    assert result["claim_status_kappa"] == 1.0
    assert cohens_kappa([0], [3], ordinal=True, categories=[0, 1, 2, 3]) == 0.0


def test_scorer_reports_retrieval_claim_alignment_and_human_rubric_metrics(tmp_path):
    dataset = _ready_dataset(tmp_path / "snapshots" / "bm25.txt")
    case = dataset.cases[0]
    run_annotation = _run_annotation()
    report = {
        "sources": [{"id": "S1", "url": "https://example.org/bm25"}],
        "claim_verifications": [
            {
                "claim_id": "C100",
                "status": "supported",
                "supporting_evidence": [
                    {
                        "source_id": "S1",
                        "quote": QUOTE,
                        "explanation": "The passage states the ranking factors.",
                    }
                ],
                "contradicting_evidence": [],
            },
            {
                "claim_id": "C200",
                "status": "uncertain",
                "supporting_evidence": [],
                "contradicting_evidence": [],
            },
        ],
    }

    result = score_report_against_case(report, case, run_annotation, k=3)

    assert result["retrieval"]["recall_at_k"] == 1.0
    assert result["retrieval"]["precision_at_k"] == pytest.approx(1 / 3)
    assert result["retrieval"]["ndcg_at_k"] == 1.0
    claims = result["claim_evaluation"]
    assert claims["status_accuracy_on_matched_claims"] == 1.0
    assert claims["claim_coverage_recall"] == 1.0
    assert claims["claim_alignment_precision"] == 0.5
    assert claims["extra_generated_claim_count"] == 1
    assert claims["unresolved_generated_claim_count"] == 1
    assert result["evidence_grounding"]["claim_source_precision"] == 1.0
    assert result["evidence_grounding"]["claim_source_recall"] == 1.0
    assert result["evidence_grounding"]["exact_quote_match_rate"] == 1.0
    assert result["human_report_rubric"]["mean_across_scored_dimensions"] == 3.0
    assert result["system"]["prompt_versions"]["claim-verifier"] == 4


def test_scorer_rejects_missing_second_human_review(tmp_path):
    dataset = _ready_dataset(tmp_path / "snapshots" / "bm25.txt")
    annotation = _run_annotation()
    annotation.annotations = annotation.annotations[:1]

    with pytest.raises(ValueError, match="two independent"):
        score_report_against_case(
            {"sources": [], "claim_verifications": []},
            dataset.cases[0],
            annotation,
        )


def test_scorer_rejects_unaccounted_report_claims(tmp_path):
    dataset = _ready_dataset(tmp_path / "snapshots" / "bm25.txt")
    annotation = _run_annotation()
    annotation.adjudication.claim_alignments = annotation.adjudication.claim_alignments[:1]

    with pytest.raises(ValueError, match="every verified/generated"):
        score_report_against_case(
            {
                "sources": [{"id": "S1", "url": "https://example.org/bm25"}],
                "claim_verifications": [
                    {"claim_id": "C100", "status": "supported"},
                    {"claim_id": "C200", "status": "uncertain"},
                ],
            },
            dataset.cases[0],
            annotation,
        )
