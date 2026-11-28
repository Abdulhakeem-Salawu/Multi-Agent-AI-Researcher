from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


ClaimStatus = Literal["supported", "contradicted", "insufficient_evidence", "uncertain"]
RubricDimension = Literal[
    "completeness",
    "analytical_depth",
    "factual_support",
    "citation_quality",
    "uncertainty_calibration",
    "temporal_accuracy",
    "presentation",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetProvenance(StrictModel):
    collection_method: str = Field(min_length=8)
    corpus_description: str = Field(min_length=8)
    collected_at: datetime
    dataset_license: str
    curation_notes: str = ""


class SnapshotSource(StrictModel):
    source_id: str = Field(pattern=r"^[A-Za-z0-9_.:-]+$")
    title: str = Field(min_length=1)
    url: HttpUrl
    publisher: str = ""
    source_type: Literal[
        "primary", "secondary", "official_record", "dataset", "documentation", "other"
    ] = "other"
    published_at: datetime | None = None
    retrieved_at: datetime
    content_path: str | None = None
    sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    controlled_snapshot_ref: str | None = None
    license_id: str = ""
    redistribution: Literal[
        "redistributable", "excerpt_only", "metadata_only", "restricted", "unknown"
    ] = "unknown"


class EvidenceSnapshot(StrictModel):
    snapshot_id: str = Field(min_length=3)
    frozen: bool = False
    frozen_at: datetime | None = None
    capture_method: str = Field(min_length=3)
    sources: list[SnapshotSource] = Field(default_factory=list)


class GoldClaim(StrictModel):
    claim_id: str = Field(pattern=r"^[A-Za-z0-9_.:-]+$")
    proposition: str = Field(min_length=8)
    scope_notes: str = ""


class GoldEvidenceSpan(StrictModel):
    source_id: str
    quote: str = Field(min_length=1, max_length=3000)
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=1)


class SourceRelevanceJudgment(StrictModel):
    source_id: str
    grade: int = Field(ge=0, le=3)
    rationale: str = Field(min_length=5)


class ClaimJudgment(StrictModel):
    claim_id: str
    status: ClaimStatus
    evidence_spans: list[GoldEvidenceSpan] = Field(default_factory=list)
    rationale: str = Field(min_length=8)


class CaseAnnotation(StrictModel):
    annotator_id: str = Field(min_length=2)
    annotated_at: datetime
    source_relevance: list[SourceRelevanceJudgment] = Field(default_factory=list)
    claim_judgments: list[ClaimJudgment] = Field(default_factory=list)
    notes: str = ""


class CaseAdjudication(StrictModel):
    adjudicator_id: str = Field(min_length=2)
    adjudicated_at: datetime
    source_relevance: list[SourceRelevanceJudgment] = Field(default_factory=list)
    claim_judgments: list[ClaimJudgment] = Field(default_factory=list)
    rationale_for_disagreements: str = ""


class BenchmarkCase(StrictModel):
    case_id: str = Field(pattern=r"^[A-Za-z0-9_.:-]+$")
    question: str = Field(min_length=8)
    scope: str = Field(min_length=5)
    as_of: datetime | None = None
    topic_tags: list[str] = Field(default_factory=list)
    snapshot: EvidenceSnapshot
    expected_claims: list[GoldClaim] = Field(default_factory=list)
    annotations: list[CaseAnnotation] = Field(default_factory=list)
    adjudication: CaseAdjudication | None = None


class BenchmarkDataset(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    benchmark_id: str = Field(min_length=3)
    version: str = Field(min_length=1)
    status: Literal["draft", "ready"] = "draft"
    description: str = Field(min_length=8)
    provenance: DatasetProvenance
    cases: list[BenchmarkCase] = Field(default_factory=list)


class ClaimAlignment(StrictModel):
    benchmark_claim_id: str | None = None
    report_claim_id: str | None = None
    relation: Literal["equivalent", "partial", "missing", "extra"]
    rationale: str = Field(min_length=5)


class RubricScore(StrictModel):
    dimension: RubricDimension
    score: int = Field(ge=0, le=4)
    rationale: str = Field(min_length=8)
    evidence: str = ""


class RunOutputAnnotation(StrictModel):
    annotator_id: str = Field(min_length=2)
    annotated_at: datetime
    claim_alignments: list[ClaimAlignment] = Field(default_factory=list)
    rubric_scores: list[RubricScore] = Field(default_factory=list)
    notes: str = ""


class RunOutputAdjudication(StrictModel):
    adjudicator_id: str = Field(min_length=2)
    adjudicated_at: datetime
    claim_alignments: list[ClaimAlignment] = Field(default_factory=list)
    rubric_scores: list[RubricScore] = Field(default_factory=list)
    rationale_for_disagreements: str = ""


class RunAnnotation(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    benchmark_id: str
    benchmark_version: str
    case_id: str
    report_run_id: int | str
    git_commit: str
    model_name: str
    model_version: str = ""
    prompt_versions: dict[str, str | int] = Field(default_factory=dict)
    created_at: datetime
    annotations: list[RunOutputAnnotation] = Field(default_factory=list)
    adjudication: RunOutputAdjudication | None = None
