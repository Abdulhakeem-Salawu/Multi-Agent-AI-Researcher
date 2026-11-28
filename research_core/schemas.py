from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class SearchHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    url: HttpUrl
    content: str = ""
    raw_content: str | None = None
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    published_date: str | None = None
    docid: str | None = None


class RetrievedHit(SearchHit):
    query: str
    query_index: int = Field(ge=0)
    result_index: int = Field(ge=0)
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EvidenceSource(BaseModel):
    id: str
    title: str
    url: HttpUrl
    published_date: str | None = None
    docid: str | None = None
    retrieved_at: datetime
    search_score: float = Field(ge=0.0, le=1.0)
    relevance_score: float = Field(ge=0.0)
    retrieved_excerpt: str = ""


class ClaimAssessment(BaseModel):
    claim: str
    cited_sources: list[str] = Field(default_factory=list)
    lexical_support_score: float = Field(ge=0.0, le=1.0)
    status: Literal[
        "uncited",
        "invalid_citation",
        "needs_review",
        "candidate_supported",
    ]
    reason: str


class CandidateClaim(BaseModel):
    claim_id: str
    claim: str = Field(min_length=5)
    cited_source_ids: list[str] = Field(default_factory=list)


class EvidenceSpan(BaseModel):
    source_id: str
    quote: str = Field(min_length=1, max_length=2000)
    explanation: str = ""


class ClaimVerification(BaseModel):
    claim_id: str
    claim: str
    status: Literal[
        "supported",
        "contradicted",
        "insufficient_evidence",
        "uncertain",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_evidence: list[EvidenceSpan] = Field(default_factory=list)
    contradicting_evidence: list[EvidenceSpan] = Field(default_factory=list)
    reasoning: str
    research_needed: bool = False
    span_warnings: list[str] = Field(default_factory=list)


class ReportReview(BaseModel):
    status: Literal["reviewed", "not_run", "failed"] = "not_run"
    overall_score: float | None = Field(default=None, ge=0.0, le=5.0)
    completeness_score: float | None = Field(default=None, ge=0.0, le=5.0)
    analytical_depth_score: float | None = Field(default=None, ge=0.0, le=5.0)
    citation_quality_score: float | None = Field(default=None, ge=0.0, le=5.0)
    ready_for_publication: bool | None = None
    strengths: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    rationale: str = ""


class ResearchReport(BaseModel):
    question: str
    answer: str
    sources: list[EvidenceSource] = Field(default_factory=list)
    claims: list[ClaimAssessment] = Field(default_factory=list)
    claim_verifications: list[ClaimVerification] = Field(default_factory=list)
    report_review: ReportReview = Field(default_factory=ReportReview)
    limitations: list[str] = Field(default_factory=list)
    query_plan: list[str] = Field(default_factory=list)
    research_rounds: int = Field(default=0, ge=0)
    run_id: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
