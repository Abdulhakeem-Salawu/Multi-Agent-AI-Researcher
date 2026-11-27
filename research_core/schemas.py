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


class ResearchReport(BaseModel):
    question: str
    answer: str
    sources: list[EvidenceSource] = Field(default_factory=list)
    claims: list[ClaimAssessment] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    query_plan: list[str] = Field(default_factory=list)
    run_id: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
