from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def merge_verification_results(
    existing: dict[str, dict[str, Any]],
    updates: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Merge parallel verifier outputs; newer verification for a claim replaces older output."""
    return {**existing, **updates}


class ResearchState(TypedDict, total=False):
    question: str
    max_queries: int
    max_results_per_query: int
    max_research_rounds: int
    research_round: int
    total_research_tasks: int
    query_plan: Annotated[list[str], operator.add]
    tasks: list[dict[str, Any]]
    search_results: Annotated[list[dict[str, Any]], operator.add]
    search_errors: Annotated[list[str], operator.add]
    planning_errors: Annotated[list[str], operator.add]
    ranked_results: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    source_map: dict[str, str]
    draft_answer: str
    claims: list[dict[str, Any]]
    claim_candidates: list[dict[str, Any]]
    verification_results: Annotated[
        dict[str, dict[str, Any]],
        merge_verification_results,
    ]
    verification_errors: Annotated[list[str], operator.add]
    report_review: dict[str, Any]
    limitations: list[str]
    report: dict[str, Any]
