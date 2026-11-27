from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class ResearchState(TypedDict, total=False):
    question: str
    max_queries: int
    max_results_per_query: int
    query_plan: list[str]
    tasks: list[dict[str, Any]]
    search_results: Annotated[list[dict[str, Any]], operator.add]
    search_errors: Annotated[list[str], operator.add]
    planning_errors: Annotated[list[str], operator.add]
    ranked_results: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    source_map: dict[str, str]
    draft_answer: str
    claims: list[dict[str, Any]]
    limitations: list[str]
    report: dict[str, Any]
