from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from research_core.audit import audit_claims
from research_core.retrieval import rank_hits
from research_core.schemas import EvidenceSource, ResearchReport, RetrievedHit, SearchHit

from .dependencies import ResearchDependencies
from .state import ResearchState


def _unique_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))


def _fallback_plan(question: str) -> list[str]:
    parts = re.split(
        r"\b(?:and|versus|vs\.?|compared with|impact of|relationship between)\b",
        question,
        flags=re.IGNORECASE,
    )
    return _unique_strings([question, *[part.strip(" ?.,;:") for part in parts if len(part.split()) >= 3]])


def _parse_query_list(raw: str) -> list[str]:
    cleaned = raw.strip()
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("Planner response did not contain a JSON array.")
    parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, list):
        raise TypeError("Planner response must be a JSON array.")
    return _unique_strings([str(item) for item in parsed if str(item).strip()])


def build_research_graph(
    dependencies: ResearchDependencies,
    checkpointer: Any | None = None,
):
    """Build the graph. Pass a LangGraph checkpointer to persist execution state."""

    def intake(state: ResearchState) -> dict[str, Any]:
        question = str(state.get("question", "")).strip()
        if len(question) < 8:
            raise ValueError("Please provide a research question of at least 8 characters.")
        if len(question) > 2000:
            raise ValueError("Research questions must be 2,000 characters or fewer.")
        return {
            "question": question,
            "max_queries": dependencies.max_queries,
            "max_results_per_query": dependencies.max_results_per_query,
            "search_results": [],
            "search_errors": [],
            "planning_errors": [],
            "limitations": [],
        }

    def plan_research(state: ResearchState) -> dict[str, Any]:
        question = state["question"]
        planning_errors: list[str] = []
        queries: list[str] = []
        if dependencies.llm:
            try:
                raw = dependencies.llm.complete(
                    "You are a research planner. Return only a JSON array of 2 to 4 concise, "
                    "non-overlapping web search queries. Do not answer the question.",
                    f"Research question: {question}",
                    temperature=0.0,
                )
                queries = _parse_query_list(raw)
            except Exception as exc:  # noqa: BLE001 - convert planner/model failures into a visible deterministic fallback.
                planning_errors.append(
                    f"Planner failed ({type(exc).__name__}: {exc}); deterministic query planning was used."
                )

        if not queries:
            queries = _fallback_plan(question)

        queries = _unique_strings(queries)[: dependencies.max_queries]
        if not queries:
            raise RuntimeError("The planner could not produce any research queries.")

        tasks = [
            {"task_id": f"research-{index}", "query": query, "query_index": index}
            for index, query in enumerate(queries)
        ]
        return {
            "query_plan": queries,
            "tasks": tasks,
            "planning_errors": planning_errors,
        }

    def dispatch_research(state: ResearchState) -> list[Send]:
        return [Send("researcher", {"task": task}) for task in state.get("tasks", [])]

    def researcher(task_state: dict[str, Any]) -> dict[str, Any]:
        task = task_state["task"]
        query = task["query"]
        query_index = int(task["query_index"])
        try:
            hits = dependencies.search_provider.search(
                query,
                max_results=dependencies.max_results_per_query,
            )
            retrieved_at = datetime.now(timezone.utc)
            results = []
            for result_index, hit in enumerate(hits):
                parsed = hit if isinstance(hit, SearchHit) else SearchHit.model_validate(hit)
                results.append(
                    RetrievedHit(
                        **parsed.model_dump(),
                        query=query,
                        query_index=query_index,
                        result_index=result_index,
                        retrieved_at=retrieved_at,
                    ).model_dump(mode="json")
                )
            return {"search_results": results}
        except Exception as exc:  # noqa: BLE001 - all provider failures must become explicit task errors.
            # Preserve the task failure in state; never turn an error into a fabricated finding.
            return {
                "search_errors": [
                    f"{task['task_id']} ({query}): {type(exc).__name__}: {exc}"
                ]
            }

    def rank_evidence(state: ResearchState) -> dict[str, Any]:
        raw_hits = [
            RetrievedHit.model_validate(item)
            for item in sorted(
                state.get("search_results", []),
                key=lambda item: (
                    int(item["query_index"]),
                    int(item["result_index"]),
                    str(item["url"]),
                ),
            )
        ]
        ranked = rank_hits(
            state["question"],
            raw_hits,
            top_k=dependencies.max_evidence_sources,
            mode=dependencies.retrieval_mode,
        )

        source_map: dict[str, str] = {}
        ranked_results: list[dict[str, Any]] = []
        sources: list[dict[str, Any]] = []
        for index, (hit, score) in enumerate(ranked, start=1):
            source_id = f"S{index}"
            evidence_text = (hit.raw_content or hit.content).strip()
            source_map[source_id] = evidence_text
            ranked_results.append(
                {
                    **hit.model_dump(mode="json"),
                    "source_id": source_id,
                    "relevance_score": round(score, 6),
                    "evidence_text": evidence_text,
                }
            )
            sources.append(
                EvidenceSource(
                    id=source_id,
                    title=hit.title,
                    url=hit.url,
                    published_date=hit.published_date,
                    retrieved_at=hit.retrieved_at,
                    search_score=hit.score,
                    relevance_score=max(0.0, score),
                    retrieved_excerpt=evidence_text[:1200],
                ).model_dump(mode="json")
            )

        limitations = list(state.get("limitations", []))
        limitations.extend(state.get("planning_errors", []))
        if state.get("search_errors"):
            limitations.append(
                f"{len(state['search_errors'])} of {len(state.get('tasks', []))} research tasks failed; "
                "the evidence set may be incomplete."
            )
        if not ranked:
            limitations.append("No usable search evidence was returned.")
        if any(not item.get("published_date") for item in sources):
            limitations.append(
                "Publication dates were unavailable for one or more sources; recency could not be "
                "assessed consistently."
            )

        return {
            "ranked_results": ranked_results,
            "sources": sources,
            "source_map": source_map,
            "limitations": _unique_strings(limitations),
        }

    def write_report(state: ResearchState) -> dict[str, Any]:
        ranked_results = state.get("ranked_results", [])
        limitations = list(state.get("limitations", []))

        if not ranked_results:
            return {
                "draft_answer": (
                    "I could not find usable retrieved evidence for this question. "
                    "The system cannot provide a reliable answer from the available search results."
                ),
                "limitations": _unique_strings(limitations),
            }

        if dependencies.llm:
            evidence_blocks = [
                (
                    f"[{item['source_id']}] {item['title']}\n"
                    f"URL: {item['url']}\n"
                    f"Published: {item.get('published_date') or 'Not provided'}\n"
                    f"Evidence: {item['evidence_text'][:2400]}"
                )
                for item in ranked_results
            ]
            try:
                answer = dependencies.llm.complete(
                    "You are an evidence-grounded research analyst. Use only the supplied evidence. "
                    "Cite factual paragraphs and bullets with the exact source IDs such as [S1]. "
                    "Do not invent facts, dates, quotations, or URLs. Distinguish evidence from inference, "
                    "surface disagreements, and explicitly state when the evidence is insufficient.",
                    f"Research question: {state['question']}\n\n"
                    + "\n\n".join(evidence_blocks)
                    + "\n\nWrite a structured, balanced answer with concise analysis.",
                    temperature=0.1,
                ).strip()
                if not answer:
                    raise RuntimeError("The configured model returned an empty answer.")
                return {"draft_answer": answer}
            except Exception as exc:  # noqa: BLE001 - expose model failures and use the documented extractive fallback.
                limitations.append(
                    f"LLM synthesis failed ({type(exc).__name__}: {exc}); an extractive evidence summary was used."
                )
        else:
            limitations.append(
                "No LLM was configured; an extractive evidence summary was used instead of model synthesis."
            )

        lines = [
            "## Retrieved evidence summary",
            "These points are extracted from retrieved source text and have not been independently verified:",
        ]
        for item in ranked_results[:5]:
            text = item["evidence_text"].strip()
            first_sentence = re.split(r"(?<=[.!?])\s+", text)[0] if text else ""
            if first_sentence:
                lines.append(f"- {first_sentence[:500]} [{item['source_id']}]")
        return {
            "draft_answer": "\n".join(lines),
            "limitations": _unique_strings(limitations),
        }

    def audit_report(state: ResearchState) -> dict[str, Any]:
        checks = audit_claims(state.get("draft_answer", ""), state.get("source_map", {}))
        limitations = list(state.get("limitations", []))
        if any(
            check.status in {"uncited", "invalid_citation", "needs_review"}
            for check in checks
        ):
            limitations.append(
                "One or more candidate claims need citation review. The lexical audit is a screening "
                "heuristic, not an entailment or truth verifier."
            )
        elif checks:
            limitations.append(
                "Claims marked candidate_supported passed lexical-overlap screening only; entailment "
                "has not been established."
            )

        return {
            "claims": [check.model_dump(mode="json") for check in checks],
            "limitations": _unique_strings(limitations),
        }

    def finalize_report(state: ResearchState) -> dict[str, Any]:
        report = ResearchReport(
            question=state["question"],
            answer=state.get("draft_answer", ""),
            sources=[
                EvidenceSource.model_validate(source)
                for source in state.get("sources", [])
            ],
            claims=state.get("claims", []),
            limitations=state.get("limitations", []),
            query_plan=state.get("query_plan", []),
        )
        if dependencies.store:
            report.run_id = dependencies.store.save(report)
        return {"report": report.model_dump(mode="json")}

    builder = StateGraph(ResearchState)
    builder.add_node("intake", intake)
    builder.add_node("planner", plan_research)
    builder.add_node("researcher", researcher)
    builder.add_node("rank_evidence", rank_evidence)
    builder.add_node("writer", write_report)
    builder.add_node("claim_audit", audit_report)
    builder.add_node("finalize", finalize_report)

    builder.add_edge(START, "intake")
    builder.add_edge("intake", "planner")
    builder.add_conditional_edges("planner", dispatch_research, ["researcher"])
    builder.add_edge("researcher", "rank_evidence")
    builder.add_edge("rank_evidence", "writer")
    builder.add_edge("writer", "claim_audit")
    builder.add_edge("claim_audit", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)


def build_default_graph(checkpointer: Any | None = None):
    from .dependencies import build_default_dependencies

    return build_research_graph(build_default_dependencies(), checkpointer=checkpointer)
