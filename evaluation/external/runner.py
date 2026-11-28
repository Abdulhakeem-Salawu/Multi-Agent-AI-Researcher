from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research_core.llm import OpenAICompatibleChatModel

from .adapters import load_browsecomp_plus_jsonl, load_deepresearch_bench_ii_jsonl, load_frames_tsv
from .corpus import BrowseCompPlusCorpusSearchProvider
from .reporting import write_run_summary
from .scoring import (
    score_browsecomp_plus_retrieval,
    score_drb2_rubric_judgments,
    score_frames_answer,
    score_frames_retrieval,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    if os.getenv("GITHUB_SHA"):
        return os.getenv("GITHUB_SHA")
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _parse_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Judge response did not contain a JSON object.")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise TypeError("Judge response must be a JSON object.")
    return value


def _drb2_judge(item: Any, report: dict[str, Any], judge: OpenAICompatibleChatModel):
    rubric_rows = (item.annotations or {}).get("rubrics", [])
    expected = {row["rubric_id"]: row for row in rubric_rows}
    report_text = str(report.get("answer", ""))[:150000]
    blocked = (item.annotations or {}).get("blocked_reference") or {
        "urls": (item.annotations or {}).get("blocked_source_urls", [])
    }
    scores: dict[str, int] = {}
    details = []
    sources = report.get("sources", [])
    for start in range(0, len(rubric_rows), 25):
        batch = rubric_rows[start:start + 25]
        raw = judge.complete(
            "You are a strict independent report judge using the DeepResearch Bench II three-way "
            "rubric. Return JSON only with results: rubric_id, score, reason and evidence. Score 1 "
            "if clearly satisfied with non-blocked supporting text; 0 if not mentioned; -1 if "
            "mentioned but the supporting sentence cites a blocked reference. Nonzero evidence "
            "must be an exact substring of the report. Map [S#] citations to supplied source URLs. "
            "Include every input rubric ID exactly once.",
            json.dumps({
                "task": item.question,
                "rubrics": batch,
                "blocked_reference": blocked,
                "report": report_text,
                "sources": sources,
            }, ensure_ascii=False),
            temperature=0.0,
        )
        results = _parse_json(raw).get("results")
        if not isinstance(results, list):
            raise TypeError("DRB II judge did not return a results list.")
        batch_ids = {row["rubric_id"] for row in batch}
        received: set[str] = set()
        for result in results:
            rubric_id = str(result.get("rubric_id", ""))
            score = result.get("score")
            evidence = str(result.get("evidence") or "")
            if rubric_id not in batch_ids or rubric_id in received:
                raise ValueError(f"Unknown or duplicate DRB II rubric ID: {rubric_id}")
            if type(score) is not int or score not in {-1, 0, 1}:
                raise ValueError(f"Invalid DRB II score for {rubric_id}.")
            if score == 0 and evidence:
                raise ValueError(f"Score 0 must not include evidence for {rubric_id}.")
            if score != 0 and (not evidence or evidence not in report_text):
                raise ValueError(f"Nonzero score lacks exact report evidence for {rubric_id}.")
            received.add(rubric_id)
            scores[rubric_id] = score
            details.append({
                "rubric_id": rubric_id,
                "dimension": expected[rubric_id]["dimension"],
                "score": score,
                "reason": str(result.get("reason") or ""),
                "evidence": evidence,
            })
        if received != batch_ids:
            raise ValueError("DRB II judge omitted one or more rubric IDs.")
    return score_drb2_rubric_judgments(item, scores), details


def _browsecomp_answer_judge(item: Any, report: dict[str, Any], judge: OpenAICompatibleChatModel):
    raw = judge.complete(
        "Judge if the report materially and correctly answers this fact-seeking question compared "
        "with the reference answer. Accept equivalent wording; reject vague, incomplete or contradictory "
        "answers. Return JSON only with a boolean correct and a short reason.",
        json.dumps({
            "question": item.question,
            "reference_answer": item.reference_answer,
            "report": report.get("answer", ""),
        }, ensure_ascii=False),
        temperature=0.0,
    )
    result = _parse_json(raw)
    if type(result.get("correct")) is not bool:
        raise ValueError("BrowseComp-Plus judge must return boolean correct.")
    return {
        "correct": result["correct"],
        "reason": str(result.get("reason") or ""),
        "judge_model": judge.model,
        "judge_type": "custom_llm_judge",
    }


def _frames_answer_extract(item: Any, report: dict[str, Any], judge: OpenAICompatibleChatModel):
    raw = judge.complete(
        "Extract the report's direct final answer to the FRAMES question. Do not solve the question "
        "again or add facts. Return the shortest answer supported by the report, preserving names, "
        "numbers and units. If the report does not provide an answer, return an empty string. "
        "Return JSON only with predicted_answer and reason.",
        json.dumps({
            "question": item.question,
            "report": report.get("answer", ""),
        }, ensure_ascii=False),
        temperature=0.0,
    )
    result = _parse_json(raw)
    answer = result.get("predicted_answer")
    if not isinstance(answer, str):
        raise TypeError("FRAMES answer extractor must return a text predicted_answer.")
    return {
        "predicted_answer": answer,
        "reason": str(result.get("reason") or ""),
        "extractor_model": judge.model,
        "extractor_type": "custom_llm_answer_extraction",
    }

def _prepare_resume_manifest(
    previous_manifest: dict[str, Any],
    current_manifest: dict[str, Any],
    completed_task_ids: set[str],
) -> dict[str, Any]:
    immutable_fields = (
        "dataset_id",
        "input_path",
        "input_sha256",
        "task_ids",
        "offset",
        "limit",
        "language_filter",
        "retrieval_k",
        "agent_model",
        "agent_base_url",
        "agent_max_queries",
        "agent_max_results_per_query",
        "agent_max_evidence_sources",
        "agent_max_research_rounds",
        "retrieval_mode",
        "judge_model",
        "judge_base_url",
        "judge_type",
        "search_provider",
        "corpus_revision",
        "corpus_index",
        "corpus_index_sha256",
    )
    mismatches = [
        field
        for field in immutable_fields
        if previous_manifest.get(field) != current_manifest.get(field)
    ]
    if mismatches:
        raise ValueError(
            "Cannot resume this run because its inputs/configuration changed: "
            + ", ".join(mismatches)
        )

    selected_ids = set(current_manifest.get("task_ids", []))
    unexpected_ids = completed_task_ids - selected_ids
    if unexpected_ids:
        raise ValueError(
            "Cannot resume: predictions contain task IDs outside the selected run: "
            + ", ".join(sorted(unexpected_ids)[:5])
        )

    current_manifest["started_at"] = previous_manifest.get(
        "started_at",
        current_manifest.get("started_at"),
    )
    old_history = previous_manifest.get("resume_history", [])
    history = list(old_history) if isinstance(old_history, list) else []
    history.append({
        "resumed_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_manifest.get("git_commit"),
        "previous_status": previous_manifest.get("status"),
    })
    current_manifest["resume_history"] = history
    return current_manifest


def run_experiment(
    dataset_id: str,
    input_path: str | Path,
    output_dir: str | Path,
    *,
    limit: int = 10,
    offset: int = 0,
    k: int = 10,
    include_noncommercial: bool = False,
    corpus_index: str | Path | None = None,
    resume: bool = False,
    language: str | None = None,
) -> dict[str, Any]:
    """Run a slice of an external dataset through the actual LangGraph research graph."""
    from research_agent.dependencies import build_default_dependencies
    from research_agent.graph import build_research_graph

    if limit < 1 or offset < 0 or k < 1:
        raise ValueError("limit and k must be positive; offset cannot be negative.")
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if dataset_id == "deepresearch-bench-ii":
        records = load_deepresearch_bench_ii_jsonl(
            input_path, include_noncommercial=include_noncommercial
        )
    elif dataset_id == "frames":
        records = load_frames_tsv(input_path)
    elif dataset_id == "browsecomp-plus":
        records = load_browsecomp_plus_jsonl(input_path)
    else:
        raise ValueError(f"Unknown benchmark: {dataset_id}")
    if language and dataset_id != "deepresearch-bench-ii":
        raise ValueError("--language is only supported for DeepResearch Bench II.")
    if language:
        requested_language = language.strip().lower()
        records = [
            record
            for record in records
            if str((record.metadata or {}).get("language", "")).lower() == requested_language
        ]
        if not records:
            raise ValueError(
                f"No records found for language {requested_language!r}; use en or zh for DRB II."
            )
    selected = records[offset:offset + limit]
    if not selected:
        raise ValueError("No records selected; check --offset, --limit and --language.")

    judge = None
    if dataset_id in {"deepresearch-bench-ii", "browsecomp-plus", "frames"}:
        base_url = os.getenv("EVAL_JUDGE_LLM_BASE_URL")
        model = os.getenv("EVAL_JUDGE_LLM_MODEL")
        if not base_url or not model:
            raise RuntimeError("Set EVAL_JUDGE_LLM_BASE_URL and EVAL_JUDGE_LLM_MODEL.")
        judge = OpenAICompatibleChatModel(
            base_url,
            model,
            api_key=os.getenv("EVAL_JUDGE_LLM_API_KEY"),
            timeout=float(os.getenv("EVAL_JUDGE_TIMEOUT_SECONDS", "120")),
        )

    provider = None
    provider_name = "tavily"
    fixed_corpus = None
    if dataset_id == "browsecomp-plus":
        if corpus_index is None:
            raise ValueError("BrowseComp-Plus requires --corpus-index built from the fixed corpus.")
        fixed_corpus = BrowseCompPlusCorpusSearchProvider(corpus_index)
        provider = fixed_corpus
        provider_name = "browsecomp-plus-fts5"

    dependencies = build_default_dependencies(
        search_provider=provider,
        search_provider_name=provider_name,
    )
    dependencies.max_question_chars = 20000
    dependencies.store = None
    graph = build_research_graph(dependencies)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "predictions.jsonl"
    already_done: set[str] = set()
    if results_path.exists():
        if not resume:
            raise FileExistsError(f"{results_path} exists; pass --resume to continue.")
        with results_path.open("r", encoding="utf-8") as old:
            for line in old:
                if line.strip():
                    old_row = json.loads(line)
                    if old_row.get("status") == "completed":
                        already_done.add(str(old_row["task_id"]))

    manifest = {
        "dataset_id": dataset_id,
        "input_path": str(input_path.resolve()),
        "input_sha256": _sha256(input_path),
        "upstream_input_manifest": None,
        "git_commit": _git_commit(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "task_ids": [record.task_id for record in selected],
        "offset": offset,
        "limit": limit,
        "language_filter": language,
        "retrieval_k": k,
        "agent_model": dependencies.llm_model,
        "agent_base_url": os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
        "agent_max_queries": dependencies.max_queries,
        "agent_max_results_per_query": dependencies.max_results_per_query,
        "agent_max_evidence_sources": dependencies.max_evidence_sources,
        "agent_max_research_rounds": dependencies.max_research_rounds,
        "retrieval_mode": dependencies.retrieval_mode,
        "judge_model": judge.model if judge else None,
        "judge_base_url": os.getenv("EVAL_JUDGE_LLM_BASE_URL"),
        "judge_type": "custom_llm_judge" if judge else None,
        "search_provider": provider_name,
        "corpus_revision": getattr(fixed_corpus, "revision", None),
        "corpus_index_sha256": _sha256(Path(corpus_index)) if corpus_index else None,
        "status": "running",
    }
    input_manifest_path = input_path.with_name(input_path.name + ".manifest.json")
    if input_manifest_path.is_file():
        manifest["upstream_input_manifest"] = json.loads(
            input_manifest_path.read_text(encoding="utf-8")
        )
    manifest_path = output_dir / "run_manifest.json"
    if resume and results_path.exists():
        if not manifest_path.is_file():
            raise ValueError("Cannot resume: run_manifest.json is missing.")
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = _prepare_resume_manifest(
            previous_manifest,
            manifest,
            already_done,
        )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    successful = failed = skipped = 0
    mode = "a" if resume else "w"
    with results_path.open(mode, encoding="utf-8") as output:
        for record in selected:
            if record.task_id in already_done:
                skipped += 1
                continue
            row = {
                "dataset_id": dataset_id,
                "task_id": record.task_id,
                "question": record.question,
                "license_id": record.license_id,
                "annotation_type": record.annotation_type,
                "metadata": record.metadata,
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            try:
                report = graph.invoke({"question": record.question})["report"]
                row["report"] = report
                urls = [str(source["url"]) for source in report.get("sources", [])]
                docids = [
                    str(source["docid"])
                    for source in report.get("sources", [])
                    if source.get("docid")
                ]
                row["retrieved_source_urls"] = urls
                row["retrieved_docids"] = docids
                if dataset_id == "frames":
                    extracted_answer = _frames_answer_extract(record, report, judge)
                    row["extracted_answer"] = extracted_answer
                    row["metrics"] = {
                        "answer": score_frames_answer(
                            extracted_answer["predicted_answer"],
                            str(record.reference_answer or ""),
                        ),
                        "retrieval": score_frames_retrieval(record, urls, k=k),
                    }
                elif dataset_id == "browsecomp-plus":
                    row["metrics"] = {
                        "retrieval": score_browsecomp_plus_retrieval(record, docids, k=k),
                        "answer_judgment": _browsecomp_answer_judge(record, report, judge),
                    }
                else:
                    rubric_metrics, details = _drb2_judge(record, report, judge)
                    row["metrics"] = {"report_rubric": rubric_metrics}
                    row["rubric_judgments"] = details
                row["status"] = "completed"
                successful += 1
            except Exception as exc:  # noqa: BLE001 - record each task failure and continue the batch.
                row["status"] = "failed"
                row["error"] = f"{type(exc).__name__}: {exc}"
                failed += 1
            row["finished_at"] = datetime.now(timezone.utc).isoformat()
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()

    from research_observability.tracing import flush_telemetry

    flush_telemetry()
    summary_path = output_dir / "summary.json"
    summary = write_run_summary(results_path, summary_path)
    manifest.update({
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed_with_failures" if failed else "completed",
        "completed_this_run": successful,
        "failed_this_run": failed,
        "skipped_already_completed": skipped,
        "completed_total": summary["completed_count"],
        "failed_total": summary["failed_count"],
        "predictions_path": str(results_path),
        "summary_path": str(summary_path),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
