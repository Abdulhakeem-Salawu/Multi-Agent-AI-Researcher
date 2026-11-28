from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _mean(values: list[float]) -> float | None:
    return round(mean(values), 6) if values else None


def _average_metric_rows(rows: list[dict[str, Any]], metric_path: tuple[str, ...]) -> dict[str, float | int | None]:
    values_by_name: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value: Any = row.get("metrics", {})
        for part in metric_path:
            value = value.get(part, {}) if isinstance(value, dict) else {}
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, (int, float)) and not isinstance(item, bool):
                    values_by_name[key].append(float(item))
    return {key: _mean(values) for key, values in sorted(values_by_name.items())}


def summarize_predictions(predictions_path: str | Path) -> dict[str, Any]:
    """Summarise the latest outcome per task so retries do not double-count."""
    path = Path(predictions_path)
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed predictions JSONL at line {line_number}: {exc}") from exc
            if not isinstance(row, dict) or not row.get("dataset_id") or not row.get("task_id"):
                raise ValueError(f"Predictions line {line_number} lacks dataset_id/task_id.")
            key = (str(row["dataset_id"]), str(row["task_id"]))
            latest[key] = row

    if not latest:
        raise ValueError("No task predictions are available to summarise.")
    dataset_ids = {key[0] for key in latest}
    if len(dataset_ids) != 1:
        raise ValueError("A single run directory cannot mix different dataset IDs.")
    dataset_id = next(iter(dataset_ids))
    rows = list(latest.values())
    completed = [row for row in rows if row.get("status") == "completed"]
    failed = [row for row in rows if row.get("status") != "completed"]
    summary: dict[str, Any] = {
        "dataset_id": dataset_id,
        "task_count": len(rows),
        "completed_count": len(completed),
        "failed_count": len(failed),
        "success_rate": round(len(completed) / len(rows), 6) if rows else 0.0,
        "failed_task_ids": [str(row["task_id"]) for row in failed],
    }

    if dataset_id == "frames":
        summary["answer_metrics"] = {
            "exact_match": _mean([
                float(row["metrics"]["answer"]["exact_match"])
                for row in completed
                if "answer" in row.get("metrics", {})
            ]),
            "token_f1": _mean([
                float(row["metrics"]["answer"]["token_f1"])
                for row in completed
                if "answer" in row.get("metrics", {})
            ]),
        }
        summary["retrieval_metrics"] = _average_metric_rows(completed, ("retrieval",))
    elif dataset_id == "browsecomp-plus":
        summary["evidence_retrieval_metrics"] = _average_metric_rows(
            completed, ("retrieval", "evidence")
        )
        summary["gold_retrieval_metrics"] = _average_metric_rows(
            completed, ("retrieval", "gold")
        )
        answer_verdicts = [
            bool(row["metrics"]["answer_judgment"]["correct"])
            for row in completed
            if isinstance(row.get("metrics", {}).get("answer_judgment"), dict)
        ]
        summary["answer_judgment"] = {
            "judged_count": len(answer_verdicts),
            "correct_count": sum(answer_verdicts),
            "accuracy": _mean([float(value) for value in answer_verdicts]),
            "judge_type": "custom_llm_judge",
        }
    elif dataset_id == "deepresearch-bench-ii":
        dimensions: dict[str, dict[str, int]] = {}
        totals = {
            "rubric_count": 0,
            "satisfied_count": 0,
            "not_mentioned_count": 0,
            "blocked_reference_count": 0,
        }
        for row in completed:
            report_rubric = row.get("metrics", {}).get("report_rubric", {})
            for key in totals:
                totals[key] += int(report_rubric.get(key, 0))
            for name, values in report_rubric.get("dimensions", {}).items():
                target = dimensions.setdefault(name, {
                    "rubric_count": 0,
                    "satisfied_count": 0,
                    "not_mentioned_count": 0,
                    "blocked_reference_count": 0,
                })
                for key in target:
                    target[key] += int(values.get(key, 0))
        denominator = totals["rubric_count"]
        totals["satisfied_rate"] = (
            round(totals["satisfied_count"] / denominator, 6) if denominator else None
        )
        totals["blocked_reference_rate"] = (
            round(totals["blocked_reference_count"] / denominator, 6) if denominator else None
        )
        for values in dimensions.values():
            count = values["rubric_count"]
            values["satisfied_rate"] = (
                round(values["satisfied_count"] / count, 6) if count else None
            )
            values["blocked_reference_rate"] = (
                round(values["blocked_reference_count"] / count, 6) if count else None
            )
        summary["rubric_metrics"] = totals
        summary["rubric_metrics_by_dimension"] = dimensions
        summary["judge_type"] = "custom_llm_judge"

    return summary


def write_run_summary(predictions_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    summary = summarize_predictions(predictions_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary
