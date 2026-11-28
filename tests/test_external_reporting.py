from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.external.reporting import summarize_predictions, write_run_summary


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_frames_summary_aggregates_metrics_and_counts_latest_retry(tmp_path: Path):
    predictions = _write(
        tmp_path / "predictions.jsonl",
        [
            {
                "dataset_id": "frames",
                "task_id": "f-1",
                "status": "failed",
                "error": "temporary provider issue",
            },
            {
                "dataset_id": "frames",
                "task_id": "f-1",
                "status": "completed",
                "metrics": {
                    "answer": {"exact_match": 1.0, "token_f1": 0.8},
                    "retrieval": {
                        "precision_at_k": 0.5,
                        "recall_at_k": 1.0,
                        "mrr_at_k": 0.5,
                        "ndcg_at_k": 0.75,
                    },
                },
            },
            {
                "dataset_id": "frames",
                "task_id": "f-2",
                "status": "completed",
                "metrics": {
                    "answer": {"exact_match": 0.0, "token_f1": 0.4},
                    "retrieval": {
                        "precision_at_k": 0.25,
                        "recall_at_k": 0.5,
                        "mrr_at_k": 0.25,
                        "ndcg_at_k": 0.5,
                    },
                },
            },
        ],
    )

    summary = summarize_predictions(predictions)
    assert summary["task_count"] == 2
    assert summary["completed_count"] == 2
    assert summary["failed_count"] == 0
    assert summary["answer_metrics"]["exact_match"] == 0.5
    assert summary["answer_metrics"]["token_f1"] == 0.6
    assert summary["retrieval_metrics"]["recall_at_k"] == 0.75

    output = tmp_path / "summary.json"
    written = write_run_summary(predictions, output)
    assert json.loads(output.read_text(encoding="utf-8")) == written


def test_drb2_summary_aggregates_native_rubric_categories(tmp_path: Path):
    predictions = _write(
        tmp_path / "drb.jsonl",
        [
            {
                "dataset_id": "deepresearch-bench-ii",
                "task_id": "d-1",
                "status": "completed",
                "metrics": {
                    "report_rubric": {
                        "rubric_count": 3,
                        "satisfied_count": 1,
                        "not_mentioned_count": 1,
                        "blocked_reference_count": 1,
                        "dimensions": {
                            "analysis": {
                                "rubric_count": 2,
                                "satisfied_count": 1,
                                "not_mentioned_count": 0,
                                "blocked_reference_count": 1,
                            },
                            "presentation": {
                                "rubric_count": 1,
                                "satisfied_count": 0,
                                "not_mentioned_count": 1,
                                "blocked_reference_count": 0,
                            },
                        },
                    }
                },
            }
        ],
    )
    summary = summarize_predictions(predictions)
    assert summary["rubric_metrics"]["satisfied_rate"] == pytest.approx(1 / 3)
    assert summary["rubric_metrics"]["blocked_reference_rate"] == pytest.approx(1 / 3)
    assert summary["rubric_metrics_by_dimension"]["analysis"]["satisfied_rate"] == 0.5


def test_browsecomp_summary_keeps_evidence_and_gold_metrics_separate(tmp_path: Path):
    predictions = _write(
        tmp_path / "browse.jsonl",
        [
            {
                "dataset_id": "browsecomp-plus",
                "task_id": "q-1",
                "status": "completed",
                "metrics": {
                    "retrieval": {
                        "evidence": {"recall_at_k": 0.5, "ndcg_at_k": 0.4},
                        "gold": {"recall_at_k": 1.0, "ndcg_at_k": 0.9},
                    },
                    "answer_judgment": {"correct": True, "judge_type": "same_model_llm_judge"},
                },
            }
        ],
    )
    summary = summarize_predictions(predictions)
    assert summary["evidence_retrieval_metrics"]["recall_at_k"] == 0.5
    assert summary["gold_retrieval_metrics"]["recall_at_k"] == 1.0
    assert summary["answer_judgment"]["accuracy"] == 1.0
    assert summary["answer_judgment"]["judge_type"] == "same_model_llm_judge"


def test_summary_rejects_mixed_datasets_and_malformed_jsonl(tmp_path: Path):
    mixed = _write(
        tmp_path / "mixed.jsonl",
        [
            {"dataset_id": "frames", "task_id": "1", "status": "completed"},
            {"dataset_id": "browsecomp-plus", "task_id": "1", "status": "completed"},
        ],
    )
    with pytest.raises(ValueError, match="mix different dataset IDs"):
        summarize_predictions(mixed)

    malformed = tmp_path / "bad.jsonl"
    malformed.write_text("{not-json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed predictions"):
        summarize_predictions(malformed)



def test_frames_summary_identifies_same_model_answer_extraction(tmp_path: Path):
    predictions = _write(
        tmp_path / "frames-same-model.jsonl",
        [
            {
                "dataset_id": "frames",
                "task_id": "f-same",
                "status": "completed",
                "extracted_answer": {
                    "extractor_model": "Qwen/Qwen3-1.7B",
                    "extractor_type": "same_model_answer_extraction",
                    "judge_type": "same_model_llm_judge",
                },
                "metrics": {
                    "answer": {"exact_match": 1.0, "token_f1": 1.0},
                    "retrieval": {"recall_at_k": 0.5},
                },
            }
        ],
    )

    summary = summarize_predictions(predictions)

    assert summary["answer_extraction"]["model"] == "Qwen/Qwen3-1.7B"
    assert summary["answer_extraction"]["judge_type"] == "same_model_llm_judge"


def test_drb2_summary_discloses_same_model_judge(tmp_path: Path):
    predictions = _write(
        tmp_path / "drb-same-model.jsonl",
        [
            {
                "dataset_id": "deepresearch-bench-ii",
                "task_id": "d-same",
                "status": "completed",
                "judge": {
                    "model": "Qwen/Qwen3-1.7B",
                    "type": "same_model_llm_judge",
                },
                "metrics": {
                    "report_rubric": {
                        "rubric_count": 1,
                        "satisfied_count": 1,
                        "not_mentioned_count": 0,
                        "blocked_reference_count": 0,
                        "dimensions": {},
                    }
                },
            }
        ],
    )

    summary = summarize_predictions(predictions)

    assert summary["judge_type"] == "same_model_llm_judge"
