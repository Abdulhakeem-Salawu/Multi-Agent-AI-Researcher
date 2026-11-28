from __future__ import annotations

import json

import pytest

from evaluation.external.models import ExternalBenchmarkItem
from evaluation.external.runner import (
    _browsecomp_answer_judge,
    _drb2_judge,
    _frames_answer_extract,
    _prepare_resume_manifest,
)


class FakeJudge:
    model = "offline-test-judge"

    def __init__(self, callback):
        self.callback = callback

    def complete(self, system_prompt, user_prompt, temperature=0.0):
        return self.callback(system_prompt, json.loads(user_prompt))


def test_frames_answer_extraction_returns_separate_concise_answer():
    def extract_without_gold(system, payload):
        assert "reference_answer" not in payload
        assert payload["question"] == item.question
        return json.dumps({
            "predicted_answer": "37th",
            "reason": "The final section states the rank directly.",
        })

    judge = FakeJudge(extract_without_gold)
    item = ExternalBenchmarkItem(
        dataset_id="frames",
        track="multi_hop_qa",
        task_id="frames-1",
        question="What rank does the building have?",
        reference_answer="37th",
        annotation_type="gold_answer_and_relevant_articles",
    )
    result = _frames_answer_extract(
        item,
        {"answer": "The report concludes that the building ranks 37th."},
        judge,
        judge_type="same_model_llm_judge",
    )
    assert result["predicted_answer"] == "37th"
    assert result["extractor_model"] == "offline-test-judge"
    assert result["extractor_type"] == "same_model_answer_extraction"
    assert result["judge_type"] == "same_model_llm_judge"


def test_drb2_judge_requires_complete_rubric_ids_and_exact_report_evidence():
    item = ExternalBenchmarkItem(
        dataset_id="deepresearch-bench-ii",
        track="report_rubric",
        task_id="task-1",
        question="Assess this research report.",
        annotation_type="expert_refined_rubric_scored_by_llm_judge",
        annotations={
            "rubrics": [
                {"rubric_id": "1:info_recall:1", "dimension": "info_recall", "text": "Reports the main point."},
                {"rubric_id": "1:analysis:1", "dimension": "analysis", "text": "Explains the causal mechanism."},
                {"rubric_id": "1:presentation:1", "dimension": "presentation", "text": "Uses clear headings."},
            ],
            "blocked_source_urls": ["https://blocked.example/source"],
        },
    )
    report_text = "The system uses a durable checkpoint store."
    def callback(system, payload):
        scores = [1, 0, -1]
        return json.dumps({
            "results": [
                {
                    "rubric_id": row["rubric_id"],
                    "score": scores[index],
                    "reason": "Reviewed.",
                    "evidence": report_text if scores[index] != 0 else "",
                }
                for index, row in enumerate(payload["rubrics"])
            ]
        })
    metrics, details = _drb2_judge(item, {"answer": report_text}, FakeJudge(callback))
    assert metrics["satisfied_count"] == 1
    assert metrics["not_mentioned_count"] == 1
    assert metrics["blocked_reference_count"] == 1
    assert len(details) == 3


def test_browsecomp_answer_judge_requires_boolean_verdict():
    item = ExternalBenchmarkItem(
        dataset_id="browsecomp-plus",
        track="fixed_corpus_retrieval_and_research",
        task_id="q1",
        question="Who created the system?",
        reference_answer="Ada Lovelace",
        annotation_type="human_verified_document_labels",
    )
    def compare_with_reference(system, payload):
        assert payload["reference_answer"] == "Ada Lovelace"
        return json.dumps({
            "correct": True,
            "reason": "The report identifies the same person.",
        })

    judge = FakeJudge(compare_with_reference)
    result = _browsecomp_answer_judge(
        item,
        {"answer": "Ada Lovelace created it."},
        judge,
        judge_type="same_model_llm_judge",
    )
    assert result["correct"] is True
    assert result["judge_type"] == "same_model_llm_judge"



def _resume_manifest() -> dict:
    return {
        "dataset_id": "frames",
        "input_path": "/data/frames.tsv",
        "input_sha256": "a" * 64,
        "task_ids": ["f-1", "f-2"],
        "offset": 0,
        "limit": 2,
        "language_filter": None,
        "retrieval_k": 10,
        "agent_model": "agent-v1",
        "agent_base_url": "http://agent.local/v1",
        "agent_max_queries": 4,
        "agent_max_results_per_query": 8,
        "agent_max_evidence_sources": 8,
        "agent_max_research_rounds": 1,
        "retrieval_mode": "hybrid",
        "judge_model": "judge-v1",
        "judge_base_url": "http://judge.local/v1",
        "judge_type": "custom_llm_judge",
        "search_provider": "tavily",
        "corpus_revision": None,
        "corpus_index": None,
        "corpus_index_sha256": None,
        "started_at": "2026-11-28T09:00:00+00:00",
        "status": "completed_with_failures",
        "git_commit": "old-commit",
    }


def test_resume_manifest_preserves_original_start_and_records_code_revision():
    previous = _resume_manifest()
    current = {
        **_resume_manifest(),
        "started_at": "2026-11-28T10:00:00+00:00",
        "git_commit": "new-commit",
        "status": "running",
    }
    resumed = _prepare_resume_manifest(previous, current, {"f-1"})

    assert resumed["started_at"] == previous["started_at"]
    assert resumed["resume_history"][-1]["git_commit"] == "new-commit"
    assert resumed["resume_history"][-1]["previous_status"] == "completed_with_failures"


def test_resume_manifest_rejects_changed_inputs_or_task_selection():
    previous = _resume_manifest()
    changed = {**_resume_manifest(), "input_sha256": "b" * 64}
    with pytest.raises(ValueError, match="input_sha256"):
        _prepare_resume_manifest(previous, changed, set())

    changed_model_config = {**_resume_manifest(), "agent_max_queries": 5}
    with pytest.raises(ValueError, match="agent_max_queries"):
        _prepare_resume_manifest(previous, changed_model_config, set())

    with pytest.raises(ValueError, match="outside the selected run"):
        _prepare_resume_manifest(previous, _resume_manifest(), {"f-3"})
