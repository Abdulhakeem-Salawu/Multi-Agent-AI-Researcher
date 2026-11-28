from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from evaluation.external.adapters import (
    ExternalDatasetError,
    load_browsecomp_plus_jsonl,
    load_deepresearch_bench_ii_jsonl,
    load_frames_tsv,
)
from evaluation.external.models import ExternalBenchmarkItem
from evaluation.external.scoring import (
    score_browsecomp_plus_retrieval,
    score_drb2_rubric_judgments,
    score_frames_answer,
    score_frames_retrieval,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_drb2_loader_preserves_rubric_dimensions_and_excludes_noncommercial_by_default(
    tmp_path: Path,
):
    data = [
        {
            "id": "task-a",
            "idx": 1,
            "language": "en",
            "theme": "Engineering",
            "description": "A production systems research task",
            "prompt": "Write a report about durable workflow execution.",
            "rubric": {
                "info_recall": ["Explains durable state restoration accurately."],
                "analysis": ["Compares checkpoint trade-offs with concrete reasoning."],
                "presentation": ["Uses clear, traceable section headings."],
            },
            "blocked": {"title": "Blocked source", "urls": ["https://blocked.example/source"]},
            "license": "CC BY 4.0",
        },
        {
            "id": "task-b",
            "idx": 26,
            "prompt": "A separate task that is non-commercial.",
            "rubric": {"analysis": ["Explains the relevant trade-off in sufficient detail."]},
            "license": "CC BY-NC 4.0",
        },
    ]
    path = _write_jsonl(tmp_path / "drb2.jsonl", data)
    items = load_deepresearch_bench_ii_jsonl(path)

    assert len(items) == 1
    item = items[0]
    assert isinstance(item, ExternalBenchmarkItem)
    assert item.task_id == "1"
    assert item.license_id == "CC BY 4.0"
    assert {rubric["dimension"] for rubric in item.annotations["rubrics"]} == {
        "info_recall",
        "analysis",
        "presentation",
    }
    assert item.annotations["blocked_source_urls"] == ["https://blocked.example/source"]
    assert item.annotation_type == "expert_refined_rubric_scored_by_llm_judge"


def test_drb2_can_include_noncommercial_records_only_when_explicit(tmp_path: Path):
    row = {
        "id": "task-nc",
        "idx": 26,
        "prompt": "Explain a non-commercial research topic.",
        "rubric": {"analysis": ["Explains the core argument with a valid chain of reasoning."]},
        "license": "CC BY-NC 4.0",
    }
    path = _write_jsonl(tmp_path / "drb2-nc.jsonl", [row])
    assert load_deepresearch_bench_ii_jsonl(path) == []
    items = load_deepresearch_bench_ii_jsonl(path, include_noncommercial=True)
    assert len(items) == 1
    assert items[0].license_id == "CC BY-NC 4.0"


def test_drb2_rejects_missing_license_and_duplicate_ids(tmp_path: Path):
    missing_license = {
        "idx": 1,
        "prompt": "A task with no explicit data licence.",
        "rubric": {"analysis": ["This rubric has enough descriptive text."]},
    }
    path = _write_jsonl(tmp_path / "missing-license.jsonl", [missing_license])
    with pytest.raises(ExternalDatasetError, match="per-task license"):
        load_deepresearch_bench_ii_jsonl(path)

    repeated = {
        "idx": 2,
        "prompt": "A task with a complete licence record.",
        "rubric": {"analysis": ["This rubric has enough descriptive text."]},
        "license": "CC0",
    }
    path = _write_jsonl(tmp_path / "duplicate.jsonl", [repeated, repeated])
    with pytest.raises(ExternalDatasetError, match="duplicate task id"):
        load_deepresearch_bench_ii_jsonl(path)


def test_drb2_scorer_requires_full_rubric_coverage_and_preserves_three_labels(
    tmp_path: Path,
):
    row = {
        "idx": 7,
        "prompt": "Evaluate a research report about observability design.",
        "rubric": {
            "info_recall": ["Identifies the relevant metrics and tracing components."],
            "analysis": ["Explains a credible failure mode and its mitigation."],
            "presentation": ["Presents the answer in a traceable structure."],
        },
        "license": "CC BY 4.0",
    }
    item = load_deepresearch_bench_ii_jsonl(_write_jsonl(tmp_path / "drb.jsonl", [row]))[0]
    rubrics = item.annotations["rubrics"]
    judgments = {
        rubrics[0]["rubric_id"]: 1,
        rubrics[1]["rubric_id"]: 0,
        rubrics[2]["rubric_id"]: -1,
    }
    result = score_drb2_rubric_judgments(item, judgments)
    assert result["satisfied_count"] == 1
    assert result["not_mentioned_count"] == 1
    assert result["blocked_reference_count"] == 1
    assert result["dimensions"]["analysis"]["not_mentioned_count"] == 1
    with pytest.raises(ValueError, match="coverage mismatch"):
        score_drb2_rubric_judgments(item, {rubrics[0]["rubric_id"]: 1})


def _write_frames_tsv(path: Path) -> Path:
    fields = [
        "",
        "Prompt",
        "Answer",
        "wikipedia_link_1",
        "wikipedia_link_2",
        "reasoning_types",
        "wiki_links",
    ]
    rows = [
        {
            "": "0",
            "Prompt": "How many years are there between event A and event B?",
            "Answer": "42",
            "wikipedia_link_1": "https://en.wikipedia.org/wiki/Event_A",
            "wikipedia_link_2": "https://en.wikipedia.org/wiki/Event_B",
            "reasoning_types": "Numerical reasoning | Temporal reasoning",
            "wiki_links": (
                "['https://en.wikipedia.org/wiki/Event_A', "
                "'https://en.wikipedia.org/wiki/Event_B']"
            ),
        },
        {
            "": "1",
            "Prompt": "Which person is linked to both records?",
            "Answer": "Ada Lovelace",
            "wikipedia_link_1": "https://en.wikipedia.org/wiki/Person_A",
            "wikipedia_link_2": "",
            "reasoning_types": "Multiple constraints",
            "wiki_links": "['https://en.wikipedia.org/wiki/Person_A']",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_frames_loader_preserves_row_id_answers_reasoning_types_and_article_labels(
    tmp_path: Path,
):
    items = load_frames_tsv(_write_frames_tsv(tmp_path / "frames.tsv"))
    assert [item.task_id for item in items] == ["0", "1"]
    assert items[0].reference_answer == "42"
    assert items[0].annotations["reasoning_types"] == [
        "Numerical reasoning",
        "Temporal reasoning",
    ]
    assert items[0].annotations["relevant_article_urls"] == [
        "https://en.wikipedia.org/wiki/Event_A",
        "https://en.wikipedia.org/wiki/Event_B",
    ]
    assert items[0].license_id == "Apache-2.0"


def test_frames_answer_and_retrieval_metrics_are_deterministic(tmp_path: Path):
    items = load_frames_tsv(_write_frames_tsv(tmp_path / "frames.tsv"))
    answer = score_frames_answer("The answer is 42!", items[0].reference_answer)
    retrieval = score_frames_retrieval(
        items[0],
        [
            "https://en.wikipedia.org/wiki/Unrelated",
            "https://en.wikipedia.org/wiki/Event_A",
            "https://en.wikipedia.org/wiki/Event_B",
        ],
        k=3,
    )
    assert answer == {"exact_match": 1.0, "token_f1": 1.0}
    assert retrieval["recall_at_k"] == 1.0
    assert retrieval["mrr_at_k"] == 0.5


def test_browsecomp_plus_loader_keeps_document_judgements_but_discards_text(
    tmp_path: Path,
):
    path = _write_jsonl(
        tmp_path / "browsecomp.jsonl",
        [
            {
                "query_id": "q-001",
                "query": "Which system published the first stable release in 2024?",
                "answer": "Example System",
                "evidence_docs": [
                    {
                        "docid": "e-1",
                        "text": "PRIVATE DOCUMENT BODY MUST NOT BE RETAINED",
                        "url": "https://example.org/evidence",
                    }
                ],
                "gold_docs": [
                    {
                        "docid": "g-1",
                        "text": "PRIVATE GOLD DOCUMENT BODY MUST NOT BE RETAINED",
                        "url": "https://example.org/gold",
                    }
                ],
                "negative_docs": [
                    {
                        "docid": "n-1",
                        "text": "PRIVATE NEGATIVE DOCUMENT BODY MUST NOT BE RETAINED",
                        "url": "https://example.org/negative",
                    }
                ],
            }
        ],
    )
    item = load_browsecomp_plus_jsonl(path)[0]
    assert item.task_id == "q-001"
    assert item.annotations["evidence_docids"] == ["e-1"]
    assert item.annotations["gold_docids"] == ["g-1"]
    assert item.annotations["negative_docids"] == ["n-1"]
    assert item.annotations["evidence_urls"] == ["https://example.org/evidence"]
    assert item.metadata["document_text_loaded"] is False
    serialized = json.dumps(item.to_dict())
    assert "PRIVATE DOCUMENT BODY" not in serialized


def test_browsecomp_plus_scores_evidence_and_gold_qrels_separately(tmp_path: Path):
    row = {
        "query_id": "q-002",
        "query": "Find the source for the technical claim.",
        "answer": "A verified answer",
        "evidence_docs": [
            {"docid": "e-1", "text": "not retained", "url": "https://example.org/e1"},
            {"docid": "e-2", "text": "not retained", "url": "https://example.org/e2"},
        ],
        "gold_docs": [{"docid": "g-1", "text": "not retained", "url": "https://example.org/g1"}],
        "negative_docs": [],
    }
    item = load_browsecomp_plus_jsonl(_write_jsonl(tmp_path / "browse.jsonl", [row]))[0]
    result = score_browsecomp_plus_retrieval(item, ["n-1", "e-1", "g-1"], k=3)
    assert result["evidence"]["recall_at_k"] == 0.5
    assert result["gold"]["recall_at_k"] == 1.0
    assert result["evidence"]["mrr_at_k"] == 0.5


def test_browsecomp_plus_rejects_undecoded_or_incomplete_records(tmp_path: Path):
    path = _write_jsonl(
        tmp_path / "bad-browse.jsonl",
        [{"query_id": "q-003", "query": "Question text", "answer": "Answer"}],
    )
    with pytest.raises(ExternalDatasetError, match="evidence_docs must be a list"):
        load_browsecomp_plus_jsonl(path)
