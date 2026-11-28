from __future__ import annotations

import ast
import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .models import ExternalBenchmarkItem


class ExternalDatasetError(ValueError):
    """Raised when an upstream file does not match its documented schema."""


def _jsonl_records(path: str | Path) -> Iterable[tuple[int, dict[str, Any]]]:
    source = Path(path)
    try:
        handle = source.open("r", encoding="utf-8")
    except OSError as exc:
        raise ExternalDatasetError(f"Cannot open external dataset file {source}: {exc}") from exc

    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ExternalDatasetError(
                    f"Invalid JSON on line {line_number} of {source}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ExternalDatasetError(
                    f"Expected a JSON object on line {line_number} of {source}."
                )
            yield line_number, record


def _required_text(value: Any, field: str, *, row: int | str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExternalDatasetError(f"Record {row}: required field {field!r} is empty or not text.")
    return value.strip()


def _as_document_records(value: Any, field: str, *, row: str) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ExternalDatasetError(f"BrowseComp-Plus record {row}: {field} must be a list.")
    documents: list[dict[str, str]] = []
    for position, item in enumerate(value):
        if not isinstance(item, dict):
            raise ExternalDatasetError(
                f"BrowseComp-Plus record {row}: {field}[{position}] must be an object."
            )
        docid = _required_text(item.get("docid"), f"{field}[{position}].docid", row=row)
        url_value = item.get("url")
        url = url_value.strip() if isinstance(url_value, str) else ""
        documents.append({"docid": docid, "url": url})
    return documents


def load_deepresearch_bench_ii_jsonl(
    path: str | Path,
    *,
    include_noncommercial: bool = False,
    permitted_licenses: tuple[str, ...] = ("CC BY 4.0", "CC0"),
) -> list[ExternalBenchmarkItem]:
    """Load DRB II tasks; omit CC BY-NC tasks unless explicitly requested."""
    allowed = set(permitted_licenses)
    if include_noncommercial:
        allowed.add("CC BY-NC 4.0")

    items: list[ExternalBenchmarkItem] = []
    seen_ids: set[str] = set()
    for line_number, record in _jsonl_records(path):
        raw_id = record.get("idx", record.get("id"))
        if raw_id is None:
            raise ExternalDatasetError(f"DRB II line {line_number}: missing idx/id.")
        task_id = str(raw_id)
        if task_id in seen_ids:
            raise ExternalDatasetError(f"DRB II contains duplicate task id {task_id!r}.")
        seen_ids.add(task_id)

        license_id = record.get("license")
        if not isinstance(license_id, str) or not license_id.strip():
            raise ExternalDatasetError(
                f"DRB II task {task_id}: per-task license metadata is required."
            )
        license_id = license_id.strip()
        if license_id not in {"CC BY 4.0", "CC BY-NC 4.0", "CC0"}:
            raise ExternalDatasetError(
                f"DRB II task {task_id}: unrecognised license {license_id!r}; review upstream metadata."
            )
        if license_id not in allowed:
            continue

        content = record.get("content")
        content = content if isinstance(content, dict) else {}
        prompt = record.get("prompt") or content.get("task")
        prompt = _required_text(prompt, "prompt/content.task", row=task_id)

        rubric_groups = record.get("rubric") or content.get("rubric")
        if not isinstance(rubric_groups, dict) or not rubric_groups:
            raise ExternalDatasetError(f"DRB II task {task_id}: missing rubric groups.")

        rubrics: list[dict[str, str]] = []
        for dimension, values in rubric_groups.items():
            if not isinstance(dimension, str) or not isinstance(values, list):
                raise ExternalDatasetError(
                    f"DRB II task {task_id}: rubric dimensions must map to lists of strings."
                )
            for rubric_index, rubric_text in enumerate(values, start=1):
                text_value = _required_text(
                    rubric_text,
                    f"rubric.{dimension}[{rubric_index - 1}]",
                    row=task_id,
                )
                rubrics.append(
                    {
                        "rubric_id": f"{task_id}:{dimension}:{rubric_index}",
                        "dimension": dimension,
                        "text": text_value,
                    }
                )
        if not rubrics:
            raise ExternalDatasetError(f"DRB II task {task_id}: rubric groups are empty.")

        blocked = record.get("blocked", content.get("blocked", {}))
        blocked = blocked if isinstance(blocked, dict) else {}
        blocked_urls = blocked.get("urls", [])
        if not isinstance(blocked_urls, list):
            blocked_urls = []

        items.append(
            ExternalBenchmarkItem(
                dataset_id="deepresearch-bench-ii",
                track="report_rubric",
                task_id=task_id,
                question=prompt,
                annotation_type="expert_refined_rubric_scored_by_llm_judge",
                annotations={
                    "rubrics": rubrics,
                    "score_labels": {
                        "1": "rubric satisfied with valid report evidence",
                        "0": "rubric not mentioned",
                        "-1": "rubric mentioned but relies on a blocked reference",
                    },
                    "blocked_source_urls": [
                        value.strip() for value in blocked_urls if isinstance(value, str)
                    ],
                    "blocked_reference": {
                        "title": str(blocked.get("title") or ""),
                        "authors": (
                            [str(author) for author in blocked.get("authors", [])]
                            if isinstance(blocked.get("authors", []), list)
                            else []
                        ),
                        "urls": [
                            value.strip() for value in blocked_urls if isinstance(value, str)
                        ],
                    },
                },
                metadata={
                    "upstream_id": record.get("id"),
                    "theme": record.get("theme"),
                    "language": record.get("language"),
                    "description": record.get("description"),
                    "rubric_count": len(rubrics),
                },
                license_id=license_id,
            )
        )
    return items


def _parse_list_literal(value: Any) -> list[str]:
    if isinstance(value, list):
        return [entry.strip() for entry in value if isinstance(entry, str) and entry.strip()]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, (list, tuple)):
        return []
    return [entry.strip() for entry in parsed if isinstance(entry, str) and entry.strip()]


def _frames_article_urls(row: dict[str, Any]) -> list[str]:
    urls = _parse_list_literal(row.get("wiki_links"))
    if not urls:
        urls = [
            value.strip()
            for key, value in row.items()
            if key.lower().startswith("wikipedia_link_")
            and isinstance(value, str)
            and value.strip().startswith(("http://", "https://"))
        ]
    return list(dict.fromkeys(urls))


def load_frames_tsv(path: str | Path) -> list[ExternalBenchmarkItem]:
    """Load FRAMES questions and article URL labels, not Wikipedia article bodies."""
    source = Path(path)
    try:
        handle = source.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise ExternalDatasetError(f"Cannot open FRAMES TSV {source}: {exc}") from exc

    items: list[ExternalBenchmarkItem] = []
    seen_ids: set[str] = set()
    with handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ExternalDatasetError("FRAMES TSV is empty or has no header.")
        normalized_headers = {field.strip().lower() for field in reader.fieldnames if field}
        if not {"prompt", "answer"}.issubset(normalized_headers):
            raise ExternalDatasetError("FRAMES TSV must contain Prompt and Answer columns.")

        for row_number, row in enumerate(reader, start=1):
            question = _required_text(row.get("Prompt"), "Prompt", row=row_number)
            answer = _required_text(row.get("Answer"), "Answer", row=row_number)
            raw_index = next(
                (
                    value
                    for key, value in row.items()
                    if key in {"", "index", "Unnamed: 0"} and value is not None
                ),
                None,
            )
            task_id = str(raw_index).strip() if raw_index is not None else str(row_number - 1)
            if task_id in seen_ids:
                raise ExternalDatasetError(f"FRAMES contains duplicate row id {task_id!r}.")
            seen_ids.add(task_id)

            article_urls = _frames_article_urls(row)
            raw_reasoning_types = row.get("reasoning_types", "") or ""
            parsed_reasoning_types = _parse_list_literal(raw_reasoning_types)
            if parsed_reasoning_types:
                reasoning_types = parsed_reasoning_types
            else:
                reasoning_text = str(raw_reasoning_types).strip()
                reasoning_types = [
                    part.strip()
                    for part in reasoning_text.split("|")
                    if part.strip()
                ]
            if not article_urls:
                raise ExternalDatasetError(
                    f"FRAMES row {task_id}: no relevant Wikipedia article URLs were found."
                )

            items.append(
                ExternalBenchmarkItem(
                    dataset_id="frames",
                    track="multi_hop_qa",
                    task_id=task_id,
                    question=question,
                    reference_answer=answer,
                    annotation_type="dataset_provided_gold_answer_and_relevant_articles",
                    annotations={
                        "relevant_article_urls": article_urls,
                        "reasoning_types": reasoning_types,
                    },
                    metadata={
                        "upstream_row_id": raw_index,
                        "relevant_article_count": len(article_urls),
                    },
                    license_id="Apache-2.0",
                )
            )
    return items


def load_browsecomp_plus_jsonl(path: str | Path) -> list[ExternalBenchmarkItem]:
    """Load de-obfuscated labels; deliberately discard all document text."""
    items: list[ExternalBenchmarkItem] = []
    seen_ids: set[str] = set()
    for line_number, record in _jsonl_records(path):
        task_id = _required_text(record.get("query_id"), "query_id", row=line_number)
        if task_id in seen_ids:
            raise ExternalDatasetError(
                f"BrowseComp-Plus contains duplicate query_id {task_id!r}."
            )
        seen_ids.add(task_id)
        question = _required_text(record.get("query"), "query", row=task_id)
        answer = _required_text(record.get("answer"), "answer", row=task_id)
        evidence_docs = _as_document_records(
            record.get("evidence_docs"), "evidence_docs", row=task_id
        )
        gold_docs = _as_document_records(record.get("gold_docs"), "gold_docs", row=task_id)
        negative_value = record.get("negative_docs", [])
        negative_docs = (
            _as_document_records(negative_value, "negative_docs", row=task_id)
            if negative_value is not None
            else []
        )

        items.append(
            ExternalBenchmarkItem(
                dataset_id="browsecomp-plus",
                track="fixed_corpus_retrieval_and_research",
                task_id=task_id,
                question=question,
                reference_answer=answer,
                annotation_type="human_verified_evidence_and_gold_document_labels",
                annotations={
                    "evidence_docids": [doc["docid"] for doc in evidence_docs],
                    "gold_docids": [doc["docid"] for doc in gold_docs],
                    "negative_docids": [doc["docid"] for doc in negative_docs],
                    "evidence_urls": [doc["url"] for doc in evidence_docs if doc["url"]],
                    "gold_urls": [doc["url"] for doc in gold_docs if doc["url"]],
                },
                metadata={
                    "corpus_dataset": "Tevatron/browsecomp-plus-corpus",
                    "document_text_loaded": False,
                    "evidence_doc_count": len(evidence_docs),
                    "gold_doc_count": len(gold_docs),
                    "negative_doc_count": len(negative_docs),
                },
                license_id="MIT dataset-card label; underlying web documents require separate rights review",
            )
        )
    return items
