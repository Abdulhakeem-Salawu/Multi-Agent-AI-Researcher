from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .adapters import (
    load_browsecomp_plus_jsonl,
    load_deepresearch_bench_ii_jsonl,
    load_frames_tsv,
)


def _registry() -> dict[str, Any]:
    path = Path(__file__).with_name("registry.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect(dataset_id: str, input_path: Path, include_noncommercial: bool) -> dict[str, Any]:
    if dataset_id == "deepresearch-bench-ii":
        items = load_deepresearch_bench_ii_jsonl(
            input_path,
            include_noncommercial=include_noncommercial,
        )
        licenses = Counter(item.license_id for item in items)
        dimensions: Counter[str] = Counter()
        for item in items:
            for rubric in (item.annotations or {}).get("rubrics", []):
                dimensions[rubric["dimension"]] += 1
        details = {
            "per_task_license_counts": dict(sorted(licenses.items())),
            "rubric_counts_by_dimension": dict(sorted(dimensions.items())),
            "noncommercial_records_included": include_noncommercial,
        }
    elif dataset_id == "frames":
        items = load_frames_tsv(input_path)
        reasoning_types: Counter[str] = Counter()
        article_count = 0
        for item in items:
            annotations = item.annotations or {}
            reasoning_types.update(annotations.get("reasoning_types", []))
            article_count += len(annotations.get("relevant_article_urls", []))
        details = {
            "reasoning_type_counts": dict(sorted(reasoning_types.items())),
            "relevant_article_url_count": article_count,
        }
    elif dataset_id == "browsecomp-plus":
        items = load_browsecomp_plus_jsonl(input_path)
        evidence_count = sum(
            item.metadata["evidence_doc_count"] for item in items if item.metadata
        )
        gold_count = sum(item.metadata["gold_doc_count"] for item in items if item.metadata)
        negative_count = sum(
            item.metadata["negative_doc_count"] for item in items if item.metadata
        )
        details = {
            "evidence_document_judgments": evidence_count,
            "gold_document_judgments": gold_count,
            "negative_document_judgments": negative_count,
            "document_text_loaded": False,
        }
    else:
        raise ValueError(f"Unknown external dataset id: {dataset_id}")

    return {
        "dataset_id": dataset_id,
        "input_path": str(input_path),
        "input_sha256": _sha256(input_path),
        "loaded_item_count": len(items),
        "track": items[0].track if items else None,
        "annotation_type": items[0].annotation_type if items else None,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect and validate local copies of public external research benchmarks."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List registered external benchmark tracks.")
    inspect = commands.add_parser("inspect", help="Validate a downloaded upstream file.")
    inspect.add_argument(
        "dataset_id",
        choices=["deepresearch-bench-ii", "frames", "browsecomp-plus"],
    )
    inspect.add_argument("--input", type=Path, required=True)
    inspect.add_argument(
        "--include-noncommercial",
        action="store_true",
        help="Include DRB II CC BY-NC 4.0 tasks, only for a non-commercial use case.",
    )
    inspect.add_argument("--output", type=Path, help="Optional JSON summary output path.")

    args = parser.parse_args()
    try:
        if args.command == "list":
            registry = _registry()
            result = {
                "registry_status": registry["registry_status"],
                "datasets": [
                    {
                        "dataset_id": item["dataset_id"],
                        "name": item["name"],
                        "track": item["track"],
                        "expected_records": item["expected_records"],
                        "upstream_format": item["upstream_format"],
                    }
                    for item in registry["datasets"]
                ],
            }
        else:
            result = _inspect(
                args.dataset_id,
                args.input,
                include_noncommercial=args.include_noncommercial,
            )
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
