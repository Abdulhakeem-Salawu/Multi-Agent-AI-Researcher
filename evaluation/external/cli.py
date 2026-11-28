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
from .corpus import build_browsecomp_plus_index
from .downloads import download_benchmark_file
from .preparation import prepare_browsecomp_plus_queries
from .runner import run_experiment


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

    download = commands.add_parser("download", help="Download a pinned small benchmark input.")
    download.add_argument("dataset_id", choices=["deepresearch-bench-ii", "frames"])
    download.add_argument("--output-dir", type=Path, default=Path("data/external"))
    download.add_argument("--overwrite", action="store_true")

    prepare = commands.add_parser(
        "prepare-browsecomp-plus",
        help="Use the upstream decoder to prepare IDs/URLs without retaining document text.",
    )
    prepare.add_argument("--upstream-repo", type=Path, default=Path("external/BrowseComp-Plus"))
    prepare.add_argument("--output", type=Path, default=Path("data/external/browsecomp-plus/queries.jsonl"))
    prepare.add_argument("--revision", default="144cff8e35b5eaef7e526346aa60774a9deb941f")
    prepare.add_argument("--limit", type=int, help="Optional smoke-test limit; omit for all 830 queries.")

    index = commands.add_parser("build-browsecomp-index", help="Build a local FTS5 index from the fixed corpus.")
    index.add_argument("--output", type=Path, default=Path("data/external/browsecomp-plus/corpus.sqlite3"))
    index.add_argument("--revision", default="89e412cede7d9904f0e8b717aebd9b317c46dfdc")
    index.add_argument("--limit", type=int, help="Optional document limit for smoke testing.")

    run = commands.add_parser("run", help="Run a dataset slice through the LangGraph agent and score it.")
    run.add_argument("dataset_id", choices=["deepresearch-bench-ii", "frames", "browsecomp-plus"])
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--limit", type=int, default=10)
    run.add_argument("--offset", type=int, default=0)
    run.add_argument("--k", type=int, default=10)
    run.add_argument("--include-noncommercial", action="store_true")
    run.add_argument("--corpus-index", type=Path)
    run.add_argument("--resume", action="store_true")

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
        elif args.command == "inspect":
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
        elif args.command == "download":
            result = download_benchmark_file(
                args.dataset_id,
                args.output_dir,
                overwrite=args.overwrite,
            )
        elif args.command == "prepare-browsecomp-plus":
            result = prepare_browsecomp_plus_queries(
                args.upstream_repo,
                args.output,
                revision=args.revision,
                max_records=args.limit,
            )
        elif args.command == "build-browsecomp-index":
            result = build_browsecomp_plus_index(
                args.output,
                revision=args.revision,
                max_documents=args.limit,
            )
        else:
            result = run_experiment(
                args.dataset_id,
                args.input,
                args.output_dir,
                limit=args.limit,
                offset=args.offset,
                k=args.k,
                include_noncommercial=args.include_noncommercial,
                corpus_index=args.corpus_index,
                resume=args.resume,
            )
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
