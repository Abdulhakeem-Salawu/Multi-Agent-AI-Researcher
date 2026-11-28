from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .adapters import load_deepresearch_bench_ii_jsonl, load_frames_tsv

DRB2_REVISION = "ca2d511449646318535fd88c00790f4e85f18a7f"
FRAMES_REVISION = "2383b440f3de33d9c003f089fb66f6677436d696"

DOWNLOADS = {
    "deepresearch-bench-ii": {
        "url": "https://raw.githubusercontent.com/imlrz/DeepResearch-Bench-II/"
        + DRB2_REVISION + "/tasks_and_rubrics.jsonl",
        "relative_path": "drb2/tasks_and_rubrics.jsonl",
        "revision": DRB2_REVISION,
        "expected_records": 132,
    },
    "frames": {
        "url": "https://huggingface.co/datasets/google/frames-benchmark/resolve/"
        + FRAMES_REVISION + "/test.tsv",
        "relative_path": "frames/test.tsv",
        "revision": FRAMES_REVISION,
        "expected_records": 824,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_benchmark_file(dataset_id: str, output_dir: str | Path, overwrite: bool = False):
    if dataset_id not in DOWNLOADS:
        raise ValueError("Use the BrowseComp-Plus preparation command for that dataset.")
    spec = DOWNLOADS[dataset_id]
    target = Path(output_dir) / spec["relative_path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists; use --overwrite to replace it.")
    temporary = target.with_suffix(target.suffix + ".part")
    try:
        with httpx.stream("GET", spec["url"], follow_redirects=True, timeout=120.0) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_bytes():
                    if chunk:
                        output.write(chunk)
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if dataset_id == "deepresearch-bench-ii":
        count = len(load_deepresearch_bench_ii_jsonl(target, include_noncommercial=True))
    else:
        count = len(load_frames_tsv(target))
    if count != spec["expected_records"]:
        raise ValueError(f"Expected {spec['expected_records']} records; parsed {count}.")
    manifest = {
        "dataset_id": dataset_id,
        "source_url": spec["url"],
        "upstream_revision": spec["revision"],
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "path": str(target),
        "size_bytes": target.stat().st_size,
        "sha256": sha256_file(target),
        "record_count": count,
        "licence_note": "Local-only; follow per-task and dataset-card terms.",
    }
    target.with_name(target.name + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
