from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BROWSECOMP_PLUS_CODE_REVISION = "046949032b0328319cc9a02663a759ec601d9402"
BROWSECOMP_PLUS_QUERY_REVISION = "144cff8e35b5eaef7e526346aa60774a9deb941f"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_browsecomp_plus_queries(
    upstream_repo: str | Path,
    output_path: str | Path,
    *,
    revision: str = BROWSECOMP_PLUS_QUERY_REVISION,
    max_records: int | None = None,
) -> dict[str, Any]:
    """Use the official decoder and retain only labels, IDs and URLs, never document text."""
    if not revision or revision in {"main", "master"}:
        raise ValueError("Pass a pinned BrowseComp-Plus dataset revision.")
    if max_records is not None and max_records < 1:
        raise ValueError("max_records must be at least 1.")
    repo_root = Path(upstream_repo).resolve()
    decoder_path = repo_root / "scripts_build_index" / "decrypt_dataset.py"
    if not decoder_path.is_file():
        raise FileNotFoundError(
            "Official decoder not found. Clone https://github.com/texttron/BrowseComp-Plus."
        )
    try:
        checked_out_revision = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            "Could not verify the official BrowseComp-Plus decoder checkout revision."
        ) from exc
    if checked_out_revision != BROWSECOMP_PLUS_CODE_REVISION:
        raise ValueError(
            "The BrowseComp-Plus source repository must be checked out at "
            f"{BROWSECOMP_PLUS_CODE_REVISION}; found {checked_out_revision}."
        )
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the benchmark extra with pip install -e '.[benchmark]'") from exc

    decoder = runpy.run_path(str(decoder_path))
    transform = decoder["transform_decrypt"]
    canary = decoder["DEFAULT_CANARY"]
    dataset = load_dataset(
        "Tevatron/browsecomp-plus",
        split="test",
        streaming=True,
        revision=revision,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_suffix(output.suffix + ".part")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for raw_record in dataset:
                decoded = transform(dict(raw_record), canary, {"query_id"})
                safe_docs = {}
                for key in ("evidence_docs", "gold_docs", "negative_docs"):
                    safe_docs[key] = [
                        {
                            "docid": str(doc.get("docid", "")),
                            "url": str(doc.get("url", "")),
                        }
                        for doc in decoded.get(key, [])
                        if isinstance(doc, dict)
                    ]
                safe_record = {
                    "query_id": decoded.get("query_id"),
                    "query": decoded.get("query"),
                    "answer": decoded.get("answer"),
                    **safe_docs,
                }
                handle.write(json.dumps(safe_record, ensure_ascii=False) + "\n")
                count += 1
                if max_records is not None and count >= max_records:
                    break
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
        raise

    if max_records is None and count != 830:
        raise ValueError(
            f"Expected the full BrowseComp-Plus test split to contain 830 queries; prepared {count}."
        )
    manifest = {
        "dataset_id": "browsecomp-plus",
        "upstream_revision": revision,
        "upstream_decoder_revision": BROWSECOMP_PLUS_CODE_REVISION,
        "path": str(output),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "sha256": _sha256(output),
        "size_bytes": output.stat().st_size,
        "record_count": count,
        "document_text_retained": False,
        "complete_split": max_records is None,
        "policy": "Local controlled evaluation only; do not redistribute.",
    }
    output.with_name(output.name + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
