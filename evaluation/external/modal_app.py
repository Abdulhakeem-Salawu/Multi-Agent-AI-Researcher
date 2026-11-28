from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal
from dotenv import load_dotenv

from .adapters import load_browsecomp_plus_jsonl, load_deepresearch_bench_ii_jsonl, load_frames_tsv
from .corpus import BrowseCompPlusCorpusSearchProvider, build_browsecomp_plus_index
from .downloads import DOWNLOADS, download_benchmark_file
from .preparation import (
    BROWSECOMP_PLUS_CODE_REVISION,
    BROWSECOMP_PLUS_QUERY_REVISION,
    prepare_browsecomp_plus_queries,
)
from .runner import run_experiment

load_dotenv()

APP_NAME = "multi-agent-ai-researcher-benchmarks"
VOLUME_NAME = "multi-agent-ai-research-benchmarks"
QWEN_CACHE_VOLUME_NAME = "multi-agent-ai-research-qwen-cache"
DEFAULT_MODEL_ID = "Qwen/Qwen3-1.7B"
VOLUME_ROOT = Path("/mnt/research-benchmarks")
DATA_ROOT = VOLUME_ROOT / "data" / "external"
RUNS_ROOT = VOLUME_ROOT / "runs"
BROWSECOMP_CORPUS_REVISION = "89e412cede7d9904f0e8b717aebd9b317c46dfdc"
BROWSECOMP_CODE_URL = "https://github.com/texttron/BrowseComp-Plus.git"

# These values are copied from the local Modal CLI process (GitHub Actions in CI)
# into a short-lived Modal Secret. The secret values are never printed or returned.
RUNTIME_SECRET_KEYS = [
    "TAVILY_KEY",
    "TAVILY_API_KEY",
    "HF_TOKEN",
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
    "EVAL_JUDGE_LLM_BASE_URL",
    "EVAL_JUDGE_LLM_API_KEY",
    "EVAL_JUDGE_LLM_MODEL",
    "EVAL_JUDGE_TIMEOUT_SECONDS",
    "LANGFUSE_BASE_URL",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_TRACE_CONTENT",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
]

def _normalize_secret_values(values: dict[str, str | None]) -> dict[str, str | None]:
    """Trim copy/paste line endings and omit absent optional secrets."""
    return {
        key: (value.strip() or None) if value is not None else None
        for key, value in values.items()
    }


runtime_secret = modal.Secret.from_dict(
    _normalize_secret_values({key: os.getenv(key) for key in RUNTIME_SECRET_KEYS})
)
benchmark_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
qwen_cache_volume = modal.Volume.from_name(QWEN_CACHE_VOLUME_NAME, create_if_missing=True)

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "modal>=1.6.4,<2.0.0",
        "langgraph>=1.2.24,<2.0.0",
        "httpx>=0.27",
        "pydantic>=2.8",
        "python-dotenv>=1.0",
        "datasets>=3.0,<5.0",
        "prometheus-client>=0.21",
        "opentelemetry-api>=1.27",
        "opentelemetry-sdk>=1.27",
        "opentelemetry-exporter-otlp-proto-http>=1.27",
        "langfuse>=5.0,<6",
    )
)

LOCAL_PYTHON_PACKAGES = (
    "research_agent",
    "research_core",
    "research_observability",
    "evaluation",
)

image = base_image.add_local_python_source(*LOCAL_PYTHON_PACKAGES)

qwen_image = (
    base_image
    .pip_install(
        "torch>=2.6",
        "transformers>=4.51,<6.0",
        "accelerate>=1.0",
        "safetensors>=0.4",
    )
    .add_local_python_source(*LOCAL_PYTHON_PACKAGES)
)

app = modal.App(APP_NAME)


def validate_hf_model_id(model_id: str) -> str:
    value = model_id.strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value):
        raise ValueError("model_id must be a Hugging Face repo ID such as Qwen/Qwen3-1.7B.")
    return value


def safe_run_tag(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        normalized = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", normalized):
        raise ValueError("run_tag must contain only letters, numbers, dot, underscore or hyphen (max 80 chars).")
    return normalized


def validate_remote_model_endpoint(endpoint: str, name: str) -> None:
    from urllib.parse import urlparse

    parsed = urlparse(endpoint)
    hostname = (parsed.hostname or "").lower()
    if hostname in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}:
        raise ValueError(
            f"{name} points to localhost. Modal runs in the cloud; use a publicly reachable HTTPS endpoint."
        )
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"{name} must be an absolute HTTPS URL.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            f"{name} must not embed credentials, query parameters or fragments; pass credentials separately."
        )


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_small_dataset(dataset_id: str) -> tuple[Path, dict[str, Any]]:
    if dataset_id not in DOWNLOADS:
        raise ValueError(f"Unsupported small benchmark dataset: {dataset_id}")
    spec = DOWNLOADS[dataset_id]
    path = DATA_ROOT / spec["relative_path"]
    manifest_path = path.with_name(path.name + ".manifest.json")

    if path.is_file() and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            valid_revision = manifest.get("upstream_revision") == spec["revision"]
            valid_hash = manifest.get("sha256") == _file_sha256(path)
            if dataset_id == "deepresearch-bench-ii":
                count = len(load_deepresearch_bench_ii_jsonl(path, include_noncommercial=True))
            else:
                count = len(load_frames_tsv(path))
            if valid_revision and valid_hash and count == spec["expected_records"]:
                return path, {"downloaded": False, **manifest}
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass

    result = download_benchmark_file(dataset_id, DATA_ROOT, overwrite=path.exists())
    return path, {"downloaded": True, **result}


def _clone_pinned_browsecomp_code() -> Path:
    temporary_root = Path(tempfile.mkdtemp(prefix="browsecomp-plus-upstream-"))
    checkout = temporary_root / "BrowseComp-Plus"
    try:
        subprocess.run(
            ["git", "clone", "--quiet", "--filter=blob:none", BROWSECOMP_CODE_URL, str(checkout)],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        subprocess.run(
            ["git", "-C", str(checkout), "checkout", "--quiet", BROWSECOMP_PLUS_CODE_REVISION],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return checkout
    except (OSError, subprocess.SubprocessError):
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


def _ensure_browsecomp_queries() -> tuple[Path, dict[str, Any]]:
    path = DATA_ROOT / "browsecomp-plus" / "queries.jsonl"
    manifest_path = path.with_name(path.name + ".manifest.json")
    if path.is_file() and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            valid = (
                manifest.get("upstream_revision") == BROWSECOMP_PLUS_QUERY_REVISION
                and manifest.get("upstream_decoder_revision") == BROWSECOMP_PLUS_CODE_REVISION
                and manifest.get("complete_split") is True
                and manifest.get("sha256") == _file_sha256(path)
                and manifest.get("record_count") == 830
            )
            if valid and len(load_browsecomp_plus_jsonl(path)) == 830:
                return path, {"prepared": False, **manifest}
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass
        path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)

    checkout = _clone_pinned_browsecomp_code()
    try:
        result = prepare_browsecomp_plus_queries(
            checkout,
            path,
            revision=BROWSECOMP_PLUS_QUERY_REVISION,
        )
    finally:
        shutil.rmtree(checkout.parent, ignore_errors=True)
    return path, {"prepared": True, **result}


def _ensure_browsecomp_corpus() -> tuple[Path, dict[str, Any]]:
    path = DATA_ROOT / "browsecomp-plus" / "corpus.sqlite3"
    if path.is_file():
        try:
            provider = BrowseCompPlusCorpusSearchProvider(path)
            if (
                provider.revision == BROWSECOMP_CORPUS_REVISION
                and provider.document_count == 100195
            ):
                return path, {
                    "indexed": False,
                    "revision": provider.revision,
                    "document_count": provider.document_count,
                    "index_path": str(path),
                    "reused_existing_index": True,
                }
        except (OSError, RuntimeError, ValueError):
            pass
        path.unlink(missing_ok=True)
        Path(str(path) + "-wal").unlink(missing_ok=True)
        Path(str(path) + "-shm").unlink(missing_ok=True)

    result = build_browsecomp_plus_index(
        path,
        revision=BROWSECOMP_CORPUS_REVISION,
    )
    result["indexed"] = True
    return path, result


def _validate_runtime_config(dataset_id: str) -> None:
    if dataset_id in {"frames", "deepresearch-bench-ii"}:
        tavily_key = os.getenv("TAVILY_API_KEY") or os.getenv("TAVILY_KEY")
        if not tavily_key:
            raise RuntimeError(
                "Missing TAVILY_KEY. Add it as a GitHub Actions secret; the workflow forwards it to Modal."
            )
        os.environ["TAVILY_API_KEY"] = tavily_key


@app.function(
    image=qwen_image,
    gpu="L4",
    cpu=8,
    memory=32768,
    timeout=6 * 60 * 60,
    max_containers=1,
    volumes={
        str(VOLUME_ROOT): benchmark_volume,
        "/root/.cache/huggingface": qwen_cache_volume,
    },
    secrets=[runtime_secret],
)
def execute_qwen_experiment(
    dataset_id: str,
    input_path: str,
    output_dir: str,
    model_id: str = DEFAULT_MODEL_ID,
    limit: int = 3,
    offset: int = 0,
    language: str = "",
    include_noncommercial: bool = False,
    corpus_index: str = "",
    source_commit: str = "",
) -> dict[str, Any]:
    """Load the selected Hugging Face model once and score a benchmark slice on an L4."""
    model_id = validate_hf_model_id(model_id)
    benchmark_volume.reload()
    qwen_cache_volume.reload()
    os.environ["RESEARCH_DB_PATH"] = "/tmp/modal-research-benchmark-runs.sqlite3"
    if source_commit:
        os.environ["GITHUB_SHA"] = source_commit

    from research_core.llm import TransformersQwen3ChatModel

    try:
        chat_model = TransformersQwen3ChatModel(
            model_id,
            revision="main",
            hf_token=os.getenv("HF_TOKEN"),
            max_new_tokens=4096,
            max_input_tokens=24576,
            enable_thinking=False,
        )
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if (output_path / "predictions.jsonl").exists():
            raise FileExistsError(
                f"Run output already exists at {output_path}; choose a different run_tag."
            )

        manifest = run_experiment(
            dataset_id,
            input_path,
            output_path,
            limit=limit,
            offset=offset,
            language=language or None,
            include_noncommercial=include_noncommercial,
            corpus_index=corpus_index or None,
            agent_model=chat_model,
            agent_model_id=model_id,
            evaluator_model=chat_model,
            evaluator_model_id=model_id,
            model_revision=chat_model.resolved_revision,
        )
        summary_path = output_path / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return {
            "status": manifest["status"],
            "dataset_id": dataset_id,
            "model_id": model_id,
            "model_revision": chat_model.resolved_revision,
            "completed_count": summary["completed_count"],
            "failed_count": summary["failed_count"],
            "summary": summary,
            "run_manifest_path": str(output_path / "run_manifest.json"),
            "predictions_path": str(output_path / "predictions.jsonl"),
        }
    finally:
        qwen_cache_volume.commit()
        benchmark_volume.commit()


@app.function(
    image=image,
    volumes={str(VOLUME_ROOT): benchmark_volume},
    secrets=[runtime_secret],
    cpu=4,
    memory=16384,
    max_containers=1,
    timeout=6 * 60 * 60,
)
def execute_external_benchmark(
    dataset_id: str,
    stage: str = "run",
    limit: int = 5,
    offset: int = 0,
    language: str = "",
    include_noncommercial: bool = False,
    run_tag: str = "",
    source_commit: str = "",
    model_id: str = DEFAULT_MODEL_ID,
) -> dict[str, Any]:
    """Download, index and/or evaluate an external benchmark entirely inside Modal."""
    if dataset_id not in {"frames", "deepresearch-bench-ii", "browsecomp-plus"}:
        raise ValueError("dataset_id must be frames, deepresearch-bench-ii or browsecomp-plus.")
    if stage not in {"prepare", "build-corpus", "run", "all"}:
        raise ValueError("stage must be prepare, build-corpus, run or all.")
    if limit < 1 or offset < 0:
        raise ValueError("limit must be positive and offset must be non-negative.")
    model_id = validate_hf_model_id(model_id)
    if language and (dataset_id != "deepresearch-bench-ii" or language not in {"en", "zh"}):
        raise ValueError("language is only supported for DRB II and must be en or zh.")
    if include_noncommercial and dataset_id != "deepresearch-bench-ii":
        raise ValueError("include_noncommercial is only supported for DRB II.")
    if stage == "build-corpus" and dataset_id != "browsecomp-plus":
        raise ValueError("build-corpus is only supported for BrowseComp-Plus.")

    benchmark_volume.reload()
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    os.environ["RESEARCH_DB_PATH"] = "/tmp/modal-research-benchmark-runs.sqlite3"
    if source_commit:
        os.environ["GITHUB_SHA"] = source_commit

    input_path: Path | None = None
    corpus_index: Path | None = None
    preparation_details: dict[str, Any] = {}

    if dataset_id in {"frames", "deepresearch-bench-ii"}:
        if stage == "build-corpus":
            raise ValueError("This dataset does not have a fixed corpus index.")
        if stage in {"prepare", "run", "all"}:
            input_path, detail = _ensure_small_dataset(dataset_id)
            preparation_details["dataset_input"] = detail
            benchmark_volume.commit()
            if stage == "prepare":
                return {
                    "status": "prepared",
                    "dataset_id": dataset_id,
                    "input_path": str(input_path),
                    "preparation": preparation_details,
                    "storage": "Modal Volume (persistent)",
                }

    else:
        if stage in {"prepare", "all"}:
            input_path, detail = _ensure_browsecomp_queries()
            preparation_details["query_dataset"] = detail
            benchmark_volume.commit()
            if stage == "prepare":
                return {
                    "status": "prepared",
                    "dataset_id": dataset_id,
                    "input_path": str(input_path),
                    "preparation": preparation_details,
                    "storage": "Modal Volume (persistent)",
                }

        if stage in {"build-corpus", "all"}:
            corpus_index, detail = _ensure_browsecomp_corpus()
            preparation_details["fixed_corpus"] = detail
            benchmark_volume.commit()
            if stage == "build-corpus":
                return {
                    "status": "corpus_ready",
                    "dataset_id": dataset_id,
                    "corpus_index": str(corpus_index),
                    "preparation": preparation_details,
                    "storage": "Modal Volume (persistent)",
                }

        if stage == "run":
            input_path = DATA_ROOT / "browsecomp-plus" / "queries.jsonl"
            corpus_index = DATA_ROOT / "browsecomp-plus" / "corpus.sqlite3"
            if not input_path.is_file():
                raise FileNotFoundError(
                    "BrowseComp-Plus queries are not prepared. Run this workflow with stage=prepare first, "
                    "or stage=all to prepare, build the full corpus index and evaluate."
                )
            if not corpus_index.is_file():
                raise FileNotFoundError(
                    "The full BrowseComp-Plus corpus index is not built. Run this workflow with "
                    "stage=build-corpus first, or stage=all."
                )

    if stage not in {"run", "all"}:
        raise RuntimeError("The selected stage did not produce a runnable benchmark input.")

    _validate_runtime_config(dataset_id)
    if dataset_id == "browsecomp-plus":
        provider = BrowseCompPlusCorpusSearchProvider(corpus_index)
        if provider.document_count != 100195 or provider.revision != BROWSECOMP_CORPUS_REVISION:
            raise RuntimeError(
                "Refusing to score BrowseComp-Plus with a partial corpus or an unexpected revision."
            )

    selected_tag = safe_run_tag(run_tag)
    output_dir = RUNS_ROOT / dataset_id / selected_tag
    result = execute_qwen_experiment.remote(
        dataset_id=dataset_id,
        input_path=str(input_path),
        output_dir=str(output_dir),
        model_id=model_id,
        limit=limit,
        offset=offset,
        language=language,
        include_noncommercial=include_noncommercial,
        corpus_index=str(corpus_index) if corpus_index else "",
        source_commit=source_commit,
    )
    return {
        **result,
        "run_tag": selected_tag,
        "storage": "Modal Volume (persistent)",
        "preparation": preparation_details,
        "inference": {
            "runtime": "Modal L4 GPU",
            "weights_source": "Hugging Face Hub",
            "weights_cache": QWEN_CACHE_VOLUME_NAME,
            "agent_and_evaluator_share_model": True,
            "judge_type": "same_model_llm_judge",
        },
    }


@app.local_entrypoint()
def main(
    dataset_id: str = "frames",
    stage: str = "run",
    limit: int = 5,
    offset: int = 0,
    language: str = "",
    include_noncommercial: bool = False,
    run_tag: str = "",
    source_commit: str = "",
    model_id: str = DEFAULT_MODEL_ID,
) -> None:
    result = execute_external_benchmark.remote(
        dataset_id=dataset_id,
        stage=stage,
        limit=limit,
        offset=offset,
        language=language,
        include_noncommercial=include_noncommercial,
        run_tag=run_tag,
        source_commit=source_commit,
        model_id=model_id,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
