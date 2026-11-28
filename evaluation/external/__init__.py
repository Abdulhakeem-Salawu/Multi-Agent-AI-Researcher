"""Adapters for public, externally maintained research benchmarks."""

from .adapters import (
    ExternalDatasetError,
    load_browsecomp_plus_jsonl,
    load_deepresearch_bench_ii_jsonl,
    load_frames_tsv,
)
from .models import ExternalBenchmarkItem
from .corpus import BrowseCompPlusCorpusSearchProvider, build_browsecomp_plus_index
from .downloads import download_benchmark_file
from .preparation import prepare_browsecomp_plus_queries
from .runner import run_experiment
from .reporting import summarize_predictions, write_run_summary
from .scoring import (
    score_browsecomp_plus_retrieval,
    score_drb2_rubric_judgments,
    score_frames_answer,
    score_frames_retrieval,
)

__all__ = [
    "ExternalBenchmarkItem",
    "BrowseCompPlusCorpusSearchProvider",
    "build_browsecomp_plus_index",
    "download_benchmark_file",
    "prepare_browsecomp_plus_queries",
    "run_experiment",
    "summarize_predictions",
    "write_run_summary",
    "ExternalDatasetError",
    "load_browsecomp_plus_jsonl",
    "load_deepresearch_bench_ii_jsonl",
    "load_frames_tsv",
    "score_browsecomp_plus_retrieval",
    "score_drb2_rubric_judgments",
    "score_frames_answer",
    "score_frames_retrieval",
]
