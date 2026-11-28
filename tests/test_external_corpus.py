from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from evaluation.external.corpus import (
    BrowseCompIndexError,
    BrowseCompPlusCorpusSearchProvider,
)
from research_agent.dependencies import ResearchDependencies
from research_agent.graph import build_research_graph
from research_core.retrieval import rank_hits
from research_core.schemas import RetrievedHit, SearchHit
from research_core.search import StaticSearchProvider


def _make_index(path: Path) -> Path:
    connection = sqlite3.connect(path)
    try:
        try:
            connection.execute(
                "CREATE VIRTUAL TABLE documents USING fts5("
                "docid UNINDEXED, url UNINDEXED, title, text)"
            )
        except sqlite3.OperationalError:
            pytest.skip("SQLite FTS5 is not available in this environment.")
        connection.execute(
            "CREATE TABLE index_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO index_metadata(key, value) VALUES (?, ?)",
            [
                ("dataset_id", "browsecomp-plus-corpus"),
                ("revision", "89e412c"),
                ("document_count", "2"),
            ],
        )
        connection.executemany(
            "INSERT INTO documents(docid, url, title, text) VALUES (?, ?, ?, ?)",
            [
                (
                    "doc-1",
                    "https://example.org/sqlite",
                    "SQLite query planner",
                    "SQLite uses a cost based query planner and indexes to optimize query execution.",
                ),
                (
                    "doc-2",
                    "https://example.org/python",
                    "Python concurrency",
                    "Python provides asyncio tasks and asynchronous concurrency primitives.",
                ),
            ],
        )
        connection.commit()
    finally:
        connection.close()
    return path


def test_browsecomp_provider_returns_retrieved_document_ids(tmp_path: Path):
    provider = BrowseCompPlusCorpusSearchProvider(_make_index(tmp_path / "corpus.sqlite3"))

    results = provider.search("SQLite query planner indexes", max_results=5)

    assert results
    assert results[0].docid == "doc-1"
    assert str(results[0].url) == "https://example.org/sqlite"
    assert "SQLite" in results[0].content
    assert provider.revision == "89e412c"
    assert provider.document_count == 2


def test_browsecomp_provider_rejects_unrelated_sqlite_file(tmp_path: Path):
    path = tmp_path / "not-a-corpus.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE unrelated (value TEXT)")
    connection.close()

    with pytest.raises(BrowseCompIndexError, match="not a BrowseComp-Plus"):
        BrowseCompPlusCorpusSearchProvider(path)


def test_langgraph_report_preserves_corpus_document_ids():
    provider = StaticSearchProvider([
        SearchHit(
            title="SQLite query planner",
            url="https://example.org/sqlite",
            content="SQLite uses a cost based query planner to optimize query execution.",
            score=0.95,
            docid="doc-42",
        )
    ])
    graph = build_research_graph(
        ResearchDependencies(
            search_provider=provider,
            llm=None,
            max_queries=1,
            max_results_per_query=5,
            max_evidence_sources=5,
        )
    )

    result = graph.invoke({"question": "How does SQLite optimize query execution?"})

    assert result["report"]["sources"][0]["docid"] == "doc-42"

def test_graph_accepts_longer_benchmark_questions_when_configured():
    provider = StaticSearchProvider([
        SearchHit(
            title="Technical reference",
            url="https://example.org/reference",
            content="A reference describes the system and its verified properties.",
            score=0.9,
        )
    ])
    question = "Please research the following detailed question " + ("in depth " * 350)
    assert len(question) > 2000
    graph = build_research_graph(
        ResearchDependencies(
            search_provider=provider,
            llm=None,
            max_queries=1,
            max_question_chars=4000,
        )
    )
    result = graph.invoke({"question": question})
    assert result["report"]["question"] == question.strip()



def test_corpus_retrieval_deduplicates_by_document_id_not_url():
    from datetime import datetime, timezone

    hits = [
        RetrievedHit(
            title="First archive copy",
            url="https://example.org/shared",
            content="SQLite uses a cost based query planner and indexes.",
            score=0.8,
            docid="document-a",
            query="SQLite query planner",
            query_index=0,
            result_index=0,
            retrieved_at=datetime.now(timezone.utc),
        ),
        RetrievedHit(
            title="Second archive copy",
            url="https://example.org/shared",
            content="SQLite uses a query planner to select indexes.",
            score=0.9,
            docid="document-b",
            query="SQLite query planner",
            query_index=0,
            result_index=1,
            retrieved_at=datetime.now(timezone.utc),
        ),
    ]

    ranked = rank_hits("SQLite query planner", hits, top_k=10, mode="hybrid")

    assert len(ranked) == 2
    assert {hit.docid for hit, _ in ranked} == {"document-a", "document-b"}
