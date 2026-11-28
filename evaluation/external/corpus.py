from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from research_core.schemas import SearchHit

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'-]*")


class BrowseCompIndexError(RuntimeError):
    """Raised when a local fixed-corpus index cannot be read or built."""


def build_browsecomp_plus_index(
    output_path: str | Path,
    *,
    revision: str,
    max_documents: int | None = None,
    batch_size: int = 250,
) -> dict[str, Any]:
    """Stream a pinned upstream corpus into SQLite FTS5 in a local environment."""
    if not revision.strip() or revision in {"main", "master"}:
        raise ValueError("Pass an immutable corpus revision, not main/master.")
    if max_documents is not None and max_documents < 1:
        raise ValueError("max_documents must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise BrowseCompIndexError(
            "Install the optional dependency with: pip install -e '.[benchmark]'"
        ) from exc

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Index already exists: {output}")

    dataset = load_dataset(
        "Tevatron/browsecomp-plus-corpus",
        split="train",
        streaming=True,
        revision=revision,
    )
    connection = sqlite3.connect(output)
    try:
        try:
            connection.execute("CREATE VIRTUAL TABLE temp.fts5_probe USING fts5(value)")
            connection.execute("DROP TABLE temp.fts5_probe")
        except sqlite3.OperationalError as exc:
            raise BrowseCompIndexError(
                "This Python SQLite build lacks FTS5 support."
            ) from exc

        connection.execute(
            "CREATE VIRTUAL TABLE documents USING fts5("
            "docid UNINDEXED, url UNINDEXED, title, text, tokenize='unicode61'"
            ")"
        )
        connection.execute(
            "CREATE TABLE index_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        metadata = {
            "dataset_id": "browsecomp-plus-corpus",
            "revision": revision,
            "content_policy": "local controlled evaluation only; do not redistribute",
        }
        connection.executemany(
            "INSERT INTO index_metadata(key, value) VALUES (?, ?)",
            list(metadata.items()),
        )

        batch: list[tuple[str, str, str, str]] = []
        count = 0
        for row in dataset:
            docid = str(row.get("docid") or "").strip()
            url = str(row.get("url") or "").strip()
            text = row.get("text")
            if not docid or not url or not isinstance(text, str) or not text.strip():
                raise BrowseCompIndexError(
                    f"Corpus record {count} lacks a valid docid, URL or text."
                )
            parsed = urlparse(url)
            title = parsed.netloc + parsed.path
            batch.append((docid, url, title[:1000], text))
            count += 1
            if len(batch) >= batch_size:
                connection.executemany(
                    "INSERT INTO documents(docid, url, title, text) VALUES (?, ?, ?, ?)",
                    batch,
                )
                connection.commit()
                batch.clear()
            if max_documents is not None and count >= max_documents:
                break
        if batch:
            connection.executemany(
                "INSERT INTO documents(docid, url, title, text) VALUES (?, ?, ?, ?)",
                batch,
            )
        if max_documents is None and count != 100195:
            raise BrowseCompIndexError(
                f"Expected the complete pinned corpus to contain 100195 documents; indexed {count}."
            )
        connection.execute(
            "INSERT INTO index_metadata(key, value) VALUES (?, ?)",
            ("document_count", str(count)),
        )
        connection.commit()
        return {
            "index_path": str(output),
            "dataset_id": metadata["dataset_id"],
            "revision": revision,
            "document_count": count,
            "complete_corpus": max_documents is None,
            "content_policy": metadata["content_policy"],
        }
    except Exception:
        output.unlink(missing_ok=True)
        raise
    finally:
        connection.close()


class BrowseCompPlusCorpusSearchProvider:
    """Search a local BrowseComp-Plus FTS5 index and return the upstream docid."""

    def __init__(self, index_path: str | Path, *, max_text_chars: int = 18000) -> None:
        self.index_path = Path(index_path)
        self.max_text_chars = max_text_chars
        self.calls: list[str] = []
        if not self.index_path.is_file():
            raise BrowseCompIndexError(f"Index not found: {self.index_path}")
        try:
            with sqlite3.connect(self.index_path) as connection:
                metadata = dict(connection.execute("SELECT key, value FROM index_metadata"))
        except sqlite3.DatabaseError as exc:
            raise BrowseCompIndexError(
                "This file is not a BrowseComp-Plus corpus index: required SQLite metadata is missing or invalid."
            ) from exc
        if metadata.get("dataset_id") != "browsecomp-plus-corpus":
            raise BrowseCompIndexError("This is not a BrowseComp-Plus corpus index.")
        try:
            self.revision = metadata["revision"]
            self.document_count = int(metadata["document_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BrowseCompIndexError(
                "The BrowseComp-Plus corpus index metadata is incomplete or invalid."
            ) from exc

    @staticmethod
    def _match_expression(query: str) -> str:
        terms = list(dict.fromkeys(token.lower() for token in _TOKEN_RE.findall(query)))
        if not terms:
            return ""
        return " OR ".join('"' + term.replace('"', '""') + '"' for term in terms[:32])

    def search(self, query: str, max_results: int = 8) -> list[SearchHit]:
        if max_results < 1:
            raise ValueError("max_results must be at least 1.")
        self.calls.append(query)
        expression = self._match_expression(query)
        if not expression:
            return []
        with sqlite3.connect(self.index_path) as connection:
            rows = connection.execute(
                "SELECT docid, url, title, text, bm25(documents) AS rank "
                "FROM documents WHERE documents MATCH ? ORDER BY rank LIMIT ?",
                (expression, max_results),
            ).fetchall()
        if not rows:
            return []
        scores = [max(0.0, -float(row[4])) for row in rows]
        maximum = max(scores, default=0.0)
        results: list[SearchHit] = []
        for row, raw_score in zip(rows, scores):
            evidence_text = str(row[3])[: self.max_text_chars]
            results.append(
                SearchHit(
                    title=str(row[2] or row[1]),
                    url=str(row[1]),
                    content=evidence_text,
                    raw_content=evidence_text,
                    score=(raw_score / maximum) if maximum else 0.0,
                    docid=str(row[0]),
                )
            )
        return results
