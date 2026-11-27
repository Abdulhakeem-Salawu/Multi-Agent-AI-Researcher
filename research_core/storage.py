from __future__ import annotations

from contextlib import closing
from pathlib import Path
import json
import sqlite3

from .schemas import ResearchReport


class ResearchRunStore:
    """SQLite store for completed research reports, separate from graph checkpoints."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS research_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    question TEXT NOT NULL,
                    report_json TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def save(self, report: ResearchReport) -> int:
        payload = report.model_dump(mode="json")
        payload["run_id"] = None
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                "INSERT INTO research_runs(created_at, question, report_json) VALUES (?, ?, ?)",
                (
                    report.created_at.isoformat(),
                    report.question,
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def get(self, run_id: int) -> dict | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT report_json FROM research_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(row[0])
        payload["run_id"] = run_id
        return payload
