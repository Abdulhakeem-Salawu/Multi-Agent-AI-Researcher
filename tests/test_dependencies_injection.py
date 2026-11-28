from __future__ import annotations

from pathlib import Path

from research_agent.dependencies import build_default_dependencies
from research_core.search import StaticSearchProvider


class FakeChatModel:
    model_id = "Qwen/Qwen3-1.7B"

    def complete(self, system_prompt, user_prompt, temperature=0.1):
        return "ok"


def test_default_dependencies_accept_an_injected_chat_model(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "runs.sqlite3"))
    chat_model = FakeChatModel()

    dependencies = build_default_dependencies(
        search_provider=StaticSearchProvider([]),
        llm=chat_model,
        llm_model="Qwen/Qwen3-1.7B",
    )

    assert dependencies.llm is chat_model
    assert dependencies.llm_model == "Qwen/Qwen3-1.7B"
    assert dependencies.store is not None
