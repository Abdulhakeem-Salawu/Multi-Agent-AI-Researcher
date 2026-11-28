from __future__ import annotations

from research_agent.dependencies import ResearchSettings


def test_tavily_key_secret_alias_is_supported(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("TAVILY_KEY", "tavily-from-github-secret")

    settings = ResearchSettings.from_environment()

    assert settings.tavily_api_key == "tavily-from-github-secret"


def test_tavily_api_key_takes_precedence_when_both_names_exist(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "canonical-key")
    monkeypatch.setenv("TAVILY_KEY", "compatibility-alias")

    settings = ResearchSettings.from_environment()

    assert settings.tavily_api_key == "canonical-key"
