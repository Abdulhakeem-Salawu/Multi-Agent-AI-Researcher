from __future__ import annotations

import pytest

modal = pytest.importorskip("modal")

from evaluation.external.modal_app import (
    _normalize_secret_values,
    safe_run_tag,
    validate_remote_model_endpoint,
)  # noqa: E402


def test_modal_run_tag_is_safe_for_volume_paths():
    assert safe_run_tag("github-123-1") == "github-123-1"
    with pytest.raises(ValueError, match="run_tag"):
        safe_run_tag("../escape-volume")


@pytest.mark.parametrize("endpoint", ["http://localhost:8000/v1", "http://127.0.0.1:11434/v1"])
def test_modal_runner_rejects_localhost_model_endpoints(endpoint):
    with pytest.raises(ValueError, match="localhost"):
        validate_remote_model_endpoint(endpoint, "LLM_BASE_URL")


def test_modal_runner_accepts_remote_openai_compatible_https_endpoint():
    validate_remote_model_endpoint("https://models.example.com/v1", "LLM_BASE_URL")


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://models.example.com/v1",
        "https://user:password@models.example.com/v1",
        "https://models.example.com/v1?api_key=secret",
    ],
)
def test_modal_runner_rejects_insecure_or_secret_bearing_endpoint_urls(endpoint):
    with pytest.raises(ValueError):
        validate_remote_model_endpoint(endpoint, "LLM_BASE_URL")



def test_modal_secret_values_trim_newlines_and_omit_blank_optional_values():
    cleaned = _normalize_secret_values({
        "TAVILY_KEY": " tavily-key-value\r\n",
        "LLM_API_KEY": "  ",
        "LANGFUSE_SECRET_KEY": None,
    })
    assert cleaned == {
        "TAVILY_KEY": "tavily-key-value",
        "LLM_API_KEY": None,
        "LANGFUSE_SECRET_KEY": None,
    }
