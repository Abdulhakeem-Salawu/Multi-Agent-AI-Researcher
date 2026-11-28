from __future__ import annotations

from contextlib import contextmanager

import pytest

from scripts.langfuse_smoke import run_smoke_test


class FakeSpan:
    def __init__(self):
        self.output = None

    def update(self, **kwargs):
        self.output = kwargs.get("output")


class FakeLangfuseClient:
    def __init__(self, auth_result: bool = True):
        self.auth_result = auth_result
        self.auth_checked = False
        self.flushed = False
        self.name = None
        self.metadata = None
        self.span = FakeSpan()

    def auth_check(self):
        self.auth_checked = True
        return self.auth_result

    @contextmanager
    def start_as_current_observation(self, *, as_type, name, metadata):
        self.name = name
        self.metadata = metadata
        yield self.span

    def flush(self):
        self.flushed = True


def set_required_credentials(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")


def test_langfuse_smoke_test_checks_auth_sends_synthetic_observation_and_flushes(monkeypatch):
    set_required_credentials(monkeypatch)
    monkeypatch.setenv("LANGFUSE_TRACE_CONTENT", "false")
    client = FakeLangfuseClient()

    run_smoke_test(client)

    assert client.auth_checked is True
    assert client.flushed is True
    assert client.name == "github-actions.langfuse-connection-check"
    assert client.metadata["content_capture_enabled"] is False
    assert client.span.output["status"] == "connected"


def test_langfuse_smoke_test_fails_when_required_credentials_are_missing(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    with pytest.raises(RuntimeError, match="LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY"):
        run_smoke_test(FakeLangfuseClient())


def test_langfuse_smoke_test_fails_when_authentication_is_rejected(monkeypatch):
    set_required_credentials(monkeypatch)

    with pytest.raises(RuntimeError, match="rejected"):
        run_smoke_test(FakeLangfuseClient(auth_result=False))
