from __future__ import annotations

import sys
import threading
from contextlib import nullcontext
from types import SimpleNamespace

from research_core.llm import TransformersQwen3ChatModel


class FakeTensor:
    shape = (1, 3)

    def to(self, device):
        self.device = device
        return self


class FakeTokenizer:
    eos_token_id = 0

    def __init__(self):
        self.template_kwargs = None
        self.decode_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.template_kwargs = kwargs
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        return "rendered chat prompt"

    def __call__(self, prompt, **kwargs):
        assert prompt == "rendered chat prompt"
        assert kwargs["truncation"] is True
        return {"input_ids": FakeTensor()}

    def decode(self, tokens, **kwargs):
        self.decode_kwargs = kwargs
        assert tokens == [4, 5]
        return "generated answer"


class FakeModel:
    def __init__(self):
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return [[1, 2, 3, 4, 5]]


def _adapter() -> tuple[TransformersQwen3ChatModel, FakeModel, FakeTokenizer]:
    model = TransformersQwen3ChatModel.__new__(TransformersQwen3ChatModel)
    fake_model = FakeModel()
    fake_tokenizer = FakeTokenizer()
    model.model_id = "Qwen/Qwen3-1.7B"
    model.tokenizer = fake_tokenizer
    model.model = fake_model
    model.device = "cuda:0"
    model.max_new_tokens = 128
    model.max_input_tokens = 4096
    model.enable_thinking = False
    model._torch = SimpleNamespace(inference_mode=nullcontext)
    model._lock = threading.Lock()
    return model, fake_model, fake_tokenizer


def test_transformers_qwen_chat_adapter_uses_qwen_chat_template_and_decodes_only_new_tokens():
    adapter, fake_model, tokenizer = _adapter()

    answer = adapter.complete("System role", "User prompt", temperature=0.0)

    assert answer == "generated answer"
    assert tokenizer.template_kwargs["enable_thinking"] is False
    assert fake_model.kwargs["max_new_tokens"] == 128
    assert fake_model.kwargs["do_sample"] is False
    assert fake_model.kwargs["input_ids"].device == "cuda:0"


def test_transformers_qwen_chat_adapter_uses_non_thinking_sampling_settings():
    adapter, fake_model, _ = _adapter()

    adapter.complete("System role", "User prompt", temperature=0.7)

    assert fake_model.kwargs["do_sample"] is True
    assert fake_model.kwargs["temperature"] == 0.7
    assert fake_model.kwargs["top_p"] == 0.8
    assert fake_model.kwargs["top_k"] == 20


def test_transformers_qwen_resolves_and_loads_both_components_from_immutable_sha(monkeypatch):
    resolved_sha = "a" * 40
    calls = []

    class FakeHub:
        def __init__(self, token=None):
            assert token == "hf-test-token"

        def model_info(self, *, repo_id, revision):
            assert repo_id == "Qwen/Qwen3-1.7B"
            assert revision == "main"
            return SimpleNamespace(sha=resolved_sha)

    class FakeTokenizerLoader:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append(("tokenizer", model_id, kwargs.copy()))
            return FakeTokenizer()

    class FakeModelLoader:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append(("model", model_id, kwargs.copy()))
            model = FakeModel()
            model.config = SimpleNamespace(_commit_hash=None)
            model.eval = lambda: None
            model.parameters = lambda: iter([SimpleNamespace(device="cuda:0")])
            return model

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeHub))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=FakeTokenizerLoader,
            AutoModelForCausalLM=FakeModelLoader,
        ),
    )

    adapter = TransformersQwen3ChatModel(
        "Qwen/Qwen3-1.7B",
        revision="main",
        hf_token="hf-test-token",
    )

    assert adapter.requested_revision == "main"
    assert adapter.resolved_revision == resolved_sha
    assert [call[0] for call in calls] == ["tokenizer", "model"]
    assert all(call[2]["revision"] == resolved_sha for call in calls)
    assert all(call[2]["token"] == "hf-test-token" for call in calls)
