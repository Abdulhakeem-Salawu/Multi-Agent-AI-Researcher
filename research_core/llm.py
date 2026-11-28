from __future__ import annotations

from typing import Protocol

import httpx


class ChatModel(Protocol):
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> str: ...


class OpenAICompatibleChatModel:
    """Adapter for Ollama, vLLM, LM Studio and hosted OpenAI-compatible APIs."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 45.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise TypeError("The chat completion endpoint returned non-text content.")
        return content



class TransformersQwen3ChatModel:
    """Run a Hugging Face Transformers chat model inside a GPU worker.

    Qwen3's thinking mode is disabled by default to return concise final content
    for the existing agent and JSON-oriented benchmark evaluators.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-1.7B",
        *,
        revision: str = "main",
        hf_token: str | None = None,
        max_new_tokens: int = 4096,
        max_input_tokens: int = 24576,
        enable_thinking: bool = False,
    ) -> None:
        import threading

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not model_id or "/" not in model_id:
            raise ValueError("model_id must be a Hugging Face repository ID such as Qwen/Qwen3-1.7B.")
        if max_new_tokens < 1 or max_input_tokens < 1:
            raise ValueError("Token limits must be positive.")

        # Resolve mutable refs (for example, "main") to an immutable commit
        # before loading any files. Both tokenizer and weights must use this SHA.
        from huggingface_hub import HfApi

        model_info = HfApi(token=hf_token).model_info(
            repo_id=model_id,
            revision=revision,
        )
        resolved_revision = getattr(model_info, "sha", None)
        if not isinstance(resolved_revision, str) or not resolved_revision:
            raise RuntimeError(
                f"Could not resolve an immutable Hugging Face commit for {model_id}@{revision}."
            )

        hf_kwargs: dict[str, str] = {"revision": resolved_revision}
        if hf_token:
            hf_kwargs["token"] = hf_token

        self.model_id = model_id
        self.requested_revision = revision
        self.resolved_revision = resolved_revision
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self.enable_thinking = enable_thinking
        self._torch = torch
        self._lock = threading.Lock()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **hf_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            low_cpu_mem_usage=True,
            **hf_kwargs,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        # Keep the Hub-resolved SHA as the authoritative revision. Transformers
        # may omit _commit_hash for some cached/offline loading paths.

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        generation: dict[str, object] = {
            "max_new_tokens": self.max_new_tokens,
            "use_cache": True,
            "repetition_penalty": 1.05,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation.update({
                "do_sample": True,
                "temperature": max(0.05, float(temperature)),
                "top_p": 0.8,
                "top_k": 20,
            })
        else:
            generation["do_sample"] = False

        with self._lock, self._torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation)

        prompt_tokens = inputs["input_ids"].shape[-1]
        new_tokens = output_ids[0][prompt_tokens:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if not answer:
            raise RuntimeError(f"{self.model_id} returned an empty completion.")
        return answer
