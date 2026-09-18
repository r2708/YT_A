from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from video_dataset.utils.logging import get_logger

log = get_logger("llm")


@dataclass
class LLMResponse:
    text: str
    refusal: bool = False
    usage: dict[str, Any] = field(default_factory=dict)
    raw: Any = None
    model: str | None = None


class LLMClient(Protocol):
    name: str
    model: str
    supports_images: bool

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        images: list[Path] | None = None,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse: ...


def resolve_api_key(env_name: str | None, default_env: str) -> str | None:
    return os.environ.get(env_name or default_env)


def create_llm_client(
    provider: str,
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    timeout: float = 120.0,
    **kwargs: Any,
) -> LLMClient:
    provider = (provider or "none").lower()
    if provider == "anthropic":
        from video_dataset.llm.anthropic_client import AnthropicClient

        return AnthropicClient(model=model or "claude-opus-5", api_key_env=api_key_env, timeout=timeout, base_url=base_url, **kwargs)
    if provider in ("openai", "openai_compatible"):
        from video_dataset.llm.openai_client import OpenAICompatibleClient

        return OpenAICompatibleClient(model=model or "gpt-4o", base_url=base_url, api_key_env=api_key_env, timeout=timeout, **kwargs)
    if provider == "mock":
        from video_dataset.llm.mock import MockLLMClient

        return MockLLMClient(model=model or "mock")
    raise ValueError(f"Unknown LLM provider '{provider}'")
