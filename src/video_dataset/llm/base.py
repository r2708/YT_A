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
    requests_per_minute: float = 0.0,
    **kwargs: Any,
) -> LLMClient:
    """Build a client; with `requests_per_minute` > 0 it is wrapped in a process-wide token bucket
    shared by every caller using the same provider+model."""
    provider = (provider or "none").lower()
    client: LLMClient
    if provider == "anthropic":
        from video_dataset.llm.anthropic_client import AnthropicClient

        client = AnthropicClient(model=model or "claude-opus-5", api_key_env=api_key_env, timeout=timeout, base_url=base_url, **kwargs)
    elif provider in ("openai", "openai_compatible"):
        from video_dataset.llm.openai_client import OpenAICompatibleClient

        client = OpenAICompatibleClient(model=model or "gpt-4o", base_url=base_url, api_key_env=api_key_env, timeout=timeout, **kwargs)
    elif provider == "mock":
        from video_dataset.llm.mock import MockLLMClient

        client = MockLLMClient(model=model or "mock")
    else:
        raise ValueError(f"Unknown LLM provider '{provider}'")
    return rate_limited(client, requests_per_minute)


def rate_limited(client: LLMClient, requests_per_minute: float) -> LLMClient:
    from video_dataset.utils.ratelimit import RateLimitedLLMClient, get_rate_limiter

    limiter = get_rate_limiter(f"{client.name}:{client.model}", requests_per_minute)
    if limiter is None:
        return client
    log.info("rate limiting %s/%s to %.1f requests/minute", client.name, client.model, requests_per_minute)
    return RateLimitedLLMClient(client, limiter)  # type: ignore[return-value]
