"""Anthropic Claude client (text + images, JSON-schema structured output)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from video_dataset.llm.base import LLMResponse, resolve_api_key
from video_dataset.llm.images import encode_image_base64
from video_dataset.utils.logging import get_logger

log = get_logger("llm.anthropic")

_FALLBACK_MODELS = ("claude-opus-5", "claude-fable-5", "claude-fable-5-1", "claude-mythos-5", "claude-mythos-5-1")


class AnthropicClient:
    name = "anthropic"
    supports_images = True

    def __init__(
        self,
        model: str = "claude-opus-5",
        api_key_env: str | None = None,
        timeout: float = 120.0,
        base_url: str | None = None,
        image_max_side: int = 1024,
        use_fallbacks: bool = True,
        **_: Any,
    ):
        import anthropic

        key = resolve_api_key(api_key_env, "ANTHROPIC_API_KEY")
        kwargs: dict[str, Any] = {"timeout": timeout, "max_retries": 2}
        if key:
            kwargs["api_key"] = key
        if base_url:
            kwargs["base_url"] = base_url
        self.client = anthropic.Anthropic(**kwargs)
        self.model = model
        self.image_max_side = image_max_side
        self.use_fallbacks = use_fallbacks and any(model.startswith(m) for m in _FALLBACK_MODELS)

    def _content(self, prompt: str, images: list[Path] | None) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for p in images or []:
            data, media = encode_image_base64(p, self.image_max_side)
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": media, "data": data}})
        blocks.append({"type": "text", "text": prompt})
        return blocks

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        images: list[Path] | None = None,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        import anthropic

        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": self._content(prompt, images)}],
        }
        if system:
            params["system"] = system
        if json_schema:
            params["output_config"] = {"format": {"type": "json_schema", "schema": json_schema}}

        response = None
        if self.use_fallbacks:
            try:
                response = self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **params
                )
            except (TypeError, anthropic.BadRequestError) as exc:
                log.debug("fallbacks unsupported here (%s); retrying without", exc)
                response = None
        if response is None:
            response = self.client.messages.create(**params)

        refusal = getattr(response, "stop_reason", None) == "refusal"
        text = "".join(getattr(b, "text", "") for b in response.content if getattr(b, "type", "") == "text")
        usage = {}
        if getattr(response, "usage", None) is not None:
            usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
        return LLMResponse(text=text, refusal=refusal, usage=usage, raw=response, model=getattr(response, "model", self.model))
