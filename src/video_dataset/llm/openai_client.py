"""OpenAI-compatible chat client: OpenAI, vLLM, Ollama, LM Studio, llama.cpp server..."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from video_dataset.llm.base import LLMResponse, resolve_api_key
from video_dataset.llm.images import encode_image_base64
from video_dataset.utils.logging import get_logger

log = get_logger("llm.openai_compatible")


class OpenAICompatibleClient:
    name = "openai_compatible"
    supports_images = True

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key_env: str | None = None,
        timeout: float = 120.0,
        image_max_side: int = 1024,
        json_mode: str = "auto",  # auto | json_schema | json_object | none
        **_: Any,
    ):
        from openai import OpenAI

        key = resolve_api_key(api_key_env, "OPENAI_API_KEY") or "not-needed"
        self.client = OpenAI(api_key=key, base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None, timeout=timeout, max_retries=2)
        self.model = model
        self.image_max_side = image_max_side
        self.json_mode = json_mode

    def _content(self, prompt: str, images: list[Path] | None) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for p in images or []:
            data, media = encode_image_base64(p, self.image_max_side)
            blocks.append({"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}})
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
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": self._content(prompt, images)})
        base: dict[str, Any] = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}

        attempts: list[dict[str, Any] | None] = []
        if json_schema and self.json_mode in ("auto", "json_schema"):
            attempts.append({"type": "json_schema", "json_schema": {"name": "annotation", "schema": json_schema, "strict": False}})
        if json_schema and self.json_mode in ("auto", "json_object"):
            attempts.append({"type": "json_object"})
        attempts.append(None)

        last_exc: Exception | None = None
        for fmt in attempts:
            params = dict(base)
            if fmt is not None:
                params["response_format"] = fmt
            try:
                resp = self.client.chat.completions.create(**params)
                choice = resp.choices[0]
                text = choice.message.content or ""
                usage = {}
                if resp.usage is not None:
                    usage = {"input_tokens": resp.usage.prompt_tokens, "output_tokens": resp.usage.completion_tokens}
                return LLMResponse(text=text, refusal=(choice.finish_reason == "content_filter"), usage=usage, raw=resp, model=resp.model)
            except Exception as exc:
                last_exc = exc
                if fmt is None:
                    raise
                log.debug("response_format %s rejected (%s); retrying with a simpler format", fmt.get("type"), exc)
        assert last_exc is not None
        raise last_exc
