"""Mock LLM client for tests. Returns queued responses or a callable's output."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from video_dataset.llm.base import LLMResponse


class MockLLMClient:
    name = "mock"
    supports_images = True

    def __init__(self, responses: list[str] | None = None, handler: Callable[[str, list[Path] | None], str] | None = None, model: str = "mock"):
        self.responses = list(responses or [])
        self.handler = handler
        self.model = model
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append({"prompt": prompt, "system": system, "images": list(images or []), "json_schema": json_schema})
        if self.handler is not None:
            return LLMResponse(text=self.handler(prompt, images), model=self.model)
        text = self.responses.pop(0) if self.responses else "{}"
        return LLMResponse(text=text, model=self.model)
