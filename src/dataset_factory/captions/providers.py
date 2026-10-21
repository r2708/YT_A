"""Caption review providers behind one interface. Only ``none`` ships: it makes no calls and
returns nothing, so the pipeline runs end to end without credentials. A real provider (remote
API or local multimodal model) registers with ``register_provider`` and must return
``ProviderOutput`` built from *strictly validated* JSON; invalid output never replaces a caption.

Nothing here sends frames or text anywhere unless a provider that does so is configured
explicitly in ``provider.name``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from dataset_factory.config import ProviderConfig
from dataset_factory.schema.canonical import CanonicalRecord


class ProviderOutput(BaseModel):
    """What a provider must return. ``observed`` facts are separated from ``assumed`` ones."""

    model_config = ConfigDict(extra="forbid")
    caption: str = Field(min_length=1)
    observed: list[str] = Field(default_factory=list)
    assumed: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    verdict: str = Field(pattern="^(supported|contradicted|uncertain)$")
    confidence: float = Field(ge=0.0, le=1.0)


@dataclass
class ReviewRequest:
    record: CanonicalRecord
    caption: str
    frame_paths: list[str] = field(default_factory=list)


@dataclass
class ReviewResult:
    called: bool
    output: ProviderOutput | None
    raw: str = ""
    error: str | None = None


class CaptionProvider(Protocol):
    name: str
    model: str | None

    def review(self, request: ReviewRequest) -> ReviewResult: ...


class NoneProvider:
    """Reuses source annotations; never calls anything."""

    name = "none"
    model = None

    def __init__(self, cfg: ProviderConfig) -> None:
        self.cfg = cfg

    def review(self, request: ReviewRequest) -> ReviewResult:
        return ReviewResult(called=False, output=None)


_REGISTRY: dict[str, Callable[[ProviderConfig], Any]] = {"none": NoneProvider}


def register_provider(name: str, factory: Callable[[ProviderConfig], Any]) -> None:
    _REGISTRY[name] = factory


def available_providers() -> list[str]:
    return sorted(_REGISTRY)


def make_provider(cfg: ProviderConfig, offline: bool) -> Any:
    name = (cfg.name or "none").lower()
    if offline and name != "none":
        raise RuntimeError(f"offline mode: provider '{name}' may not run (set offline: false and configure it deliberately)")
    if name not in _REGISTRY:
        raise RuntimeError(f"unknown caption provider '{name}'; available: {', '.join(available_providers())}")
    return _REGISTRY[name](cfg)


def parse_provider_json(raw: str) -> tuple[ProviderOutput | None, str | None]:
    """Strict parse of a provider's raw text; (None, reason) on any problem."""
    try:
        data = json.loads(raw)
    except ValueError as exc:
        return None, f"invalid_json:{str(exc)[:60]}"
    try:
        return ProviderOutput.model_validate(data), None
    except ValidationError as exc:
        return None, "schema:" + "; ".join(f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in exc.errors())[:200]
