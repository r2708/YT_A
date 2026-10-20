"""Shared schema primitives used across all pipeline stages."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Version of the exported record schemas. Bump the minor part whenever a field is added, the major
# part whenever a field changes meaning or is removed; every exported record and manifest carries it.
SCHEMA_VERSION = "2.1.0"


class StrEnum(str, Enum):
    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.value)


class ConfidenceSource(StrEnum):
    """Where a confidence value came from. Never invent a number without one of these."""

    VERIFIER = "verifier"  # an explicit verification pass (VLM / heuristic verifier)
    MODEL_SELF_REPORT = "model_self_report"  # the generating model reported it (weak signal)
    MEASUREMENT = "measurement"  # derived from a direct signal measurement (optical flow, pixel stats)
    ASR_LOGPROB = "asr_logprob"  # derived from ASR log-probabilities / no-speech probability
    OCR_SCORE = "ocr_score"  # OCR recognition score
    DETECTOR_SCORE = "detector_score"  # classifier / detector probability
    INTERVAL_ALGEBRA = "interval_algebra"  # deterministic from timestamps; inherits evidence confidence
    DERIVED_MIN = "derived_min"  # min() over the confidences of the supporting records
    UNAVAILABLE = "unavailable"  # no reliable estimate exists; value must be None


class BaseSchema(BaseModel):
    """Base for every record. Extra keys are rejected so schema drift is caught early."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, use_enum_values=True)

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=False)


class TimeSpan(BaseSchema):
    """An interval in seconds relative to the start of the source video.

    Convention (used everywhere in the pipeline): ``start_time`` is inclusive, ``end_time`` is the
    first instant *not* covered, i.e. half-open ``[start_time, end_time)``. Consecutive shots share
    their boundary time, a clip cut for ``[s, e)`` holds ``round((e - s) * fps)`` frames, and
    ``contains_time`` / ``overlaps`` treat the end as exclusive up to the tolerance."""

    start_time: float = Field(ge=0.0)
    end_time: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _check_order(self) -> TimeSpan:
        if self.end_time < self.start_time:
            raise ValueError(f"end_time ({self.end_time}) must be >= start_time ({self.start_time})")
        return self

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def overlaps(self, other: TimeSpan, tolerance: float = 0.0) -> bool:
        return self.start_time < other.end_time + tolerance and other.start_time < self.end_time + tolerance

    def contains_time(self, t: float, tolerance: float = 0.0) -> bool:
        return self.start_time - tolerance <= t <= self.end_time + tolerance


class Provenance(BaseSchema):
    """Records which component produced a record so that models can be swapped and compared."""

    source: str
    model: str | None = None
    provider: str | None = None
    version: str | None = None
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
