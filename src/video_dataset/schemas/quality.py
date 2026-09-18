"""Quality scoring and validation schemas."""

from __future__ import annotations

from pydantic import Field

from video_dataset.schemas.common import BaseSchema, StrEnum


class QualityScore(BaseSchema):
    """Component scores are None when they cannot be estimated from real signals.

    `overall` is the mean of the available components only, and `components_used`
    lists which ones contributed, so consumers can tell a measured 0.9 from a guess.
    """

    grounding: float | None = Field(default=None, ge=0, le=1)
    temporal_accuracy: float | None = Field(default=None, ge=0, le=1)
    description_quality: float | None = Field(default=None, ge=0, le=1)
    overall: float | None = Field(default=None, ge=0, le=1)
    components_used: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @classmethod
    def from_components(
        cls,
        grounding: float | None = None,
        temporal_accuracy: float | None = None,
        description_quality: float | None = None,
        notes: list[str] | None = None,
    ) -> QualityScore:
        comps = {
            "grounding": grounding,
            "temporal_accuracy": temporal_accuracy,
            "description_quality": description_quality,
        }
        used = [k for k, v in comps.items() if v is not None]
        values = [float(v) for v in comps.values() if v is not None]
        overall = round(sum(values) / len(values), 4) if values else None
        return cls(
            grounding=grounding,
            temporal_accuracy=temporal_accuracy,
            description_quality=description_quality,
            overall=overall,
            components_used=used,
            notes=notes or [],
        )


class ValidationStatus(StrEnum):
    ACCEPTED = "accepted"
    REVIEW = "review"
    REJECTED = "rejected"
    DUPLICATE = "duplicate"


class ValidationInfo(BaseSchema):
    status: ValidationStatus
    issues: list[str] = Field(default_factory=list)
    checks: dict[str, bool | None] = Field(default_factory=dict)  # None = check not applicable / not run
    verifier_score: float | None = None
    duplicate_of: str | None = None
