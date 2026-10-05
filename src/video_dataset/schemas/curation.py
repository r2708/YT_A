"""Dataset curation schemas: train/validation/test split, quality tiers, hard negatives, subsets.

These fields are derived from signals the pipeline already measured (validation status, quality
components, confidence provenance, aesthetic score). They never introduce new claims about the
video; they only organise the records so a consumer can filter and split them without leakage.
"""

from __future__ import annotations

from pydantic import Field

from video_dataset.schemas.common import BaseSchema, StrEnum


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class Tier(StrEnum):
    """Gold: accepted, high quality, grounded by a verifier or a direct measurement.
    Silver: accepted with a measured quality score above the silver threshold.
    Bronze: everything else that is still exportable (review status, unscored quality, ...)."""

    GOLD = "gold"
    SILVER = "silver"
    BRONZE = "bronze"


class HardNegative(BaseSchema):
    """A text that is deliberately wrong for the record it is attached to, built by a rule that
    knows *why* it is wrong (swapped order, shifted time, caption of another shot, ...)."""

    text: str
    kind: str  # swapped_order | shifted_time | wrong_duration | other_shot | attribute_swap | other_event
    source_ids: list[str] = Field(default_factory=list)  # scene / event ids the negative was built from
    note: str | None = None  # the measured difference that makes it a negative, e.g. "camera pan_left vs static"
