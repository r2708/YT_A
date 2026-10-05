"""Temporal question-answer schemas."""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from video_dataset.schemas.common import BaseSchema, ConfidenceSource, StrEnum
from video_dataset.schemas.curation import HardNegative, Split, Tier
from video_dataset.schemas.quality import QualityScore, ValidationInfo


class QAType(StrEnum):
    TIMESTAMP = "timestamp"
    BEFORE_AFTER = "before_after"
    TEMPORAL_ORDERING = "temporal_ordering"
    DURATION = "duration"
    EVENT_LOCALIZATION = "event_localization"
    STATE_CHANGE = "state_change"
    MULTI_EVENT = "multi_event"
    LONG_RANGE = "long_range"


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Evidence(BaseSchema):
    start_time: float = Field(ge=0)
    end_time: float = Field(ge=0)
    timestamps: list[float] = Field(default_factory=list)
    scene_ids: list[str] = Field(default_factory=list)
    event_ids: list[str] = Field(default_factory=list)
    frame_ids: list[str] = Field(default_factory=list)
    relation_ids: list[str] = Field(default_factory=list)
    transcript_segment_ids: list[str] = Field(default_factory=list)
    ocr_track_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> Evidence:
        if self.end_time < self.start_time:
            raise ValueError("evidence end_time must be >= start_time")
        if not (self.event_ids or self.scene_ids or self.timestamps):
            raise ValueError("evidence must reference at least one event, scene or timestamp")
        return self


class QARecord(BaseSchema):
    question_id: str
    record_id: str | None = None  # always equal to question_id; present so every exported record has the same key
    record_type: str = "temporal_qa"  # temporal_qa | long_video_qa (set by the export builder)
    video_id: str
    type: QAType
    question: str
    answer: str
    evidence: Evidence
    difficulty: Difficulty = Difficulty.MEDIUM
    confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_source: ConfidenceSource = ConfidenceSource.UNAVAILABLE
    generator: str = "template"
    template_id: str | None = None
    options: list[str] = Field(default_factory=list)  # for ordering / multiple-choice style questions
    is_long_range: bool = False
    quality: QualityScore | None = None
    validation: ValidationInfo | None = None
    hard_negatives: list[HardNegative] = Field(default_factory=list)  # rule-built wrong answers (export only)
    split: Split | None = None  # train / validation / test, assigned per source video
    tier: Tier | None = None  # gold / silver / bronze, derived from validation + quality
    tier_reasons: list[str] = Field(default_factory=list)
    subsets: list[str] = Field(default_factory=list)  # e.g. ["cinematic"]

    @model_validator(mode="before")
    @classmethod
    def _fill_record_id(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("record_id") and data.get("question_id"):
            data = {**data, "record_id": data["question_id"]}
        return data


class QAResult(BaseSchema):
    video_id: str
    generator: str
    questions: list[QARecord] = Field(default_factory=list)
