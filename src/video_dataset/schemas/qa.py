"""Temporal question-answer schemas."""

from __future__ import annotations

from pydantic import Field, model_validator

from video_dataset.schemas.common import BaseSchema, ConfidenceSource, StrEnum
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


class QAResult(BaseSchema):
    video_id: str
    generator: str
    questions: list[QARecord] = Field(default_factory=list)
