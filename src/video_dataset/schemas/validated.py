"""Validated-record wrappers written by the VALIDATION stage."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from video_dataset.schemas.common import BaseSchema
from video_dataset.schemas.events import Event, TemporalRelation
from video_dataset.schemas.qa import QARecord
from video_dataset.schemas.quality import QualityScore, ValidationInfo
from video_dataset.schemas.vision import SceneAnalysis


class ValidatedScene(BaseSchema):
    analysis: SceneAnalysis
    validation: ValidationInfo
    quality: QualityScore


class ValidatedEvent(BaseSchema):
    event: Event
    validation: ValidationInfo
    quality: QualityScore


class ValidatedRelation(BaseSchema):
    relation: TemporalRelation
    validation: ValidationInfo
    quality: QualityScore


class ValidatedVideo(BaseSchema):
    video_id: str
    duration: float
    scenes: list[ValidatedScene] = Field(default_factory=list)
    events: list[ValidatedEvent] = Field(default_factory=list)
    relations: list[ValidatedRelation] = Field(default_factory=list)
    qa: list[QARecord] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
