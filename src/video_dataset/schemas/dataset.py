"""Final dataset record schemas (what is exported to JSONL / Parquet)."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from video_dataset.schemas.common import BaseSchema, ConfidenceSource
from video_dataset.schemas.quality import QualityScore, ValidationInfo
from video_dataset.schemas.vision import CameraAnnotation, Environment, Measurements, VisualStyle


class FrameCaptionRecord(BaseSchema):
    record_id: str
    video_id: str
    scene_id: str
    frame_id: str
    frame_path: str
    timestamp: float
    caption: str
    caption_source: str = "scene_summary"  # frame_analysis | scene_summary | measurement
    objects: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    camera: CameraAnnotation | None = None
    environment: Environment | None = None
    measurements: Measurements | None = None
    ocr_text: list[str] = Field(default_factory=list)
    provider: str | None = None
    confidence: float | None = None
    confidence_source: ConfidenceSource = ConfidenceSource.UNAVAILABLE
    quality: QualityScore | None = None
    validation: ValidationInfo | None = None


class ClipCaptionRecord(BaseSchema):
    record_id: str
    video_id: str
    scene_id: str
    clip_id: str
    clip_path: str
    start_time: float
    end_time: float
    description: str
    actions: list[str] = Field(default_factory=list)
    camera: CameraAnnotation | None = None
    transcript: str | None = None
    frame_ids: list[str] = Field(default_factory=list)
    provider: str | None = None
    confidence: float | None = None
    confidence_source: ConfidenceSource = ConfidenceSource.UNAVAILABLE
    quality: QualityScore | None = None
    validation: ValidationInfo | None = None


class SceneRecord(BaseSchema):
    record_id: str
    video_id: str
    scene_id: str
    start: float
    end: float
    duration: float
    summary: str
    environment: Environment
    objects: list[str] = Field(default_factory=list)
    object_details: list[dict[str, Any]] = Field(default_factory=list)
    people: dict[str, Any] | None = None
    actions: list[str] = Field(default_factory=list)
    camera: CameraAnnotation
    visual_style: VisualStyle
    temporal_progression: str | None = None
    transcript: str | None = None
    ocr_text: list[str] = Field(default_factory=list)
    frame_ids: list[str] = Field(default_factory=list)
    clip_id: str | None = None
    measurements: Measurements | None = None
    provider: str | None = None
    confidence: float | None = None
    confidence_source: ConfidenceSource = ConfidenceSource.UNAVAILABLE
    quality: QualityScore | None = None
    validation: ValidationInfo | None = None


class EventRecord(BaseSchema):
    record_id: str
    video_id: str
    event_id: str
    event_type: str
    start_time: float
    end_time: float
    event: str
    entities: list[str] = Field(default_factory=list)
    action: str | None = None
    scene_ids: list[str] = Field(default_factory=list)
    frame_ids: list[str] = Field(default_factory=list)
    clip_ids: list[str] = Field(default_factory=list)
    source: str
    confidence: float | None = None
    confidence_source: ConfidenceSource = ConfidenceSource.UNAVAILABLE
    relations: list[dict[str, Any]] = Field(default_factory=list)  # outgoing relations
    quality: QualityScore | None = None
    validation: ValidationInfo | None = None


class VideoDescriptionRecord(BaseSchema):
    """A generation-oriented description of the whole video or a multi-scene segment."""

    record_id: str
    video_id: str
    start_time: float
    end_time: float
    scene_ids: list[str]
    subject: str | None = None
    environment: str | None = None
    action: str | None = None
    camera: str | None = None
    composition: str | None = None
    lighting: str | None = None
    motion: str | None = None
    temporal_progression: str
    transitions: str | None = None
    prompt: str  # a single flowing generation prompt assembled from the fields above
    shots: list[dict[str, Any]] = Field(default_factory=list)  # per-scene compact descriptors
    confidence: float | None = None
    confidence_source: ConfidenceSource = ConfidenceSource.UNAVAILABLE
    quality: QualityScore | None = None
    validation: ValidationInfo | None = None
