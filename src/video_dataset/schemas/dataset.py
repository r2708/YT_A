"""Final dataset record schemas (what is exported to JSONL / Parquet).

Every record type shares the same envelope so one loader works for all of them:
``record_id``, ``record_type``, ``video_id``, ``start_time`` / ``end_time`` (frames: ``timestamp``),
``provider``, ``confidence`` (+ ``confidence_source``), ``quality``, ``validation``,
``split``, ``tier`` (+ ``tier_reasons``) and ``subsets``.
"""

from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, Field, model_validator

from video_dataset.schemas.common import BaseSchema, ConfidenceSource
from video_dataset.schemas.curation import HardNegative, Split, Tier
from video_dataset.schemas.quality import QualityScore, ValidationInfo
from video_dataset.schemas.vision import (
    CameraAnnotation,
    Environment,
    Measurements,
    SubjectMotion,
    VisualStyle,
    camera_fields_from_style,
)


class _Curated(BaseSchema):
    """Fields filled by the export / aggregation step for every record type."""

    split: Split | None = None  # train / validation / test, assigned per source video (no leakage across scenes)
    tier: Tier | None = None  # gold / silver / bronze, derived from validation status + quality + confidence provenance
    tier_reasons: list[str] = Field(default_factory=list)  # why the record did not reach a higher tier
    subsets: list[str] = Field(default_factory=list)  # named subsets the record belongs to, e.g. ["cinematic"]


class FrameCaptionRecord(_Curated):
    record_id: str
    record_type: str = "frame"
    video_id: str
    scene_id: str
    clip_id: str | None = None  # the clip this frame belongs to (video -> clip -> frame); filled at export time
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


class ClipCaptionRecord(_Curated):
    record_id: str
    record_type: str = "clip"
    video_id: str
    scene_id: str
    clip_id: str
    clip_path: str
    start_time: float
    end_time: float
    media_duration: float | None = None  # measured length of the clip file
    exact: bool = True  # the file covers exactly start_time..end_time (within frame_sampling.clip_tolerance_seconds)
    description: str  # raw analysis description (what the analyzer saw / measured)
    generation_prompt: str | None = None  # the same shot as a text-to-video prompt, built from the structured fields only
    actions: list[str] = Field(default_factory=list)
    camera: CameraAnnotation | None = None
    subject_motion: SubjectMotion | None = None  # measured subject motion separated from the camera motion
    transcript: str | None = None
    frame_ids: list[str] = Field(default_factory=list)
    provider: str | None = None
    confidence: float | None = None
    confidence_source: ConfidenceSource = ConfidenceSource.UNAVAILABLE
    quality: QualityScore | None = None
    validation: ValidationInfo | None = None
    hard_negatives: list[HardNegative] = Field(default_factory=list)  # captions that are wrong for this clip


class SceneRecord(_Curated):
    record_id: str
    record_type: str = "scene"
    video_id: str
    scene_id: str
    # `start` / `end` were the field names of earlier exports; they are still accepted on input
    start_time: float = Field(validation_alias=AliasChoices("start_time", "start"))
    end_time: float = Field(validation_alias=AliasChoices("end_time", "end"))
    duration: float
    summary: str  # raw analysis description
    generation_prompt: str | None = None  # text-to-video prompt assembled from the structured fields (no free-form claims)
    environment: Environment
    objects: list[str] = Field(default_factory=list)
    object_details: list[dict[str, Any]] = Field(default_factory=list)
    people: dict[str, Any] | None = None
    actions: list[str] = Field(default_factory=list)
    camera: CameraAnnotation
    subject_motion: SubjectMotion | None = None
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
    hard_negatives: list[HardNegative] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _camera_from_style(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("camera"), dict):
            data = {**data, "camera": camera_fields_from_style(data["camera"], data.get("visual_style"))}
        return data


class EventRecord(_Curated):
    record_id: str
    record_type: str = "event"
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


class VideoDescriptionRecord(_Curated):
    """A generation-oriented description of the whole video or a multi-scene segment."""

    record_id: str
    record_type: str = "video_description"
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
