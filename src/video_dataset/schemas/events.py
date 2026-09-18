"""Temporal event and relation schemas - the core of the temporal timeline."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from video_dataset.schemas.common import BaseSchema, ConfidenceSource, StrEnum, TimeSpan
from video_dataset.schemas.vision import VerificationResult


class EventType(StrEnum):
    ACTION = "action"  # something an entity does (from vision analysis)
    APPEARANCE = "appearance"  # an entity/object becomes visible in a scene
    TRANSITION = "transition"  # shot change (cut / fade)
    CAMERA = "camera"  # camera movement (measured from optical flow or reported by VLM)
    SPEECH = "speech"  # a spoken utterance (from ASR)
    TEXT_ON_SCREEN = "text_on_screen"  # OCR text appears
    SOUND = "sound"  # non-speech audio event
    STATE_CHANGE = "state_change"  # environment / lighting / setting changes between scenes


class EventSource(StrEnum):
    VISION = "vision"
    SCENE_DETECTOR = "scene_detector"
    MOTION = "motion"
    ASR = "asr"
    OCR = "ocr"
    AUDIO = "audio"
    DERIVED = "derived"


class Event(TimeSpan):
    event_id: str
    video_id: str
    event_type: EventType
    event: str  # natural-language description, e.g. "The car enters a tunnel."
    entities: list[str] = Field(default_factory=list)
    action: str | None = None
    scene_ids: list[str] = Field(default_factory=list)
    frame_ids: list[str] = Field(default_factory=list)
    clip_ids: list[str] = Field(default_factory=list)
    source: EventSource
    source_ids: list[str] = Field(default_factory=list)  # ids of the transcript segment / ocr track / etc.
    confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_source: ConfidenceSource = ConfidenceSource.UNAVAILABLE
    boundary_precision: str = "scene"  # "scene" | "measured" | "asr" | "frame" | "approximate"
    attributes: dict[str, Any] = Field(default_factory=dict)
    verification: VerificationResult | None = None


class RelationType(StrEnum):
    BEFORE = "BEFORE"
    AFTER = "AFTER"
    DURING = "DURING"
    OVERLAPS = "OVERLAPS"
    STARTS = "STARTS"
    ENDS = "ENDS"
    CONTINUES = "CONTINUES"
    INTERRUPTS = "INTERRUPTS"
    CAUSES = "CAUSES"
    CHANGES_TO = "CHANGES_TO"


class TemporalRelation(BaseSchema):
    relation_id: str
    video_id: str
    event_a: str  # event_id
    event_b: str  # event_id
    relation: RelationType
    event_a_text: str
    event_b_text: str
    gap_seconds: float | None = None  # signed: b.start - a.end for BEFORE/AFTER
    confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_source: ConfidenceSource = ConfidenceSource.UNAVAILABLE
    derivation: str = "interval_algebra"  # how the relation was inferred
    rationale: str | None = None


class Timeline(BaseSchema):
    video_id: str
    duration: float
    events: list[Event] = Field(default_factory=list)
    relations: list[TemporalRelation] = Field(default_factory=list)

    def event_by_id(self, event_id: str) -> Event | None:
        for e in self.events:
            if e.event_id == event_id:
                return e
        return None

    def events_at(self, t: float, tolerance: float = 0.0) -> list[Event]:
        return [e for e in self.events if e.contains_time(t, tolerance)]
