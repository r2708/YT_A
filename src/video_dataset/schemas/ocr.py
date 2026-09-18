"""OCR schemas."""

from __future__ import annotations

from pydantic import Field

from video_dataset.schemas.common import BaseSchema, ConfidenceSource


class OCRDetection(BaseSchema):
    detection_id: str
    frame_id: str
    scene_id: str
    timestamp: float = Field(ge=0)
    text: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_source: ConfidenceSource = ConfidenceSource.OCR_SCORE
    bbox: list[float] = Field(default_factory=list)  # [x_min, y_min, x_max, y_max] in frame pixels


class OCRTrack(BaseSchema):
    """The same on-screen text persisting across consecutive frames."""

    track_id: str
    text: str
    normalized_text: str
    first_seen: float
    last_seen: float
    frame_ids: list[str]
    scene_ids: list[str]
    detection_ids: list[str]
    num_detections: int
    mean_confidence: float | None = None
    bbox: list[float] = Field(default_factory=list)  # union bbox


class OCRResult(BaseSchema):
    video_id: str
    engine: str
    frames_processed: int
    detections: list[OCRDetection] = Field(default_factory=list)
    tracks: list[OCRTrack] = Field(default_factory=list)
