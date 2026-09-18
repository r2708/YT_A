"""Vision / multimodal analysis schemas."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from video_dataset.schemas.common import BaseSchema, ConfidenceSource, StrEnum

UNKNOWN = "unknown"


class Setting(StrEnum):
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ShotType(StrEnum):
    EXTREME_WIDE = "extreme_wide"
    WIDE = "wide"
    MEDIUM = "medium"
    CLOSE_UP = "close_up"
    EXTREME_CLOSE_UP = "extreme_close_up"
    UNKNOWN = "unknown"


class CameraAngle(StrEnum):
    EYE_LEVEL = "eye_level"
    LOW = "low"
    HIGH = "high"
    OVERHEAD = "overhead"
    AERIAL = "aerial"
    DUTCH = "dutch"
    UNKNOWN = "unknown"


class CameraMovement(StrEnum):
    STATIC = "static"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    TILT_UP = "tilt_up"
    TILT_DOWN = "tilt_down"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    TRACKING = "tracking"
    TRACKING_FORWARD = "tracking_forward"
    TRACKING_BACKWARD = "tracking_backward"
    HANDHELD = "handheld"
    COMPLEX = "complex"
    UNKNOWN = "unknown"


class Environment(BaseSchema):
    location: str | None = None
    setting: Setting = Setting.UNKNOWN
    weather: str | None = None
    lighting: str | None = None
    time_of_day: str | None = None
    background: str | None = None


class ObjectAnnotation(BaseSchema):
    name: str
    attributes: list[str] = Field(default_factory=list)
    location: str | None = None  # e.g. "center", "left foreground"
    count: int | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_source: ConfidenceSource = ConfidenceSource.UNAVAILABLE


class PersonAnnotation(BaseSchema):
    """People are described by count / visible action / clothing only. No identities."""

    count: int | None = None
    actions: list[str] = Field(default_factory=list)
    clothing: list[str] = Field(default_factory=list)
    body_position: str | None = None
    interactions: list[str] = Field(default_factory=list)


class CameraAnnotation(BaseSchema):
    shot_type: ShotType = ShotType.UNKNOWN
    camera_angle: CameraAngle = CameraAngle.UNKNOWN
    movement: CameraMovement = CameraMovement.UNKNOWN
    zoom: str | None = None
    stability: str | None = None  # "static" | "handheld" | "stabilized" | None
    is_aerial: bool | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_source: ConfidenceSource = ConfidenceSource.UNAVAILABLE


class VisualStyle(BaseSchema):
    composition: str | None = None
    lighting: str | None = None
    color: str | None = None
    depth_of_field: str | None = None
    framing: str | None = None
    perspective: str | None = None
    motion: str | None = None
    transitions: str | None = None


class Measurements(BaseSchema):
    """Directly measured, model-free signals. Always grounded."""

    brightness_mean: float | None = None  # 0..255
    brightness_std: float | None = None
    contrast: float | None = None  # 0..1 (std / 128 clipped)
    saturation_mean: float | None = None  # 0..255
    colorfulness: float | None = None  # Hasler & Süsstrunk metric
    dominant_colors: list[str] = Field(default_factory=list)  # hex strings
    color_temperature: str | None = None  # warm | neutral | cool
    edge_density: float | None = None  # 0..1
    sharpness: float | None = None  # variance of Laplacian (normalized)
    motion_magnitude: float | None = None  # mean optical flow magnitude (analysis px/frame)
    camera_motion_label: str | None = None  # from optical flow interpretation
    camera_motion_consistency: float | None = None  # 0..1 fraction of samples agreeing with label
    brightness_trend: str | None = None  # "brightening" | "darkening" | "stable"
    lighting_level: str | None = None  # dark | dim | normal | bright


class FrameAnalysis(BaseSchema):
    frame_id: str
    scene_id: str
    timestamp: float
    caption: str | None = None
    objects: list[ObjectAnnotation] = Field(default_factory=list)
    people: PersonAnnotation | None = None
    actions: list[str] = Field(default_factory=list)
    environment: Environment | None = None
    camera: CameraAnnotation | None = None
    measurements: Measurements | None = None
    provider: str | None = None
    model: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_source: ConfidenceSource = ConfidenceSource.UNAVAILABLE


class VerificationVerdict(StrEnum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"  # verifier could not judge -> no confidence contribution


class VerificationResult(BaseSchema):
    claim: str
    verdict: VerificationVerdict = VerificationVerdict.UNKNOWN
    score: float | None = Field(default=None, ge=0, le=1)
    rationale: str | None = None
    verifier: str | None = None
    evidence_frame_ids: list[str] = Field(default_factory=list)


class SceneAnalysis(BaseSchema):
    scene_id: str
    video_id: str
    start_time: float
    end_time: float
    summary: str | None = None
    environment: Environment = Field(default_factory=Environment)
    objects: list[ObjectAnnotation] = Field(default_factory=list)
    people: PersonAnnotation | None = None
    actions: list[str] = Field(default_factory=list)
    camera: CameraAnnotation = Field(default_factory=CameraAnnotation)
    visual_style: VisualStyle = Field(default_factory=VisualStyle)
    temporal_progression: str | None = None  # what changes from the start to the end of the scene
    frame_analyses: list[FrameAnalysis] = Field(default_factory=list)
    measurements: Measurements | None = None
    frame_ids: list[str] = Field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_source: ConfidenceSource = ConfidenceSource.UNAVAILABLE
    verification: VerificationResult | None = None
    enrichments: dict[str, Any] = Field(default_factory=dict)  # e.g. CLIP zero-shot scores
    errors: list[str] = Field(default_factory=list)


class VisionResult(BaseSchema):
    video_id: str
    provider: str
    model: str | None = None
    scenes: list[SceneAnalysis] = Field(default_factory=list)
