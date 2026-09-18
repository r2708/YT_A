"""Scene / frame / clip schemas."""

from __future__ import annotations

from pydantic import Field

from video_dataset.schemas.common import BaseSchema, StrEnum, TimeSpan


class TransitionType(StrEnum):
    START = "start"  # first scene of the video
    CUT = "cut"  # hard cut detected by content change
    FADE = "fade"  # fade in/out detected by luminance threshold
    ADAPTIVE = "adaptive"  # adaptive detector (rolling-average) cut
    SPLIT = "split"  # artificial split of an over-long shot (not a real boundary)


class Scene(TimeSpan):
    scene_id: str
    video_id: str
    index: int = Field(ge=0)
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    transition_in: TransitionType = TransitionType.CUT
    detector: str | None = None
    detector_score: float | None = None  # content_val at the boundary when available
    split_from: str | None = None  # scene_id of the original shot if this was a SPLIT

    @property
    def scene_duration(self) -> float:
        return self.duration


class SamplingReason(StrEnum):
    SCENE_START = "scene_start"
    SCENE_MIDDLE = "scene_middle"
    SCENE_END = "scene_end"
    VISUAL_CHANGE = "visual_change"
    MOTION_PEAK = "motion_peak"
    UNIFORM = "uniform"


class Frame(BaseSchema):
    frame_id: str
    video_id: str
    scene_id: str
    timestamp: float = Field(ge=0)
    frame_index: int = Field(ge=0)
    frame_path: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    sampling_reason: SamplingReason = SamplingReason.UNIFORM
    change_score: float | None = None  # visual change vs previous analysed frame (0..1)
    motion_score: float | None = None  # mean optical-flow magnitude (px/frame at analysis res)


class ScanSample(BaseSchema):
    """One analysed time point from the scene scan pass (low-res decode)."""

    timestamp: float
    frame_index: int
    change: float  # 0..1 histogram/pixel change vs previous sample
    brightness: float  # 0..255 mean gray
    flow_dx: float | None = None  # median horizontal flow (px at analysis res, +x = content moves right)
    flow_dy: float | None = None
    flow_mag: float | None = None  # mean flow magnitude
    flow_div: float | None = None  # divergence proxy: >0 expanding (zoom in / forward), <0 contracting


class SceneScan(BaseSchema):
    scene_id: str
    analysis_width: int
    analysis_fps: float
    samples: list[ScanSample] = Field(default_factory=list)


class Clip(TimeSpan):
    clip_id: str
    video_id: str
    scene_id: str
    clip_path: str
    frame_ids: list[str] = Field(default_factory=list)


class SceneDetectionResult(BaseSchema):
    video_id: str
    detector: str
    threshold: float | None = None
    duration: float
    scenes: list[Scene]


class FrameSamplingResult(BaseSchema):
    video_id: str
    frames: list[Frame]
    scans: list[SceneScan] = Field(default_factory=list)
    clips: list[Clip] = Field(default_factory=list)
