"""Scene / frame / clip schemas."""

from __future__ import annotations

from pydantic import Field

from video_dataset.schemas.common import BaseSchema, StrEnum, TimeSpan


class TransitionType(StrEnum):
    START = "start"  # first scene of the video
    CUT = "cut"  # hard cut detected by content change
    FADE = "fade"  # fade in/out detected by luminance threshold
    DISSOLVE = "dissolve"  # gradual transition: the content change is spread over several frames
    ADAPTIVE = "adaptive"  # adaptive detector (rolling-average) cut
    SPLIT = "split"  # artificial split of an over-long shot (not a real boundary)


REAL_TRANSITIONS = (TransitionType.CUT, TransitionType.FADE, TransitionType.DISSOLVE, TransitionType.ADAPTIVE)


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
    dhash: str | None = None  # 64-bit perceptual hash (hex) of the written image
    duplicate_of: str | None = None  # frame_id of an earlier frame with (nearly) the same hash


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
    dhash: str | None = None  # perceptual hash of the analysed frame (video fingerprint input)


class SceneScan(BaseSchema):
    scene_id: str
    analysis_width: int
    analysis_fps: float
    samples: list[ScanSample] = Field(default_factory=list)


class Clip(TimeSpan):
    """``start_time`` / ``end_time`` are the *requested* half-open interval of the source video. What
    the file really holds is measured after cutting: ``media_duration`` (ffprobe) and, when the
    boundary check ran, ``actual_start`` / ``actual_end`` (the source timestamps of the file's first
    and last frames, found by matching their perceptual hashes against source frames around the
    requested boundaries)."""

    clip_id: str
    video_id: str
    scene_id: str
    clip_path: str
    frame_ids: list[str] = Field(default_factory=list)
    media_duration: float | None = None  # measured length of the file (stream copy cuts at keyframes)
    exact: bool = True  # media_duration matches end_time - start_time within clip_tolerance_seconds
    codec: str | None = None  # "copy" | "libx264": how this file was produced
    fps: float | None = None  # measured on the written file
    width: int | None = None
    height: int | None = None
    normalized: bool = False  # re-encoded to frame_sampling.target_fps / target size
    actual_start: float | None = None  # source time of the first frame in the file (boundary check)
    actual_end: float | None = None  # source time of the last frame in the file (boundary check)
    boundary_error_seconds: float | None = None  # max(|actual_start - start_time|, |actual_end - end_time|)
    boundary_check: str = "not_run"  # passed | failed | not_run | error (frame-hash comparison with the source)


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
    fingerprint: list[str] = Field(default_factory=list)  # distinct scan-frame hashes: the video's perceptual fingerprint
    duplicate_frames_removed: int = 0
    near_duplicate_of: str | None = None  # video_id with a matching fingerprint already in the state DB
    near_duplicate_similarity: float | None = None
