"""Resolution policy: which bucket a source falls in and what size its training clip gets.

Driven by the ``resolution`` config section (``resolution.target`` is the one knob). The short edge
classifies the source, so portrait and landscape footage are treated alike and the aspect ratio is
always preserved. A source below the target is never upscaled unless ``allow_upscale`` is set, and
then the asset is labelled ``upscaled``. This is a dataset policy: a model-specific preprocessing
profile must still validate the final width / height / frame count."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from video_dataset.config import ResolutionConfig

_LADDER = (2160, 1440, 1080, 720)


@dataclass(frozen=True)
class ResolutionDecision:
    bucket: str
    action: str  # downscaled | kept | keep_native | keep_native_experimental | upscaled | excluded
    target_width: int
    target_height: int
    upscaled: bool
    include_in_primary_training: bool

    @property
    def excluded(self) -> bool:
        return self.action == "excluded"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _even(value: float) -> int:
    return max(2, int(round(value / 2.0)) * 2)


def target_bucket(target: int) -> str:
    return f"{int(target)}p"


def _native_bucket(short_edge: int) -> str:
    for step in _LADDER:
        if short_edge >= step:
            return f"native_{step}"
    return "native_480_576"


def choose_resolution(width: int, height: int, cfg: ResolutionConfig | None = None) -> ResolutionDecision:
    """Bucket + output size for a ``width`` x ``height`` source."""
    cfg = cfg or ResolutionConfig()
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive")
    target = int(cfg.target)
    short = min(width, height)
    if short >= target:
        scale = target / short
        tw, th = (_even(width * scale), _even(height * scale)) if short > target else (_even(width), _even(height))
        return ResolutionDecision(target_bucket(target), "downscaled" if short > target else "kept", tw, th, False, True)
    if cfg.allow_upscale and short >= int(cfg.experimental_min):
        scale = target / short
        return ResolutionDecision(f"{target_bucket(target)}_upscaled", "upscaled", _even(width * scale), _even(height * scale), True, False)
    if short >= int(cfg.native_min):
        return ResolutionDecision(_native_bucket(short), "keep_native", _even(width), _even(height), False, True)
    if short >= int(cfg.experimental_min):
        return ResolutionDecision("lowres_360", "keep_native_experimental", _even(width), _even(height), False, False)
    return ResolutionDecision("excluded_lowres", "excluded", width, height, False, False)


def scale_filter(decision: ResolutionDecision, width: int, height: int) -> str | None:
    """FFmpeg ``-vf`` for the decision, or None when the frame size is already right."""
    if (decision.target_width, decision.target_height) == (width, height):
        return None
    return f"scale={decision.target_width}:{decision.target_height}:flags=lanczos"


def resolution_metadata(width: int, height: int, fps: float | None, decision: ResolutionDecision, training_fps: float | None, profile: str) -> dict[str, Any]:
    """The resolution block stored with every asset (spec section 6.5); measured values only."""
    return {
        "source_width": int(width),
        "source_height": int(height),
        "source_fps": round(float(fps), 3) if fps else None,
        "training_width": decision.target_width,
        "training_height": decision.target_height,
        "training_fps": round(float(training_fps), 3) if training_fps else None,
        "resolution_bucket": decision.bucket,
        "resolution_action": decision.action,
        "aspect_ratio_preserved": True,
        "upscaled": decision.upscaled,
        "crop_applied": False,
        "pad_applied": False,
        "frame_sampling_profile": profile,
    }
