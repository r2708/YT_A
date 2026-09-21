from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from video_dataset.config import VisionConfig
from video_dataset.schemas.scene import Clip, Frame, Scene, SceneScan
from video_dataset.schemas.vision import FrameAnalysis, Measurements, SceneAnalysis, VerificationResult
from video_dataset.utils.logging import get_logger

log = get_logger("vision")


@dataclass
class AnalysisContext:
    video_id: str
    transcript_text: str | None = None
    ocr_texts: list[str] = field(default_factory=list)
    measurements: Measurements | None = None
    scan: SceneScan | None = None
    previous_summary: str | None = None
    video_title: str | None = None
    extra: dict = field(default_factory=dict)


class VisionAnalyzer(ABC):
    """Interface every vision backend implements. Swap models by swapping this object."""

    name: str = "base"
    model: str | None = None
    supports_video: bool = False
    supports_verification: bool = False
    is_generative: bool = True  # False for measurement-only analyzers

    @abstractmethod
    def analyze_frame(self, frame: Frame, context: AnalysisContext) -> FrameAnalysis: ...

    @abstractmethod
    def analyze_scene(self, scene: Scene, frames: list[Frame], context: AnalysisContext) -> SceneAnalysis: ...

    def analyze_clip(self, clip: Clip, scene: Scene, frames: list[Frame], context: AnalysisContext) -> SceneAnalysis:
        """Default: analyse the clip through its sampled frames."""
        return self.analyze_scene(scene, frames, context)

    def verify(self, claim: str, frames: list[Frame], context: AnalysisContext) -> VerificationResult:
        return VerificationResult(claim=claim, verifier=self.name)  # UNKNOWN by default

    def close(self) -> None:  # release GPU memory etc.
        return None


def select_evenly(frames: list[Frame], k: int) -> list[Frame]:
    if k <= 0 or len(frames) <= k:
        return list(frames)
    if k == 1:
        return [frames[len(frames) // 2]]
    step = (len(frames) - 1) / (k - 1)
    idxs = sorted({int(round(i * step)) for i in range(k)})
    return [frames[i] for i in idxs]


def existing_paths(frames: list[Frame]) -> list[Path]:
    return [Path(f.frame_path) for f in frames if Path(f.frame_path).exists()]


def create_vision_analyzer(cfg: VisionConfig, device_pref: str = "auto") -> VisionAnalyzer:
    provider = (cfg.provider or "heuristic").lower()
    if provider == "heuristic":
        from video_dataset.vision.heuristic import HeuristicVisionAnalyzer

        return HeuristicVisionAnalyzer()
    if provider == "mock":
        from video_dataset.vision.mock import MockVisionAnalyzer

        return MockVisionAnalyzer()
    if provider == "hf":
        from video_dataset.vision.hf_analyzer import HFVisionAnalyzer

        return HFVisionAnalyzer(cfg, cfg.device or device_pref)
    if provider in ("anthropic", "openai_compatible", "openai"):
        from video_dataset.llm.base import create_llm_client
        from video_dataset.vision.api_analyzer import APIVisionAnalyzer

        client = create_llm_client(
            provider, cfg.model, cfg.base_url, cfg.api_key_env, float(cfg.timeout_seconds),
            requests_per_minute=float(cfg.requests_per_minute), image_max_side=cfg.image_max_side,
        )
        return APIVisionAnalyzer(client, cfg)
    raise ValueError(f"Unknown vision provider '{cfg.provider}'")
