"""Verifier interface. A verifier judges whether a textual claim is supported by frames."""

from __future__ import annotations

from typing import Protocol

from video_dataset.config import ValidationConfig, VisionConfig
from video_dataset.schemas.scene import Frame
from video_dataset.schemas.vision import VerificationResult, VerificationVerdict
from video_dataset.utils.logging import get_logger
from video_dataset.vision.base import AnalysisContext, VisionAnalyzer, create_vision_analyzer
from video_dataset.vision.heuristic import HeuristicVisionAnalyzer

log = get_logger("validation.verifier")


class Verifier(Protocol):
    name: str

    def verify(self, claim: str, frames: list[Frame], context: AnalysisContext) -> VerificationResult: ...


class NullVerifier:
    name = "none"

    def verify(self, claim: str, frames: list[Frame], context: AnalysisContext) -> VerificationResult:
        return VerificationResult(claim=claim, verifier=self.name)


class HeuristicVerifier:
    """Checks camera-motion claims against optical flow; other claims -> unknown."""

    name = "heuristic"

    def __init__(self, min_motion: float = 0.35, min_consistency: float = 0.6):
        self.analyzer = HeuristicVisionAnalyzer(min_motion, min_consistency)

    def verify(self, claim: str, frames: list[Frame], context: AnalysisContext) -> VerificationResult:
        return self.analyzer.verify(claim, frames, context)


class VLMVerifier:
    name = "vlm"

    def __init__(self, analyzer: VisionAnalyzer):
        self.analyzer = analyzer
        self.name = f"vlm:{analyzer.name}"

    def verify(self, claim: str, frames: list[Frame], context: AnalysisContext) -> VerificationResult:
        return self.analyzer.verify(claim, frames, context)


class CompositeVerifier:
    """Cheap grounded checks first; the model verifier only when they cannot decide."""

    name = "composite"

    def __init__(self, verifiers: list[Verifier]):
        self.verifiers = verifiers

    def verify(self, claim: str, frames: list[Frame], context: AnalysisContext) -> VerificationResult:
        last = VerificationResult(claim=claim, verifier=self.name)
        for v in self.verifiers:
            res = v.verify(claim, frames, context)
            if res.verdict != VerificationVerdict.UNKNOWN:
                return res
            last = res
        return last


def create_verifier(cfg: ValidationConfig, vision_cfg: VisionConfig, device: str, analyzer: VisionAnalyzer | None = None, min_motion: float = 0.35, min_consistency: float = 0.6) -> Verifier:
    kind = (cfg.verifier or "none").lower()
    if kind == "none":
        return NullVerifier()
    heuristic = HeuristicVerifier(min_motion, min_consistency)
    if kind == "heuristic":
        return heuristic
    model_analyzer = analyzer
    if model_analyzer is None or not model_analyzer.is_generative:
        try:
            model_analyzer = create_vision_analyzer(vision_cfg, device) if vision_cfg.provider not in ("heuristic",) else None
        except Exception as exc:
            log.warning("could not create model verifier (%s); using heuristic verifier only", exc)
            model_analyzer = None
    if model_analyzer is None or not model_analyzer.supports_verification:
        log.info("verifier '%s' requested but no model verifier available; using heuristic verifier", kind)
        return heuristic
    vlm = VLMVerifier(model_analyzer)
    if kind == "vlm":
        return vlm
    return CompositeVerifier([heuristic, vlm])
