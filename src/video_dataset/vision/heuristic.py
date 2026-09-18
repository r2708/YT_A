"""Measurement-only analyzer. Always available, never hallucinates: it reports lighting, colour,
motion and camera movement derived from pixels and optical flow, and says 'unknown' for the rest."""

from __future__ import annotations

from pathlib import Path

from video_dataset.schemas.common import ConfidenceSource
from video_dataset.schemas.scene import Frame, Scene
from video_dataset.schemas.vision import (
    CameraAnnotation,
    CameraMovement,
    Environment,
    FrameAnalysis,
    Measurements,
    SceneAnalysis,
    VerificationResult,
    VerificationVerdict,
    VisualStyle,
)
from video_dataset.vision.base import AnalysisContext, VisionAnalyzer
from video_dataset.vision.measurements import measure_frame_file, measure_scene
from video_dataset.vision.motion import MOVEMENT_PHRASES, dominant_motion

_MOTION_WORDS = {
    CameraMovement.PAN_LEFT: ("pan left", "pans left", "panning left"),
    CameraMovement.PAN_RIGHT: ("pan right", "pans right", "panning right"),
    CameraMovement.TILT_UP: ("tilt up", "tilts up", "tilting up"),
    CameraMovement.TILT_DOWN: ("tilt down", "tilts down", "tilting down"),
    CameraMovement.ZOOM_IN: ("zoom in", "zooms in", "zooming in", "pushes in", "moves forward"),
    CameraMovement.ZOOM_OUT: ("zoom out", "zooms out", "zooming out", "pulls back", "moves backward"),
    CameraMovement.STATIC: ("static camera", "camera is static", "camera is still", "stationary camera", "fixed camera"),
    CameraMovement.HANDHELD: ("handheld", "shaky"),
}


def _palette_phrase(m: Measurements) -> str:
    sat = m.saturation_mean or 0.0
    sat_word = "desaturated" if sat < 50 else ("moderately saturated" if sat < 120 else "highly saturated")
    contrast = m.contrast or 0.0
    con_word = "low-contrast" if contrast < 0.25 else ("medium-contrast" if contrast < 0.5 else "high-contrast")
    colors = ", ".join(m.dominant_colors[:3]) if m.dominant_colors else "unmeasured colors"
    return f"a {sat_word}, {con_word}, {m.color_temperature or 'neutral'}-toned palette (dominant colors {colors})"


def _motion_level(mag: float | None) -> str:
    if mag is None:
        return "unmeasured"
    if mag < 0.3:
        return "very little"
    if mag < 1.0:
        return "moderate"
    return "strong"


class HeuristicVisionAnalyzer(VisionAnalyzer):
    name = "heuristic"
    model = None
    is_generative = False
    supports_verification = True

    def __init__(self, min_motion: float = 0.35, min_consistency: float = 0.6):
        self.min_motion = min_motion
        self.min_consistency = min_consistency

    def camera_annotation(self, context: AnalysisContext) -> CameraAnnotation:
        label, consistency, _mag = dominant_motion(context.scan, self.min_motion)
        known = label != CameraMovement.UNKNOWN and consistency >= self.min_consistency
        return CameraAnnotation(
            movement=label if known else CameraMovement.UNKNOWN,
            stability="static" if label == CameraMovement.STATIC and known else ("handheld" if label == CameraMovement.HANDHELD and known else None),
            confidence=round(consistency, 3) if label != CameraMovement.UNKNOWN else None,
            confidence_source=ConfidenceSource.MEASUREMENT if label != CameraMovement.UNKNOWN else ConfidenceSource.UNAVAILABLE,
        )

    def analyze_frame(self, frame: Frame, context: AnalysisContext) -> FrameAnalysis:
        m = measure_frame_file(Path(frame.frame_path))
        caption = None
        if m is not None:
            focus = "sharp" if (m.sharpness or 0) > 100 else "soft"
            caption = (
                f"Frame at {frame.timestamp:.1f}s: {m.lighting_level} lighting with {_palette_phrase(m)}; "
                f"{focus} focus and {'dense' if (m.edge_density or 0) > 0.08 else 'sparse'} visual detail."
            )
        return FrameAnalysis(
            frame_id=frame.frame_id,
            scene_id=frame.scene_id,
            timestamp=frame.timestamp,
            caption=caption,
            measurements=m,
            provider=self.name,
            confidence=None,
            confidence_source=ConfidenceSource.UNAVAILABLE,
        )

    def analyze_scene(self, scene: Scene, frames: list[Frame], context: AnalysisContext) -> SceneAnalysis:
        m = context.measurements or measure_scene([Path(f.frame_path) for f in frames], context.scan)
        label, consistency, mag = dominant_motion(context.scan, self.min_motion)
        camera = self.camera_annotation(context)
        m = m.model_copy(update={"camera_motion_label": str(label) if label != CameraMovement.UNKNOWN else None, "camera_motion_consistency": round(consistency, 3) if label != CameraMovement.UNKNOWN else None, "motion_magnitude": m.motion_magnitude if m.motion_magnitude is not None else (round(mag, 4) if mag is not None else None)})

        movement_phrase = MOVEMENT_PHRASES[camera.movement]
        summary = (
            f"A {scene.duration:.1f}-second shot with {m.lighting_level or 'unmeasured'} lighting and {_palette_phrase(m)}; "
            f"{movement_phrase}, with {_motion_level(m.motion_magnitude)} on-screen motion."
        )
        progression = None
        if m.brightness_trend in ("brightening", "darkening"):
            progression = f"The image gets {'brighter' if m.brightness_trend == 'brightening' else 'darker'} over the course of the shot."
        if context.scan and context.scan.samples:
            peak = max(context.scan.samples, key=lambda s: s.change)
            if peak.change >= 0.3:
                progression = (progression + " " if progression else "") + f"The largest visual change within the shot occurs around {peak.timestamp:.1f}s."
        actions = []
        if camera.movement not in (CameraMovement.UNKNOWN, CameraMovement.STATIC):
            actions.append(MOVEMENT_PHRASES[camera.movement].replace("the camera ", "camera "))
        style = VisualStyle(
            lighting=f"{m.lighting_level} overall brightness (mean {m.brightness_mean:.0f}/255), {m.color_temperature} color temperature" if m.brightness_mean is not None else None,
            color=_palette_phrase(m),
            motion=f"{_motion_level(m.motion_magnitude)} motion; {movement_phrase}",
            transitions=f"shot begins with a {scene.transition_in}" if scene.transition_in not in ("start", "split") else None,
        )
        return SceneAnalysis(
            scene_id=scene.scene_id,
            video_id=scene.video_id,
            start_time=scene.start_time,
            end_time=scene.end_time,
            summary=summary,
            environment=Environment(lighting=m.lighting_level),
            actions=actions,
            camera=camera,
            visual_style=style,
            temporal_progression=progression,
            measurements=m,
            frame_ids=[f.frame_id for f in frames],
            provider=self.name,
            model=None,
            confidence=None,
            confidence_source=ConfidenceSource.UNAVAILABLE,
        )

    def verify(self, claim: str, frames: list[Frame], context: AnalysisContext) -> VerificationResult:
        """Only camera-motion claims can be checked against optical flow; everything else is UNKNOWN."""
        low = claim.lower()
        claimed = [lab for lab, words in _MOTION_WORDS.items() if any(w in low for w in words)]
        if not claimed or context.scan is None:
            return VerificationResult(claim=claim, verifier=self.name)
        label, consistency, _ = dominant_motion(context.scan, self.min_motion)
        if label == CameraMovement.UNKNOWN or consistency < 0.5:
            return VerificationResult(claim=claim, verifier=self.name, rationale="insufficient flow evidence")
        if label in claimed:
            return VerificationResult(claim=claim, verdict=VerificationVerdict.SUPPORTED, score=round(consistency, 3), verifier=self.name, rationale=f"optical flow: {label} ({consistency:.2f} of samples)")
        if CameraMovement.STATIC in claimed and label in (CameraMovement.HANDHELD,):
            return VerificationResult(claim=claim, verdict=VerificationVerdict.PARTIALLY_SUPPORTED, score=0.5, verifier=self.name, rationale="camera mostly still but unsteady")
        return VerificationResult(claim=claim, verdict=VerificationVerdict.UNSUPPORTED, score=round(1.0 - consistency, 3), verifier=self.name, rationale=f"optical flow indicates {label} instead")
