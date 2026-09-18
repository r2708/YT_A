"""Deterministic mock VLM for tests. Produces stable objects/actions derived from the scene index."""

from __future__ import annotations

from video_dataset.schemas.common import ConfidenceSource
from video_dataset.schemas.scene import Frame, Scene
from video_dataset.schemas.vision import (
    CameraAnnotation,
    CameraMovement,
    Environment,
    FrameAnalysis,
    ObjectAnnotation,
    SceneAnalysis,
    Setting,
    ShotType,
    VerificationResult,
    VerificationVerdict,
    VisualStyle,
)
from video_dataset.vision.base import AnalysisContext, VisionAnalyzer

_OBJECTS = ["red circle", "blue square", "green triangle", "yellow star", "white rectangle"]
_ACTIONS = ["the red circle moves right", "the blue square moves left", "the green triangle moves down", "the yellow star rotates", "the white rectangle grows"]
_LOCATIONS = ["studio backdrop", "plain wall", "open field", "city street", "beach"]


class MockVisionAnalyzer(VisionAnalyzer):
    name = "mock"
    model = "mock-vlm"
    supports_verification = True

    def analyze_frame(self, frame: Frame, context: AnalysisContext) -> FrameAnalysis:
        idx = int(frame.scene_id.split("_")[-1]) - 1
        return FrameAnalysis(
            frame_id=frame.frame_id,
            scene_id=frame.scene_id,
            timestamp=frame.timestamp,
            caption=f"A {_OBJECTS[idx % len(_OBJECTS)]} on a {_LOCATIONS[idx % len(_LOCATIONS)]} at {frame.timestamp:.1f}s.",
            objects=[ObjectAnnotation(name=_OBJECTS[idx % len(_OBJECTS)])],
            actions=[_ACTIONS[idx % len(_ACTIONS)]],
            provider=self.name,
            model=self.model,
            confidence=0.9,
            confidence_source=ConfidenceSource.MODEL_SELF_REPORT,
        )

    def analyze_scene(self, scene: Scene, frames: list[Frame], context: AnalysisContext) -> SceneAnalysis:
        idx = scene.index
        obj = _OBJECTS[idx % len(_OBJECTS)]
        return SceneAnalysis(
            scene_id=scene.scene_id,
            video_id=scene.video_id,
            start_time=scene.start_time,
            end_time=scene.end_time,
            summary=f"A {obj} sits on a {_LOCATIONS[idx % len(_LOCATIONS)]}. {_ACTIONS[idx % len(_ACTIONS)].capitalize()}.",
            environment=Environment(location=_LOCATIONS[idx % len(_LOCATIONS)], setting=Setting.INDOOR if idx % 2 == 0 else Setting.OUTDOOR, lighting="even", time_of_day="day" if idx % 3 else "night"),
            objects=[ObjectAnnotation(name=obj, attributes=obj.split()[:1], location="center")],
            actions=[_ACTIONS[idx % len(_ACTIONS)]],
            camera=CameraAnnotation(shot_type=ShotType.WIDE, movement=CameraMovement.STATIC),
            visual_style=VisualStyle(composition="centered subject", lighting="flat, even lighting", color="saturated primary colours"),
            temporal_progression=f"The {obj} moves across the frame during the shot.",
            frame_ids=[f.frame_id for f in frames],
            provider=self.name,
            model=self.model,
            confidence=0.9,
            confidence_source=ConfidenceSource.MODEL_SELF_REPORT,
        )

    def verify(self, claim: str, frames: list[Frame], context: AnalysisContext) -> VerificationResult:
        return VerificationResult(claim=claim, verdict=VerificationVerdict.SUPPORTED, score=0.9, verifier=self.name, rationale="mock verifier", evidence_frame_ids=[f.frame_id for f in frames[:3]])
