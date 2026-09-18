import json
from pathlib import Path

from video_dataset.config import load_config
from video_dataset.llm.mock import MockLLMClient
from video_dataset.schemas.common import ConfidenceSource
from video_dataset.schemas.scene import Frame, Scene, TransitionType
from video_dataset.schemas.vision import CameraMovement, ShotType, VerificationVerdict
from video_dataset.vision.api_analyzer import APIVisionAnalyzer
from video_dataset.vision.base import AnalysisContext
from video_dataset.vision.parsing import (
    coerce_enum,
    extract_json,
    scene_analysis_from_dict,
    verification_from_dict,
)


def test_extract_json_tolerates_fences_and_trailing_commas():
    assert extract_json('```json\n{"a": 1, "b": [1,2,],}\n```') == {"a": 1, "b": [1, 2]}
    assert extract_json('Sure! Here it is: {"summary": "x"} thanks') == {"summary": "x"}
    assert extract_json("no json here") is None


def test_coerce_enum_aliases():
    assert coerce_enum("Wide Shot", ShotType, ShotType.UNKNOWN) == ShotType.WIDE
    assert coerce_enum("dolly in", CameraMovement, CameraMovement.UNKNOWN) == CameraMovement.TRACKING_FORWARD
    assert coerce_enum("weird", CameraMovement, CameraMovement.UNKNOWN) == CameraMovement.UNKNOWN


def test_scene_analysis_from_dict_is_tolerant():
    scene = Scene(scene_id="scene_001", video_id="v", index=0, start_time=0, end_time=5, start_frame=0, end_frame=120, transition_in=TransitionType.START)
    data = {
        "summary": "A red car drives down a street.",
        "environment": {"location": "city street", "setting": "Outdoors", "weather": None, "lighting": "overcast", "time_of_day": "unknown", "background": "shops"},
        "objects": ["red car", {"name": "traffic light", "attributes": ["green"], "location": "top right", "count": 1}],
        "people": {"count": 0, "actions": [], "clothing": [], "body_position": None, "interactions": []},
        "actions": "the car drives forward; a pedestrian waits",
        "camera": {"shot_type": "medium shot", "camera_angle": "eye level", "movement": "static", "zoom": None, "stability": "static", "is_aerial": False},
        "visual_style": {"composition": "car centred"},
        "temporal_progression": "the car gets closer",
        "confidence": 85,
    }
    a = scene_analysis_from_dict(data, scene, [], "api:test", "m")
    assert a.environment.setting == "outdoor" and a.environment.time_of_day is None
    assert [o.name for o in a.objects] == ["red car", "traffic light"] and a.objects[1].attributes == ["green"]
    assert a.people is None and a.actions == ["the car drives forward", "a pedestrian waits"]
    assert a.camera.shot_type == ShotType.MEDIUM and a.camera.movement == CameraMovement.STATIC and a.camera.is_aerial is False
    assert a.confidence == 0.85 and a.confidence_source == ConfidenceSource.MODEL_SELF_REPORT


def test_verification_from_dict():
    v = verification_from_dict({"verdict": "supported", "score": 0.9, "rationale": "ok"}, "claim", "ver", ["f1"])
    assert v.verdict == VerificationVerdict.SUPPORTED and v.score == 0.9
    u = verification_from_dict({"verdict": "unknown", "score": 0.7}, "claim", "ver", [])
    assert u.verdict == VerificationVerdict.UNKNOWN and u.score is None  # unknown never contributes a score
    p = verification_from_dict({"verdict": "partially_supported"}, "claim", "ver", [])
    assert p.score == 0.5


def test_api_analyzer_roundtrip_with_mock_client(tmp_path: Path):
    import cv2
    import numpy as np

    img = tmp_path / "f.jpg"
    cv2.imwrite(str(img), np.full((90, 160, 3), 120, dtype=np.uint8))
    scene = Scene(scene_id="scene_001", video_id="v", index=0, start_time=0, end_time=5, start_frame=0, end_frame=120, transition_in=TransitionType.START)
    frame = Frame(frame_id="frame_001_000", video_id="v", scene_id="scene_001", timestamp=2.5, frame_index=60, frame_path=str(img), width=160, height=90)
    scene_json = json.dumps({"summary": "A grey wall fills the frame.", "environment": {"location": None, "setting": "indoor", "weather": None, "lighting": "flat", "time_of_day": None, "background": "wall"}, "objects": [{"name": "wall", "attributes": ["grey"], "location": "everywhere", "count": 1}], "people": {"count": None, "actions": [], "clothing": [], "body_position": None, "interactions": []}, "actions": [], "camera": {"shot_type": "close_up", "camera_angle": "eye_level", "movement": "static", "zoom": None, "stability": "static", "is_aerial": False}, "visual_style": {"composition": None, "lighting": "flat", "color": "grey", "depth_of_field": None, "framing": None, "perspective": None, "motion": "none", "transitions": None}, "temporal_progression": None, "confidence": 0.7})
    verify_json = json.dumps({"verdict": "supported", "score": 0.95, "rationale": "the frame is a grey wall"})
    client = MockLLMClient([scene_json, verify_json])
    cfg = load_config(None, {"vision.provider": "mock"}).vision
    analyzer = APIVisionAnalyzer(client, cfg)
    ctx = AnalysisContext(video_id="v", transcript_text="hello", ocr_texts=["EXIT"])
    a = analyzer.analyze_scene(scene, [frame], ctx)
    assert a.summary == "A grey wall fills the frame." and a.camera.shot_type == ShotType.CLOSE_UP and a.confidence == 0.7
    assert client.calls[0]["images"] == [img] and client.calls[0]["json_schema"] is not None
    assert "EXIT" in client.calls[0]["prompt"] and "hello" in client.calls[0]["prompt"]
    v = analyzer.verify(a.summary, [frame], ctx)
    assert v.verdict == VerificationVerdict.SUPPORTED and v.score == 0.95 and v.evidence_frame_ids == ["frame_001_000"]
