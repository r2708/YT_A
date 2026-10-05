"""cv_models provider (YOLO + CLIP) exercised with fake models: no weights, no network."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from video_dataset.config import load_config
from video_dataset.schemas.common import ConfidenceSource
from video_dataset.schemas.scene import Frame, Scene, TransitionType
from video_dataset.schemas.vision import CameraAngle, Setting, ShotType
from video_dataset.vision import cv_vision
from video_dataset.vision.base import AnalysisContext, create_vision_analyzer
from video_dataset.vision.cv_analyzer import aggregate_detections, bbox_location


def _frames(tmp_path: Path, n: int = 3) -> tuple[Scene, list[Frame]]:
    scene = Scene(scene_id="scene_001", video_id="vid_t", index=0, start_time=0.0, end_time=6.0, start_frame=0, end_frame=144, transition_in=TransitionType.START)
    frames = []
    for i in range(n):
        p = tmp_path / f"frame_001_{i:03d}.jpg"
        img = np.full((90, 160, 3), 40 + 60 * i, dtype=np.uint8)
        cv2.imwrite(str(p), img)
        frames.append(Frame(frame_id=f"frame_001_{i:03d}", video_id="vid_t", scene_id="scene_001", timestamp=1.0 + 2.0 * i, frame_index=24 * i, frame_path=str(p), width=160, height=90))
    return scene, frames


class FakeDetector:
    """Stands in for YOLODetector: two people and a car in every frame, a dog in one."""

    def __init__(self, model_name="fake.pt", confidence=0.5, device="cpu", image_size=640):
        self.model_name, self.confidence, self.device = model_name, confidence, device
        self.loaded = False
        self.calls: list[list[Path]] = []

    def load(self):
        self.loaded = True

    def detect_many(self, paths):
        self.calls.append(list(paths))
        out = []
        for i, _p in enumerate(paths):
            size = [160, 90]
            dets = [
                {"name": "person", "confidence": 0.9, "bbox": [10, 10, 40, 80], "frame_size": size, "location": "left part of the frame"},
                {"name": "person", "confidence": 0.7, "bbox": [60, 10, 90, 80], "frame_size": size, "location": "center of the frame"},
                {"name": "car", "confidence": 0.8, "bbox": [120 - 30 * i, 40, 158 - 30 * i, 88], "frame_size": size, "location": "right part of the frame"},
            ]
            if i == 0:
                dets.append({"name": "dog", "confidence": 0.55, "bbox": [70, 60, 90, 88], "frame_size": size, "location": "center of the frame"})
            out.append(dets)
        return out

    def close(self):
        self.loaded = False


class FakeCLIP:
    name = "clip"

    def __init__(self, model_name="fake-clip", device="cpu", aesthetic=None):
        self.model_name = model_name
        self.aesthetic = aesthetic

    def enrich(self, scene, frames):
        return {
            "model": self.model_name,
            "frames": len(frames),
            "aesthetic": {"model": "fake-aesthetic", "mean": 5.25, "min": 5.0, "max": 5.5, "per_frame": [5.0, 5.5]},
            "attributes": {
                "setting": {"label": "outdoor", "score": 0.93, "scores": {"indoor": 0.07, "outdoor": 0.93}},
                "time_of_day": {"label": "day", "score": 0.6, "scores": {"day": 0.6, "night": 0.4}},  # below threshold
                "shot_type": {"label": "wide", "score": 0.81, "scores": {}},
                "camera_angle": {"label": "aerial", "score": 0.9, "scores": {}},
                "people_present": {"label": "yes", "score": 0.99, "scores": {}},
            },
        }


class BrokenCLIP:
    def __init__(self, *a, **k):
        raise RuntimeError("transformers missing")


@pytest.fixture()
def fake_models(monkeypatch):
    monkeypatch.setattr(cv_vision, "YOLODetector", FakeDetector)
    monkeypatch.setattr(cv_vision, "CLIPZeroShotEnricher", FakeCLIP)


def test_bbox_location_and_aggregation():
    assert bbox_location([0, 0, 20, 20], 160, 90) == "left part of the frame"
    assert bbox_location([70, 0, 90, 20], 160, 90) == "center of the frame"
    assert bbox_location([140, 0, 160, 20], 160, 90) == "right part of the frame"
    agg = aggregate_detections(FakeDetector().detect_many([Path("a"), Path("b")]))
    by = {o["name"]: o for o in agg}
    assert by["person"]["count"] == 2  # max simultaneous, not the sum over frames
    assert by["person"]["confidence"] == 0.8 and by["person"]["frame_fraction"] == 1.0
    assert by["dog"]["frame_fraction"] == 0.5 and by["dog"]["count"] == 1
    assert [o["name"] for o in agg][:2] == ["person", "car"]  # sorted by presence, then how often detected
    # composition from the largest box per frame: the 30x70 person box at x=10..40 on a 160x90 frame
    assert by["person"]["position"] == "left" and by["person"]["scale"] == "medium" and by["person"]["center"] == [0.1562, 0.5]
    assert by["person"]["area_fraction"] == round(30 * 70 / (160 * 90), 4)
    assert by["dog"]["scale"] == "small" and by["dog"]["position"] == "lower"
    assert aggregate_detections([]) == []
    bare = aggregate_detections([[{"name": "cat", "confidence": 0.9, "bbox": [0, 0, 10, 10]}]])  # no frame size -> no composition
    assert bare[0]["position"] is None and bare[0]["scale"] is None


def test_cv_models_scene_analysis_is_grounded(tmp_path, fake_models):
    cfg = load_config(None, {"vision.provider": "cv_models", "vision.yolo_model": "fake.pt", "vision.clip_model": "fake-clip", "vision.yolo_max_frames": "2", "vision.aesthetic": "false"}).vision
    analyzer = create_vision_analyzer(cfg, "cpu")
    assert analyzer.name == "cv_models" and analyzer.model == "fake.pt+fake-clip"
    assert not analyzer.is_generative and analyzer.detector.loaded
    scene, frames = _frames(tmp_path, 3)
    a = analyzer.analyze_scene(scene, frames, AnalysisContext(video_id="vid_t"))

    assert len(analyzer.detector.calls) == 1 and len(analyzer.detector.calls[0]) == 2  # yolo_max_frames honoured
    names = {o.name: o for o in a.objects}
    assert names["person"].count == 2 and names["person"].confidence_source == ConfidenceSource.DETECTOR_SCORE
    assert names["car"].location == "right part of the frame"
    assert names["person"].position == "left" and names["person"].scale == "medium" and names["person"].frame_fraction == 1.0
    assert a.subject_motion is None  # only two analysed frames: not enough to measure subject motion
    assert a.measurements is not None and a.measurements.aesthetic_score == 5.25 and a.measurements.aesthetic_model == "fake-aesthetic"
    assert a.people is not None and a.people.count == 2
    assert a.confidence is not None and a.confidence_source == ConfidenceSource.DETECTOR_SCORE
    # CLIP labels above the threshold fill unknown fields; the 0.6 time-of-day score does not
    assert a.environment.setting == Setting.OUTDOOR and a.environment.time_of_day is None
    assert a.camera.shot_type == ShotType.WIDE and a.camera.camera_angle == CameraAngle.AERIAL and a.camera.is_aerial is True
    assert "clip" in a.enrichments and a.enrichments["yolo"]["frames"] == 2
    assert "Detected objects: 2 people, a car and a dog." in (a.summary or "")
    assert "CLIP classifies the shot as outdoors, framed as a wide shot, from an aerial viewpoint." in (a.summary or "")
    assert a.provider == "cv_models" and a.model == analyzer.model and a.measurements is not None
    assert a.frame_ids == [f.frame_id for f in frames] and not a.errors
    # every record survives a schema round-trip
    type(a).model_validate(a.model_dump(mode="json"))


def test_cv_models_measures_subject_motion_with_enough_frames(tmp_path, fake_models):
    cfg = load_config(None, {"vision.provider": "cv_models", "vision.yolo_max_frames": "4", "vision.aesthetic": "false"}).vision
    analyzer = cv_vision.CVVisionAnalyzer(cfg, "cpu")
    scene, frames = _frames(tmp_path, 4)
    a = analyzer.analyze_scene(scene, frames, AnalysisContext(video_id="vid_t"))
    # the person boxes never move, the car box slides 30 px left per frame; the person is the larger subject
    sm = a.subject_motion
    assert sm is not None and sm.subject == "person" and sm.label == "static" and sm.n_frames == 4 and sm.camera_relation == "camera_static"
    assert sm.confidence == 1.0 and str(sm.confidence_source) == "measurement"
    assert a.summary and a.summary.endswith("The person stays in place.")
    type(a).model_validate(a.model_dump(mode="json"))


def test_cv_models_frame_analysis_and_missing_files(tmp_path, fake_models):
    cfg = load_config(None, {"vision.provider": "cv_models", "vision.aesthetic": "false"}).vision
    analyzer = cv_vision.CVVisionAnalyzer(cfg, "cpu")
    scene, frames = _frames(tmp_path, 1)
    fa = analyzer.analyze_frame(frames[0], AnalysisContext(video_id="vid_t"))
    assert {o.name for o in fa.objects} == {"person", "car", "dog"} and fa.people.count == 2
    assert fa.caption and "Detected objects" in fa.caption and fa.provider == "cv_models"
    missing = [f.model_copy(update={"frame_path": str(tmp_path / "nope.jpg")}) for f in frames]
    a = analyzer.analyze_scene(scene, missing, AnalysisContext(video_id="vid_t"))
    assert a.errors == ["no_frame_files"] and a.objects == [] and a.provider == "cv_models"


def test_cv_models_without_clip_keeps_yolo(tmp_path, monkeypatch):
    monkeypatch.setattr(cv_vision, "YOLODetector", FakeDetector)
    monkeypatch.setattr(cv_vision, "CLIPZeroShotEnricher", BrokenCLIP)
    cfg = load_config(None, {"vision.provider": "cv_models", "vision.yolo_model": "fake.pt", "vision.aesthetic": "false"}).vision
    analyzer = cv_vision.CVVisionAnalyzer(cfg, "cpu")
    assert analyzer.clip is None and analyzer.model == "fake.pt"
    scene, frames = _frames(tmp_path, 2)
    a = analyzer.analyze_scene(scene, frames, AnalysisContext(video_id="vid_t"))
    assert {o.name for o in a.objects} == {"person", "car", "dog"}
    assert "clip" not in a.enrichments and a.environment.setting == Setting.UNKNOWN
    assert "CLIP classifies" not in (a.summary or "")
