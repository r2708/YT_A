"""``cv_models`` vision provider: YOLO object detection + CLIP zero-shot scene attributes on top of
the measurement-only heuristic analyzer.

It is not generative: objects come from the detector, setting / time of day / shot type from CLIP
softmax scores, lighting and camera movement from pixel statistics and optical flow. Anything the
models do not cover stays ``unknown``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from video_dataset.config import VisionConfig
from video_dataset.schemas.common import ConfidenceSource
from video_dataset.schemas.scene import Frame, Scene
from video_dataset.schemas.vision import FrameAnalysis, ObjectAnnotation, PersonAnnotation, SceneAnalysis
from video_dataset.utils.logging import get_logger
from video_dataset.vision.base import AnalysisContext, VisionAnalyzer, existing_paths, select_evenly
from video_dataset.vision.cv_analyzer import YOLODetector, aggregate_detections
from video_dataset.vision.enrichers.aesthetic import head_from_config
from video_dataset.vision.enrichers.clip import (
    CLIPZeroShotEnricher,
    apply_clip_aesthetic,
    apply_clip_attributes,
    clip_phrase,
)
from video_dataset.vision.heuristic import HeuristicVisionAnalyzer
from video_dataset.vision.subject_motion import analyze_subject_motion, apply_subject_motion

log = get_logger("vision.cv_models")

PERSON_CLASSES = {"person"}


def _count_phrase(name: str, count: int) -> str:
    if count <= 1:
        return f"a {name}" if name[:1] not in "aeiou" else f"an {name}"
    plural = "people" if name == "person" else (name + ("es" if name.endswith(("s", "x", "sh", "ch")) else "s"))
    return f"{count} {plural}"


def _join(parts: list[str]) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def objects_sentence(objects: list[ObjectAnnotation], limit: int = 6) -> str:
    if not objects:
        return "No objects were detected above the confidence threshold."
    parts = [_count_phrase(o.name, o.count or 1) for o in objects[:limit]]
    more = len(objects) - limit
    text = f"Detected objects: {_join(parts)}"
    if more > 0:
        text += f" and {more} more"
    return text + "."


class CVVisionAnalyzer(VisionAnalyzer):
    name = "cv_models"
    is_generative = False
    supports_verification = False
    parallel_safe = False  # the ultralytics predictor keeps state between calls; torch already uses all cores

    def __init__(self, cfg: VisionConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self.detector = YOLODetector(cfg.yolo_model, cfg.yolo_confidence, device)
        self.detector.load()  # fail at stage start, not on the first scene
        self.heuristic = HeuristicVisionAnalyzer()
        self.clip: CLIPZeroShotEnricher | None = None
        try:
            self.clip = CLIPZeroShotEnricher(cfg.clip_model, device, aesthetic=head_from_config(cfg))
        except Exception as exc:
            log.warning("CLIP unavailable for cv_models (%s); continuing with YOLO only", str(exc)[:160])
        self.model = cfg.yolo_model + (f"+{cfg.clip_model}" if self.clip else "")

    # ------------------------------------------------------------------ helpers
    def _detect(self, frames: list[Frame]) -> tuple[list[ObjectAnnotation], dict[str, Any], list[list[dict[str, Any]]]]:
        paths = existing_paths(frames)
        per_frame = self.detector.detect_many(paths) if paths else []
        agg = aggregate_detections(per_frame)
        objects = [
            ObjectAnnotation(
                name=o["name"], location=o["location"], count=o["count"],
                confidence=o["confidence"], confidence_source=ConfidenceSource.DETECTOR_SCORE,
                position=o.get("position"), center=o.get("center"), area_fraction=o.get("area_fraction"),
                scale=o.get("scale"), frame_fraction=o.get("frame_fraction"),
            )
            for o in agg
        ]
        info = {"model": self.cfg.yolo_model, "confidence_threshold": self.cfg.yolo_confidence, "frames": len(paths), "objects": agg}
        return objects, info, per_frame

    @staticmethod
    def _people(objects: list[ObjectAnnotation]) -> PersonAnnotation | None:
        n = sum(o.count or 1 for o in objects if o.name.lower() in PERSON_CLASSES)
        return PersonAnnotation(count=n) if n else None

    @staticmethod
    def _confidence(objects: list[ObjectAnnotation]) -> tuple[float | None, ConfidenceSource]:
        confs = [o.confidence for o in objects if o.confidence is not None]
        if not confs:
            return None, ConfidenceSource.UNAVAILABLE
        return round(sum(confs) / len(confs), 3), ConfidenceSource.DETECTOR_SCORE

    # ------------------------------------------------------------------ interface
    def analyze_frame(self, frame: Frame, context: AnalysisContext) -> FrameAnalysis:
        analysis = self.heuristic.analyze_frame(frame, context)
        objects, _info, _per_frame = self._detect([frame])
        analysis.objects = objects
        analysis.people = self._people(objects)
        analysis.provider = self.name
        analysis.model = self.model
        analysis.confidence, analysis.confidence_source = self._confidence(objects)
        base = analysis.caption or f"Frame at {frame.timestamp:.1f}s."
        analysis.caption = f"{base} {objects_sentence(objects)}"
        return analysis

    def analyze_scene(self, scene: Scene, frames: list[Frame], context: AnalysisContext) -> SceneAnalysis:
        analysis = self.heuristic.analyze_scene(scene, frames, context)
        analysis.provider = self.name
        analysis.model = self.model
        chosen = select_evenly(frames, int(self.cfg.yolo_max_frames))
        if not existing_paths(chosen):
            analysis.errors.append("no_frame_files")
            return analysis

        objects, yolo_info, per_frame = self._detect(chosen)
        analysis.objects = objects
        analysis.people = self._people(objects)
        analysis.enrichments["yolo"] = yolo_info
        analysis.confidence, analysis.confidence_source = self._confidence(objects)
        present = [f for f in chosen if f.frame_path and Path(f.frame_path).exists()]
        subject_motion = analyze_subject_motion(per_frame, [f.timestamp for f in present], context.scan, analysis.camera.movement) if len(present) == len(per_frame) else None

        clip_data: dict[str, Any] = {}
        if self.clip is not None:
            try:
                clip_data = self.clip.enrich(scene, chosen)
            except Exception as exc:
                log.warning("CLIP failed on %s: %s", scene.scene_id, str(exc)[:160])
                analysis.errors.append(f"clip_error: {type(exc).__name__}")
            if clip_data:
                analysis.enrichments["clip"] = clip_data
                apply_clip_attributes(analysis, clip_data)
                apply_clip_aesthetic(analysis, clip_data)

        parts = [analysis.summary or "", objects_sentence(objects)]
        clip_text = clip_phrase(clip_data)
        if clip_text:
            parts.append(clip_text)
        analysis.summary = " ".join(p for p in parts if p).strip()
        apply_subject_motion(analysis, subject_motion)  # appends the measured subject-motion sentence
        return analysis

    def close(self) -> None:
        self.detector.close()
        self.clip = None
