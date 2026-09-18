"""Tolerant parsing of model output into our schemas."""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, TypeVar

from video_dataset.schemas.common import ConfidenceSource
from video_dataset.schemas.scene import Frame, Scene
from video_dataset.schemas.vision import (
    CameraAngle,
    CameraAnnotation,
    CameraMovement,
    Environment,
    FrameAnalysis,
    ObjectAnnotation,
    PersonAnnotation,
    SceneAnalysis,
    Setting,
    ShotType,
    VerificationResult,
    VerificationVerdict,
    VisualStyle,
)

E = TypeVar("E", bound=Enum)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for cand in candidates:
        cand = cand.strip()
        start = cand.find("{")
        end = cand.rfind("}")
        if start == -1 or end == -1 or end <= start:
            continue
        body = cand[start : end + 1]
        for attempt in (body, re.sub(r",\s*([}\]])", r"\1", body)):
            try:
                obj = json.loads(attempt)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    return None


def coerce_enum(value: Any, enum_cls: type[E], default: E) -> E:
    if value is None:
        return default
    v = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    for member in enum_cls:
        if member.value == v:
            return member
    aliases = {
        "wide_shot": "wide", "long_shot": "wide", "full_shot": "wide", "medium_shot": "medium", "closeup": "close_up",
        "close": "close_up", "extreme_closeup": "extreme_close_up", "establishing": "extreme_wide", "birds_eye": "overhead",
        "top_down": "overhead", "drone": "aerial", "low_angle": "low", "high_angle": "high", "eye": "eye_level",
        "pan": "pan_right", "tilt": "tilt_up", "dolly_in": "tracking_forward", "dolly_out": "tracking_backward",
        "push_in": "zoom_in", "pull_out": "zoom_out", "steady": "static", "still": "static", "fixed": "static",
        "inside": "indoor", "outside": "outdoor", "interior": "indoor", "exterior": "outdoor", "shaky": "handheld",
        "indoors": "indoor", "outdoors": "outdoor", "in": "indoor", "out": "outdoor", "both": "mixed",
        "dolly": "tracking", "truck": "tracking", "follow": "tracking", "orbit": "complex", "crane": "complex",
    }
    v2 = aliases.get(v)
    if v2:
        for member in enum_cls:
            if member.value == v2:
                return member
    return default


def _str(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s if s and s.lower() not in {"null", "none", "unknown", "n/a", "not visible"} else None


def _str_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [s.strip() for s in re.split(r"[;,]", v) if s.strip()]
    if isinstance(v, list):
        out = []
        for item in v:
            if isinstance(item, dict):
                name = item.get("name") or item.get("text") or item.get("description")
                if name:
                    out.append(str(name).strip())
            elif item is not None and str(item).strip():
                out.append(str(item).strip())
        return out
    return []


def _int(v: Any) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _conf(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f > 1.0 and f <= 100.0:
        f /= 100.0
    return round(max(0.0, min(1.0, f)), 3)


def parse_objects(v: Any) -> list[ObjectAnnotation]:
    out: list[ObjectAnnotation] = []
    if isinstance(v, str):
        v = _str_list(v)
    for item in v or []:
        if isinstance(item, str):
            if item.strip():
                out.append(ObjectAnnotation(name=item.strip()))
        elif isinstance(item, dict) and item.get("name"):
            out.append(
                ObjectAnnotation(
                    name=str(item["name"]).strip(),
                    attributes=_str_list(item.get("attributes")),
                    location=_str(item.get("location")),
                    count=_int(item.get("count")),
                    confidence=_conf(item.get("confidence")),
                    confidence_source=ConfidenceSource.MODEL_SELF_REPORT if item.get("confidence") is not None else ConfidenceSource.UNAVAILABLE,
                )
            )
    return out[:40]


def parse_camera(v: Any) -> CameraAnnotation:
    if not isinstance(v, dict):
        v = {}
    return CameraAnnotation(
        shot_type=coerce_enum(v.get("shot_type"), ShotType, ShotType.UNKNOWN),
        camera_angle=coerce_enum(v.get("camera_angle") or v.get("angle"), CameraAngle, CameraAngle.UNKNOWN),
        movement=coerce_enum(v.get("movement"), CameraMovement, CameraMovement.UNKNOWN),
        zoom=_str(v.get("zoom")),
        stability=_str(v.get("stability")),
        is_aerial=bool(v["is_aerial"]) if isinstance(v.get("is_aerial"), bool) else None,
        confidence=None,
        confidence_source=ConfidenceSource.UNAVAILABLE,
    )


def parse_environment(v: Any) -> Environment:
    if not isinstance(v, dict):
        return Environment()
    return Environment(
        location=_str(v.get("location")),
        setting=coerce_enum(v.get("setting"), Setting, Setting.UNKNOWN),
        weather=_str(v.get("weather")),
        lighting=_str(v.get("lighting")),
        time_of_day=_str(v.get("time_of_day")),
        background=_str(v.get("background")),
    )


def parse_people(v: Any) -> PersonAnnotation | None:
    if not isinstance(v, dict):
        return None
    p = PersonAnnotation(
        count=_int(v.get("count")),
        actions=_str_list(v.get("actions")),
        clothing=_str_list(v.get("clothing")),
        body_position=_str(v.get("body_position")),
        interactions=_str_list(v.get("interactions")),
    )
    if p.count in (None, 0) and not p.actions and not p.clothing:
        return None
    return p


def parse_style(v: Any) -> VisualStyle:
    if not isinstance(v, dict):
        return VisualStyle()
    return VisualStyle(**{k: _str(v.get(k)) for k in VisualStyle.model_fields})


def scene_analysis_from_dict(data: dict[str, Any], scene: Scene, frames: list[Frame], provider: str, model: str | None) -> SceneAnalysis:
    conf = _conf(data.get("confidence"))
    return SceneAnalysis(
        scene_id=scene.scene_id,
        video_id=scene.video_id,
        start_time=scene.start_time,
        end_time=scene.end_time,
        summary=_str(data.get("summary") or data.get("description")),
        environment=parse_environment(data.get("environment")),
        objects=parse_objects(data.get("objects")),
        people=parse_people(data.get("people")),
        actions=_str_list(data.get("actions"))[:20],
        camera=parse_camera(data.get("camera")),
        visual_style=parse_style(data.get("visual_style") or data.get("style")),
        temporal_progression=_str(data.get("temporal_progression")),
        frame_ids=[f.frame_id for f in frames],
        provider=provider,
        model=model,
        confidence=conf,
        confidence_source=ConfidenceSource.MODEL_SELF_REPORT if conf is not None else ConfidenceSource.UNAVAILABLE,
    )


def frame_analysis_from_dict(data: dict[str, Any], frame: Frame, provider: str, model: str | None) -> FrameAnalysis:
    conf = _conf(data.get("confidence"))
    return FrameAnalysis(
        frame_id=frame.frame_id,
        scene_id=frame.scene_id,
        timestamp=frame.timestamp,
        caption=_str(data.get("caption") or data.get("description")),
        objects=[ObjectAnnotation(name=o) for o in _str_list(data.get("objects"))[:30]],
        actions=_str_list(data.get("actions"))[:10],
        camera=CameraAnnotation(shot_type=coerce_enum(data.get("shot_type"), ShotType, ShotType.UNKNOWN)) if data.get("shot_type") else None,
        provider=provider,
        model=model,
        confidence=conf,
        confidence_source=ConfidenceSource.MODEL_SELF_REPORT if conf is not None else ConfidenceSource.UNAVAILABLE,
    )


def verification_from_dict(data: dict[str, Any] | None, claim: str, verifier: str, frame_ids: list[str]) -> VerificationResult:
    if not data:
        return VerificationResult(claim=claim, verifier=verifier, evidence_frame_ids=frame_ids, rationale="unparseable verifier output")
    verdict = coerce_enum(data.get("verdict"), VerificationVerdict, VerificationVerdict.UNKNOWN)
    score = _conf(data.get("score"))
    if verdict == VerificationVerdict.UNKNOWN:
        score = None
    elif score is None:
        score = {VerificationVerdict.SUPPORTED: 1.0, VerificationVerdict.PARTIALLY_SUPPORTED: 0.5, VerificationVerdict.UNSUPPORTED: 0.0}[verdict]
    return VerificationResult(claim=claim, verdict=verdict, score=score, rationale=_str(data.get("rationale")), verifier=verifier, evidence_frame_ids=frame_ids)
