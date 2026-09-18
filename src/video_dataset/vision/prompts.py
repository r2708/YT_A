"""Prompts and JSON schemas for generative vision backends. Prompts insist on observable facts only."""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = (
    "You annotate video frames to build a training dataset. Describe ONLY what is visibly present in the "
    "provided frames. Never guess what is outside the frame, never infer names or identities of people, and "
    "never describe things that are only mentioned in the transcript unless they are visible. If an attribute "
    "cannot be determined from the images, use null (or \"unknown\" for enum fields). Prefer concrete, "
    "observable wording (\"low-angle shot lit by a single window on the left\") over generic labels "
    "(\"cinematic\"). Return valid JSON only."
)

_ENUM_SETTING = ["indoor", "outdoor", "mixed", "unknown"]
_ENUM_SHOT = ["extreme_wide", "wide", "medium", "close_up", "extreme_close_up", "unknown"]
_ENUM_ANGLE = ["eye_level", "low", "high", "overhead", "aerial", "dutch", "unknown"]
_ENUM_MOVEMENT = [
    "static", "pan_left", "pan_right", "tilt_up", "tilt_down", "zoom_in", "zoom_out", "tracking",
    "tracking_forward", "tracking_backward", "handheld", "complex", "unknown",
]


def _str_or_null() -> dict[str, Any]:
    return {"type": ["string", "null"]}


SCENE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "2-4 sentences describing what is visible and what happens across the frames, in temporal order."},
        "environment": {
            "type": "object",
            "properties": {
                "location": _str_or_null(),
                "setting": {"type": "string", "enum": _ENUM_SETTING},
                "weather": _str_or_null(),
                "lighting": _str_or_null(),
                "time_of_day": _str_or_null(),
                "background": _str_or_null(),
            },
            "required": ["location", "setting", "weather", "lighting", "time_of_day", "background"],
            "additionalProperties": False,
        },
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "attributes": {"type": "array", "items": {"type": "string"}},
                    "location": _str_or_null(),
                    "count": {"type": ["integer", "null"]},
                },
                "required": ["name", "attributes", "location", "count"],
                "additionalProperties": False,
            },
        },
        "people": {
            "type": "object",
            "properties": {
                "count": {"type": ["integer", "null"]},
                "actions": {"type": "array", "items": {"type": "string"}},
                "clothing": {"type": "array", "items": {"type": "string"}},
                "body_position": _str_or_null(),
                "interactions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["count", "actions", "clothing", "body_position", "interactions"],
            "additionalProperties": False,
        },
        "actions": {"type": "array", "items": {"type": "string"}, "description": "Short verb phrases with their subject, e.g. 'a red car drives toward the camera'."},
        "camera": {
            "type": "object",
            "properties": {
                "shot_type": {"type": "string", "enum": _ENUM_SHOT},
                "camera_angle": {"type": "string", "enum": _ENUM_ANGLE},
                "movement": {"type": "string", "enum": _ENUM_MOVEMENT},
                "zoom": _str_or_null(),
                "stability": _str_or_null(),
                "is_aerial": {"type": ["boolean", "null"]},
            },
            "required": ["shot_type", "camera_angle", "movement", "zoom", "stability", "is_aerial"],
            "additionalProperties": False,
        },
        "visual_style": {
            "type": "object",
            "properties": {
                "composition": _str_or_null(),
                "lighting": _str_or_null(),
                "color": _str_or_null(),
                "depth_of_field": _str_or_null(),
                "framing": _str_or_null(),
                "perspective": _str_or_null(),
                "motion": _str_or_null(),
                "transitions": _str_or_null(),
            },
            "required": ["composition", "lighting", "color", "depth_of_field", "framing", "perspective", "motion", "transitions"],
            "additionalProperties": False,
        },
        "temporal_progression": _str_or_null(),
        "confidence": {"type": "number", "minimum": 0, "maximum": 1, "description": "Your own estimate that every statement above is visibly supported."},
    },
    "required": ["summary", "environment", "objects", "people", "actions", "camera", "visual_style", "temporal_progression", "confidence"],
    "additionalProperties": False,
}

FRAME_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "caption": {"type": "string"},
        "objects": {"type": "array", "items": {"type": "string"}},
        "actions": {"type": "array", "items": {"type": "string"}},
        "shot_type": {"type": "string", "enum": _ENUM_SHOT},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["caption", "objects", "actions", "shot_type", "confidence"],
    "additionalProperties": False,
}

VERIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["supported", "partially_supported", "unsupported", "unknown"]},
        "score": {"type": "number", "minimum": 0, "maximum": 1, "description": "Probability that the claim is fully supported by the frames."},
        "rationale": {"type": "string"},
    },
    "required": ["verdict", "score", "rationale"],
    "additionalProperties": False,
}


def scene_prompt(
    n_frames: int,
    timestamps: list[float],
    duration: float,
    transcript: str | None,
    ocr_texts: list[str],
    measured_camera: str | None,
    previous_summary: str | None,
) -> str:
    parts = [
        f"These {n_frames} frames come from ONE continuous shot lasting {duration:.1f} seconds, in temporal order, "
        f"sampled at {', '.join(f'{t:.1f}s' for t in timestamps)} (video time).",
        "Describe the shot as a structured JSON annotation with these fields: summary, environment, objects, people, actions, camera, visual_style, temporal_progression, confidence.",
        "For temporal_progression, say what changes between the first and last frame (movement, entrances/exits, state changes). Say null if nothing changes.",
        "Describe people only by count, clothing, posture and visible actions.",
    ]
    if measured_camera:
        parts.append(f"Optical-flow measurement suggests the camera motion is '{measured_camera}'. Use it only if the frames agree.")
    if ocr_texts:
        parts.append("On-screen text detected by OCR: " + "; ".join(f'"{t}"' for t in ocr_texts[:8]) + ".")
    if transcript:
        parts.append(f'Speech during this shot (context only; do not describe anything that is not visible): "{transcript[:600]}"')
    if previous_summary:
        parts.append(f"For continuity, the previous shot was described as: {previous_summary[:300]}")
    parts.append("Return JSON only.")
    return "\n".join(parts)


def frame_prompt(timestamp: float) -> str:
    return (
        f"This frame is from video time {timestamp:.1f}s. Return JSON with: caption (one precise sentence of what is visible), "
        "objects (list of visible object names), actions (visible actions, may be empty), shot_type, confidence (0-1)."
    )


def verify_prompt(claim: str, timestamps: list[float]) -> str:
    return (
        "You are a strict fact checker for video annotations. Look at the frames "
        f"(video times {', '.join(f'{t:.1f}s' for t in timestamps)}) and judge this claim:\n\n"
        f'"{claim}"\n\n'
        "Return JSON with verdict (supported / partially_supported / unsupported / unknown), score (0-1 probability the "
        "claim is fully supported by what is visible) and a one-sentence rationale. Use 'unknown' if the frames cannot settle it."
    )
