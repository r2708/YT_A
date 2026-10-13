"""Cross-field consistency checks on one scene analysis.

Every rule compares two things the pipeline already recorded (a label and the measurement it should
agree with, or two labels that cannot both hold). A contradiction means at least one of them is
wrong, so the shot must not be served as a clean positive. Returns ``(hard, soft)`` lists of
human-readable issues: hard ones reject (or review, see ``validation.reject_contradictions``), soft
ones only go to review.
"""

from __future__ import annotations

import re

from video_dataset.schemas.vision import MOVEMENT_TAXONOMY, CameraMovement, CameraMovementFamily, SceneAnalysis
from video_dataset.vision.subject_motion import TRACKING_SET

# words in free text that commit to a camera movement family (+ direction)
_TEXT_MOVEMENT = [
    (re.compile(r"\bpan(?:s|ning)?\s+(?:to\s+the\s+)?left\b"), CameraMovement.PAN_LEFT),
    (re.compile(r"\bpan(?:s|ning)?\s+(?:to\s+the\s+)?right\b"), CameraMovement.PAN_RIGHT),
    (re.compile(r"\btilt(?:s|ing)?\s+up\b"), CameraMovement.TILT_UP),
    (re.compile(r"\btilt(?:s|ing)?\s+down\b"), CameraMovement.TILT_DOWN),
    (re.compile(r"\bzoom(?:s|ing)?\s+in\b"), CameraMovement.ZOOM_IN),
    (re.compile(r"\bzoom(?:s|ing)?\s+out\b"), CameraMovement.ZOOM_OUT),
    (re.compile(r"\b(?:static|stationary|fixed|still)\s+camera\b|\bcamera\s+(?:is|stays|remains)\s+(?:static|still|stationary|fixed)\b"), CameraMovement.STATIC),
]
_OPPOSITE_DIRECTION = {"left": "right", "right": "left", "up": "down", "down": "up", "in": "out", "out": "in"}
_LIGHT_RANGES = {"dark": (0.0, 50.0), "dim": (50.0, 95.0), "normal": (95.0, 170.0), "bright": (170.0, 256.0)}
_LIGHT_SLACK = 20.0  # a label is contradicted only when the measured mean is this far outside its band


def _family(m: CameraMovement | str | None) -> tuple[CameraMovementFamily, str | None]:
    try:
        return MOVEMENT_TAXONOMY[CameraMovement(str(m))]
    except (ValueError, KeyError):
        return CameraMovementFamily.UNKNOWN, None


def text_movements(text: str) -> list[CameraMovement]:
    low = text.lower()
    return [label for rx, label in _TEXT_MOVEMENT if rx.search(low)]


def _light_conflict(label: str | None, mean: float | None) -> str | None:
    if not label or mean is None:
        return None
    band = _LIGHT_RANGES.get(str(label).strip().lower())
    if band is None:
        return None
    lo, hi = band
    if mean < lo - _LIGHT_SLACK or mean > hi + _LIGHT_SLACK:
        return f"lighting labelled '{label}' but the measured mean brightness is {mean:.0f}/255"
    return None


def check_scene_consistency(a: SceneAnalysis) -> tuple[list[str], list[str]]:
    hard: list[str] = []
    soft: list[str] = []
    cam = a.camera
    m = a.measurements
    sm = a.subject_motion
    family, direction = _family(cam.movement)

    # --- camera label vs the measured optical flow (both known, families disagree) -------------
    if m is not None and m.camera_motion_label and cam.movement not in (CameraMovement.UNKNOWN, CameraMovement.TRACKING) and cam.movement not in TRACKING_SET:
        flow_family, flow_dir = _family(m.camera_motion_label)
        if flow_family not in (CameraMovementFamily.UNKNOWN, CameraMovementFamily.COMPLEX, CameraMovementFamily.HANDHELD) and family not in (CameraMovementFamily.UNKNOWN, CameraMovementFamily.COMPLEX):
            consistent = (m.camera_motion_consistency or 0.0) >= 0.6
            if family == flow_family and direction and flow_dir and direction != flow_dir and consistent:
                hard.append(f"camera labelled {cam.movement} but optical flow measures {m.camera_motion_label}")
            elif family != flow_family and consistent and {family, flow_family} != {CameraMovementFamily.ZOOM, CameraMovementFamily.DOLLY}:
                (hard if flow_family == CameraMovementFamily.STATIC or family == CameraMovementFamily.STATIC else soft).append(
                    f"camera labelled {cam.movement} but optical flow measures {m.camera_motion_label}"
                )

    # --- camera vs subject motion ---------------------------------------------------------------
    if sm is not None:
        if cam.movement == CameraMovement.STATIC and sm.camera_relation in ("tracked", "moves_with_camera"):
            hard.append(f"camera labelled static but the subject motion says '{sm.label}' ({sm.camera_relation})")
        if cam.movement in TRACKING_SET and sm.camera_relation == "camera_static":
            hard.append(f"camera labelled {cam.movement} but the subject motion was measured against a static camera")
        if sm.label in ("moves_left", "moves_right", "moves_up", "moves_down") and sm.relative_velocity:
            vx, vy = (list(sm.relative_velocity) + [0.0, 0.0])[:2]
            want = sm.label.replace("moves_", "")
            got = ("right" if vx > 0 else "left") if abs(vx) >= abs(vy) else ("down" if vy > 0 else "up")
            if (abs(vx) + abs(vy)) > 0 and got != want:
                hard.append(f"subject motion label '{sm.label}' disagrees with its own relative velocity ({vx:.3f}, {vy:.3f})")

    # --- speed / stability only make sense with a moving camera ---------------------------------
    if cam.movement == CameraMovement.STATIC and cam.speed in ("moderate", "fast"):
        soft.append(f"camera labelled static but with a '{cam.speed}' movement speed")
    if cam.movement == CameraMovement.HANDHELD and cam.stability in ("tripod", "stabilized", "gimbal", "steadicam"):
        soft.append(f"camera labelled handheld but stabilization '{cam.stability}'")

    # --- lighting labels vs the measured brightness ---------------------------------------------
    if m is not None:
        for label in {a.environment.lighting, m.lighting_level}:
            msg = _light_conflict(label, m.brightness_mean)
            if msg:
                (hard if label == m.lighting_level else soft).append(msg)
        if a.environment.lighting and m.lighting_level and str(a.environment.lighting).strip().lower() in _LIGHT_RANGES and str(a.environment.lighting).strip().lower() != m.lighting_level:
            soft.append(f"environment.lighting '{a.environment.lighting}' differs from the measured level '{m.lighting_level}'")
        if (a.environment.time_of_day or "").strip().lower() in ("night", "nighttime") and m.brightness_mean is not None and m.brightness_mean > 170:
            soft.append(f"time of day 'night' on a bright frame (mean {m.brightness_mean:.0f}/255)")

    # --- people vs objects ------------------------------------------------------------------------
    if a.people and a.people.count and a.objects and not any(o.name.lower() in ("person", "people", "man", "woman", "child") for o in a.objects):
        soft.append(f"{a.people.count} people counted but no person among the detected objects")

    # --- free text vs the camera field (caption-annotation alignment) -----------------------------
    texts = [a.summary or "", a.temporal_progression or "", a.visual_style.motion or "", *a.actions]
    claimed = {lab for t in texts for lab in text_movements(t)}
    if claimed and cam.movement != CameraMovement.UNKNOWN:
        for lab in claimed:
            lab_family, lab_dir = _family(lab)
            if lab == cam.movement or lab_family == family and (lab_dir is None or direction is None):
                continue
            if lab_family == family and lab_dir and direction and lab_dir == _OPPOSITE_DIRECTION.get(direction):
                hard.append(f"caption says '{lab}' but the camera field is {cam.movement}")
            elif lab == CameraMovement.STATIC and family not in (CameraMovementFamily.STATIC, CameraMovementFamily.UNKNOWN):
                hard.append(f"caption says the camera is static but the camera field is {cam.movement}")
            elif lab_family != family and family == CameraMovementFamily.STATIC:
                hard.append(f"caption says '{lab}' but the camera field is static")
            elif lab_family != family and lab_family != CameraMovementFamily.STATIC and family in (CameraMovementFamily.PAN, CameraMovementFamily.TILT, CameraMovementFamily.ZOOM) and lab_family in (CameraMovementFamily.PAN, CameraMovementFamily.TILT, CameraMovementFamily.ZOOM):
                soft.append(f"caption says '{lab}' but the camera field is {cam.movement}")
    return hard, soft
