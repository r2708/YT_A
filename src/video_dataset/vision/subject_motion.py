"""Subject motion separated from camera motion, and box-derived composition (position, scale).

Inputs are detector boxes on the analysed frames (``cv_models``) and the optical-flow scan of the
scene. Units: fractions of the frame width per second, in frame coordinates (+x right, +y down).

* box velocity: how the main subject's box centre moves across the frame
* camera velocity: how the background content moves (global optical flow)
* relative velocity = box - camera: the subject's own motion

A subject that keeps its place in the frame while the background streams past is being tracked.
A subject whose box moves exactly with the background is static in the world; the camera pans past it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from video_dataset.schemas.common import ConfidenceSource
from video_dataset.schemas.scene import SceneScan
from video_dataset.schemas.vision import CameraMovement, SceneAnalysis, SubjectMotion

TRACKING_SET = {CameraMovement.TRACKING, CameraMovement.TRACKING_LEFT, CameraMovement.TRACKING_RIGHT, CameraMovement.TRACKING_FORWARD, CameraMovement.TRACKING_BACKWARD}
TRACKABLE = {CameraMovement.PAN_LEFT, CameraMovement.PAN_RIGHT, CameraMovement.TILT_UP, CameraMovement.TILT_DOWN, CameraMovement.HANDHELD, CameraMovement.COMPLEX, *TRACKING_SET}
ZOOMS = {CameraMovement.ZOOM_IN, CameraMovement.ZOOM_OUT}
EPS = 0.03  # fraction of the frame per second below which motion counts as none
SCALE_EPS = 0.02  # area-fraction change per second that counts as approaching / receding


# ------------------------------------------------------------------------------- composition
def position_label(cx: float, cy: float) -> str:
    col = "left" if cx < 1 / 3 else ("right" if cx > 2 / 3 else "center")
    row = "upper" if cy < 1 / 3 else ("lower" if cy > 2 / 3 else "middle")
    if row == "middle":
        return col
    if col == "center":
        return row
    return f"{row}-{col}"


def scale_class(area_fraction: float) -> str:
    if area_fraction < 0.01:
        return "tiny"
    if area_fraction < 0.05:
        return "small"
    if area_fraction < 0.2:
        return "medium"
    if area_fraction < 0.5:
        return "large"
    return "dominant"


def normalized_box(det: dict[str, Any]) -> tuple[float, float, float] | None:
    """(cx, cy, area_fraction) in 0..1 from a detection with ``bbox`` [x1,y1,x2,y2] and ``frame_size`` [w,h]."""
    bbox = det.get("bbox")
    size = det.get("frame_size")
    if not bbox or not size or len(bbox) != 4 or size[0] <= 0 or size[1] <= 0:
        return None
    w, h = float(size[0]), float(size[1])
    x1, y1, x2, y2 = (float(v) for v in bbox)
    cx = min(1.0, max(0.0, (x1 + x2) / 2.0 / w))
    cy = min(1.0, max(0.0, (y1 + y2) / 2.0 / h))
    area = min(1.0, max(0.0, (x2 - x1) * (y2 - y1) / (w * h)))
    return cx, cy, area


# ------------------------------------------------------------------------------- camera flow
def camera_velocity(scan: SceneScan | None, t0: float | None = None, t1: float | None = None) -> tuple[float, float] | None:
    """Median background motion (vx, vy) as fraction of frame width per second over [t0, t1]."""
    if scan is None or not scan.samples or not scan.analysis_width or not scan.analysis_fps:
        return None
    samples = [s for s in scan.samples if s.flow_dx is not None and s.flow_dy is not None and (t0 is None or s.timestamp >= t0 - 1e-6) and (t1 is None or s.timestamp <= t1 + 1e-6)]
    if len(samples) < 2:
        samples = [s for s in scan.samples if s.flow_dx is not None and s.flow_dy is not None]
    if not samples:
        return None
    k = float(scan.analysis_fps) / float(scan.analysis_width)
    vx = float(np.median([s.flow_dx for s in samples])) * k  # type: ignore[misc]
    vy = float(np.median([s.flow_dy for s in samples])) * k  # type: ignore[misc]
    return vx, vy


def _slope(ts: list[float], ys: list[float]) -> float:
    if len(ts) < 2 or max(ts) - min(ts) <= 0:
        return 0.0
    return float(np.polyfit(ts, ys, 1)[0])


# ------------------------------------------------------------------------------- analysis
def _primary_subject(per_frame: list[list[dict[str, Any]]]) -> str | None:
    """Class with the largest mean box area weighted by presence; people win ties."""
    score: dict[str, float] = {}
    presence: dict[str, int] = {}
    for dets in per_frame:
        best: dict[str, float] = {}
        for d in dets:
            nb = normalized_box(d)
            if nb is None:
                continue
            best[d["name"]] = max(best.get(d["name"], 0.0), nb[2])
        for name, area in best.items():
            score[name] = score.get(name, 0.0) + area
            presence[name] = presence.get(name, 0) + 1
    if not score:
        return None
    n = len(per_frame)
    return max(score, key=lambda k: (score[k] / n * (presence[k] / n), k == "person", -len(k)))


def analyze_subject_motion(
    per_frame: list[list[dict[str, Any]]],
    timestamps: list[float],
    scan: SceneScan | None,
    camera_label: CameraMovement,
    min_frames: int = 3,
) -> SubjectMotion | None:
    if len(per_frame) != len(timestamps) or len(per_frame) < min_frames:
        return None
    subject = _primary_subject(per_frame)
    if subject is None:
        return None
    ts: list[float] = []
    cxs: list[float] = []
    cys: list[float] = []
    areas: list[float] = []
    for dets, t in zip(per_frame, timestamps):
        boxes = [nb for d in dets if d["name"] == subject and (nb := normalized_box(d)) is not None]
        if not boxes:
            continue
        cx, cy, area = max(boxes, key=lambda b: b[2])
        ts.append(float(t))
        cxs.append(cx)
        cys.append(cy)
        areas.append(area)
    if len(ts) < min_frames or max(ts) - min(ts) <= 0:
        return None
    vx, vy = _slope(ts, cxs), _slope(ts, cys)
    da = _slope(ts, areas)
    cam = camera_velocity(scan, min(ts), max(ts)) or (0.0, 0.0)
    rel = (vx - cam[0], vy - cam[1])
    box_speed = float(np.hypot(vx, vy))
    cam_speed = float(np.hypot(*cam))
    rel_speed = float(np.hypot(*rel))
    camera_moving = cam_speed >= EPS and camera_label in TRACKABLE

    if camera_moving and box_speed < max(0.35 * cam_speed, EPS):
        label, relation = "tracked_by_camera", "tracked"
    elif camera_moving and rel_speed < max(0.35 * cam_speed, EPS):
        label, relation = "panned_past", "moves_with_camera"
    elif rel_speed >= EPS:
        label = ("moves_right" if rel[0] > 0 else "moves_left") if abs(rel[0]) >= abs(rel[1]) else ("moves_down" if rel[1] > 0 else "moves_up")
        if camera_moving:
            relation = "moves_against_camera" if (rel[0] * cam[0] + rel[1] * cam[1]) < 0 else "independent"
        else:
            relation = "camera_static" if camera_label in (CameraMovement.STATIC, CameraMovement.UNKNOWN) else "independent"
    elif abs(da) >= SCALE_EPS and camera_label not in ZOOMS:
        label = "approaches" if da > 0 else "recedes"
        relation = "camera_static" if not camera_moving else "independent"
    else:
        label = "static"
        relation = "camera_static" if not camera_moving else "independent"
    frame_fraction = len(ts) / len(per_frame)
    return SubjectMotion(
        subject=subject,
        label=label,
        camera_relation=relation,
        box_velocity=[round(vx, 4), round(vy, 4)],
        camera_velocity=[round(cam[0], 4), round(cam[1], 4)],
        relative_velocity=[round(rel[0], 4), round(rel[1], 4)],
        scale_change_per_second=round(da, 4),
        n_frames=len(ts),
        confidence=round(frame_fraction * min(1.0, len(ts) / 4.0), 3),
        confidence_source=ConfidenceSource.MEASUREMENT,
    )


# ------------------------------------------------------------------------------- phrasing
def _noun(subject: str) -> str:
    return "person" if subject == "person" else subject


def subject_motion_phrase(sm: SubjectMotion) -> str:
    n = _noun(sm.subject)
    if sm.label == "tracked_by_camera":
        return f"the camera tracks the {n}, which stays in place in the frame while the background moves"
    if sm.label == "panned_past":
        return f"the {n} stays still while the camera moves past it"
    if sm.label == "static":
        return f"the {n} stays in place"
    if sm.label in ("approaches", "recedes"):
        return f"the {n} {'grows larger, approaching the camera' if sm.label == 'approaches' else 'shrinks, moving away from the camera'}"
    direction = sm.label.replace("moves_", "")
    tail = ""
    if sm.camera_relation == "moves_against_camera":
        tail = " against the camera movement"
    elif sm.camera_relation == "camera_static":
        tail = " while the camera stays still"
    return f"the {n} moves {direction}{tail}"


def tracking_direction(movement: CameraMovement, sm: SubjectMotion) -> CameraMovement:
    """Which way the camera travels while it tracks: from the pan direction it was labelled with, else
    from the background flow (content streaming right = camera moving left), else undetermined."""
    if movement == CameraMovement.PAN_LEFT:
        return CameraMovement.TRACKING_LEFT
    if movement == CameraMovement.PAN_RIGHT:
        return CameraMovement.TRACKING_RIGHT
    cam = list(sm.camera_velocity) + [0.0, 0.0]
    if abs(cam[0]) >= EPS and abs(cam[0]) >= abs(cam[1]):
        return CameraMovement.TRACKING_LEFT if cam[0] > 0 else CameraMovement.TRACKING_RIGHT
    return CameraMovement.TRACKING


def apply_subject_motion(analysis: SceneAnalysis, sm: SubjectMotion | None) -> None:
    """Attach the measurement; promote pan/handheld to TRACKING when the subject is held in frame."""
    if sm is None:
        return
    analysis.subject_motion = sm
    if sm.label == "tracked_by_camera" and analysis.camera.movement in TRACKABLE and analysis.camera.movement not in TRACKING_SET:
        confs = [c for c in (analysis.camera.confidence, sm.confidence) if c is not None]
        analysis.camera = analysis.camera.model_copy(update={
            "movement": tracking_direction(analysis.camera.movement, sm),
            "tracked_subject": sm.subject,
            "confidence": round(min(confs), 3) if confs else None,
            "confidence_source": ConfidenceSource.MEASUREMENT,
        })
        analysis.camera = type(analysis.camera).model_validate(analysis.camera.model_dump())  # re-derive the taxonomy block
    phrase = subject_motion_phrase(sm)
    action = phrase[0].upper() + phrase[1:]
    if action not in analysis.actions and sm.label != "static":
        analysis.actions.append(action)
    analysis.visual_style.motion = (analysis.visual_style.motion + "; " if analysis.visual_style.motion else "") + phrase
    if analysis.summary:
        analysis.summary = analysis.summary.rstrip() + " " + action + "."
