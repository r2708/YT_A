"""Turn optical-flow statistics into camera-motion labels with a measured consistency score.

Convention: flow is the apparent motion of image content. When the camera pans LEFT, content moves
RIGHT (positive dx). When the camera tilts UP, content moves DOWN (positive dy). Positive radial
divergence = content expanding = zoom in / camera moving forward.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from video_dataset.schemas.scene import ScanSample, SceneScan
from video_dataset.schemas.vision import CameraMovement


@dataclass
class MotionSegment:
    label: CameraMovement
    start_time: float
    end_time: float
    consistency: float  # fraction of samples in the run agreeing with the label
    mean_magnitude: float
    n_samples: int


def classify_sample(s: ScanSample, min_motion: float) -> CameraMovement | None:
    if s.flow_mag is None or s.flow_dx is None or s.flow_dy is None:
        return None
    if s.flow_mag < min_motion:
        return CameraMovement.STATIC
    div = s.flow_div or 0.0
    dx, dy = s.flow_dx, s.flow_dy
    translation = float(np.hypot(dx, dy))
    # global translation should account for most of the flow to call it a pan/tilt
    if abs(div) > 0.35 and translation < 0.6 * s.flow_mag:
        return CameraMovement.ZOOM_IN if div > 0 else CameraMovement.ZOOM_OUT
    if translation < 0.35 * s.flow_mag:
        # lots of motion but no coherent global direction: moving subjects or shaky camera
        return CameraMovement.HANDHELD if s.flow_mag > 2 * min_motion else CameraMovement.STATIC
    if abs(dx) >= abs(dy):
        return CameraMovement.PAN_LEFT if dx > 0 else CameraMovement.PAN_RIGHT
    return CameraMovement.TILT_UP if dy > 0 else CameraMovement.TILT_DOWN


def label_samples(scan: SceneScan, min_motion: float) -> list[tuple[ScanSample, CameraMovement]]:
    out = []
    for s in scan.samples:
        lab = classify_sample(s, min_motion)
        if lab is not None:
            out.append((s, lab))
    return out


def dominant_motion(scan: SceneScan | None, min_motion: float) -> tuple[CameraMovement, float, float | None]:
    """(label, consistency, mean magnitude) for the whole scene."""
    if scan is None:
        return CameraMovement.UNKNOWN, 0.0, None
    labelled = label_samples(scan, min_motion)
    if not labelled:
        return CameraMovement.UNKNOWN, 0.0, None
    mags = [s.flow_mag for s, _ in labelled if s.flow_mag is not None]
    mean_mag = float(np.mean(mags)) if mags else None
    moving = [lab for _, lab in labelled if lab != CameraMovement.STATIC]
    if len(moving) < max(2, 0.3 * len(labelled)):
        static_frac = sum(1 for _, lab in labelled if lab == CameraMovement.STATIC) / len(labelled)
        return CameraMovement.STATIC, round(static_frac, 3), mean_mag
    counts: dict[CameraMovement, int] = {}
    for lab in moving:
        counts[lab] = counts.get(lab, 0) + 1
    label = max(counts, key=lambda k: counts[k])
    consistency = counts[label] / len(labelled)
    if consistency < 0.4 and len(counts) >= 3:
        return CameraMovement.COMPLEX, round(1.0 - consistency, 3), mean_mag
    return label, round(consistency, 3), mean_mag


def motion_segments(scan: SceneScan | None, min_motion: float, min_run: int = 2, min_consistency: float = 0.6) -> list[MotionSegment]:
    """Contiguous runs of the same non-static camera motion inside a scene."""
    if scan is None:
        return []
    labelled = label_samples(scan, min_motion)
    if not labelled:
        return []
    step = 1.0 / scan.analysis_fps if scan.analysis_fps else 0.25
    segments: list[MotionSegment] = []
    run: list[tuple[ScanSample, CameraMovement]] = []

    def flush() -> None:
        if len(run) < min_run:
            return
        labs = [lab for _, lab in run]
        label = max(set(labs), key=labs.count)
        cons = labs.count(label) / len(labs)
        if label == CameraMovement.STATIC or cons < min_consistency:
            return
        mags = [s.flow_mag for s, _ in run if s.flow_mag is not None]
        segments.append(
            MotionSegment(
                label=label,
                start_time=round(run[0][0].timestamp - step, 3) if run[0][0].timestamp - step >= 0 else run[0][0].timestamp,
                end_time=round(run[-1][0].timestamp, 3),
                consistency=round(cons, 3),
                mean_magnitude=round(float(np.mean(mags)), 4) if mags else 0.0,
                n_samples=len(run),
            )
        )

    # Merge runs allowing single-sample hiccups (e.g. one HANDHELD sample inside a pan)
    for s, lab in labelled:
        if not run:
            run = [(s, lab)]
            continue
        labels_in_run = [lab_ for _, lab_ in run]
        current = max(set(labels_in_run), key=labels_in_run.count)
        if lab == current or (lab != CameraMovement.STATIC and current != CameraMovement.STATIC and len(run) >= 2 and run[-1][1] == current and lab in (CameraMovement.HANDHELD, CameraMovement.COMPLEX)):
            run.append((s, lab))
        else:
            flush()
            run = [(s, lab)]
    flush()
    return segments


def camera_speed(scan: SceneScan | None, label: CameraMovement) -> tuple[str | None, float | None]:
    """(speed label, mean global translation as fraction of frame width per second) for a moving camera.

    slow < 0.08 frame/s, moderate < 0.25 frame/s, fast otherwise. None when the camera is static,
    unknown, or the scan carries no flow.
    """
    if scan is None or label in (CameraMovement.STATIC, CameraMovement.UNKNOWN) or not scan.analysis_width or not scan.analysis_fps:
        return None, None
    mags = [s.flow_mag for s in scan.samples if s.flow_mag is not None]
    if not mags:
        return None, None
    frac = float(np.mean(mags)) * float(scan.analysis_fps) / float(scan.analysis_width)
    word = "slow" if frac < 0.08 else ("moderate" if frac < 0.25 else "fast")
    return word, round(frac, 4)


MOVEMENT_PHRASES: dict[CameraMovement, str] = {
    CameraMovement.STATIC: "the camera is static",
    CameraMovement.PAN_LEFT: "the camera pans left",
    CameraMovement.PAN_RIGHT: "the camera pans right",
    CameraMovement.TILT_UP: "the camera tilts up",
    CameraMovement.TILT_DOWN: "the camera tilts down",
    CameraMovement.ZOOM_IN: "the camera zooms in or moves forward",
    CameraMovement.ZOOM_OUT: "the camera zooms out or moves backward",
    CameraMovement.DOLLY_IN: "the camera dollies in",
    CameraMovement.DOLLY_OUT: "the camera dollies out",
    CameraMovement.TRACKING: "the camera tracks the subject",
    CameraMovement.TRACKING_LEFT: "the camera tracks the subject, moving left",
    CameraMovement.TRACKING_RIGHT: "the camera tracks the subject, moving right",
    CameraMovement.TRACKING_FORWARD: "the camera tracks forward",
    CameraMovement.TRACKING_BACKWARD: "the camera tracks backward",
    CameraMovement.ORBIT: "the camera orbits the subject",
    CameraMovement.ORBIT_CLOCKWISE: "the camera orbits the subject clockwise",
    CameraMovement.ORBIT_COUNTERCLOCKWISE: "the camera orbits the subject counterclockwise",
    CameraMovement.CRANE: "the camera cranes vertically",
    CameraMovement.CRANE_UP: "the camera cranes up",
    CameraMovement.CRANE_DOWN: "the camera cranes down",
    CameraMovement.HANDHELD: "the camera is handheld and unsteady",
    CameraMovement.FPV: "the camera is a first-person view",
    CameraMovement.DRONE: "the camera is a flying drone",
    CameraMovement.COMPLEX: "the camera makes a complex movement",
    CameraMovement.UNKNOWN: "camera movement could not be determined",
}
