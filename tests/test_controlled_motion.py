"""Controlled-motion validation of the optical-flow camera-motion labels.

Synthetic frame sequences with a known camera move (pan right, zoom in, static) are pushed through
the real flow statistics (``scan.compute_flow_stats``) and classifier (``motion.classify_sample`` /
``dominant_motion``). Convention under test (motion.py): flow is the apparent motion of image
content, so a camera pan to the RIGHT moves content LEFT (negative ``flow_dx``) and must be
labelled ``pan_right``; expanding content (positive ``flow_div``) is ``zoom_in``.
"""

from __future__ import annotations

import time

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from video_dataset.frame_sampling.scan import _radial_unit_vectors, compute_flow_stats  # noqa: E402
from video_dataset.schemas.scene import ScanSample, SceneScan  # noqa: E402
from video_dataset.schemas.vision import MOVEMENT_TAXONOMY, CameraMovement, CameraMovementFamily  # noqa: E402
from video_dataset.vision.motion import camera_speed, classify_sample, dominant_motion  # noqa: E402

W, H = 320, 180
MIN_MOTION = 0.35  # HeuristicVisionAnalyzer default
N_FRAMES = 8
PAN_PX = 4.0  # camera pan speed: window moves right by this many px per frame
ZOOM_STEP = 1.03


def _texture(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
    blurred = cv2.GaussianBlur(noise, (0, 0), 1.5)
    return cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _pan_right_frames() -> list[np.ndarray]:
    """The camera pans right: the viewing window slides right over a wide world image."""
    world = _texture(H, W + int(PAN_PX * N_FRAMES) + 8)
    return [world[:, int(i * PAN_PX): int(i * PAN_PX) + W].copy() for i in range(N_FRAMES)]


def _zoom_in_frames() -> list[np.ndarray]:
    base = _texture(H, W)
    out = []
    for i in range(N_FRAMES):
        m = cv2.getRotationMatrix2D(((W - 1) / 2.0, (H - 1) / 2.0), 0.0, ZOOM_STEP**i)
        out.append(cv2.warpAffine(base, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
    return out


def _static_frames() -> list[np.ndarray]:
    base = _texture(H, W)
    return [base.copy() for _ in range(N_FRAMES)]


def _scan(frames: list[np.ndarray], fps: float = 4.0) -> SceneScan:
    rx, ry = _radial_unit_vectors(H, W)
    scan = SceneScan(scene_id="synthetic", analysis_width=W, analysis_fps=fps)
    for i, gray in enumerate(frames):
        dx = dy = mag = div = None
        if i > 0:
            dx, dy, mag, div = compute_flow_stats(frames[i - 1], gray, rx, ry)
        scan.samples.append(ScanSample(timestamp=i / fps, frame_index=i, change=0.0, brightness=float(gray.mean()), flow_dx=dx, flow_dy=dy, flow_mag=mag, flow_div=div))
    return scan


def _flow_samples(scan: SceneScan) -> list[ScanSample]:
    return [s for s in scan.samples if s.flow_mag is not None]


def test_pan_right_moves_content_left_and_is_labelled_pan_right():
    t0 = time.perf_counter()
    scan = _scan(_pan_right_frames())
    samples = _flow_samples(scan)
    assert len(samples) == N_FRAMES - 1
    # sign convention: camera right -> content left -> negative median dx, |dx| ~ PAN_PX, no vertical drift
    for s in samples:
        assert s.flow_dx is not None and s.flow_dx < -0.5 * PAN_PX
        assert abs(s.flow_dy or 0.0) < 0.5
        assert s.flow_mag is not None and 0.5 * PAN_PX < s.flow_mag < 2.0 * PAN_PX  # px/frame at analysis res
        assert classify_sample(s, MIN_MOTION) == CameraMovement.PAN_RIGHT
    label, consistency, mean_mag = dominant_motion(scan, MIN_MOTION)
    assert label == CameraMovement.PAN_RIGHT
    assert MOVEMENT_TAXONOMY[label] == (CameraMovementFamily.PAN, "right")
    assert consistency == 1.0
    assert mean_mag is not None and abs(mean_mag - PAN_PX) < 0.5 * PAN_PX
    word, frac = camera_speed(scan, label)
    # fraction of frame width per second = mean px/frame * fps / width
    assert frac is not None and abs(frac - mean_mag * scan.analysis_fps / W) < 1e-3
    assert word in ("slow", "moderate", "fast")
    assert time.perf_counter() - t0 < 2.0


def test_zoom_in_expands_content_and_is_labelled_zoom_in():
    scan = _scan(_zoom_in_frames())
    samples = _flow_samples(scan)
    for s in samples:
        assert s.flow_div is not None and s.flow_div > 0.35  # expanding = zoom in / forward
        assert abs(s.flow_dx or 0.0) < 1.0 and abs(s.flow_dy or 0.0) < 1.0  # no global translation
        assert classify_sample(s, MIN_MOTION) == CameraMovement.ZOOM_IN
    label, consistency, _ = dominant_motion(scan, MIN_MOTION)
    assert label == CameraMovement.ZOOM_IN
    assert MOVEMENT_TAXONOMY[label] == (CameraMovementFamily.ZOOM, "in")
    assert consistency == 1.0


def test_static_frames_are_labelled_static():
    scan = _scan(_static_frames())
    samples = _flow_samples(scan)
    for s in samples:
        assert s.flow_mag is not None and s.flow_mag < MIN_MOTION
        assert classify_sample(s, MIN_MOTION) == CameraMovement.STATIC
    label, consistency, _ = dominant_motion(scan, MIN_MOTION)
    assert label == CameraMovement.STATIC
    assert MOVEMENT_TAXONOMY[label] == (CameraMovementFamily.STATIC, None)
    assert consistency == 1.0
    assert camera_speed(scan, label) == (None, None)


def test_documented_convention_matches_schema_comment():
    """motion.py and scene.py must describe the same sign convention for flow_dx."""
    import inspect

    from video_dataset.schemas import scene as scene_mod
    from video_dataset.vision import motion as motion_mod

    doc = " ".join((motion_mod.__doc__ or "").split())
    assert "When the camera pans LEFT, content moves RIGHT (positive dx)" in doc
    assert "+x = content moves right" in inspect.getsource(scene_mod)
