"""Video-to-video pairs: a real source clip, an editing instruction and a real transformed target.

Only deterministic FFmpeg edits are produced, because their result can be measured: a colour grade,
film grain, a monochrome look and slow motion (the same frames at a lower frame rate). Generative edits (day-to-night,
weather, object or scene changes) need a generation model and are deliberately not faked here.
Every pair is validated on decoded frames: the target must really differ from the source, the
picture structure must be preserved and the timing must match the declared temporal change."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from video_dataset.utils.ffmpeg import run_ffmpeg
from video_dataset.v3.media import read_frames

TRANSFORMS: dict[str, dict[str, Any]] = {
    "cinematic_grade": {
        "filter": "eq=contrast=1.12:saturation=1.2:gamma=0.96,colorbalance=rs=-0.07:bs=0.09:rh=0.08:bh=-0.08",
        "instruction": "Apply a cinematic teal-and-orange colour grade with slightly higher contrast and saturation, keeping the content, framing, camera movement and timing unchanged.",
        "preserve": ["scene_content", "composition", "camera_trajectory", "motion_timing"],
        "modify": ["color_grade", "contrast", "saturation"],
        "speed_factor": 1.0,
    },
    "film_grain": {
        "filter": "noise=alls=16:allf=t+u",
        "instruction": "Add visible, temporally varying film grain to the footage without changing the content, colours, camera movement or timing.",
        "preserve": ["scene_content", "composition", "camera_trajectory", "motion_timing", "color_grade"],
        "modify": ["film_grain", "texture"],
        "speed_factor": 1.0,
    },
    "monochrome": {
        "filter": "hue=s=0,eq=contrast=1.08",
        "instruction": "Convert the footage to black and white with slightly higher contrast, keeping the content, framing, camera movement and timing unchanged.",
        "preserve": ["scene_content", "composition", "camera_trajectory", "motion_timing"],
        "modify": ["color_saturation", "color_grade"],
        "speed_factor": 1.0,
    },
    "slow_motion": {
        "filter": "setpts=2.0*PTS",
        "instruction": "Play the footage at half speed (2x slow motion) without changing the content, colours or framing.",
        "preserve": ["scene_content", "composition", "color_grade", "camera_trajectory"],
        "modify": ["motion_speed", "duration"],
        "speed_factor": 0.5,
        "temporal_method": "the same frames at half the frame rate (no interpolated or repeated frames)",
    },
}


def pick_transforms(clip_id: str, names: list[str], count: int) -> list[str]:
    """``count`` transforms for a clip, chosen by a stable hash so a re-run picks the same ones."""
    known = [n for n in names if n in TRANSFORMS]
    if not known or count <= 0:
        return []
    start = int(hashlib.sha1(clip_id.encode()).hexdigest(), 16) % len(known)
    return [known[(start + i) % len(known)] for i in range(min(count, len(known)))]


def apply_transform(src: Path, out: Path, name: str, cfg: Any, ffmpeg_path: str | None = None) -> None:
    spec = TRANSFORMS[name]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.mp4")
    run_ffmpeg(
        ["-i", str(src), "-map", "0:v:0", "-vf", spec["filter"], "-c:v", "libx264", "-preset", str(cfg.preset), "-crf", str(int(cfg.crf)), "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(tmp)],
        ffmpeg_path=ffmpeg_path,
    )
    os.replace(tmp, out)


def structure_correlation(a: np.ndarray, b: np.ndarray, size: int = 32) -> float:
    """Pearson correlation of the two frames' luminance layout (size x size block means). Robust to
    grading, grain and saturation changes; 1.0 when both frames are flat (nothing to compare)."""
    ga = cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (size, size), interpolation=cv2.INTER_AREA).astype(np.float64).ravel()
    gb = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (size, size), interpolation=cv2.INTER_AREA).astype(np.float64).ravel()
    if ga.std() < 1.0 and gb.std() < 1.0:
        return 1.0
    if ga.std() < 1e-6 or gb.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(ga, gb)[0, 1])


def validate_pair(src: Path, tgt: Path, name: str, src_metrics: Any, tgt_metrics: Any, v2v_cfg: Any, duration_tolerance: float, frame_tolerance: int) -> tuple[str, dict[str, Any]]:
    """('accepted' | 'pair_invalid' | 'decode_failed', measured checks)."""
    spec = TRANSFORMS[name]
    speed = float(spec["speed_factor"])
    checks: dict[str, Any] = {"transform": name}
    if not tgt_metrics.ok:
        return "decode_failed", {**checks, "error": tgt_metrics.error}
    expected_duration = src_metrics.duration / speed
    expected_frames = src_metrics.frame_count  # a speed change re-times the same frames; nothing is added or dropped
    checks["target_fps"] = round(tgt_metrics.fps, 3)
    checks["fps_ok"] = abs(tgt_metrics.fps - src_metrics.fps * speed) <= 0.05 * max(1.0, src_metrics.fps)
    checks["duration_error_seconds"] = round(abs(tgt_metrics.duration - expected_duration), 3)
    checks["frame_count_error"] = abs(tgt_metrics.frame_count - expected_frames)
    checks["duration_ok"] = checks["duration_error_seconds"] <= duration_tolerance + (1.0 / max(1.0, src_metrics.fps))
    checks["frame_count_ok"] = checks["frame_count_error"] <= frame_tolerance
    fractions = [0.1, 0.3, 0.5, 0.7, 0.9]
    a, _na, _fa = read_frames(src, fractions, max_width=4096)  # full size: grain must not be averaged away
    b, _nb, _fb = read_frames(tgt, fractions, max_width=4096)  # the same relative position = the same moment at any speed
    if not a or len(a) != len(b) or any(x.shape != y.shape for x, y in zip(a, b)):
        return "decode_failed", {**checks, "error": "aligned frames could not be read"}
    change = float(np.mean([np.mean(np.abs(x.astype(np.int16) - y.astype(np.int16))) for x, y in zip(a, b)])) / 255.0
    structure = min(structure_correlation(x, y) for x, y in zip(a, b))
    checks["mean_change"] = round(change, 4)
    checks["structure_correlation_min"] = round(structure, 4)
    checks["structure_preserved"] = structure >= float(v2v_cfg.min_structure_correlation)
    # a temporal edit changes timing, not pixels; every other edit must visibly change the picture
    checks["target_differs"] = True if speed != 1.0 else change >= float(v2v_cfg.min_change)
    timing = checks["duration_ok"] and checks["frame_count_ok"] and checks["fps_ok"]
    ok = timing and checks["structure_preserved"] and checks["target_differs"]
    checks["alignment_status"] = "verified" if timing and checks["structure_preserved"] else "failed"
    return ("accepted" if ok else "pair_invalid"), checks
