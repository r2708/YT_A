"""Verify a cut clip against the source it was cut from.

``ffmpeg`` exiting 0 says nothing about *where* a clip starts: a stream copy begins at the nearest
keyframe, a variable-frame-rate source can shift timestamps, and a wrong ``-ss`` silently cuts the
neighbouring shot. The only trustworthy check decodes the file and compares it with the source:

1. the clip's first and last frames are decoded (OpenCV) and perceptually hashed (dHash);
2. source frames inside ``[boundary - search, boundary + search]`` are decoded at the same
   analysis width and hashed;
3. the source frame with the smallest Hamming distance (ties -> closest to the requested time)
   gives the clip's *actual* start / end in source time.

The error is ``max(|actual_start - start|, |actual_end - end|)``; the check passes when it is within
the configured tolerance. In a perfectly static shot every candidate hashes the same, so the match
falls back to the requested time, which is the honest answer: a shift inside a static run changes
no pixel. When no candidate is close (distance above ``max_hamming``) the result is ``error``
rather than a guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from video_dataset.utils.imagehash import dhash, hamming
from video_dataset.utils.logging import get_logger

log = get_logger("frame_sampling.boundary")


@dataclass
class BoundaryResult:
    state: str  # passed | failed | error | not_run
    actual_start: float | None = None
    actual_end: float | None = None
    error_seconds: float | None = None
    start_distance: int | None = None  # Hamming distance of the best start match
    end_distance: int | None = None
    detail: str | None = None


def _small(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= width:
        return img
    return cv2.resize(img, (width, max(1, int(round(h * width / w)))), interpolation=cv2.INTER_AREA)


def clip_edge_frames(path: Path, width: int = 160) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    """(first frame, last frame, frame count) of a clip, downscaled. None when the file cannot be decoded."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None, None, 0
        ok, first = cap.read()
        if not ok or first is None:
            return None, None, 0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        last = None
        if n > 1:
            # the frame count can overshoot (container metadata); step back until a frame decodes
            for idx in range(n - 1, max(0, n - 6), -1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, img = cap.read()
                if ok and img is not None:
                    last = img
                    n = idx + 1
                    break
        if last is None:
            # short / unindexed file: read it through
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            count = 0
            img = None
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                img, count = frame, count + 1
            last, n = (img if img is not None else first), max(count, 1)
        return _small(first, width), _small(last, width), n
    finally:
        cap.release()


def source_hashes(video: Path, fps: float, t_lo: float, t_hi: float, width: int = 160) -> list[tuple[float, str]]:
    """(timestamp, dHash) of every source frame in [t_lo, t_hi]."""
    cap = cv2.VideoCapture(str(video))
    out: list[tuple[float, str]] = []
    try:
        if not cap.isOpened() or fps <= 0:
            return out
        first = max(0, int(np.floor(max(0.0, t_lo) * fps)))
        last = int(np.ceil(max(t_lo, t_hi) * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if pos < 0 or abs(pos - first) > 5:
            pos = first
        idx = pos
        while idx <= last:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            out.append((idx / fps, dhash(_small(frame, width))))
            idx += 1
    finally:
        cap.release()
    return out


def _best(target: str, candidates: list[tuple[float, str]], requested: float) -> tuple[float | None, int | None]:
    if not candidates:
        return None, None
    best_t, best_d = None, 10**9
    for t, h in candidates:
        d = hamming(target, h)
        if d < best_d or (d == best_d and best_t is not None and abs(t - requested) < abs(best_t - requested)):
            best_t, best_d = t, d
    return best_t, best_d


def check_clip_boundaries(
    video: Path, clip: Path, start: float, end: float, fps: float, *, tolerance: float = 0.25, search: float = 1.0, max_hamming: int = 14, width: int = 160,
) -> BoundaryResult:
    """Compare the clip's first / last frames with the source around the requested boundaries."""
    try:
        first, last, _n = clip_edge_frames(clip, width)
        if first is None or last is None:
            return BoundaryResult("error", detail="clip could not be decoded")
        h_first, h_last = dhash(first), dhash(last)
        step = 1.0 / fps if fps > 0 else 0.04
        cands_start = source_hashes(video, fps, start - search, start + search, width)
        cands_end = source_hashes(video, fps, end - search - step, end + search, width)
        t0, d0 = _best(h_first, cands_start, start)
        # the last frame of a half-open [start, end) clip is the source frame just before `end`
        t1, d1 = _best(h_last, cands_end, end - step)
        if t0 is None or t1 is None:
            return BoundaryResult("error", detail="no source frames decoded around the boundaries")
        if (d0 is not None and d0 > max_hamming) or (d1 is not None and d1 > max_hamming):
            return BoundaryResult("error", actual_start=t0, actual_end=t1 + step, start_distance=d0, end_distance=d1, detail=f"no close source frame (distances {d0}/{d1} > {max_hamming})")
        actual_end = t1 + step
        err = max(abs(t0 - start), abs(actual_end - end))
        state = "passed" if err <= tolerance + 1e-6 else "failed"
        detail = None if state == "passed" else f"first frame at {t0:.3f}s (requested {start:.3f}s), last at {actual_end:.3f}s (requested {end:.3f}s)"
        return BoundaryResult(state, round(t0, 3), round(actual_end, 3), round(err, 3), d0, d1, detail)
    except Exception as exc:  # the check must never take the stage down
        log.debug("boundary check failed on %s: %s", clip.name, exc)
        return BoundaryResult("error", detail=f"{type(exc).__name__}: {str(exc)[:120]}")
