"""Bounded frame decoding with OpenCV: a handful of frames per clip, never the whole file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class DecodedFrame:
    index: int
    timestamp_seconds: float
    image: np.ndarray  # BGR


def frame_stats(image: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(gray.mean()), float(gray.std())


def is_black(image: np.ndarray, mean_max: float, std_max: float) -> bool:
    mean, std = frame_stats(image)
    return mean <= mean_max and std <= std_max


def decode_sampled_frames(path: Path, count: int, max_width: int = 320) -> tuple[list[DecodedFrame], int, float]:
    """``count`` evenly spaced frames (downscaled to ``max_width``), the frame count OpenCV
    reports and its fps. Returns ([], 0, 0.0) when the file cannot be opened."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return [], 0, 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if n <= 0:
            n = _count_by_reading(cap)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if n <= 0:
            return [], 0, fps
        want = sorted({int(round(i * (n - 1) / max(1, count - 1))) for i in range(max(1, count))})
        out: list[DecodedFrame] = []
        for idx in want:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, img = cap.read()
            if not ok or img is None:
                continue
            out.append(DecodedFrame(idx, idx / fps if fps > 0 else 0.0, _small(img, max_width)))
        return out, n, fps
    finally:
        cap.release()


def decode_from(path: Path, start_seconds: float, max_seconds: float):  # type: ignore[no-untyped-def]
    """Yield full-resolution frames from ``start_seconds`` for at most ``max_seconds``."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
        first = int(round(max(0.0, start_seconds) * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        idx = first
        last = first + int(round(max_seconds * fps))
        while idx <= last:
            ok, img = cap.read()
            if not ok or img is None:
                return
            yield DecodedFrame(idx, idx / fps, img)
            idx += 1
    finally:
        cap.release()


def _count_by_reading(cap: cv2.VideoCapture, cap_frames: int = 100000) -> int:
    n = 0
    while n < cap_frames:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    return n


def _small(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= width:
        return img
    return cv2.resize(img, (width, max(1, int(round(h * width / w)))), interpolation=cv2.INTER_AREA)
