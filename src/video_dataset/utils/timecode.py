"""Interval utilities."""

from __future__ import annotations


def interval_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0:
        return 1.0 if a_start == b_start else 0.0
    return inter / union


def interval_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def approx_equal(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol
