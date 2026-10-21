"""Lightweight, objective frame metrics. Every metric reports a status and a reason; nothing is
reduced to one opaque number without its components."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from dataset_factory.config import MediaRules
from dataset_factory.media.decode import DecodedFrame, is_black
from dataset_factory.media.fingerprints import duplicate_fraction, frame_hashes


@dataclass
class VisualQuality:
    status: str  # passed | failed | needs_review | not_checked
    score: float | None  # 1 - failed_metrics / metrics_checked
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    fingerprint: list[str] = field(default_factory=list)


def sharpness(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def measure(frames: list[DecodedFrame], rules: MediaRules) -> VisualQuality:
    if not frames:
        return VisualQuality("not_checked", None, ["no_frames_decoded"])
    hashes = frame_hashes(frames)
    black = sum(1 for f in frames if is_black(f.image, rules.black_mean_max, rules.black_std_max)) / len(frames)
    dup = duplicate_fraction(hashes, rules.duplicate_frame_hamming)
    sharp = float(np.median([sharpness(f.image) for f in frames]))
    metrics = {"black_frame_fraction": round(black, 3), "duplicate_frame_fraction": round(dup, 3), "median_sharpness": round(sharp, 2), "frames_sampled": float(len(frames))}
    reasons: list[str] = []
    checks = 2
    if black > rules.max_black_fraction:
        reasons.append(f"black_frames:{black:.2f}>{rules.max_black_fraction}")
    if dup >= rules.max_duplicate_fraction:
        reasons.append(f"frozen_or_duplicate_frames:{dup:.2f}>={rules.max_duplicate_fraction}")
    failed = len(reasons)
    status = "failed" if failed else "passed"
    if not failed and (black > rules.review_black_fraction or dup > rules.review_duplicate_fraction):
        status = "needs_review"
        reasons.append(f"partial_black_or_repeated_frames:black={black:.2f},duplicate={dup:.2f}")
    return VisualQuality(status, round(1.0 - failed / checks, 3), reasons, metrics, hashes)
