"""Subject tracking through a shot: YOLO detections on a low-rate decode of the scene, linked over
time with an IoU tracker (Hungarian assignment via SciPy; no extra packages, no learned re-id).

A track is one object followed through consecutive analysed frames. Its box centres over time give
the subject's path and direction; the longest, largest track is the shot's primary subject.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from video_dataset.utils.logging import get_logger

log = get_logger("vision.tracker")


@dataclass
class Track:
    track_id: int
    name: str
    boxes: list[tuple[float, list[float], float]] = field(default_factory=list)  # (timestamp, [x1,y1,x2,y2], confidence)
    misses: int = 0
    frame_size: tuple[int, int] = (0, 0)

    @property
    def hits(self) -> int:
        return len(self.boxes)

    @property
    def last_box(self) -> list[float]:
        return self.boxes[-1][1]

    def mean_area_fraction(self) -> float:
        w, h = self.frame_size
        if w <= 0 or h <= 0:
            return 0.0
        return float(np.mean([(b[2] - b[0]) * (b[3] - b[1]) / (w * h) for _, b, _ in self.boxes]))

    def as_detections(self) -> tuple[list[list[dict[str, Any]]], list[float]]:
        """The track as per-frame detection lists (one detection per frame) + timestamps, in the form
        ``analyze_subject_motion`` consumes."""
        w, h = self.frame_size
        per_frame = [[{"name": self.name, "confidence": c, "bbox": list(b), "frame_size": [w, h]}] for _, b, c in self.boxes]
        return per_frame, [t for t, _, _ in self.boxes]


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class IoUTracker:
    """Greedy-optimal IoU association per frame (same class only), tracks end after ``max_age`` misses."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 2, min_hits: int = 3):
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self._next_id = 1
        self.active: list[Track] = []
        self.finished: list[Track] = []

    def update(self, detections: list[dict[str, Any]], timestamp: float, frame_size: tuple[int, int]) -> None:
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        if self.active and detections:
            cost = np.ones((len(self.active), len(detections)), dtype=np.float32)
            for i, tr in enumerate(self.active):
                for j, d in enumerate(detections):
                    if d["name"] == tr.name:
                        cost[i, j] = 1.0 - iou(tr.last_box, d["bbox"])
            try:
                from scipy.optimize import linear_sum_assignment

                rows, cols = linear_sum_assignment(cost)
                pairs = list(zip(rows.tolist(), cols.tolist()))
            except Exception:  # pragma: no cover - scipy missing: greedy fallback
                pairs = []
                for i in np.argsort(cost, axis=None):
                    r, c = divmod(int(i), cost.shape[1])
                    if r not in {p[0] for p in pairs} and c not in {p[1] for p in pairs}:
                        pairs.append((r, c))
            for r, c in pairs:
                if 1.0 - cost[r, c] >= self.iou_threshold:
                    tr = self.active[r]
                    tr.boxes.append((timestamp, list(detections[c]["bbox"]), float(detections[c]["confidence"])))
                    tr.misses = 0
                    matched_tracks.add(r)
                    matched_dets.add(c)
        still: list[Track] = []
        for i, tr in enumerate(self.active):
            if i in matched_tracks:
                still.append(tr)
                continue
            tr.misses += 1
            if tr.misses > self.max_age:
                self.finished.append(tr)
            else:
                still.append(tr)
        self.active = still
        for j, d in enumerate(detections):
            if j in matched_dets:
                continue
            self.active.append(Track(track_id=self._next_id, name=d["name"], boxes=[(timestamp, list(d["bbox"]), float(d["confidence"]))], frame_size=frame_size))
            self._next_id += 1

    def tracks(self) -> list[Track]:
        """Every track with at least ``min_hits`` detections, longest first."""
        out = [t for t in self.finished + self.active if t.hits >= self.min_hits]
        out.sort(key=lambda t: (-t.hits, -t.mean_area_fraction(), t.track_id))
        return out


def decode_scene(video_path: Path, start: float, end: float, fps: float, width: int, max_seconds: float) -> tuple[list[tuple[float, np.ndarray]], tuple[int, int]]:
    """Frames of [start, min(end, start+max_seconds)] at ~``fps``, downscaled to ``width``. Returns
    ((timestamp, image) ..., (w, h) of the returned images)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    out: list[tuple[float, np.ndarray]] = []
    size = (0, 0)
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        stride = max(1, int(round(src_fps / max(0.1, fps))))
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start) * 1000.0)
        stop = min(end, start + max_seconds)
        n = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t > stop + 1e-3:
                break
            if n % stride == 0:
                ok, img = cap.retrieve()
                if ok and img is not None:
                    h, w = img.shape[:2]
                    if w > width:
                        s = width / float(w)
                        img = cv2.resize(img, (width, max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)
                    size = (img.shape[1], img.shape[0])
                    out.append((round(t, 3), img))
            n += 1
    finally:
        cap.release()
    return out, size


def track_scene(video_path: Path, start: float, end: float, detector: Any, *, fps: float, width: int, max_seconds: float, iou_threshold: float, max_age: int, min_hits: int, batch: int = 16) -> tuple[list[Track], int]:
    """Run the detector over the decoded scene and link detections into tracks. Returns (tracks,
    analysed frames). ``detector`` must offer ``detect_arrays(images) -> list[list[detection]]``."""
    frames, size = decode_scene(video_path, start, end, fps, width, max_seconds)
    tracker = IoUTracker(iou_threshold, max_age, min_hits)
    for i in range(0, len(frames), batch):
        chunk = frames[i : i + batch]
        dets = detector.detect_arrays([img for _, img in chunk])
        for (t, _img), d in zip(chunk, dets):
            tracker.update(d, t, size)
    return tracker.tracks(), len(frames)


def primary_track(tracks: list[Track]) -> Track | None:
    """Longest track; among equals the biggest, people winning ties."""
    if not tracks:
        return None
    return max(tracks, key=lambda t: (t.hits * (1.0 + t.mean_area_fraction()), t.name == "person", -t.track_id))


def trajectory(track: Track, max_points: int = 20) -> list[list[float]]:
    w, h = track.frame_size
    if w <= 0 or h <= 0:
        return []
    pts = []
    for t, b, _c in track.boxes:
        pts.append([round(t, 3), round((b[0] + b[2]) / 2.0 / w, 4), round((b[1] + b[3]) / 2.0 / h, 4), round((b[2] - b[0]) * (b[3] - b[1]) / (w * h), 4)])
    if len(pts) <= max_points:
        return pts
    step = (len(pts) - 1) / (max_points - 1)
    return [pts[int(round(i * step))] for i in range(max_points)]
