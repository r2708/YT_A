"""YOLO object detection for the ``cv_models`` vision provider.

Every value produced here is a detector score or a bounding-box measurement; nothing is described
that the detector did not see. Frames are analysed in one batched ``predict`` call per scene.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from video_dataset.utils.logging import get_logger
from video_dataset.vision.subject_motion import normalized_box, position_label, scale_class

log = get_logger("vision.yolo")

INSTALL_HINT = "install it with: pip install 'video-dataset-pipeline[vision-cv]'"


def bbox_location(bbox: list[float], width: float, height: float) -> str:
    """Coarse, grounded location phrase from a box centre ("left part of the frame")."""
    if width <= 0 or height <= 0:
        return "in frame"
    cx = (bbox[0] + bbox[2]) / 2.0 / width
    if cx < 1 / 3:
        return "left part of the frame"
    if cx > 2 / 3:
        return "right part of the frame"
    return "center of the frame"


def aggregate_detections(per_frame: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Merge per-frame detections into one entry per class.

    ``count`` is the largest number of simultaneous instances seen in a single frame (not the sum
    over frames), ``confidence`` the mean detector score, ``frame_fraction`` the share of analysed
    frames in which the class appears, ``location`` the most frequent coarse position. When the
    detections carry ``frame_size``, the largest box of the class in each frame also yields the
    composition fields ``center`` [cx, cy], ``area_fraction``, ``scale`` and ``position`` (3x3 grid).
    """
    if not per_frame:
        return []
    by_name: dict[str, dict[str, Any]] = {}
    for fi, dets in enumerate(per_frame):
        counts = Counter(d["name"] for d in dets)
        largest: dict[str, tuple[float, float, float]] = {}
        for d in dets:
            entry = by_name.setdefault(d["name"], {"confs": [], "count": 0, "frames": set(), "locations": Counter(), "boxes": []})
            entry["confs"].append(float(d["confidence"]))
            entry["count"] = max(entry["count"], counts[d["name"]])
            entry["frames"].add(fi)
            if d.get("location"):
                entry["locations"][d["location"]] += 1
            nb = normalized_box(d)
            if nb is not None and (d["name"] not in largest or nb[2] > largest[d["name"]][2]):
                largest[d["name"]] = nb
        for name, nb in largest.items():
            by_name[name]["boxes"].append(nb)
    out = []
    for name, e in by_name.items():
        boxes = e["boxes"]
        cx = sum(b[0] for b in boxes) / len(boxes) if boxes else None
        cy = sum(b[1] for b in boxes) / len(boxes) if boxes else None
        area = sum(b[2] for b in boxes) / len(boxes) if boxes else None
        out.append(
            {
                "name": name,
                "count": int(e["count"]),
                "confidence": round(sum(e["confs"]) / len(e["confs"]), 3),
                "max_confidence": round(max(e["confs"]), 3),
                "frame_fraction": round(len(e["frames"]) / len(per_frame), 3),
                "location": e["locations"].most_common(1)[0][0] if e["locations"] else None,
                "detections": len(e["confs"]),
                "center": [round(cx, 4), round(cy, 4)] if cx is not None and cy is not None else None,
                "area_fraction": round(area, 4) if area is not None else None,
                "scale": scale_class(area) if area is not None else None,
                "position": position_label(cx, cy) if cx is not None and cy is not None else None,
            }
        )
    out.sort(key=lambda o: (-o["frame_fraction"], -o["detections"], -o["confidence"], o["name"]))
    return out


class YOLODetector:
    """Thin wrapper around an Ultralytics YOLO model (yolov8n/s/m/l/x, yolo11…)."""

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.5, device: str = "cpu", image_size: int = 640):
        self.model_name = model_name
        self.confidence = float(confidence)
        self.device = "cuda:0" if device == "cuda" else device
        self.image_size = int(image_size)
        self._model: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError(f"vision.provider=cv_models needs the 'ultralytics' package; {INSTALL_HINT}") from exc
        log.info("loading YOLO model %s on %s", self.model_name, self.device)
        self._model = YOLO(self.model_name)

    def detect_many(self, image_paths: list[Path]) -> list[list[dict[str, Any]]]:
        """Run detection on several images at once. Returns one list of detections per image."""
        if not image_paths:
            return []
        self.load()
        results = self._model.predict(
            source=[str(p) for p in image_paths], conf=self.confidence, device=self.device,
            imgsz=self.image_size, verbose=False,
        )
        out: list[list[dict[str, Any]]] = []
        for res in results:
            dets: list[dict[str, Any]] = []
            h, w = (res.orig_shape or (0, 0))[:2] if getattr(res, "orig_shape", None) else (0, 0)
            boxes = getattr(res, "boxes", None)
            if boxes is not None:
                for box in boxes:
                    conf = float(box.conf[0])
                    if conf < self.confidence:
                        continue
                    bbox = [round(float(v), 1) for v in box.xyxy[0].tolist()]
                    dets.append(
                        {
                            "name": str(res.names[int(box.cls[0])]),
                            "confidence": round(conf, 3),
                            "bbox": bbox,
                            "frame_size": [int(w), int(h)],
                            "location": bbox_location(bbox, w, h),
                        }
                    )
            out.append(dets)
        return out

    def detect(self, image_path: Path) -> list[dict[str, Any]]:
        res = self.detect_many([image_path])
        return res[0] if res else []

    def close(self) -> None:
        self._model = None
