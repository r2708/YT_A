"""Merge the same on-screen text persisting across frames into tracks."""

from __future__ import annotations

from difflib import SequenceMatcher

from video_dataset.schemas.ocr import OCRDetection, OCRTrack
from video_dataset.utils.ids import track_id as make_track_id
from video_dataset.utils.text import normalize_text


def text_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _union_bbox(boxes: list[list[float]]) -> list[float]:
    boxes = [b for b in boxes if len(b) == 4]
    if not boxes:
        return []
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def merge_detections(detections: list[OCRDetection], similarity: float = 0.85, max_gap: float = 3.0) -> list[OCRTrack]:
    tracks: list[dict] = []
    for det in sorted(detections, key=lambda d: d.timestamp):
        norm = normalize_text(det.text)
        if not norm:
            continue
        best = None
        best_sim = 0.0
        for tr in tracks:
            if det.timestamp - tr["last_seen"] > max_gap:
                continue
            sim = text_similarity(norm, tr["norm"])
            if sim >= similarity and sim > best_sim:
                best, best_sim = tr, sim
        if best is None:
            tracks.append(
                {
                    "text": det.text,
                    "norm": norm,
                    "first_seen": det.timestamp,
                    "last_seen": det.timestamp,
                    "frame_ids": [det.frame_id],
                    "scene_ids": [det.scene_id],
                    "detection_ids": [det.detection_id],
                    "confs": [det.confidence] if det.confidence is not None else [],
                    "boxes": [det.bbox] if det.bbox else [],
                }
            )
        else:
            best["last_seen"] = max(best["last_seen"], det.timestamp)
            if det.frame_id not in best["frame_ids"]:
                best["frame_ids"].append(det.frame_id)
            if det.scene_id not in best["scene_ids"]:
                best["scene_ids"].append(det.scene_id)
            best["detection_ids"].append(det.detection_id)
            if det.confidence is not None:
                best["confs"].append(det.confidence)
            if det.bbox:
                best["boxes"].append(det.bbox)
            # keep the highest-confidence spelling as the canonical text
            if det.confidence is not None and best["confs"] and det.confidence >= max(best["confs"]):
                best["text"], best["norm"] = det.text, norm
    out: list[OCRTrack] = []
    for i, tr in enumerate(tracks, start=1):
        out.append(
            OCRTrack(
                track_id=make_track_id(i),
                text=tr["text"],
                normalized_text=tr["norm"],
                first_seen=tr["first_seen"],
                last_seen=tr["last_seen"],
                frame_ids=tr["frame_ids"],
                scene_ids=tr["scene_ids"],
                detection_ids=tr["detection_ids"],
                num_detections=len(tr["detection_ids"]),
                mean_confidence=round(sum(tr["confs"]) / len(tr["confs"]), 4) if tr["confs"] else None,
                bbox=_union_bbox(tr["boxes"]),
            )
        )
    return out
