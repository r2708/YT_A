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


def _family_key(norm: str) -> str:
    return norm.replace("@", "").replace(" ", "")


def _within_one_edit(a: str, b: str) -> bool:
    """Levenshtein distance <= 1 (one substitution, insertion or deletion)."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b, strict=True)) <= 1
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    i = 0
    while i < len(short) and short[i] == long_[i]:
        i += 1
    return short[i:] == long_[i + 1 :]


def classify_tracks(
    tracks: list[OCRTrack],
    total_scenes: int,
    overlay_min_scene_fraction: float = 0.3,
    overlay_min_scenes: int = 4,
    fragment_max_gap: float = 3.0,
) -> tuple[list[OCRTrack], list[str]]:
    """Flag persistent overlays (watermarks, channel handles, logos) and partial reads.

    * A text *family* is a set of normalized texts where each is a prefix of the family's longest
      member (``com`` / ``come`` / ``comed`` / ``comedy``): OCR reads a static sign differently in every
      frame. A family whose members are seen in at least ``overlay_min_scene_fraction`` of all scenes
      (and at least ``overlay_min_scenes``) is a static overlay: it says nothing about *when* anything
      happens, so it must not become events or questions.
    * A track is a *fragment* when its text is a strict substring of another track visible at the
      same time (shared frame or within ``fragment_max_gap`` seconds), or of an overlay text.
    Returns (tracks with flags set, sorted list of overlay family keys). Idempotent.
    """
    if not tracks:
        return tracks, []
    keys = {t.track_id: _family_key(t.normalized_text) for t in tracks}
    distinct = sorted({k for k in keys.values() if k}, key=len, reverse=True)
    # longest-first greedy family assignment: a text joins the first (longest) family it is a prefix of
    family_of: dict[str, str] = {}
    for k in distinct:
        for head in family_of.values():
            if head.startswith(k) and len(k) >= 2:
                family_of[k] = head
                break
        else:
            family_of[k] = k
    scenes_by_family: dict[str, set[str]] = {}
    for t in tracks:
        fam = family_of.get(keys[t.track_id], keys[t.track_id])
        scenes_by_family.setdefault(fam, set()).update(t.scene_ids)
    needed = max(int(overlay_min_scenes), int(round(overlay_min_scene_fraction * max(1, total_scenes))))
    overlays = {fam for fam, sc in scenes_by_family.items() if total_scenes >= overlay_min_scenes and len(sc) >= needed}
    # every spelling OCR produced for an overlay ("com", "come", "comed", "comedy", "the", "theatf"...)
    overlay_members = {k for k, head in family_of.items() if head in overlays}

    def misread_of_overlay(key: str) -> bool:
        """'con' for 'com', 'theatr' for 'theatf', 'eomed' for 'comed': one edit away from a spelling
        already attributed to a watermark, and at least 3 characters so real short words are not caught."""
        if len(key) < 3:
            return False
        return any(_within_one_edit(key, m) and (len(key) >= 4 or key[:2] == m[:2]) for m in overlay_members if len(m) >= 3)

    by_time = sorted(tracks, key=lambda t: t.first_seen)
    out: list[OCRTrack] = []
    for t in by_time:
        key = keys[t.track_id]
        fam = family_of.get(key, key)
        is_overlay = fam in overlays
        is_fragment = False
        if not is_overlay and key:
            if any(key != ov and key in ov for ov in overlays) or misread_of_overlay(key):
                is_fragment = True  # partial read or one-character misread of a watermark
            else:
                frames = set(t.frame_ids)
                for other in by_time:
                    ok = keys[other.track_id]
                    if other.track_id == t.track_id or len(ok) <= len(key) or key not in ok:
                        continue
                    near = bool(frames & set(other.frame_ids)) or (
                        other.first_seen - fragment_max_gap <= t.last_seen and t.first_seen <= other.last_seen + fragment_max_gap
                    )
                    if near:
                        is_fragment = True
                        break
        out.append(t.model_copy(update={"is_static_overlay": is_overlay, "is_fragment": is_fragment, "overlay_family": fam if is_overlay else None}))
    out.sort(key=lambda t: t.track_id)
    return out, sorted(overlays)


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
