"""Human review queue and reviewed corrections.

``export_review_queue`` writes the records that need a human (status ``review``, contradicted
claims, failed boundary checks, suspected missed cuts) to ``final/review_queue.jsonl`` with the
evidence an auditor needs (frames, clip, claims, reasons) and an empty ``label`` slot.

``apply_review_labels`` reads the filled-in file (``record_id`` + ``label``: accept | reject |
correct, optional ``note`` / ``corrections``) and writes ``final/reviews/labels.jsonl``; the
aggregation applies them: accepted records become ``accepted`` with ``semantic_verification:
human``, rejected ones ``rejected`` with reason ``human:rejected``. Labels are also the input of
``video-dataset calibrate``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from video_dataset.schemas.quality import ValidationStatus
from video_dataset.utils.io import read_jsonl, write_jsonl

QUEUE_FILE = "review_queue.jsonl"
LABELS_FILE = Path("reviews") / "labels.jsonl"
_REVIEW_CODES = ("failed_check:clip_boundaries", "failed_check:single_shot", "failed_check:caption_alignment", "failed_check:consistent", "failed_check:camera_motion_agreement", "semantic_verification:not_run")


def needs_review(rec: dict[str, Any]) -> list[str]:
    v = rec.get("validation") or {}
    why: list[str] = []
    if v.get("status") == ValidationStatus.REVIEW.value:
        why.append("status_review")
    for code in v.get("reason_codes") or []:
        if any(code.startswith(c) for c in _REVIEW_CODES):
            why.append(code)
    if any(c.get("status") == "contradicted" for c in rec.get("claims") or []):
        why.append("contradicted_claim")
    return why


def export_review_queue(records: dict[str, list[dict[str, Any]]], final_dir: Path, limit: int | None = None) -> int:
    rows: list[dict[str, Any]] = []
    for name, recs in records.items():
        for r in recs:
            why = needs_review(r)
            if not why:
                continue
            rows.append({
                "record_id": r.get("record_id"), "record_type": name, "video_id": r.get("video_id"),
                "start_time": r.get("start_time") or (r.get("evidence") or {}).get("start_time") or r.get("timestamp"),
                "end_time": r.get("end_time") or (r.get("evidence") or {}).get("end_time") or r.get("timestamp"),
                "text": r.get("generation_prompt") or r.get("summary") or r.get("description") or r.get("question") or r.get("event"),
                "answer": r.get("answer"),
                "media": r.get("clip_path") or r.get("frame_path"),
                "frame_ids": (r.get("frame_ids") or (r.get("evidence") or {}).get("frame_ids") or [])[:8],
                "reasons": why,
                "issues": (r.get("validation") or {}).get("issues") or [],
                "claims": [c for c in (r.get("claims") or []) if c.get("status") != "supported"][:10],
                "label": None,  # accept | reject | correct
                "note": None,
                "corrections": {},
            })
            if limit and len(rows) >= limit:
                break
    write_jsonl(final_dir / QUEUE_FILE, rows)
    return len(rows)


def read_review_labels(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    for row in read_jsonl(path):
        rid = str(row.get("record_id") or "")
        label = str(row.get("label") or "").strip().lower()
        if rid and label in ("accept", "reject", "correct"):
            out[rid] = {"label": label, "note": row.get("note"), "corrections": row.get("corrections") or {}}
    return out


def apply_review_labels(records: dict[str, list[dict[str, Any]]], labels: dict[str, dict[str, Any]]) -> dict[str, int]:
    """Apply human verdicts to record dicts in place. Returns counts."""
    counts = {"accepted": 0, "rejected": 0, "corrected": 0}
    if not labels:
        return counts
    for rows in records.values():
        for r in rows:
            lab = labels.get(str(r.get("record_id")))
            if not lab:
                continue
            v = r.setdefault("validation", {"status": "review", "issues": [], "checks": {}})
            codes = [c for c in (v.get("reason_codes") or []) if not c.startswith("human:")]
            if lab["label"] == "reject":
                v["status"] = ValidationStatus.REJECTED.value
                codes.append("human:rejected")
                counts["rejected"] += 1
            else:
                for key, value in (lab.get("corrections") or {}).items():
                    if key in r and not isinstance(r[key], (dict, list)):
                        r[key] = value
                        counts["corrected"] += 1
                v["status"] = ValidationStatus.ACCEPTED.value
                v["semantic_verification"] = "human"
                codes.append("human:accepted")
                counts["accepted"] += 1
            v["reason_codes"] = codes
    return counts


def labels_for_calibration(labels: dict[str, dict[str, Any]]) -> dict[str, bool]:
    return {rid: lab["label"] in ("accept", "correct") for rid, lab in labels.items()}
