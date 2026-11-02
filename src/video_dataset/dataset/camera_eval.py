"""Camera-movement confusion matrix: human labels against the predicted movement family.

Labels file (CSV or JSONL): ``record_id`` plus ``camera_movement`` - a family (``pan``, ``zoom``,
``static`` ...) or a fine label (``pan_left``, ``dolly_in``) which is mapped to its family through
``MOVEMENT_TAXONOMY``. The filled ``reviews/sample.jsonl`` of ``human_review`` is accepted too:
its ``camera_movement_label`` column is read when ``camera_movement`` is absent.

The report compares, for every scene / clip record whose ``record_id`` is labelled, the human
family with ``camera.camera_movement`` (falling back to the family of ``camera.movement``;
``unknown`` when neither is set). Rows of the matrix are the human labels, columns the predicted
ones, class order is sorted and always contains ``unknown``.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

from video_dataset.schemas.vision import MOVEMENT_TAXONOMY, CameraMovement, CameraMovementFamily
from video_dataset.utils.io import read_jsonl

UNKNOWN = CameraMovementFamily.UNKNOWN.value
_EVAL_TYPES = {"scene", "clip"}
_TYPE_BY_KEY = {"scenes": "scene", "clips": "clip"}


def to_family(value: Any) -> str:
    """Fine or family label -> family string (``unknown`` when it is neither)."""
    text = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    if not text:
        return UNKNOWN
    try:
        return MOVEMENT_TAXONOMY[CameraMovement(text)][0].value
    except ValueError:
        pass
    try:
        return CameraMovementFamily(text).value
    except ValueError:
        return UNKNOWN


def read_camera_labels(path: Path) -> dict[str, str]:
    """``record_id`` -> human movement family from a CSV / JSONL labels file."""
    path = Path(path)
    rows: list[dict[str, Any]]
    if path.suffix.lower() == ".jsonl":
        rows = list(read_jsonl(path))
    else:
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    labels: dict[str, str] = {}
    for r in rows:
        rid = str(r.get("record_id") or "").strip()
        raw = r.get("camera_movement")
        if raw in (None, ""):
            raw = r.get("camera_movement_label")
        if not rid or raw in (None, ""):
            continue
        labels[rid] = to_family(raw)
    return labels


def _camera(rec: dict[str, Any]) -> dict[str, Any]:
    cam = rec.get("camera")
    return cam if isinstance(cam, dict) else {}


def predicted_family(rec: dict[str, Any]) -> str:
    cam = _camera(rec)
    if cam.get("camera_movement"):
        return to_family(cam["camera_movement"])
    if cam.get("movement"):
        return to_family(cam["movement"])
    return UNKNOWN


def confusion_matrix(records: dict[str, list[dict[str, Any]]], labels: dict[str, str]) -> dict[str, Any]:
    """Confusion matrix of human vs predicted movement family over labelled scene / clip records."""
    pairs: list[tuple[str, str]] = []
    ambiguous = 0
    seen: set[str] = set()
    for key, rows in records.items():
        for rec in rows:
            rtype = str(rec.get("record_type") or _TYPE_BY_KEY.get(key) or key)
            rid = str(rec.get("record_id") or "")
            if rtype not in _EVAL_TYPES or rid not in labels or rid in seen:
                continue
            seen.add(rid)
            if _camera(rec).get("movement_ambiguous"):
                ambiguous += 1
            pairs.append((labels[rid], predicted_family(rec)))
    classes = sorted({h for h, _ in pairs} | {p for _, p in pairs} | {UNKNOWN})
    index = {c: i for i, c in enumerate(classes)}
    matrix = [[0] * len(classes) for _ in classes]
    for h, p in pairs:
        matrix[index[h]][index[p]] += 1
    n = len(pairs)
    per_class: dict[str, dict[str, Any]] = {}
    for c in classes:
        i = index[c]
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        precision = tp / predicted if predicted else None
        recall = tp / support if support else None
        f1 = (2 * precision * recall / (precision + recall)) if precision and recall else (0.0 if (precision is not None and recall is not None) else None)
        per_class[c] = {
            "precision": None if precision is None else round(precision, 4),
            "recall": None if recall is None else round(recall, 4),
            "f1": None if f1 is None else round(f1, 4),
            "support": support,
            "predicted": predicted,
        }
    off = Counter((h, p) for h, p in pairs if h != p)
    confused = sorted(off.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    return {
        "labels": classes,
        "matrix": matrix,
        "per_class": per_class,
        "accuracy": round(sum(1 for h, p in pairs if h == p) / n, 4) if n else None,
        "n": n,
        "labelled": len(labels),
        "ambiguous_predicted": ambiguous,
        "most_confused_pairs": [{"human": h, "predicted": p, "count": c} for (h, p), c in confused],
    }


def format_confusion(report: dict[str, Any]) -> str:
    """Plain-text table: rows = human label, columns = predicted family, then per-class scores."""
    classes: list[str] = list(report.get("labels") or [])
    matrix: list[list[int]] = report.get("matrix") or []
    if not classes:
        return "camera confusion matrix: no labelled scene / clip records"
    width = max(8, max(len(c) for c in classes) + 1)
    lines = [f"camera movement confusion (n={report.get('n', 0)}, accuracy={report.get('accuracy')}, ambiguous_predicted={report.get('ambiguous_predicted', 0)})"]
    lines.append("human \\ predicted".ljust(18) + "".join(c.rjust(width) for c in classes))
    for c, row in zip(classes, matrix, strict=False):
        lines.append(c.ljust(18) + "".join(str(v).rjust(width) for v in row))
    lines.append("")
    lines.append("class".ljust(18) + "precision".rjust(10) + "recall".rjust(10) + "f1".rjust(10) + "support".rjust(10))
    for c in classes:
        pc = (report.get("per_class") or {}).get(c) or {}
        fmt = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
        lines.append(c.ljust(18) + fmt(pc.get("precision")).rjust(10) + fmt(pc.get("recall")).rjust(10) + fmt(pc.get("f1")).rjust(10) + str(pc.get("support", 0)).rjust(10))
    if report.get("most_confused_pairs"):
        lines.append("")
        lines.append("most confused: " + ", ".join(f"{p['human']}->{p['predicted']} x{p['count']}" for p in report["most_confused_pairs"]))
    return "\n".join(lines)
