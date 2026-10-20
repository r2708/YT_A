"""Confidence calibration against human labels.

Scores in this dataset come from different instruments (optical-flow consistency, detector
probabilities, ASR log-probabilities, OCR scores, interval algebra) and are *not* probabilities of
correctness. This module measures how each ``confidence_source`` relates to human judgement on a
labelled sample and recommends per-source thresholds for a chosen false-accept budget.

Labels file (CSV or JSONL): ``record_id`` and ``correct`` (1 / 0 / true / false), optionally
``record_type``. Output: per confidence source and record type - count, accuracy, Brier score,
expected calibration error (10 bins), reliability diagram, precision / false-accept rate at the
current threshold and the smallest threshold that meets the target false-accept rate.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from video_dataset.utils.io import read_jsonl


def read_labels(path: Path) -> dict[str, bool]:
    path = Path(path)
    labels: dict[str, bool] = {}
    rows: list[dict[str, Any]]
    if path.suffix.lower() == ".jsonl":
        rows = list(read_jsonl(path))
    else:
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    for r in rows:
        rid = str(r.get("record_id") or "").strip()
        val = str(r.get("correct", "")).strip().lower()
        if not rid or val == "":
            continue
        labels[rid] = val in ("1", "true", "yes", "y", "correct")
    return labels


def _bins(pairs: list[tuple[float, bool]], n_bins: int = 10) -> list[dict[str, Any]]:
    out = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        sel = [(c, y) for c, y in pairs if lo <= c < hi or (b == n_bins - 1 and c == 1.0)]
        if not sel:
            continue
        out.append({"bin": f"{lo:.1f}-{hi:.1f}", "count": len(sel), "mean_confidence": round(sum(c for c, _ in sel) / len(sel), 4), "accuracy": round(sum(1 for _, y in sel if y) / len(sel), 4)})
    return out


def _metrics(pairs: list[tuple[float, bool]], threshold: float, target_far: float) -> dict[str, Any]:
    n = len(pairs)
    acc = sum(1 for _, y in pairs if y) / n
    brier = sum((c - (1.0 if y else 0.0)) ** 2 for c, y in pairs) / n
    bins = _bins(pairs)
    ece = sum(b["count"] / n * abs(b["accuracy"] - b["mean_confidence"]) for b in bins)
    accepted = [(c, y) for c, y in pairs if c >= threshold]
    precision = (sum(1 for _, y in accepted if y) / len(accepted)) if accepted else None
    false_accept = (sum(1 for _, y in accepted if not y) / len(accepted)) if accepted else None
    rejected = [(c, y) for c, y in pairs if c < threshold]
    false_reject = (sum(1 for _, y in rejected if y) / len(rejected)) if rejected else None
    # smallest threshold whose false-accept rate meets the target (None when none does)
    recommended = None
    for t in sorted({round(c, 3) for c, _ in pairs}):
        acc_t = [(c, y) for c, y in pairs if c >= t]
        if acc_t and sum(1 for _, y in acc_t if not y) / len(acc_t) <= target_far:
            recommended = t
            break
    return {
        "count": n, "accuracy": round(acc, 4), "brier": round(brier, 4), "ece": round(ece, 4), "reliability": bins,
        "threshold": threshold, "precision_at_threshold": None if precision is None else round(precision, 4),
        "false_accept_rate_at_threshold": None if false_accept is None else round(false_accept, 4),
        "false_reject_rate_at_threshold": None if false_reject is None else round(false_reject, 4),
        "recommended_threshold": recommended, "target_false_accept_rate": target_far,
    }


def calibrate(records: dict[str, list[dict[str, Any]]], labels: dict[str, bool], threshold: float = 0.75, target_far: float = 0.05) -> dict[str, Any]:
    by_key: dict[tuple[str, str], list[tuple[float, bool]]] = {}
    unlabelled_conf = 0
    matched = 0
    for name, rows in records.items():
        for r in rows:
            rid = str(r.get("record_id") or r.get("question_id") or "")
            if rid not in labels:
                continue
            matched += 1
            conf = r.get("confidence")
            if conf is None:
                unlabelled_conf += 1
                continue
            key = (str(r.get("confidence_source") or "unavailable"), name)
            by_key.setdefault(key, []).append((float(conf), labels[rid]))
    out = {
        "labels": len(labels), "matched_records": matched, "records_without_confidence": unlabelled_conf,
        "by_source": {f"{src}|{kind}": _metrics(pairs, threshold, target_far) for (src, kind), pairs in sorted(by_key.items()) if pairs},
        "note": "Scores from different confidence_source values are different instruments; thresholds are per source, never global.",
    }
    return out


def write_calibration(report: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    return path
