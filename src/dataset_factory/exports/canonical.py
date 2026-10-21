"""Canonical manifest export: approved / review / rejected manifests in Parquet (typed, from the
record model) with JSONL beside them, deterministic ordering, relative paths."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from dataset_factory.schema.canonical import CanonicalRecord
from video_dataset.dataset.arrow import write_typed_parquet

BUCKETS = ("approved", "review", "rejected")


def bucket_of(rec: CanonicalRecord) -> str:
    if rec.t2v_eligible or rec.i2v_eligible:
        return "approved"
    if rec.validation_status == "failed":
        return "rejected"
    return "review"


def write_canonical(records: Iterable[CanonicalRecord], out: Path) -> dict[str, int]:
    out.mkdir(parents=True, exist_ok=True)
    handles = {b: (out / f"{b}.jsonl").open("w", encoding="utf-8") for b in BUCKETS}
    counts = dict.fromkeys(BUCKETS, 0)
    rows: dict[str, list[dict[str, Any]]] = {b: [] for b in BUCKETS}
    try:
        for rec in sorted(records, key=lambda r: r.record_id):
            b = bucket_of(rec)
            row = rec.to_row()
            handles[b].write(json.dumps(row, ensure_ascii=False) + "\n")
            rows[b].append(row)
            counts[b] += 1
    finally:
        for fh in handles.values():
            fh.close()
    for b in BUCKETS:
        try:
            write_typed_parquet(CanonicalRecord, rows[b], out / f"{b}.parquet")
        except Exception as exc:
            (out / f"{b}.parquet.error").write_text(str(exc)[:300])
    return counts
