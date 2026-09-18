"""JSONL / Parquet writers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from video_dataset.utils.io import write_jsonl

DATASET_FILES = ["frames", "clips", "scenes", "events", "temporal_qa", "long_video_qa", "video_descriptions"]
RECORD_TYPES = {
    "frames": "frame",
    "clips": "clip",
    "scenes": "scene",
    "events": "event",
    "temporal_qa": "temporal_qa",
    "long_video_qa": "long_video_qa",
    "video_descriptions": "video_description",
}
COMBINED_FILE = "dataset.jsonl"


def write_jsonl_files(records: dict[str, list[Any]], out_dir: Path) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name in DATASET_FILES:
        counts[name] = write_jsonl(out_dir / f"{name}.jsonl", records.get(name, []))
    return counts


def iter_combined(records: dict[str, list[Any]]):  # type: ignore[no-untyped-def]
    """Every record of every type as one stream; each line starts with its record_type."""
    for name in DATASET_FILES:
        kind = RECORD_TYPES.get(name, name)
        for rec in records.get(name, []):
            d = rec.model_dump(mode="json") if isinstance(rec, BaseModel) else dict(rec)
            yield {"record_type": kind, **d}


def write_combined_jsonl(records: dict[str, list[Any]], path: Path) -> int:
    return write_jsonl(path, iter_combined(records))


def _get(d: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def normalize_record(kind: str, rec: BaseModel | dict[str, Any]) -> dict[str, Any]:
    d = rec.model_dump(mode="json") if isinstance(rec, BaseModel) else dict(rec)
    validation = d.get("validation") or {}
    quality = d.get("quality") or {}
    evidence = d.get("evidence") or {}
    if evidence:  # QA records keep their window inside `evidence`
        d = {**d, "start_time": evidence.get("start_time"), "end_time": evidence.get("end_time")}
        if evidence.get("scene_ids") and not d.get("scene_id"):
            d["scene_id"] = ",".join(evidence["scene_ids"])
    return {
        "record_id": _get(d, "record_id", "question_id"),
        "record_type": kind,
        "video_id": d.get("video_id"),
        "scene_id": _get(d, "scene_id") or (",".join(d.get("scene_ids") or []) or None),
        "start_time": _get(d, "start_time", "start") if kind != "frame" else d.get("timestamp"),
        "end_time": _get(d, "end_time", "end") if kind != "frame" else d.get("timestamp"),
        "text": _get(d, "caption", "description", "summary", "event", "question", "prompt"),
        "answer": d.get("answer"),
        "qa_type": d.get("type") if kind in ("temporal_qa", "long_video_qa") else None,
        "difficulty": d.get("difficulty"),
        "confidence": d.get("confidence"),
        "confidence_source": d.get("confidence_source"),
        "validation_status": validation.get("status"),
        "quality_overall": quality.get("overall"),
        "media_path": _get(d, "frame_path", "clip_path"),
        "payload": json.dumps(d, ensure_ascii=False, default=str),
    }


def write_parquet(records: dict[str, list[Any]], path: Path) -> int:
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for name, recs in records.items():
        for r in recs:
            rows.append(normalize_record(RECORD_TYPES.get(name, name), r))
    df = pd.DataFrame(rows, columns=[
        "record_id", "record_type", "video_id", "scene_id", "start_time", "end_time", "text", "answer", "qa_type", "difficulty",
        "confidence", "confidence_source", "validation_status", "quality_overall", "media_path", "payload",
    ])
    for col in ("start_time", "end_time", "confidence", "quality_overall"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return len(df)
