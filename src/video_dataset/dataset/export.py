"""JSONL / Parquet writers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from video_dataset.dataset.arrow import json_schemas, write_typed_parquet
from video_dataset.schemas.dataset import (
    ClipCaptionRecord,
    EventRecord,
    FrameCaptionRecord,
    SceneRecord,
    VideoDescriptionRecord,
)
from video_dataset.schemas.qa import QARecord
from video_dataset.utils.io import write_json_atomic, write_jsonl
from video_dataset.utils.logging import get_logger

log = get_logger("dataset.export")

DATASET_FILES = ["frames", "clips", "scenes", "events", "temporal_qa", "long_video_qa", "video_descriptions"]
RECORD_MODELS: dict[str, type[BaseModel]] = {
    "frames": FrameCaptionRecord,
    "clips": ClipCaptionRecord,
    "scenes": SceneRecord,
    "events": EventRecord,
    "temporal_qa": QARecord,
    "long_video_qa": QARecord,
    "video_descriptions": VideoDescriptionRecord,
}
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
SCHEMA_FILE = "schemas.json"
SPLITS_FILE = "splits.json"
CINEMATIC_DIR = "cinematic"
PARQUET_COLUMNS = [
    "record_id", "record_type", "video_id", "scene_id", "start_time", "end_time", "text", "answer", "qa_type", "difficulty",
    "provider", "confidence", "confidence_source", "validation_status", "quality_overall", "split", "tier", "subsets",
    "aesthetic_score", "hard_negatives", "media_path", "payload",
]


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
            d.pop("record_type", None)
            yield {"record_type": kind, **d}


def upgrade_record(kind: str, rec: dict[str, Any] | BaseModel) -> dict[str, Any]:
    """Bring a record (possibly written by an earlier version) to the current schema: renamed
    fields are mapped, new fields get their defaults, QA gets record_id / record_type. Records that
    do not validate are returned unchanged so a rebuild never loses data."""
    if isinstance(rec, BaseModel):
        d = rec.model_dump(mode="json")
    else:
        d = dict(rec)
    model = RECORD_MODELS.get(kind)
    if model is None:
        return d
    d.pop("record_type", None)
    try:
        out = model.model_validate(d).model_dump(mode="json")
    except Exception as exc:  # pragma: no cover - depends on legacy data
        log.warning("record %s of type %s does not match the current schema (%s); kept as is", d.get("record_id") or d.get("question_id"), kind, str(exc)[:160])
        out = d
    out["record_type"] = RECORD_TYPES.get(kind, kind)
    return out


def upgrade_records(records: dict[str, list[Any]]) -> dict[str, list[dict[str, Any]]]:
    return {kind: [upgrade_record(kind, r) for r in rows] for kind, rows in records.items()}


def write_typed_parquet_files(records: dict[str, list[Any]], out_dir: Path) -> dict[str, int]:
    """One Parquet file per record type with the schema derived from its model (loader friendly).
    A type with no records still gets an (empty, typed) file so the loader configs in the dataset
    card, which glob `shard_*/<type>.parquet`, resolve for every shard."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for name in DATASET_FILES:
        counts[name] = write_typed_parquet(RECORD_MODELS[name], records.get(name, []), out_dir / f"{name}.parquet")
    return counts


def write_schema_file(path: Path) -> Path:
    """schemas.json: JSON Schema of every record type plus the file -> type mapping."""
    return write_json_atomic(path, {"files": {f"{n}.jsonl": RECORD_TYPES[n] for n in DATASET_FILES}, "record_types": {RECORD_TYPES[n]: json_schemas({n: RECORD_MODELS[n]})[n] for n in DATASET_FILES}})


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
    measurements = d.get("measurements") or {}
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
        "provider": d.get("provider"),
        "confidence": d.get("confidence"),
        "confidence_source": d.get("confidence_source"),
        "validation_status": validation.get("status"),
        "quality_overall": quality.get("overall"),
        "split": d.get("split"),
        "tier": d.get("tier"),
        "subsets": ",".join(d.get("subsets") or []) or None,
        "aesthetic_score": measurements.get("aesthetic_score"),
        "hard_negatives": len(d.get("hard_negatives") or []),
        "media_path": _get(d, "frame_path", "clip_path"),
        "payload": json.dumps(d, ensure_ascii=False, default=str),
    }


def write_parquet(records: dict[str, list[Any]], path: Path) -> int:
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for name, recs in records.items():
        for r in recs:
            rows.append(normalize_record(RECORD_TYPES.get(name, name), r))
    df = pd.DataFrame(rows, columns=PARQUET_COLUMNS)
    for col in ("start_time", "end_time", "confidence", "quality_overall", "aesthetic_score"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["hard_negatives"] = pd.to_numeric(df["hard_negatives"], errors="coerce").fillna(0).astype("int64")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return len(df)
