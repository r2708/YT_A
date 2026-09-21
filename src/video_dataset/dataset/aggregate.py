"""Merge per-video exports into the final dataset files: <final>/*.jsonl, dataset.parquet, statistics.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from video_dataset.config import PipelineConfig
from video_dataset.dataset.export import (
    COMBINED_FILE,
    DATASET_FILES,
    RECORD_TYPES,
    write_combined_jsonl,
    write_jsonl_files,
    write_parquet,
)
from video_dataset.dataset.stats import compute_statistics
from video_dataset.storage.state_db import StateDB
from video_dataset.utils.io import read_json, read_jsonl, write_json_atomic
from video_dataset.utils.logging import get_logger

log = get_logger("dataset.aggregate")


def _record_key(rec: dict[str, Any]) -> str | None:
    return rec.get("record_id") or rec.get("question_id")


def load_existing_records(final_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Records already in the final dataset: per-type files when present, else the combined file."""
    existing: dict[str, list[dict[str, Any]]] = {name: [] for name in DATASET_FILES}
    found = False
    for name in DATASET_FILES:
        f = final_dir / f"{name}.jsonl"
        if f.exists():
            found = True
            existing[name].extend(read_jsonl(f))
    if found:
        return existing
    combined = final_dir / COMBINED_FILE
    if combined.exists():
        by_type = {kind: name for name, kind in RECORD_TYPES.items()}
        for row in read_jsonl(combined):
            kind = by_type.get(str(row.pop("record_type", "")))
            if kind:
                existing[kind].append(row)
    return existing


def merge_records(
    existing: dict[str, list[dict[str, Any]]],
    fresh: dict[str, list[dict[str, Any]]],
    fresh_video_ids: set[str],
) -> dict[str, list[dict[str, Any]]]:
    """Existing records of videos that were NOT re-exported now, followed by the fresh ones.

    A video re-exported in this run replaces all of its earlier records (they may have changed after
    `--force-from`), so the dataset never holds two versions of the same video. Duplicate record ids
    within a type are dropped, first occurrence wins.
    """
    merged: dict[str, list[dict[str, Any]]] = {}
    for name in DATASET_FILES:
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for rec in [r for r in existing.get(name, []) if r.get("video_id") not in fresh_video_ids] + list(fresh.get(name, [])):
            key = _record_key(rec)
            if key is not None:
                if key in seen:
                    continue
                seen.add(key)
            out.append(rec)
        merged[name] = out
    return merged


def aggregate_exports(config: PipelineConfig, db: StateDB, output_dir: Path | None = None) -> dict[str, Any]:
    final_dir = output_dir or config.export_dir
    per_video = config.export_dir / "per_video"
    records: dict[str, list[dict[str, Any]]] = {name: [] for name in DATASET_FILES}
    manifests: list[dict[str, Any]] = []
    if per_video.exists():
        for vdir in sorted(p for p in per_video.iterdir() if p.is_dir()):
            manifest_path = vdir / "manifest.json"
            if not manifest_path.exists():
                continue
            manifests.append(read_json(manifest_path))
            for name in DATASET_FILES:
                f = vdir / f"{name}.jsonl"
                if f.exists():
                    records[name].extend(read_jsonl(f))
    final_dir.mkdir(parents=True, exist_ok=True)

    if config.export.merge_existing:
        fresh_ids = {m["video_id"] for m in manifests}
        existing = load_existing_records(final_dir)
        existing_manifest = final_dir / "manifest.json"
        if existing_manifest.exists():
            try:
                for m in read_json(existing_manifest).get("videos", []):
                    if m.get("video_id") and m["video_id"] not in fresh_ids:
                        manifests.append(m)
            except (ValueError, AttributeError) as exc:
                log.warning("could not read previous %s: %s", existing_manifest, exc)
        kept = sum(len([r for r in rows if r.get("video_id") not in fresh_ids]) for rows in existing.values())
        if kept:
            log.info("keeping %d record(s) of previously exported videos whose per-video export is gone", kept)
        records = merge_records(existing, records, fresh_ids)
    counts = {}
    combined_rows = 0
    if "jsonl" in config.export.formats:
        if config.export.per_type_jsonl:
            counts = write_jsonl_files(records, final_dir)
        else:
            counts = {name: len(records.get(name, [])) for name in DATASET_FILES}
        if config.export.combined_jsonl:
            combined_rows = write_combined_jsonl(records, final_dir / COMBINED_FILE)
    parquet_rows = 0
    if "parquet" in config.export.formats:
        parquet_rows = write_parquet(records, final_dir / "dataset.parquet")

    videos = db.list_videos()
    exported_ids = {m["video_id"] for m in manifests}
    videos_exported = [v for v in videos if v["video_id"] in exported_ids]
    stats = compute_statistics(
        videos_exported,
        records,
        [m.get("validation_summary", {}) for m in manifests],
        extra={
            "videos_registered": len(videos),
            "transcript_segments": sum(int(m.get("transcript_segments", 0)) for m in manifests),
            "ocr_tracks": sum(int(m.get("ocr_tracks", 0)) for m in manifests),
            "parquet_rows": parquet_rows,
            "combined_jsonl_rows": combined_rows,
            "files": {
                **({name: str(final_dir / f"{name}.jsonl") for name in DATASET_FILES} if config.export.per_type_jsonl else {}),
                **({"dataset_jsonl": str(final_dir / COMBINED_FILE)} if config.export.combined_jsonl and "jsonl" in config.export.formats else {}),
                **({"dataset_parquet": str(final_dir / "dataset.parquet")} if "parquet" in config.export.formats else {}),
            },
            "jsonl_counts": counts,
        },
    )
    write_json_atomic(final_dir / "statistics.json", stats)
    write_json_atomic(final_dir / "manifest.json", {"videos": manifests, "counts": counts, "parquet_rows": parquet_rows, "combined_jsonl_rows": combined_rows})
    log.info("aggregated %d videos -> %s", len(manifests), final_dir)
    return stats


def load_statistics(config: PipelineConfig) -> dict[str, Any] | None:
    p = config.export_dir / "statistics.json"
    return json.loads(p.read_text()) if p.exists() else None
