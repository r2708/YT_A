"""Merge per-video exports into the final dataset files: <final>/*.jsonl, dataset.parquet, statistics.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from video_dataset.config import PipelineConfig
from video_dataset.dataset.export import (
    COMBINED_FILE,
    DATASET_FILES,
    write_combined_jsonl,
    write_jsonl_files,
    write_parquet,
)
from video_dataset.dataset.stats import compute_statistics
from video_dataset.storage.state_db import StateDB
from video_dataset.utils.io import read_json, read_jsonl, write_json_atomic
from video_dataset.utils.logging import get_logger

log = get_logger("dataset.aggregate")


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
