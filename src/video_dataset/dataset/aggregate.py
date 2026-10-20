"""Merge per-video exports into the final dataset files.

<final>/<type>.jsonl + dataset.jsonl   human-readable records (every type shares the same envelope)
<final>/<type>.parquet                 the same records with an explicit, model-derived schema
<final>/dataset.parquet                one normalised row per record (payload column = full JSON)
<final>/cinematic/...                  the cinematic subset (same layout, when enabled)
<final>/splits.json, schemas.json, statistics.json, manifest.json

Every record is passed through the current schema (``upgrade_records``) and then annotated with
split / tier / subsets (``annotate_records``), so records kept from earlier final files end up in
exactly the same shape as fresh ones.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from video_dataset.config import PipelineConfig
from video_dataset.dataset.curation import (
    STRICT_SUBSET,
    annotate_records,
    rights_status_for,
    split_manifest,
    subset_records,
    video_info_from_manifests,
)
from video_dataset.dataset.export import (
    CINEMATIC_DIR,
    COMBINED_FILE,
    DATASET_FILES,
    RECORD_TYPES,
    SCHEMA_FILE,
    SPLITS_FILE,
    upgrade_records,
    write_combined_jsonl,
    write_jsonl_files,
    write_parquet,
    write_schema_file,
    write_typed_parquet_files,
)
from video_dataset.dataset.media_manifest import write_media_manifest
from video_dataset.dataset.review import LABELS_FILE, apply_review_labels, read_review_labels
from video_dataset.dataset.stats import compute_diversity, compute_statistics
from video_dataset.dataset.takedown import removed_sources
from video_dataset.schemas.common import SCHEMA_VERSION
from video_dataset.storage.state_db import StateDB
from video_dataset.utils.environment import environment_manifest
from video_dataset.utils.io import read_json, read_jsonl, write_json_atomic, write_jsonl
from video_dataset.utils.logging import get_logger
from video_dataset.validation.policy import POLICY_VERSION

log = get_logger("dataset.aggregate")
STRICT_DIR = "strict"
PROVENANCE_FILE = "provenance.jsonl"


def provenance_rows(manifests: list[dict[str, Any]], video_info: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per source video: everything needed to audit or remove it later. No credentials, no
    commenter data; channel name is public metadata."""
    rows = []
    for m in manifests:
        vid = str(m.get("video_id"))
        rows.append({
            "video_id": vid, "url": m.get("url"), "youtube_id": m.get("youtube_id"), "channel": m.get("channel"), "title": m.get("title"),
            "upload_date": m.get("upload_date"), "retrieved_at": m.get("retrieved_at"), "duration": m.get("duration"),
            "license": m.get("license"), "license_status": m.get("license_status"),
            "rights_status": m.get("rights_status") or rights_status_for(m.get("license_status")),
            "source_sha256": m.get("source_sha256"), "canonical_video_id": video_info.get(vid, {}).get("canonical_video_id"),
            "coverage": m.get("coverage") or {}, "quality_flags": m.get("quality_flags") or [],
        })
    return rows


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


def _write_dataset_files(records: dict[str, list[dict[str, Any]]], out_dir: Path, config: PipelineConfig) -> tuple[dict[str, int], int, dict[str, int], int]:
    """(per-type jsonl counts, combined rows, per-type parquet counts, normalised parquet rows)."""
    counts: dict[str, int] = {name: len(records.get(name, [])) for name in DATASET_FILES}
    combined_rows = 0
    typed_counts: dict[str, int] = {}
    parquet_rows = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    if "jsonl" in config.export.formats:
        if config.export.per_type_jsonl:
            counts = write_jsonl_files(records, out_dir)
        if config.export.combined_jsonl:
            combined_rows = write_combined_jsonl(records, out_dir / COMBINED_FILE)
    if "parquet" in config.export.formats:
        if config.export.per_type_parquet:
            typed_counts = write_typed_parquet_files(records, out_dir)
        parquet_rows = write_parquet(records, out_dir / "dataset.parquet")
    return counts, combined_rows, typed_counts, parquet_rows


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

    # sources removed by a takedown never come back through merge_existing or a stale per_video export
    removed = removed_sources(final_dir)
    if removed:
        manifests = [m for m in manifests if str(m.get("video_id")) not in removed]
        for name in list(records):
            records[name] = [r for r in records[name] if str(r.get("video_id")) not in removed]

    # one schema for every record, then split / tier / subsets (near-duplicate videos share a split)
    records = upgrade_records(records)
    video_info = video_info_from_manifests(manifests, float(config.deduplication.video_similarity_threshold), bool(config.deduplication.video_fingerprint))
    for m in manifests:
        vi = video_info.setdefault(str(m.get("video_id")), {})
        vi.setdefault("rights_status", m.get("rights_status") or rights_status_for(m.get("license_status")))
        vi.setdefault("coverage", m.get("coverage") or {})
    labels = read_review_labels(final_dir / LABELS_FILE)
    review_counts = apply_review_labels(records, labels)  # human verdicts win over the automatic status
    curation = annotate_records(records, config.export, video_info, config.licensing, seed=int(config.qa.seed))
    curation["human_reviews"] = review_counts
    for m in manifests:  # record the resolved canonical ids so the manifest explains the splits
        vi = video_info.get(str(m.get("video_id")), {})
        m["canonical_video_id"] = vi.get("canonical_video_id")
        m["rights_status"] = vi.get("rights_status")
    splits = split_manifest(records)

    counts, combined_rows, typed_counts, parquet_rows = _write_dataset_files(records, final_dir, config)
    write_schema_file(final_dir / SCHEMA_FILE)
    write_json_atomic(final_dir / SPLITS_FILE, {"salt": config.export.split.salt, "fractions": {"validation": config.export.split.validation_fraction, "test": config.export.split.test_fraction}, "videos": splits, "counts": {k: len(v) for k, v in splits.items()}, "groups": {str(m.get("video_id")): video_info.get(str(m.get("video_id")), {}).get("canonical_video_id") or str(m.get("video_id")) for m in manifests}})
    write_jsonl(final_dir / PROVENANCE_FILE, provenance_rows(manifests, video_info))
    media_summary = write_media_manifest(records, final_dir, config.data_dir, bool(getattr(config.export, "hash_media", False))) if getattr(config.export, "media_manifest", True) else {}

    cinematic_counts: dict[str, int] = {}
    if config.export.cinematic.enabled and config.export.cinematic.write_files:
        subset = subset_records(records, "cinematic")
        cinematic_counts, _c, _t, _p = _write_dataset_files(subset, final_dir / CINEMATIC_DIR, config)
    strict_counts: dict[str, int] = {}
    strict_cfg = getattr(config.export, "strict", None)
    if strict_cfg is not None and strict_cfg.enabled and strict_cfg.write_files:
        strict_counts, _c, _t, _p = _write_dataset_files(subset_records(records, STRICT_SUBSET), final_dir / STRICT_DIR, config)

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
            "typed_parquet_counts": typed_counts,
            "cinematic": {**curation, "files": cinematic_counts},
            "strict": {"records": curation.get("strict", 0), "files": strict_counts},
            "curation": curation,
            "media_references": media_summary,
            "removed_sources": len(removed),
            "schema_version": SCHEMA_VERSION,
            "policy_version": POLICY_VERSION,
            "diversity": compute_diversity(records, manifests),
            "files": {
                **({name: str(final_dir / f"{name}.jsonl") for name in DATASET_FILES} if config.export.per_type_jsonl else {}),
                **({f"{name}_parquet": str(final_dir / f"{name}.parquet") for name in DATASET_FILES if typed_counts.get(name)} if config.export.per_type_parquet else {}),
                **({"dataset_jsonl": str(final_dir / COMBINED_FILE)} if config.export.combined_jsonl and "jsonl" in config.export.formats else {}),
                **({"dataset_parquet": str(final_dir / "dataset.parquet")} if "parquet" in config.export.formats else {}),
                "schemas": str(final_dir / SCHEMA_FILE),
                "splits": str(final_dir / SPLITS_FILE),
                **({"cinematic_dir": str(final_dir / CINEMATIC_DIR)} if cinematic_counts else {}),
                **({"strict_dir": str(final_dir / STRICT_DIR)} if strict_counts else {}),
                "provenance": str(final_dir / PROVENANCE_FILE),
            },
            "jsonl_counts": counts,
        },
    )
    write_json_atomic(final_dir / "statistics.json", stats)
    write_json_atomic(final_dir / "manifest.json", {
        "videos": manifests, "counts": counts, "parquet_rows": parquet_rows, "combined_jsonl_rows": combined_rows,
        "splits": {k: len(v) for k, v in splits.items()}, "cinematic": cinematic_counts, "strict": strict_counts,
        "schema_version": SCHEMA_VERSION, "policy_version": POLICY_VERSION, "media_references": media_summary,
        "removed_sources": sorted(removed), "environment": environment_manifest(config),
    })
    log.info("aggregated %d videos -> %s (splits %s, cinematic records %d)", len(manifests), final_dir, {k: len(v) for k, v in splits.items()}, curation.get("cinematic", 0))
    return stats


def load_statistics(config: PipelineConfig) -> dict[str, Any] | None:
    p = config.export_dir / "statistics.json"
    return json.loads(p.read_text()) if p.exists() else None
