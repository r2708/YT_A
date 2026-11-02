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
    RIGHTS_TERMS,
    STRICT_SUBSET,
    annotate_records,
    load_rights_evidence,
    rights_info_for,
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
    UpgradeStats,
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
from video_dataset.schemas.common import SCHEMA_VERSION, utc_now_iso
from video_dataset.storage.state_db import StateDB
from video_dataset.utils.environment import environment_manifest
from video_dataset.utils.io import read_json, read_jsonl, write_json_atomic, write_jsonl
from video_dataset.utils.logging import get_logger
from video_dataset.validation.policy import POLICY_VERSION

log = get_logger("dataset.aggregate")
STRICT_DIR = "strict"
PROVENANCE_FILE = "provenance.jsonl"


def licenses_block(config: PipelineConfig) -> dict[str, Any]:
    """What each licence in the release covers, published verbatim in manifest.json and the card."""
    lic = config.licensing
    return {
        "code": getattr(lic, "code_license", "MIT"),
        "annotations_and_metadata": getattr(lic, "annotation_license", "MIT"),
        "source_videos": "each source video keeps its own licence; see provenance.jsonl (url, license, license_status, rights_status)",
        "media": getattr(lic, "media_policy", ""),
        "release_rights": list(lic.release_rights or []),
        "note": "the annotation licence grants no rights to the underlying videos, audio, transcripts or on-screen text; obtain permission from the rights holder before redistributing media",
    }


def tier_policy_block(config: PipelineConfig) -> dict[str, Any]:
    """Minimum requirements of every tier, so `tier` is documented next to the data."""
    t = config.export.tiers
    return {
        "accepted": f"validation.status accepted = every required check of the record type passed or is explicitly not applicable under acceptance policy {POLICY_VERSION} (clips: boundary_check passed); records that only carry a confidence threshold are review",
        "gold": {"status": "accepted", "quality.overall_min": t.gold_min_overall, "requires_grounding_component": bool(t.gold_requires_grounding), "confidence_min": t.gold_min_confidence, "confidence_source": "verifier | measurement | asr_logprob | ocr_score | detector_score | interval_algebra | derived_min", "quality_flags": "none", "near_duplicate_video": False},
        "silver": {"status": "accepted", "quality.overall_min": t.silver_min_overall, "note": "or a gold candidate blocked by one gold requirement (tier_reasons says which)"},
        "bronze": {"status": "accepted with quality.overall below silver, or review / unscored", "note": "exported for inspection; not training material"},
        "strict_subset": "accepted + every required check passed + semantic verification present + single/multi-shot generation target + not a near-duplicate",
        "missing_scores": "a null quality component is unmeasured, never zero or a pass; quality.missing_components lists them",
    }


def splits_block(config: PipelineConfig, splits: dict[str, list[str]], groups: dict[str, str]) -> dict[str, Any]:
    sp = config.export.split
    return {
        "method": "sha1(salt | source_group_id) buckets, eval splits topped up to the minimums with the lowest-hash groups; assigned per source group (video + near-duplicates), never per clip",
        "salt": sp.salt,
        "fractions": {"validation": sp.validation_fraction, "test": sp.test_fraction},
        "minimums": {"validation": int(sp.min_validation_videos), "test": int(sp.min_test_videos)},
        "required_splits": list(sp.required_splits or []),
        "videos": splits,
        "counts": {k: len(v) for k, v in splits.items()},
        "groups": groups,
    }


def provenance_rows(manifests: list[dict[str, Any]], video_info: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per source video: everything needed to audit or remove it later. No credentials, no
    commenter data; channel name is public metadata."""
    rows = []
    for m in manifests:
        vid = str(m.get("video_id"))
        vi = video_info.get(vid, {})
        rights = rights_info_for({**vi, "license": m.get("license"), "url": m.get("url"), "retrieved_at": m.get("retrieved_at"), "license_status": m.get("license_status"), "rights_status": m.get("rights_status")})
        rows.append({
            "video_id": vid, "url": m.get("url"), "youtube_id": m.get("youtube_id"), "channel": m.get("channel"), "title": m.get("title"),
            "upload_date": m.get("upload_date"), "retrieved_at": m.get("retrieved_at"), "duration": m.get("duration"),
            "license": m.get("license"), "license_status": m.get("license_status"),
            "rights_status": rights["status"],
            # the rights record (issue P0-07): how the status was obtained, when, what it allows, the evidence
            "rights_method": rights["method"], "rights_verified_at": rights["verified_at"], "allowed_use": rights["allowed_use"],
            "redistribution": rights["redistribution"], "evidence_url": rights["evidence_url"],
            "rights_verified_by": (vi.get("rights_evidence") or {}).get("verified_by"),
            "source_sha256": m.get("source_sha256"), "canonical_video_id": vi.get("canonical_video_id"),
            "source_group_id": vi.get("source_group_id"),  # the split group: channel / near-duplicate cluster / the video itself
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
            m = read_json(manifest_path)
            files = {k: f"per_video/{vdir.name}/{k}.jsonl" for k in (m.get("counts") or {})}
            if m.get("files") != files:  # exports written before 2026-10-20 stored absolute paths
                m["files"] = files
                write_json_atomic(manifest_path, m)
            manifests.append(m)
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

    # one schema for every record (versioned migrations), then split / tier / subsets
    upgrade_stats = UpgradeStats()
    records = upgrade_records(records, upgrade_stats)
    video_info = video_info_from_manifests(manifests, float(config.deduplication.video_similarity_threshold), bool(config.deduplication.video_fingerprint))
    rights_evidence = load_rights_evidence(final_dir)
    for m in manifests:
        vid = str(m.get("video_id"))
        vi = video_info.setdefault(vid, {})
        vi.setdefault("rights_status", m.get("rights_status") or rights_status_for(m.get("license_status")))
        vi.setdefault("coverage", m.get("coverage") or {})
        vi.update({"channel": m.get("channel"), "duration": m.get("duration"), "license": m.get("license"), "url": m.get("url"), "retrieved_at": m.get("retrieved_at"), "media": m.get("media") or {}})
        if vid in rights_evidence:
            vi["rights_evidence"] = rights_evidence[vid]
            vi["rights_status"] = rights_evidence[vid]["status"]
    environment = environment_manifest(config)
    provenance_base = {"pipeline_version": environment.get("code", {}).get("package_version"), "git_commit": environment.get("code", {}).get("git_commit"), "models": {}}
    calibration = _read_optional_json(final_dir / "calibration.json")
    started_at = utc_now_iso()
    labels = read_review_labels(final_dir / LABELS_FILE)
    review_counts = apply_review_labels(records, labels)  # human verdicts win over the automatic status
    curation = annotate_records(records, config.export, video_info, config.licensing, seed=int(config.qa.seed), provenance_base=provenance_base, calibration=calibration)
    curation["human_reviews"] = review_counts
    curation["schema_upgrade"] = upgrade_stats.as_dict()
    fill_lighting_and_color(records, video_info)
    for m in manifests:  # record the resolved canonical ids so the manifest explains the splits
        vi = video_info.get(str(m.get("video_id")), {})
        m["canonical_video_id"] = vi.get("canonical_video_id")
        m["rights_status"] = vi.get("rights_status")
        m["rights"] = rights_info_for(vi)
    splits = split_manifest(records)
    for rows_ in records.values():
        for r in rows_:
            if r.get("video_id") and r.get("source_group_id"):
                video_info.setdefault(str(r["video_id"]), {})["source_group_id"] = r["source_group_id"]
    rights_by_video = {vid: str((vi.get("rights_evidence") or {}).get("status") or vi.get("rights_status") or "unknown") for vid, vi in video_info.items()}

    # the asset registry first: records carry asset_id / media_status before the files are written
    media_summary = write_media_manifest(records, final_dir, config.data_dir, bool(getattr(config.export, "hash_media", False)), rights=rights_by_video) if getattr(config.export, "media_manifest", True) else {}
    counts, combined_rows, typed_counts, parquet_rows = _write_dataset_files(records, final_dir, config)
    write_schema_file(final_dir / SCHEMA_FILE)
    write_json_atomic(final_dir / SPLITS_FILE, splits_block(config, splits, {str(r.get("video_id")): str(r.get("source_group_id") or r.get("video_id")) for rows in records.values() for r in rows if r.get("video_id")}))
    write_jsonl(final_dir / PROVENANCE_FILE, provenance_rows(manifests, video_info))

    cinematic_counts: dict[str, int] = {}
    if config.export.cinematic.enabled and config.export.cinematic.write_files:
        subset = subset_records(records, "cinematic")
        cinematic_counts, _c, _t, _p = _write_dataset_files(subset, final_dir / CINEMATIC_DIR, config)
    strict_counts: dict[str, int] = {}
    strict_cfg = getattr(config.export, "strict", None)
    if strict_cfg is not None and strict_cfg.enabled and strict_cfg.write_files:
        strict_counts, _c, _t, _p = _write_dataset_files(subset_records(records, STRICT_SUBSET), final_dir / STRICT_DIR, config)

    # task-specific exports (understanding / temporal reasoning / cinematic / generation-ready) and the benchmark
    task_summary: dict[str, Any] = {}
    if getattr(config.export, "task_exports", True):
        from video_dataset.dataset.task_exports import write_task_exports

        available = {str(r["path"]) for r in read_jsonl(final_dir / "media_manifest.jsonl") if r.get("exists")} if (final_dir / "media_manifest.jsonl").exists() else set()
        task_summary = write_task_exports(records, final_dir, config, media_available=lambda rec: str(rec.get("clip_path") or rec.get("frame_path") or "") in available, rights=rights_by_video, release_rights=list(config.licensing.release_rights or []), write_parquet="parquet" in config.export.formats)
    benchmark_summary: dict[str, Any] = {}
    if getattr(config.export, "benchmark", True):
        from video_dataset.dataset.benchmark import build_benchmark

        benchmark_summary = build_benchmark(records, final_dir)

    # lineage: every status change of this export (policy downgrades, human labels) with a trace
    lineage_rows = 0
    if getattr(config.export, "lineage", True):
        from video_dataset.dataset.repair import append_lineage, lineage_row

        rows = []
        for name, recs in records.items():
            for r in recs:
                for h in ((r.get("validation") or {}).get("history") or []):
                    if str(h.get("at") or "") >= started_at:
                        rows.append(lineage_row(r.get("record_id"), name, r.get("video_id"), "status", h.get("from"), h.get("to"), h.get("reason"), "export_policy" if h.get("actor") == "export_policy" else "human_review", h.get("actor"), policy_version=h.get("policy_version"), pipeline_version=provenance_base.get("git_commit")))
        lineage_rows = append_lineage(final_dir, rows) if rows else 0

    videos = db.list_videos()
    exported_ids = {m["video_id"] for m in manifests}
    videos_exported = [v for v in videos if v["video_id"] in exported_ids]
    release_id = f"{started_at.replace(':', '').replace('-', '')}-{(provenance_base.get('git_commit') or 'nogit')[:8]}"
    stats = compute_statistics(
        videos_exported,
        records,
        [m.get("validation_summary", {}) for m in manifests],
        extra={
            "release_id": release_id,
            "source_revision": provenance_base.get("git_commit"),
            "generated_at": started_at,
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
            "cinematic_coverage": cinematic_coverage(records),
            "qa_rejection_analysis": qa_rejection_analysis(records),
            "task_exports": task_summary,
            "benchmark": benchmark_summary,
            "lineage_rows_appended": lineage_rows,
            "calibration": calibration_status(calibration),
            "files": {  # relative to the export root, so the file is valid wherever the export lands
                **({name: f"{name}.jsonl" for name in DATASET_FILES} if config.export.per_type_jsonl else {}),
                **({f"{name}_parquet": f"{name}.parquet" for name in DATASET_FILES} if config.export.per_type_parquet else {}),
                **({"dataset_jsonl": COMBINED_FILE} if config.export.combined_jsonl and "jsonl" in config.export.formats else {}),
                **({"dataset_parquet": "dataset.parquet"} if "parquet" in config.export.formats else {}),
                "schemas": SCHEMA_FILE,
                "splits": SPLITS_FILE,
                **({"cinematic_dir": CINEMATIC_DIR} if cinematic_counts else {}),
                **({"strict_dir": STRICT_DIR} if strict_counts else {}),
                **({"exports_dir": "exports"} if task_summary else {}),
                **({"benchmark_dir": "benchmark"} if benchmark_summary else {}),
                "provenance": PROVENANCE_FILE,
                "media_manifest": "media_manifest.jsonl",
                "inventory": "inventory.json",
                "checksums": "checksums.sha256",
            },
            "jsonl_counts": counts,
            "empty_record_types": [name for name in DATASET_FILES if not counts.get(name)],
            "licenses": licenses_block(config),
            "tier_policy": tier_policy_block(config),
            "rights_terms": {k: {"allowed_use": v[0], "redistribution": v[1]} for k, v in RIGHTS_TERMS.items()},
        },
    )
    write_json_atomic(final_dir / "statistics.json", stats)
    if getattr(config.export, "review_history", True):
        append_review_history(final_dir, release_id, stats)
    for m in manifests:  # never publish the producer's directory layout
        m["files"] = {k: f"per_video/{m.get('video_id')}/{k}.jsonl" for k in (m.get("counts") or {})}
    write_json_atomic(final_dir / "manifest.json", {
        "release_id": release_id, "source_revision": provenance_base.get("git_commit"), "generated_at": started_at,
        "videos": manifests, "counts": counts, "typed_parquet_counts": typed_counts, "parquet_rows": parquet_rows, "combined_jsonl_rows": combined_rows,
        "empty_record_types": [name for name in DATASET_FILES if not counts.get(name)],
        "splits": {k: len(v) for k, v in splits.items()}, "cinematic": cinematic_counts, "strict": strict_counts,
        "schema_version": SCHEMA_VERSION, "policy_version": POLICY_VERSION, "media_references": media_summary,
        "licenses": licenses_block(config), "tier_policy": tier_policy_block(config),
        "rights_terms": {k: {"allowed_use": v[0], "redistribution": v[1]} for k, v in RIGHTS_TERMS.items()},
        "policy_downgraded": curation.get("policy_downgraded", 0), "schema_upgrade": upgrade_stats.as_dict(),
        "task_exports": {k: {"eligible_total": v.get("eligible_total"), "counts": v.get("counts")} for k, v in task_summary.items()},
        "benchmark": {k: benchmark_summary.get(k) for k in ("version", "counts", "frozen_hash", "note") if k in benchmark_summary},
        "removed_sources": sorted(removed), "environment": environment,
    })
    if getattr(config.export, "inventory", True):
        from video_dataset.dataset.inventory import write_inventory

        inv = write_inventory(final_dir, include_media=False)
        log.info("inventory: %d files, %.1f MB hashed -> inventory.json + checksums.sha256", inv["totals"]["files"], inv["totals"]["bytes"] / 1e6)
    log.info("aggregated %d videos -> %s (release %s, splits %s, cinematic records %d)", len(manifests), final_dir, release_id, {k: len(v) for k, v in splits.items()}, curation.get("cinematic", 0))
    return stats


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = read_json(path)
        return data if isinstance(data, dict) else None
    except (ValueError, OSError):
        return None


def calibration_status(calibration: dict[str, Any] | None) -> dict[str, Any]:
    """The calibration block of statistics.json: measured results when ``calibration.json`` exists
    (``video-dataset calibrate``), else an explicit ``uncalibrated`` status - never a silent default."""
    if not calibration or not calibration.get("by_source"):
        return {"status": "uncalibrated", "note": "confidence values are instrument scores (flow consistency, detector probability, ASR log-prob, OCR score); run `video-dataset calibrate` with human labels to measure them"}
    return {
        "status": "calibrated", "version": calibration.get("version"), "labels": calibration.get("labels"), "matched_records": calibration.get("matched_records"),
        "by_source": {k: {m: v.get(m) for m in ("count", "accuracy", "brier", "ece", "recommended_threshold")} for k, v in calibration["by_source"].items()},
        "note": "per confidence_source and record type; scores of different sources are never pooled",
    }


def fill_lighting_and_color(records: dict[str, list[dict[str, Any]]], video_info: dict[str, dict[str, Any]]) -> int:
    """Controlled lighting and colour blocks on scene records (in place): the measured level and
    the labelled appearance stay apart from the physical setup, which no instrument estimates
    (null); colour space / transfer / primaries / range come from the container when declared."""
    n = 0
    for rec in records.get("scenes", []):
        m = rec.get("measurements") if isinstance(rec.get("measurements"), dict) else {}
        env = rec.get("environment") if isinstance(rec.get("environment"), dict) else {}
        media = (video_info.get(str(rec.get("video_id")), {}) or {}).get("media") or {}
        tod = env.get("time_of_day")
        rec["lighting"] = {
            "level": m.get("lighting_level"),
            "level_method": "mean_gray_thresholds" if m.get("lighting_level") else None,
            "exposure": m.get("exposure"),
            "trend": m.get("brightness_trend"),
            "appearance_time_of_day": {"label": tod, "source": "label" if tod and str(tod) != "unknown" else None, "confidence": None},
            "source": None, "direction": None, "softness": None,  # physical setup: not estimated by any instrument
            "appearance_vs_setup": "appearance only; the physical light setup is not inferred",
            "method": "measured_brightness+environment_label",
        }
        rec["color"] = {
            "color_space": media.get("color_space"), "color_transfer": media.get("color_transfer"), "color_primaries": media.get("color_primaries"), "color_range": media.get("color_range"), "pix_fmt": media.get("pix_fmt"),
            "stats_color_space": m.get("color_stats_space") or "bgr8_opencv_decoded (display-referred, no colour management)",
            "stats_method": m.get("color_method") or "mean/std over decoded frames; saturation from HSV; colorfulness Hasler-Suesstrunk; temperature from R-B balance",
            "white_balance": None,  # never guessed
            "temperature_label": m.get("color_temperature"),
            "dominant_colors": m.get("dominant_colors") or [],
        }
        n += 1
    return n


def cinematic_coverage(records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Eligible unique clips and hours of the cinematic subset by camera movement, shot size,
    lighting level and setting: the numbers a card may claim, nothing more."""
    scenes = [r for r in records.get("scenes", []) if "cinematic" in (r.get("subsets") or [])]
    clips = [r for r in records.get("clips", []) if "cinematic" in (r.get("subsets") or [])]

    def by(key_fn: Any, rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for r in rows:
            k = str(key_fn(r) or "unknown")
            d = out.setdefault(k, {"clips": 0, "hours": 0.0})
            d["clips"] += 1
            dur = r.get("duration")
            if dur is None and r.get("start_time") is not None and r.get("end_time") is not None:
                dur = float(r["end_time"]) - float(r["start_time"])
            d["hours"] += float(dur or 0.0) / 3600.0
        return {k: {"clips": int(v["clips"]), "hours": round(v["hours"], 3)} for k, v in sorted(out.items(), key=lambda kv: -kv[1]["clips"])}

    base = clips or scenes
    total_hours = sum(float((r.get("end_time") or 0) - (r.get("start_time") or 0)) for r in base) / 3600.0
    return {
        "unique_clips": len(clips), "unique_scenes": len(scenes), "videos": len({str(r.get("video_id")) for r in base}), "hours": round(total_hours, 3),
        "by_camera_movement": by(lambda r: (r.get("camera") or {}).get("camera_movement"), base),
        "by_shot_size": by(lambda r: (r.get("camera") or {}).get("shot_size"), base),
        "by_lighting_level": by(lambda r: ((r.get("measurements") or {}).get("lighting_level")), base),
        "by_setting": by(lambda r: ((r.get("environment") or {}).get("setting")), base),
        "camera_verified": sum(1 for r in base if (r.get("camera") or {}).get("camera_movement") not in (None, "unknown") and not (r.get("camera") or {}).get("movement_ambiguous")),
        "camera_ambiguous": sum(1 for r in base if (r.get("camera") or {}).get("movement_ambiguous")),
        "note": "counts of the cinematic subset only; camera labels are optical-flow measurements, not human-verified unless validation.verification.human == accepted",
    }


def qa_rejection_analysis(records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Why QA items are not accepted, per file and question type: reason codes by type, the share
    of long-video questions accepted, and the checks that fail most (the upstream repair targets)."""
    out: dict[str, Any] = {}
    for name in ("temporal_qa", "long_video_qa"):
        rows = records.get(name, [])
        if not rows:
            continue
        by_type: dict[str, dict[str, int]] = {}
        reasons: dict[str, int] = {}
        for r in rows:
            st = str((r.get("validation") or {}).get("status"))
            t = str(r.get("type"))
            by_type.setdefault(t, {})[st] = by_type.setdefault(t, {}).get(st, 0) + 1
            if st != "accepted":
                for code in (r.get("validation") or {}).get("reason_codes") or []:
                    key = str(code).split(" ")[0][:60]
                    reasons[key] = reasons.get(key, 0) + 1
        accepted = sum(1 for r in rows if (r.get("validation") or {}).get("status") == "accepted")
        out[name] = {"records": len(rows), "accepted": accepted, "acceptance_rate": round(accepted / len(rows), 4), "by_type_status": by_type, "top_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])[:15]), "human_reviewed": sum(1 for r in rows if "human:accepted" in ((r.get("validation") or {}).get("reason_codes") or []) or "human:rejected" in ((r.get("validation") or {}).get("reason_codes") or []))}
    return out


def append_review_history(final_dir: Path, release_id: str, stats: dict[str, Any]) -> None:
    """One line per export in ``review_history.jsonl``: the review / accepted share per record type,
    so a repair can be shown to lower the review rate instead of asserting it."""
    row = {"release_id": release_id, "at": stats.get("generated_at"), "policy_version": stats.get("policy_version"), "schema_version": stats.get("schema_version"), "review_rate": {}, "accepted_rate": {}}
    for name, rep in (stats.get("validation_report") or {}).items():
        pct = rep.get("status_pct") or {}
        row["review_rate"][name] = pct.get("review")
        row["accepted_rate"][name] = pct.get("accepted")
    with (final_dir / "review_history.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_statistics(config: PipelineConfig) -> dict[str, Any] | None:
    p = config.export_dir / "statistics.json"
    return json.loads(p.read_text()) if p.exists() else None
