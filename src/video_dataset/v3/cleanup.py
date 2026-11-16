"""The V3 cleanup gate (spec section 15, "mandatory change to current cleanup logic").

Old behaviour: delete all media right after the JSONL / Parquet upload. New behaviour: media of a
source that is approved for visual training is deleted only after its assets were uploaded, read
back, checksum-verified and recorded (job flag ``cleanup_allowed``). Sources without an approved
rights entry produce no visual training asset; they are metadata-only and are cleaned as before."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from video_dataset.utils.logging import get_logger
from video_dataset.v3 import rights
from video_dataset.v3.pipeline import job_dir, open_store

log = get_logger("v3.cleanup")

_rights_cache: dict[str, tuple[float, dict[str, dict[str, Any]]]] = {}


def _rights(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    mtime = path.stat().st_mtime
    cached = _rights_cache.get(str(path))
    if cached is None or cached[0] != mtime:
        cached = (mtime, rights.load_rights(path))
        _rights_cache[str(path)] = cached
    return cached[1]


def cleanup_block_reason(config: Any, video_id: str) -> str | None:
    """Why the main pipeline must keep this video's source and clips for now (None = free to clean)."""
    if not getattr(config.cleanup, "require_verified_upload", True):
        return None
    path = rights.default_rights_path(config)
    entry = _rights(path).get(video_id)
    if not rights.is_approved(entry, path.parent):
        return None  # metadata-only source: no visual training asset is produced from it
    state_db = config.v3_dir / "state.sqlite"
    if state_db.exists():
        store = open_store(config)
        try:
            job = store.get(video_id)
        finally:
            store.close()
        if job and job["flags"].get("cleanup_allowed") and job["flags"].get("metadata_upload_verified") and (job["flags"].get("visual_asset_upload_verified") or job["flags"].get("visual_assets") == "none"):
            return None
    return "approved for visual training: media is kept until `video-dataset v3 publish` has verified its upload (then `video-dataset v3 cleanup`)"


def run_cleanup(config: Any, dry_run: bool = False) -> dict[str, Any]:
    """Delete local media of every job whose remote copy is verified. Records, probes and the state
    database stay. Safe to re-run after a crash: only paths that still exist are removed."""
    from video_dataset.storage.cleanup import apply_plan, path_size, plan_after_export
    from video_dataset.storage.paths import DataPaths

    store = open_store(config)
    freed = 0
    cleaned: list[str] = []
    kept: dict[str, str] = {}
    try:
        for job in store.all():
            vid = job["source_video_id"]
            flags = job["flags"]
            if job["state"] not in ("CLEANUP_ALLOWED", "COMPLETE") or job["failure"]:
                if job["state"] not in ("DISCOVERED", "RIGHTS_PENDING"):
                    kept[vid] = job["failure"] or job["state"]
                continue
            if not (flags.get("cleanup_allowed") and flags.get("metadata_upload_verified")):
                kept[vid] = "flags incomplete"
                continue
            targets = [p for p in (job_dir(config, vid) / d for d in ("source", "clips", "targets")) if p.exists()]
            level = (config.cleanup.after_export or "none").lower()
            plan = plan_after_export(DataPaths(config.data_dir), vid, level if level != "none" else "media", frames_and_clips=bool(config.cleanup.frames_and_clips)) if level != "none" else None
            size = sum(path_size(p) for p in targets) + (plan.bytes if plan else 0)
            if not dry_run:
                for p in targets:
                    shutil.rmtree(p, ignore_errors=True)
                if plan and plan.paths:
                    apply_plan(plan)
                if job["state"] != "COMPLETE":
                    store.advance(vid, "COMPLETE", f"local media removed ({size / 1e6:.1f} MB)")
            freed += size
            cleaned.append(vid)
    finally:
        store.close()
    return {"cleaned": cleaned, "kept": kept, "bytes": freed, "dry_run": dry_run}
