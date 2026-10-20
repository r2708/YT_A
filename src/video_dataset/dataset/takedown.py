"""Remove one source video from the working data and from every export (takedown / rights removal).

Everything derived from a source is keyed by its ``video_id``: working files (download, frames,
clips, transcripts, annotations ...), the per-video export, the records in ``final/`` and the
records in archived shards under ``final/uploaded/``. ``remove_source`` plans and deletes all of
them, rewrites the affected final files, and appends a tombstone to ``final/removed_sources.json``
so the id can never be re-exported silently (aggregation drops records of removed sources).
Shards already pushed to the Hub are listed in the result; deleting there is a separate, explicit
step (``huggingface-cli`` / the Hub UI) because it rewrites a public repository.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from video_dataset.config import PipelineConfig
from video_dataset.dataset.export import COMBINED_FILE, DATASET_FILES
from video_dataset.storage.cleanup import apply_plan, plan_video
from video_dataset.storage.paths import DataPaths
from video_dataset.storage.state_db import StateDB
from video_dataset.utils.io import read_json, read_jsonl, write_json_atomic, write_jsonl
from video_dataset.utils.logging import get_logger

log = get_logger("dataset.takedown")
REMOVED_FILE = "removed_sources.json"


def removed_sources(final_dir: Path) -> dict[str, dict[str, Any]]:
    p = final_dir / REMOVED_FILE
    if not p.exists():
        return {}
    try:
        return {str(k): v for k, v in read_json(p).items()}
    except (ValueError, AttributeError):
        return {}


def _strip_jsonl(path: Path, video_id: str) -> int:
    if not path.exists():
        return 0
    rows = list(read_jsonl(path))
    kept = [r for r in rows if str(r.get("video_id")) != video_id]
    if len(kept) != len(rows):
        write_jsonl(path, kept)
    return len(rows) - len(kept)


def _strip_dir(final: Path, video_id: str) -> dict[str, int]:
    removed: dict[str, int] = {}
    for name in [*DATASET_FILES, COMBINED_FILE.removesuffix(".jsonl")]:
        n = _strip_jsonl(final / f"{name}.jsonl", video_id)
        if n:
            removed[name] = n
    for sub in ("cinematic", "strict"):
        d = final / sub
        if d.is_dir():
            for name in [*DATASET_FILES, COMBINED_FILE.removesuffix(".jsonl")]:
                n = _strip_jsonl(d / f"{name}.jsonl", video_id)
                if n:
                    removed[f"{sub}/{name}"] = n
    manifest = final / "manifest.json"
    if manifest.exists():
        try:
            data = read_json(manifest)
            vids = [m for m in data.get("videos", []) if str(m.get("video_id")) != video_id]
            if len(vids) != len(data.get("videos", [])):
                data["videos"] = vids
                write_json_atomic(manifest, data)
                removed["manifest"] = 1
        except (ValueError, AttributeError):
            pass
    per_video = final / "per_video" / video_id
    if per_video.exists():
        import shutil

        shutil.rmtree(per_video, ignore_errors=True)
        removed["per_video"] = 1
    return removed


def remove_source(config: PipelineConfig, video_id: str, reason: str | None = None, dry_run: bool = False) -> dict[str, Any]:
    paths = DataPaths(config.data_dir)
    final = config.export_dir
    plan = plan_video(paths, video_id)
    result: dict[str, Any] = {"video_id": video_id, "working_paths": [str(p) for p in plan.paths], "working_bytes": plan.bytes, "final": {}, "archived_shards": {}, "hub_shards": [], "dry_run": dry_run}
    state_file = final / "uploads.json"
    if state_file.exists():
        try:
            for s in read_json(state_file).get("shards", []):
                if video_id in s.get("videos", []):
                    result["hub_shards"].append({"shard": s.get("name"), "repo_id": s.get("repo_id"), "path_in_repo": s.get("path_in_repo")})
        except (ValueError, AttributeError):
            pass
    if dry_run:
        return result
    apply_plan(plan)
    result["final"] = _strip_dir(final, video_id)
    uploaded = final / "uploaded"
    if uploaded.is_dir():
        for shard in sorted(p for p in uploaded.iterdir() if p.is_dir()):
            removed = _strip_dir(shard, video_id)
            if removed:
                result["archived_shards"][shard.name] = removed
    with StateDB(config.db_path) as db:
        if db.get_video(video_id) is not None:
            db.delete_video(video_id)
    tomb = removed_sources(final)
    tomb[video_id] = {"removed_at": datetime.now(UTC).isoformat(timespec="seconds"), "reason": reason, "hub_shards": result["hub_shards"]}
    final.mkdir(parents=True, exist_ok=True)
    write_json_atomic(final / REMOVED_FILE, tomb)
    log.info("removed source %s: %d working paths, final records %s, archived shards %s, hub shards to clean: %d", video_id, len(plan.paths), result["final"], list(result["archived_shards"]), len(result["hub_shards"]))
    return result
