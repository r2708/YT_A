"""Disk cleanup helpers behind `video-dataset clean`.

Everything here is *planned first, deleted second*: `plan_*` functions return the paths (with sizes)
that would go, so the CLI can show a dry run, and `apply_plan` removes them. Nothing touches the
exported dataset in data/final unless a plan explicitly includes it.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from video_dataset.stages import Stage
from video_dataset.storage.paths import DataPaths
from video_dataset.storage.state_db import StateDB
from video_dataset.utils.logging import get_logger

log = get_logger("cleanup")


@dataclass
class CleanupPlan:
    paths: list[Path] = field(default_factory=list)
    bytes: int = 0
    notes: list[str] = field(default_factory=list)

    def add(self, path: Path) -> None:
        if path in self.paths or not path.exists():
            return
        self.paths.append(path)
        self.bytes += path_size(path)

    def extend(self, other: CleanupPlan) -> None:
        for p in other.paths:
            self.add(p)
        self.notes.extend(other.notes)


def path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


# ----------------------------------------------------------------------------- per-video artifacts
def video_artifact_paths(paths: DataPaths, video_id: str) -> list[Path]:
    """Every file or directory the pipeline writes for one video (the exported dataset excluded)."""
    return [
        paths.video_dir(video_id),
        paths.audio_file(video_id),
        paths.audio_events_file(video_id),
        paths.scenes_file(video_id),
        paths.frames_dir(video_id),
        paths.clips_dir(video_id),
        paths.transcript_file(video_id),
        paths.ocr_file(video_id),
        paths.annotations_dir(video_id),
        paths.qa_file(video_id),
        paths.validated_file(video_id),
        paths.video_log_file(video_id),
    ]


def plan_video(paths: DataPaths, video_id: str) -> CleanupPlan:
    """All artifacts of one video (frames and clips included, so the exported dataset will reference
    missing media afterwards - re-export or re-run the video)."""
    plan = CleanupPlan()
    for p in video_artifact_paths(paths, video_id):
        plan.add(p)
    return plan


AFTER_EXPORT_LEVELS = ("none", "media", "all")


def plan_after_export(paths: DataPaths, video_id: str, level: str, frames_and_clips: bool = False) -> CleanupPlan:
    """What the runner deletes automatically once a video's EXPORT stage is DONE.

    * ``media``: the original download(s), the canonical video.mp4 and audio.wav.
    * ``all``:   additionally every working JSON (download metadata, scenes, audio events, transcript,
                 OCR, annotations, QA, validated).
    * ``frames_and_clips``: with either level, also the whole frames/<id>/ directory (the scene_NNN
                 folders and frames.json) and clips/<id>/. The exported records still carry those paths.

    The per-video export under final/per_video and the video log are always kept.
    """
    level = (level or "none").lower()
    if level not in AFTER_EXPORT_LEVELS:
        raise ValueError(f"cleanup.after_export must be one of {AFTER_EXPORT_LEVELS}, got '{level}'")
    plan = CleanupPlan()
    if level == "none":
        return plan
    if frames_and_clips:
        plan.add(paths.frames_dir(video_id))
        plan.add(paths.clips_dir(video_id))
    if level == "all":
        plan.add(paths.video_dir(video_id))  # source.*, video.mp4, metadata.json, media_info.json, download.json
    else:
        for src in paths.source_candidates(video_id):
            plan.add(src)
        plan.add(paths.video_file(video_id))
    plan.add(paths.audio_file(video_id))
    if level == "all":
        plan.add(paths.audio_events_file(video_id))
        plan.add(paths.scenes_file(video_id))
        plan.add(paths.transcript_file(video_id))
        plan.add(paths.ocr_file(video_id))
        plan.add(paths.annotations_dir(video_id))
        plan.add(paths.qa_file(video_id))
        plan.add(paths.validated_file(video_id))
    return plan


def plan_intermediate(paths: DataPaths, db: StateDB, video_ids: list[str] | None = None) -> CleanupPlan:
    """Bulky files the dataset does not reference and later stages do not need once EXPORT is DONE:

    * the original download (`source.*`) when a separate canonical `video.mp4` exists
    * the 16 kHz `audio.wav` (re-created by `--force-from PREPROCESS` if transcription must be redone)

    Frames, clips, scenes, transcripts, annotations and QA files are kept: they are referenced by the
    exported records or needed to resume later stages cheaply.
    """
    plan = CleanupPlan()
    videos = video_ids or [v["video_id"] for v in db.list_videos()]
    for vid in videos:
        if not db.is_done(vid, Stage.EXPORT):
            plan.notes.append(f"{vid}: EXPORT not done, kept")
            continue
        canonical = paths.video_file(vid)
        for src in paths.source_candidates(vid):
            if canonical.exists() and src.resolve() != canonical.resolve():
                # a hard-linked source shares its blocks with video.mp4: deleting it frees nothing but is harmless
                plan.add(src)
        plan.add(paths.audio_file(vid))
    return plan


def plan_orphans(paths: DataPaths, db: StateDB) -> CleanupPlan:
    """Artifacts on disk for video ids the state DB no longer knows about (after `clean VIDEO_ID`,
    a deleted DB, or crashes between registration and checkpointing)."""
    known = {v["video_id"] for v in db.list_videos()}
    plan = CleanupPlan()
    seen: set[str] = set()
    for root in (paths.downloads_dir, paths.frames_root, paths.clips_root, paths.annotations_root):
        if not root.exists():
            continue
        for d in root.iterdir():
            if d.is_dir() and d.name.startswith("vid_") and d.name not in known:
                seen.add(d.name)
    for root in (paths.audio_dir, paths.scenes_dir, paths.transcripts_dir, paths.ocr_dir, paths.qa_dir, paths.validated_dir, paths.logs_dir):
        if not root.exists():
            continue
        for f in root.iterdir():
            vid = f.name.split(".")[0]
            if f.is_file() and vid.startswith("vid_") and vid not in known:
                seen.add(vid)
    for vid in sorted(seen):
        plan.extend(plan_video(paths, vid))
    return plan


def plan_logs(paths: DataPaths) -> CleanupPlan:
    plan = CleanupPlan()
    if paths.logs_dir.exists():
        for f in paths.logs_dir.iterdir():
            if f.is_file() and f.suffix == ".log":
                plan.add(f)
    return plan


def plan_failed(paths: DataPaths, db: StateDB) -> CleanupPlan:
    """Artifacts of videos whose DOWNLOAD or PREPROCESS failed (partial downloads, rejected inputs)."""
    plan = CleanupPlan()
    for rec in db.failed_stages():
        if rec.stage in (Stage.DOWNLOAD, Stage.PREPROCESS):
            plan.extend(plan_video(paths, rec.video_id))
    return plan


# ----------------------------------------------------------------------------- apply
def apply_plan(plan: CleanupPlan) -> tuple[int, int]:
    """Delete everything in the plan. Returns (paths removed, bytes freed)."""
    removed = 0
    freed = 0
    for p in plan.paths:
        try:
            if not p.exists() and not p.is_symlink():
                continue  # removed with a parent directory earlier in the plan
            size = path_size(p)
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p)
            else:
                p.unlink()
            removed += 1
            freed += size
        except FileNotFoundError:
            continue
        except OSError as exc:
            log.warning("could not remove %s: %s", p, exc)
    return removed, freed


def data_dir_usage(paths: DataPaths) -> dict[str, int]:
    """Bytes used per top-level data sub-directory (for `clean --report`)."""
    return {sub: path_size(paths.root / sub) for sub in DataPaths.SUBDIRS if (paths.root / sub).exists()}
