"""Check that the media the exported records point at really exists and can be read.

Every ``clip_path`` / ``frame_path`` of the dataset (final/ root files, else per_video/) is resolved
against the data directory; clips are probed with ffprobe (duration must be > 0 and, when the
record carries a span, within the tolerance), frames are decoded with OpenCV. The result is a
report (``final/media_report.json``) listing missing, unreadable and mismatched files per video,
so a shard can be checked before it is used for training or uploaded with media.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from video_dataset.config import PipelineConfig
from video_dataset.dataset.export import DATASET_FILES
from video_dataset.utils.io import read_jsonl, write_json_atomic
from video_dataset.utils.logging import get_logger

log = get_logger("dataset.verify_media")
REPORT_FILE = "media_report.json"


def _records(final_dir: Path, kind: str) -> list[dict[str, Any]]:
    root = final_dir / f"{kind}.jsonl"
    if root.exists():
        return list(read_jsonl(root))
    out: list[dict[str, Any]] = []
    per_video = final_dir / "per_video"
    if per_video.exists():
        for vdir in sorted(p for p in per_video.iterdir() if p.is_dir()):
            f = vdir / f"{kind}.jsonl"
            if f.exists():
                out.extend(read_jsonl(f))
    return out


def _resolve(path: str, data_dir: Path, final_dir: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    for base in (data_dir, final_dir):
        cand = base / p
        if cand.exists():
            return cand
    return data_dir / p


def verify_media(
    config: PipelineConfig,
    final_dir: Path | None = None,
    *,
    probe_clips: bool = True,
    frames_per_video: int | None = 20,
    tolerance: float | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    final = Path(final_dir) if final_dir else config.export_dir
    data_dir = config.data_dir
    tol = float(tolerance if tolerance is not None else (config.frame_sampling.clip_tolerance_seconds or 0.25))
    ffprobe = config.project.ffprobe_path
    problems: list[dict[str, Any]] = []
    per_video: dict[str, dict[str, int]] = {}

    def note(vid: str, kind: str, what: str, path: str, detail: str | None = None) -> None:
        per_video.setdefault(vid, {})[what] = per_video.setdefault(vid, {}).get(what, 0) + 1
        if len(problems) < 500:
            problems.append({"video_id": vid, "kind": kind, "problem": what, "path": path, **({"detail": detail} if detail else {})})

    # ---- clips -----------------------------------------------------------------------------------
    clips = _records(final, "clips")
    clip_stats = {"total": len(clips), "missing": 0, "unreadable": 0, "length_mismatch": 0, "ok": 0}
    if clips and probe_clips:
        from video_dataset.frame_sampling.clips import probe_clip
    for rec in clips:
        vid = str(rec.get("video_id"))
        path = str(rec.get("clip_path") or "")
        p = _resolve(path, data_dir, final)
        if not path or not p.exists() or p.stat().st_size == 0:
            clip_stats["missing"] += 1
            note(vid, "clip", "missing", path)
            continue
        if probe_clips:
            info = probe_clip(p, ffprobe)
            dur = info.get("duration")
            if not dur or float(dur) <= 0:
                clip_stats["unreadable"] += 1
                note(vid, "clip", "unreadable", path)
                continue
            if rec.get("start_time") is not None and rec.get("end_time") is not None:
                expected = float(rec["end_time"]) - float(rec["start_time"])
                if abs(float(dur) - expected) > tol:
                    clip_stats["length_mismatch"] += 1
                    note(vid, "clip", "length_mismatch", path, f"{float(dur):.2f}s vs {expected:.2f}s")
                    continue
        clip_stats["ok"] += 1

    # ---- frames ----------------------------------------------------------------------------------
    frames = _records(final, "frames")
    frame_stats = {"total": len(frames), "checked": 0, "missing": 0, "unreadable": 0, "ok": 0}
    seen_per_video: dict[str, int] = {}
    import cv2

    for rec in frames:
        vid = str(rec.get("video_id"))
        path = str(rec.get("frame_path") or "")
        p = _resolve(path, data_dir, final)
        if not path or not p.exists() or p.stat().st_size == 0:
            frame_stats["missing"] += 1
            note(vid, "frame", "missing", path)
            continue
        if frames_per_video is not None and seen_per_video.get(vid, 0) >= int(frames_per_video):
            continue
        seen_per_video[vid] = seen_per_video.get(vid, 0) + 1
        frame_stats["checked"] += 1
        img = cv2.imread(str(p))
        if img is None or img.size == 0:
            frame_stats["unreadable"] += 1
            note(vid, "frame", "unreadable", path)
            continue
        frame_stats["ok"] += 1

    # ---- videos without any media at all --------------------------------------------------------
    video_ids = {str(r.get("video_id")) for kind in DATASET_FILES for r in _records(final, kind) if kind in ("scenes", "clips", "frames")}
    media_less = sorted(v for v in video_ids if per_video.get(v, {}).get("missing", 0) and not any(k in ("ok",) for k in per_video.get(v, {})))
    report = {
        "final_dir": str(final),
        "data_dir": str(data_dir),
        "clips": clip_stats,
        "frames": frame_stats,
        "videos_with_problems": len(per_video),
        "per_video": dict(sorted(per_video.items())),
        "videos_without_media": media_less,
        "problems": problems,
        "ok": clip_stats["missing"] == clip_stats["unreadable"] == clip_stats["length_mismatch"] == 0 and frame_stats["missing"] == frame_stats["unreadable"] == 0,
    }
    if write_report:
        final.mkdir(parents=True, exist_ok=True)
        write_json_atomic(final / REPORT_FILE, report)
    log.info(
        "media check: clips %d ok / %d missing / %d unreadable / %d length mismatch; frames %d ok / %d missing / %d unreadable (%d checked of %d)",
        clip_stats["ok"], clip_stats["missing"], clip_stats["unreadable"], clip_stats["length_mismatch"],
        frame_stats["ok"], frame_stats["missing"], frame_stats["unreadable"], frame_stats["checked"], frame_stats["total"],
    )
    return report
