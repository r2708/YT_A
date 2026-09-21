"""Cut per-scene clips with FFmpeg.

Stream copy (`-c copy`) is instant but can only start at a keyframe, so a clip may contain seconds of the
neighbouring shots while its record claims one scene. Every clip is therefore *measured* after cutting:

* ``clip_codec: auto`` (default) - stream-copy first; if the file is longer or shorter than the scene by
  more than ``clip_tolerance_seconds`` it is re-encoded with libx264 at the exact boundaries.
* ``libx264`` - always re-encode (exact, slower).
* ``copy`` - stream copy only; inexact clips are kept but flagged ``exact: false`` in the record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from video_dataset.config import FrameSamplingConfig
from video_dataset.schemas.scene import Clip, Frame, Scene
from video_dataset.utils.ffmpeg import ffprobe_json, run_ffmpeg
from video_dataset.utils.ids import clip_id as make_clip_id
from video_dataset.utils.logging import get_logger

log = get_logger("frame_sampling.clips")

CLIP_CODECS = ("auto", "copy", "libx264")


@dataclass
class ClipStats:
    total: int = 0
    stream_copied: int = 0
    reencoded: int = 0
    inexact: int = 0  # clips left with a boundary error above the tolerance (copy mode only)
    failed: int = 0
    reused: int = 0
    worst_error_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def as_metrics(self) -> dict[str, int | float]:
        return {
            "clips": self.total, "clips_stream_copied": self.stream_copied, "clips_reencoded": self.reencoded,
            "clips_inexact": self.inexact, "clips_failed": self.failed, "clips_reused": self.reused,
            "clip_worst_error_seconds": round(self.worst_error_seconds, 3),
        }


def media_duration(path: Path, ffprobe_path: str | None = None) -> float | None:
    """Length of a media file in seconds (ffprobe, OpenCV fallback)."""
    try:
        info = ffprobe_json(path, ffprobe_path)
    except Exception as exc:
        log.debug("ffprobe failed on %s: %s", path.name, exc)
        info = None
    if info:
        try:
            d = float(info.get("format", {}).get("duration") or 0.0)
            if d > 0:
                return d
        except (TypeError, ValueError):
            pass
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        finally:
            cap.release()
        return n / fps if fps and n else None
    except Exception as exc:
        log.debug("OpenCV duration probe failed on %s: %s", path.name, exc)
        return None


def _cut(video_path: Path, out: Path, start: float, end: float, *, reencode: bool, has_audio: bool, clip_height: int, ffmpeg_path: str | None) -> None:
    length = max(0.05, end - start)
    args = ["-ss", f"{start:.3f}", "-i", str(video_path), "-t", f"{length:.3f}"]
    if not reencode:
        args += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
    else:
        args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-vf", f"scale=-2:'min(ih,{clip_height})'"]
        args += ["-c:a", "aac", "-b:a", "128k"] if has_audio else ["-an"]
    args += ["-movflags", "+faststart", str(out)]
    run_ffmpeg(args, ffmpeg_path=ffmpeg_path)


def extract_clips(
    video_path: Path,
    video_id: str,
    scenes: list[Scene],
    frames: list[Frame],
    out_dir: Path,
    cfg: FrameSamplingConfig,
    has_audio: bool,
    ffmpeg_path: str | None = None,
    ffprobe_path: str | None = None,
) -> tuple[list[Clip], ClipStats]:
    out_dir.mkdir(parents=True, exist_ok=True)
    codec = (cfg.clip_codec or "auto").lower()
    if codec not in CLIP_CODECS:
        raise ValueError(f"frame_sampling.clip_codec must be one of {CLIP_CODECS}, got '{cfg.clip_codec}'")
    tolerance = float(cfg.clip_tolerance_seconds or 0.25)
    clip_height = int(cfg.clip_max_height or 720)
    max_len = float(cfg.clip_max_duration or 0) or float("inf")
    stats = ClipStats()
    clips: list[Clip] = []

    for scene in scenes:
        n_parts = max(1, int(scene.duration // max_len) + (1 if scene.duration % max_len > 0.5 else 0)) if scene.duration > max_len else 1
        part_len = scene.duration / n_parts
        for part in range(n_parts):
            start = scene.start_time + part * part_len
            end = scene.end_time if part == n_parts - 1 else start + part_len
            expected = end - start
            if expected < 0.05:
                continue
            cid = make_clip_id(scene.index, part)
            out = out_dir / f"{cid}.mp4"
            stats.total += 1
            produced_by: str | None = None
            measured: float | None = None

            try:
                if out.exists() and out.stat().st_size > 0:
                    measured = media_duration(out, ffprobe_path)
                    stats.reused += 1
                elif codec == "libx264":
                    _cut(video_path, out, start, end, reencode=True, has_audio=has_audio, clip_height=clip_height, ffmpeg_path=ffmpeg_path)
                    produced_by = "libx264"
                    measured = media_duration(out, ffprobe_path)
                    stats.reencoded += 1
                else:
                    _cut(video_path, out, start, end, reencode=False, has_audio=has_audio, clip_height=clip_height, ffmpeg_path=ffmpeg_path)
                    produced_by = "copy"
                    measured = media_duration(out, ffprobe_path)
                    stats.stream_copied += 1

                error = abs(measured - expected) if measured is not None else 0.0
                if error > tolerance and codec == "auto":
                    # keyframe-aligned copy missed the boundary: re-cut this clip exactly
                    log.debug("clip %s is %.2fs vs scene %.2fs; re-encoding", cid, measured or -1, expected)
                    _cut(video_path, out, start, end, reencode=True, has_audio=has_audio, clip_height=clip_height, ffmpeg_path=ffmpeg_path)
                    produced_by = "libx264"
                    measured = media_duration(out, ffprobe_path)
                    error = abs(measured - expected) if measured is not None else 0.0
                    stats.reencoded += 1
            except Exception as exc:
                stats.failed += 1
                stats.errors.append(f"{cid}: {exc}"[:200])
                log.warning("clip %s failed: %s", cid, exc)
                continue

            exact = error <= tolerance
            if not exact:
                stats.inexact += 1
            stats.worst_error_seconds = max(stats.worst_error_seconds, error)
            clips.append(
                Clip(
                    clip_id=cid,
                    video_id=video_id,
                    scene_id=scene.scene_id,
                    start_time=round(start, 3),
                    end_time=round(end, 3),
                    clip_path=str(out),
                    frame_ids=[f.frame_id for f in frames if f.scene_id == scene.scene_id and start - 1e-3 <= f.timestamp <= end + 1e-3],
                    media_duration=round(measured, 3) if measured is not None else None,
                    exact=exact,
                    codec=produced_by,
                )
            )
    if stats.inexact:
        log.warning(
            "%d of %d clips are off by more than %.2fs (worst %.2fs); use frame_sampling.clip_codec=auto or libx264 for exact cuts",
            stats.inexact, stats.total, tolerance, stats.worst_error_seconds,
        )
    return clips, stats
