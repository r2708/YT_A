"""Cut per-scene clips with FFmpeg (re-encode for exact boundaries, or stream-copy for speed)."""

from __future__ import annotations

from pathlib import Path

from video_dataset.config import FrameSamplingConfig
from video_dataset.schemas.scene import Clip, Frame, Scene
from video_dataset.utils.ffmpeg import run_ffmpeg
from video_dataset.utils.ids import clip_id as make_clip_id
from video_dataset.utils.logging import get_logger

log = get_logger("frame_sampling.clips")


def extract_clips(
    video_path: Path,
    video_id: str,
    scenes: list[Scene],
    frames: list[Frame],
    out_dir: Path,
    cfg: FrameSamplingConfig,
    has_audio: bool,
    ffmpeg_path: str | None = None,
) -> list[Clip]:
    out_dir.mkdir(parents=True, exist_ok=True)
    clips: list[Clip] = []
    max_len = float(cfg.clip_max_duration or 0) or float("inf")
    clip_height = int(getattr(cfg, "clip_max_height", 720) or 720)
    for scene in scenes:
        n_parts = max(1, int(scene.duration // max_len) + (1 if scene.duration % max_len > 0.5 else 0)) if scene.duration > max_len else 1
        part_len = scene.duration / n_parts
        for part in range(n_parts):
            start = scene.start_time + part * part_len
            end = scene.end_time if part == n_parts - 1 else start + part_len
            if end - start < 0.05:
                continue
            cid = make_clip_id(scene.index, part)
            out = out_dir / f"{cid}.mp4"
            if not (out.exists() and out.stat().st_size > 0):
                args = ["-ss", f"{start:.3f}", "-i", str(video_path), "-t", f"{max(0.05, end - start):.3f}"]
                if cfg.clip_codec == "copy":
                    args += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
                else:
                    args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                             "-vf", f"scale=-2:'min(ih,{clip_height})'"]
                    args += ["-c:a", "aac", "-b:a", "128k"] if has_audio else ["-an"]
                args += ["-movflags", "+faststart", str(out)]
                try:
                    run_ffmpeg(args, ffmpeg_path=ffmpeg_path)
                except Exception as exc:
                    log.warning("clip %s failed: %s", cid, exc)
                    continue
            clips.append(
                Clip(
                    clip_id=cid,
                    video_id=video_id,
                    scene_id=scene.scene_id,
                    start_time=round(start, 3),
                    end_time=round(end, 3),
                    clip_path=str(out),
                    frame_ids=[f.frame_id for f in frames if f.scene_id == scene.scene_id and start - 1e-3 <= f.timestamp <= end + 1e-3],
                )
            )
    return clips
