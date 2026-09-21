"""Create the canonical `video.mp4` and speech-ready `audio.wav` with FFmpeg."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from video_dataset.config import PreprocessConfig
from video_dataset.schemas.video import MediaInfo
from video_dataset.utils.ffmpeg import run_ffmpeg
from video_dataset.utils.logging import get_logger

log = get_logger("preprocess.normalize")


def _opencv_can_decode(path: Path) -> bool:
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        ok = cap.isOpened()
        if ok:
            ok, frame = cap.read()
            ok = bool(ok) and frame is not None
        cap.release()
        return ok
    except Exception as exc:
        log.debug("OpenCV could not open %s: %s", path, exc)
        return False


def ensure_canonical_video(
    source: Path,
    dest: Path,
    info: MediaInfo,
    config: PreprocessConfig,
    ffmpeg_path: str | None = None,
) -> tuple[Path, bool]:
    """Return (path, transcoded). Links/copies when the file is already a decodable MP4, otherwise transcodes."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and _opencv_can_decode(dest):
        return dest, False

    container_ok = (info.container or "").lower() in {"mov", "mp4", "m4a", "3gp", "3g2", "mj2"} or source.suffix.lower() == ".mp4"
    codec_ok = (info.video_codec or "").lower() in {c.lower() for c in config.decodable_codecs}
    needs_transcode = config.transcode_if_needed and (not container_ok or not codec_ok or not _opencv_can_decode(source))

    if not needs_transcode:
        if dest.exists():
            dest.unlink()
        try:
            os.link(source, dest)
        except OSError:
            shutil.copy2(source, dest)
        if _opencv_can_decode(dest):
            return dest, False
        if not config.transcode_if_needed:
            raise RuntimeError(f"{source} cannot be decoded by OpenCV and transcoding is disabled")
        dest.unlink(missing_ok=True)

    log.info("transcoding %s -> %s (container=%s codec=%s)", source.name, dest.name, info.container, info.video_codec)
    scale = f"scale=-2:'min(ih,{int(config.max_transcode_height)})'"
    args = [
        "-i", str(source),
        "-map", "0:v:0",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-vf", scale,
        "-movflags", "+faststart",
    ]
    if info.has_audio:
        args += ["-map", "0:a:0?", "-c:a", "aac", "-b:a", "160k"]
    else:
        args += ["-an"]
    tmp = dest.with_suffix(".transcoding.mp4")
    run_ffmpeg([*args, str(tmp)], ffmpeg_path=ffmpeg_path)
    os.replace(tmp, dest)
    return dest, True


def extract_audio(
    video: Path,
    wav: Path,
    sample_rate: int = 16000,
    channels: int = 1,
    ffmpeg_path: str | None = None,
) -> bool:
    """Write 16-bit PCM WAV suitable for ASR. Returns False when the video has no audio stream."""
    wav.parent.mkdir(parents=True, exist_ok=True)
    tmp = wav.with_suffix(".extracting.wav")
    try:
        run_ffmpeg(
            ["-i", str(video), "-vn", "-ac", str(channels), "-ar", str(sample_rate), "-acodec", "pcm_s16le", "-f", "wav", str(tmp)],
            ffmpeg_path=ffmpeg_path,
        )
    except RuntimeError as exc:
        msg = str(exc).lower()
        tmp.unlink(missing_ok=True)
        if "does not contain any stream" in msg or "output file is empty" in msg or "no audio" in msg or "matches no streams" in msg:
            return False
        raise
    if not tmp.exists() or tmp.stat().st_size <= 44:  # WAV header only
        tmp.unlink(missing_ok=True)
        return False
    os.replace(tmp, wav)
    return True
