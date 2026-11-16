"""Create the canonical `video.mp4` and speech-ready `audio.wav` with FFmpeg."""

from __future__ import annotations

import json
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


COMPRESSION_MARKER = "compression.json"


def compress_source(source: Path, config: PreprocessConfig, ffmpeg_path: str | None = None) -> tuple[Path, bool]:
    """Re-encode a downloaded source in place (H.264 at ``compress_crf``) when ``preprocess.compress`` is on.

    Returns (path, compressed). The original is replaced by ``source.mp4`` only when the result is
    smaller; otherwise it is kept untouched. A marker next to the file records the outcome so a
    re-run never compresses the same file twice."""
    if not config.compress:
        return source, False
    marker = source.parent / COMPRESSION_MARKER
    size = source.stat().st_size
    if marker.exists():
        try:
            done = json.loads(marker.read_text())
        except (OSError, ValueError):
            done = {}
        if done.get("size_bytes") == size and done.get("file") == source.name:
            return source, bool(done.get("compressed"))

    crf, preset = int(config.compress_crf), str(config.compress_preset)
    tmp = source.parent / "compressing.tmp.mp4"  # not matched by the `source.*` globs
    log.info("compressing %s (crf=%d preset=%s, %.1f MB)", source.name, crf, preset, size / 1e6)
    run_ffmpeg(
        [
            "-i", str(source),
            "-map", "0:v:0", "-map", "0:a:0?",
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-vf", f"scale=-2:'min(ih,{int(config.max_transcode_height or 720)})'",
            "-c:a", "aac", "-b:a", str(config.compress_audio_bitrate),
            "-movflags", "+faststart",
            str(tmp),
        ],
        ffmpeg_path=ffmpeg_path,
    )
    new_size = tmp.stat().st_size
    record = {"crf": crf, "preset": preset, "original_file": source.name, "original_bytes": size, "encoded_bytes": new_size}
    if new_size >= size or not _opencv_can_decode(tmp):
        tmp.unlink(missing_ok=True)
        log.info("compression kept the original %s (result %.1f MB was not smaller or not decodable)", source.name, new_size / 1e6)
        marker.write_text(json.dumps({**record, "compressed": False, "file": source.name, "size_bytes": size}))
        return source, False
    final = source.with_name("source.mp4") if source.stem == "source" else source.with_suffix(".mp4")
    if final != source:
        source.unlink()
    os.replace(tmp, final)
    marker.write_text(json.dumps({**record, "compressed": True, "file": final.name, "size_bytes": new_size}))
    log.info("compressed %s: %.1f MB -> %.1f MB (%.0f%% smaller)", final.name, size / 1e6, new_size / 1e6, 100 * (1 - new_size / size))
    return final, True


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

    source, compressed = compress_source(source, config, ffmpeg_path)
    container_ok = compressed or (info.container or "").lower() in {"mov", "mp4", "m4a", "3gp", "3g2", "mj2"} or source.suffix.lower() == ".mp4"
    codec_ok = compressed or (info.video_codec or "").lower() in {c.lower() for c in config.decodable_codecs}
    needs_transcode = config.transcode_if_needed and (not container_ok or not codec_ok or not _opencv_can_decode(source))

    if not needs_transcode:
        if dest.exists():
            dest.unlink()
        try:
            os.link(source, dest)
        except OSError:
            shutil.copy2(source, dest)
        if _opencv_can_decode(dest):
            return dest, compressed
        if not config.transcode_if_needed:
            raise RuntimeError(f"{source} cannot be decoded by OpenCV and transcoding is disabled")
        dest.unlink(missing_ok=True)

    log.info("transcoding %s -> %s (container=%s codec=%s)", source.name, dest.name, info.container, info.video_codec)
    scale = f"scale=-2:'min(ih,{int(config.max_transcode_height or 720)})'"
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
