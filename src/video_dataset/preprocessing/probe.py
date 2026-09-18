"""Measure technical media properties with ffprobe (fallback: OpenCV)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from video_dataset.schemas.video import AudioStreamInfo, MediaInfo
from video_dataset.utils.ffmpeg import ffprobe_json, parse_rate
from video_dataset.utils.logging import get_logger

log = get_logger("preprocess.probe")


def _int(v: Any) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _float(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def probe_media(path: str | Path, ffprobe_path: str | None = None) -> MediaInfo:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    data = None
    try:
        data = ffprobe_json(path, ffprobe_path)
    except Exception as exc:
        log.warning("ffprobe failed (%s); falling back to OpenCV", exc)
    if data:
        return _from_ffprobe(path, data)
    return _from_opencv(path)


def _from_ffprobe(path: Path, data: dict[str, Any]) -> MediaInfo:
    streams = data.get("streams") or []
    fmt = data.get("format") or {}
    video = next((s for s in streams if s.get("codec_type") == "video" and not _is_cover_art(s)), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise ValueError(f"No video stream in {path}")

    fps = parse_rate(video.get("avg_frame_rate")) or parse_rate(video.get("r_frame_rate")) or 0.0
    if not fps or fps <= 0 or fps > 480:
        fps = parse_rate(video.get("r_frame_rate")) or 30.0
    duration = _float(fmt.get("duration")) or _float(video.get("duration")) or 0.0
    frame_count = _int(video.get("nb_frames"))
    if not frame_count or frame_count <= 0:
        frame_count = int(round(duration * fps))

    audio_info = None
    if audio is not None:
        audio_info = AudioStreamInfo(
            codec=audio.get("codec_name"),
            sample_rate=_int(audio.get("sample_rate")),
            channels=_int(audio.get("channels")),
            bitrate=_int(audio.get("bit_rate")),
        )

    return MediaInfo(
        path=str(path),
        duration=duration,
        fps=fps,
        width=_int(video.get("width")) or 0,
        height=_int(video.get("height")) or 0,
        frame_count=frame_count,
        video_codec=video.get("codec_name"),
        bitrate=_int(fmt.get("bit_rate")) or _int(video.get("bit_rate")),
        container=(fmt.get("format_name") or "").split(",")[0] or None,
        has_audio=audio is not None,
        audio=audio_info,
        size_bytes=_int(fmt.get("size")) or path.stat().st_size,
    )


def _is_cover_art(stream: dict[str, Any]) -> bool:
    disp = stream.get("disposition") or {}
    return bool(disp.get("attached_pic")) or stream.get("codec_name") in {"mjpeg", "png"}


def _from_opencv(path: Path) -> MediaInfo:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"OpenCV cannot open {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    duration = frame_count / fps if fps else 0.0
    return MediaInfo(
        path=str(path),
        duration=duration,
        fps=float(fps),
        width=width,
        height=height,
        frame_count=frame_count,
        video_codec=None,
        container=path.suffix.lstrip("."),
        has_audio=False,  # unknown without ffprobe; audio extraction will detect
        size_bytes=path.stat().st_size,
    )
