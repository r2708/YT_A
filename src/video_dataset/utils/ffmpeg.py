"""Locating and invoking FFmpeg / ffprobe."""

from __future__ import annotations

import json
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from video_dataset.utils.logging import get_logger

log = get_logger("ffmpeg")


class FFmpegNotFoundError(RuntimeError):
    pass


@lru_cache(maxsize=4)
def find_ffmpeg(configured: str | None = None) -> str:
    if configured and Path(configured).exists():
        return str(configured)
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg  # type: ignore

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as exc:  # pragma: no cover - depends on environment
        raise FFmpegNotFoundError(
            "ffmpeg not found. Install it (brew install ffmpeg / apt install ffmpeg) or set project.ffmpeg_path."
        ) from exc


@lru_cache(maxsize=4)
def find_ffprobe(configured: str | None = None) -> str | None:
    if configured and Path(configured).exists():
        return str(configured)
    found = shutil.which("ffprobe")
    if found:
        return found
    ff = Path(find_ffmpeg())
    sibling = ff.with_name("ffprobe" + ff.suffix)
    if sibling.exists():
        return str(sibling)
    return None  # callers fall back to OpenCV/PyAV probing


def run_ffmpeg(args: list[str], timeout: float | None = None, ffmpeg_path: str | None = None) -> subprocess.CompletedProcess[str]:
    cmd = [find_ffmpeg(ffmpeg_path), "-hide_banner", "-loglevel", "error", "-y", *args]
    log.debug("ffmpeg %s", " ".join(cmd[1:]))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {proc.stderr.strip()[:2000]}")
    return proc


def ffprobe_json(path: str | Path, ffprobe_path: str | None = None, timeout: float = 60) -> dict[str, Any] | None:
    probe = find_ffprobe(ffprobe_path)
    if probe is None:
        return None
    cmd = [probe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr.strip()[:1000]}")
    return json.loads(proc.stdout or "{}")


def parse_rate(rate: str | None) -> float | None:
    """Parse ffprobe rational frame rates like '30000/1001'."""
    if not rate:
        return None
    try:
        if "/" in rate:
            num, den = rate.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f else None
        return float(rate)
    except ValueError:
        return None
