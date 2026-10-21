"""ffprobe-backed media probing (OpenCV fallback), reusing the producer's probe. Missing tools
produce an actionable message instead of a silent skip."""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from video_dataset.preprocessing.probe import probe_media

MACOS_INSTALL_HINT = "ffprobe/ffmpeg not found: install with `brew install ffmpeg` (macOS) and make sure /opt/homebrew/bin is on PATH; the OpenCV fallback cannot read codec, bitrate or reliable frame counts"


@dataclass
class ProbeResult:
    ok: bool
    duration: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    frame_count: int | None = None
    codec: str | None = None
    size_bytes: int | None = None
    tool: str = "ffprobe"
    error: str | None = None

    def to_dict(self) -> dict:  # type: ignore[type-arg]
        return asdict(self)


def tools_available() -> dict[str, str | None]:
    return {"ffprobe": shutil.which("ffprobe"), "ffmpeg": shutil.which("ffmpeg")}


def probe(path: Path) -> ProbeResult:
    if not path.exists():
        return ProbeResult(False, error="asset_missing")
    if path.stat().st_size == 0:
        return ProbeResult(False, error="empty_file", size_bytes=0)
    try:
        info = probe_media(path)
    except Exception as exc:
        return ProbeResult(False, error=f"unreadable:{type(exc).__name__}", size_bytes=path.stat().st_size)
    tool = "ffprobe" if tools_available()["ffprobe"] else "opencv"
    return ProbeResult(True, float(info.duration), int(info.width), int(info.height), float(info.fps), int(info.frame_count), info.video_codec, int(info.size_bytes or path.stat().st_size), tool)
