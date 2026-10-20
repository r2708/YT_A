"""Reproducibility manifest: code, dependencies, tools, models and configuration hashes.

Written into every per-video manifest and into ``final/manifest.json`` so a release can be traced
to the exact code commit, package versions, FFmpeg build, model identifiers and effective
configuration that produced it. Never contains secrets: configuration enters only as per-section
hashes, API keys / cookies / tokens are not read at all.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from video_dataset import __version__
from video_dataset.schemas.common import SCHEMA_VERSION

_PACKAGES = (
    "yt-dlp", "yt-dlp-ejs", "scenedetect", "opencv-python", "opencv-python-headless", "numpy", "torch", "transformers",
    "ultralytics", "faster-whisper", "rapidocr-onnxruntime", "onnxruntime", "spacy", "datasketch", "pyarrow", "pandas",
    "pydantic", "huggingface-hub",
)
_CONFIG_SECTIONS = (
    "download", "preprocess", "scene_detection", "frame_sampling", "transcription", "audio_events", "ocr", "vision",
    "temporal", "qa", "validation", "quality", "deduplication", "export", "licensing",
)
_SECRET_KEYS = {"api_key_env", "cookies_file", "cookies_from_browser", "token_env"}


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return (out.stdout or out.stderr).strip().splitlines()[0] if out.returncode == 0 and (out.stdout or out.stderr).strip() else None
    except Exception:
        return None


@lru_cache(maxsize=1)
def code_version() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    commit = _run(["git", "-C", str(root), "rev-parse", "HEAD"])
    dirty = _run(["git", "-C", str(root), "status", "--porcelain"])
    return {"package_version": __version__, "git_commit": commit, "git_dirty": bool(dirty) if commit else None}


@lru_cache(maxsize=1)
def package_versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {}
    for name in _PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue
    return out


@lru_cache(maxsize=4)
def tool_versions(ffmpeg_path: str | None = None, ffprobe_path: str | None = None) -> dict[str, str | None]:
    from video_dataset.utils.ffmpeg import find_ffmpeg, find_ffprobe

    out: dict[str, str | None] = {}
    try:
        out["ffmpeg"] = _run([find_ffmpeg(ffmpeg_path), "-version"])
    except Exception:
        out["ffmpeg"] = None
    try:
        probe = find_ffprobe(ffprobe_path)
        out["ffprobe"] = _run([probe, "-version"]) if probe else None
    except Exception:
        out["ffprobe"] = None
    return out


def _file_sha(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    try:
        from video_dataset.utils.io import file_sha256

        return file_sha256(p)[:16]
    except Exception:
        return None


def _section_hash(config: Any, section: str) -> str | None:
    sec = getattr(config, section, None)
    if sec is None:
        return None
    data = sec.model_dump(mode="json") if hasattr(sec, "model_dump") else dict(sec)
    for k in list(data):
        if k in _SECRET_KEYS:
            data[k] = "<redacted>"
    return hashlib.sha1(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[:12]


def model_identifiers(config: Any) -> dict[str, Any]:
    v, t, o, q = config.vision, config.transcription, config.ocr, config.qa
    return {
        "vision": {"provider": v.provider, "model": v.model, "yolo_model": v.yolo_model, "yolo_sha256_16": _file_sha(v.yolo_model), "clip_model": v.clip_model if v.provider == "cv_models" or "clip" in (v.enrichers or []) else None, "aesthetic": bool(v.aesthetic)},
        "transcription": {"provider": t.provider, "model": t.model, "language": t.language},
        "audio_events": {"provider": config.audio_events.provider, "model": config.audio_events.model if config.audio_events.provider == "transformers" else None},
        "ocr": {"provider": o.provider, "languages": list(o.languages)},
        "qa": {"generator": q.generator, "spacy_model": q.spacy_model if q.generator in ("spacy", "both") else None},
        "verifier": config.validation.verifier,
        "llm": {"provider": config.llm.provider},
    }


def environment_manifest(config: Any) -> dict[str, Any]:
    """The full reproducibility block (cached parts are computed once per process)."""
    from video_dataset.validation.policy import POLICY_VERSION

    return {
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "code": code_version(),
        "python": sys.version.split()[0],
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "tools": tool_versions(config.project.ffmpeg_path, config.project.ffprobe_path),
        "packages": package_versions(),
        "models": model_identifiers(config),
        "config_hashes": {s: _section_hash(config, s) for s in _CONFIG_SECTIONS},
    }
