"""Media work for V3: authorized acquisition, probe, decode validation, exact clip extraction and
the measurements the quality gates use. FFmpeg runs in software; no hardware decode is required."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from video_dataset.config import LicensingConfig
from video_dataset.preprocessing.probe import probe_media
from video_dataset.schemas.video import DownloadStatus
from video_dataset.storage.paths import DataPaths
from video_dataset.utils.ffmpeg import run_ffmpeg
from video_dataset.utils.imagehash import dhash, hamming
from video_dataset.utils.io import file_sha256
from video_dataset.utils.logging import get_logger
from video_dataset.utils.urls import InputItem, extract_youtube_id
from video_dataset.v3.resolution import ResolutionDecision, scale_filter

log = get_logger("v3.media")


class AcquisitionError(RuntimeError):
    def __init__(self, message: str, retryable: bool):
        super().__init__(message)
        self.retryable = retryable


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def acquire(config: Any, entry: dict[str, Any], row: dict[str, Any], job_dir: Path) -> tuple[Path, str]:
    """Get the source video of an *approved* entry. Order: a file supplied by the rightsholder
    (``local_path``), the copy the main pipeline still has, then a download. Returns (path, method)."""
    vid = row["source_video_id"]
    src_dir = job_dir / "source"
    existing = sorted(p for p in src_dir.glob("source.*") if p.is_file() and p.stat().st_size > 0 and not p.name.endswith((".part", ".ytdl"))) if src_dir.exists() else []
    if existing:
        return existing[0], "existing"
    local = entry.get("local_path")
    if local:
        p = Path(local)
        if not p.exists():
            raise AcquisitionError(f"local_path not found: {local}", retryable=False)
        dst = src_dir / f"source{p.suffix.lower() or '.mp4'}"
        _link(p, dst)
        return dst, "rightsholder_file"
    if config.v3.acquisition.reuse_local_source:
        kept = DataPaths(config.data_dir).video_file(vid)
        if kept.exists() and kept.stat().st_size > 0:
            dst = src_dir / "source.mp4"
            _link(kept, dst)
            return dst, "pipeline_copy"
    url = entry.get("source_url") or row.get("source_url")
    if not url:
        raise AcquisitionError("no source_url and no local_path", retryable=False)
    if entry.get("download_authorized") is not True:
        raise AcquisitionError("download_authorized is not true", retryable=False)
    from video_dataset.downloader.youtube import YouTubeDownloader

    yid = extract_youtube_id(url)
    item = InputItem(raw=url, kind="youtube" if yid else "url", youtube_id=yid)
    # the rights manifest is the gate here; the platform-licence filter of the main pipeline is not applied twice
    downloader = YouTubeDownloader(config.download, config.project.ffmpeg_path, config.limits, LicensingConfig(action="flag"))
    result = downloader.fetch(item, vid, src_dir)
    if result.video_path is None or result.status in (DownloadStatus.FAILED, DownloadStatus.UNAVAILABLE, DownloadStatus.REJECTED):
        raise AcquisitionError(f"{result.status}: {result.error}", retryable=result.status == DownloadStatus.FAILED)
    path = Path(result.video_path)
    if config.preprocess.compress:  # optional lossy re-encode of the download (preprocess.compress)
        from video_dataset.preprocessing.normalize import compress_source

        path, _ = compress_source(path, config.preprocess, config.project.ffmpeg_path)
    return path, "download"


def probe(path: Path, ffprobe_path: str | None = None) -> dict[str, Any]:
    info = probe_media(path, ffprobe_path)
    data = info.model_dump(mode="json")
    data["sha256"] = file_sha256(path)
    return data


def decode_check(path: Path, mode: str = "full", ffmpeg_path: str | None = None) -> tuple[bool, str | None]:
    """Does the whole file decode? ``full`` decodes every frame with FFmpeg (errors are fatal),
    ``sampled`` reads 24 frames spread over the file, ``none`` skips."""
    mode = (mode or "full").lower()
    if mode == "none":
        return True, None
    if mode == "sampled":
        cap = cv2.VideoCapture(str(path))
        try:
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if not cap.isOpened() or n <= 0:
                return False, "cannot open the file"
            for idx in sorted({int(i * (n - 1) / 23) for i in range(24)}):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, img = cap.read()
                if not ok or img is None:
                    return False, f"frame {idx} of {n} does not decode"
        finally:
            cap.release()
        return True, None
    try:
        run_ffmpeg(["-xerror", "-i", str(path), "-map", "0:v:0", "-an", "-f", "null", "-"], ffmpeg_path=ffmpeg_path)
    except RuntimeError as exc:
        return False, str(exc)[:500]
    return True, None


def extract_clip(src: Path, out: Path, start: float, end: float, decision: ResolutionDecision, src_w: int, src_h: int, cfg: Any, has_audio: bool, ffmpeg_path: str | None = None) -> None:
    """Frame-exact re-encode of [start, end) at the decided size (the source fps is kept)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    args = ["-ss", f"{start:.3f}", "-i", str(src), "-t", f"{max(0.05, end - start):.3f}", "-map", "0:v:0"]
    args += ["-c:v", "libx264", "-preset", str(cfg.preset), "-crf", str(int(cfg.crf)), "-pix_fmt", "yuv420p"]
    vf = scale_filter(decision, src_w, src_h)
    if vf:
        args += ["-vf", vf]
    if cfg.keep_audio and has_audio:
        args += ["-map", "0:a:0?", "-c:a", "aac", "-b:a", "128k"]
    else:
        args += ["-an"]
    tmp = out.with_name(out.stem + ".tmp.mp4")
    run_ffmpeg([*args, "-movflags", "+faststart", str(tmp)], ffmpeg_path=ffmpeg_path)
    os.replace(tmp, out)


@dataclass
class ClipMetrics:
    ok: bool = False
    frame_count: int = 0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    duration: float = 0.0
    black_fraction: float = 0.0
    sharpness: float = 0.0
    motion: float = 0.0
    hashes: list[str] = field(default_factory=list)
    error: str | None = None


def read_frames(path: Path, fractions: list[float], max_width: int = 320) -> tuple[list[np.ndarray], int, float]:
    """Frames at the given positions (0..1) downscaled to ``max_width``; ([], 0, 0) when unreadable."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return [], 0, 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames: list[np.ndarray] = []
        if n <= 0:
            return [], 0, fps
        for f in fractions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(n - 1, max(0, int(round(f * (n - 1))))))
            ok, img = cap.read()
            if not ok or img is None:
                return [], n, fps
            h, w = img.shape[:2]
            if w > max_width:
                img = cv2.resize(img, (max_width, max(2, int(h * max_width / w))), interpolation=cv2.INTER_AREA)
            frames.append(img)
        return frames, n, fps
    finally:
        cap.release()


def measure_clip(path: Path, ffprobe_path: str | None = None, samples: int = 8) -> ClipMetrics:
    """Probe + sampled decode of a clip: exact frame count / duration from the container, black
    fraction, sharpness (median variance of Laplacian), motion (mean change between sampled frames)
    and three perceptual hashes for duplicate detection."""
    m = ClipMetrics()
    try:
        info = probe_media(path, ffprobe_path)
    except Exception as exc:
        m.error = f"probe failed: {exc}"
        return m
    m.frame_count, m.fps, m.width, m.height, m.duration = int(info.frame_count or 0), float(info.fps or 0.0), int(info.width), int(info.height), float(info.duration or 0.0)
    frames, _n, _fps = read_frames(path, [i / (samples - 1) for i in range(samples)])
    if not frames:
        m.error = "frames do not decode"
        return m
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    m.black_fraction = sum(1 for g in grays if float(g.mean()) <= 12.0 and float(g.std()) <= 6.0) / len(grays)
    m.sharpness = float(np.median([cv2.Laplacian(g, cv2.CV_64F).var() for g in grays]))
    diffs = [float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16)))) / 255.0 for a, b in pairwise(grays) if a.shape == b.shape]
    m.motion = float(np.mean(diffs)) if diffs else 0.0
    m.hashes = [dhash(frames[i]) for i in sorted({len(frames) // 4, len(frames) // 2, (3 * len(frames)) // 4})]
    m.ok = True
    return m


def count_cuts(path: Path, threshold: float = 27.0, min_scene_seconds: float = 0.6) -> int | None:
    """Shot changes inside a clip, found by PySceneDetect's content detector (0 = one continuous
    shot). None when the clip could not be analysed."""
    try:
        from scenedetect import ContentDetector, SceneManager, open_video

        video = open_video(str(path))
        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=float(threshold), min_scene_len=max(1, int(round(min_scene_seconds * float(video.frame_rate))))))
        manager.detect_scenes(video, show_progress=False)
        return max(0, len(manager.get_scene_list()) - 1)
    except Exception as exc:
        log.warning("scene check failed for %s: %s", path, exc)
        return None


_nlp_cache: dict[str, Any] = {}


def prompt_problem(prompt: str, model: str) -> tuple[str, str | None]:
    """spaCy check of a training prompt: it must name something (a noun) and contain at least three
    content words. Returns (passed | failed | not_run, reason)."""
    if model not in _nlp_cache:
        try:
            import spacy

            _nlp_cache[model] = spacy.load(model, disable=["ner", "lemmatizer"])
        except Exception as exc:
            log.warning("spaCy pipeline '%s' not available (%s): prompt check not run", model, exc)
            _nlp_cache[model] = None
    nlp = _nlp_cache[model]
    if nlp is None:
        return "not_run", None
    doc = nlp(prompt)
    words = [t for t in doc if t.is_alpha and not t.is_stop]
    if not any(t.pos_ in ("NOUN", "PROPN") for t in doc):
        return "failed", "prompt names no subject or object (no noun)"
    if len(words) < 3:
        return "failed", f"prompt has only {len(words)} content word(s)"
    return "passed", None


def near_duplicate(a: list[str], b: list[str], max_hamming: int) -> bool:
    return bool(a) and len(a) == len(b) and all(hamming(x, y) <= max_hamming for x, y in zip(a, b))
