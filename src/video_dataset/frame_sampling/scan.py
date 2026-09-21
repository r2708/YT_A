"""Streaming low-resolution scan of the whole video.

FFmpeg decodes and downscales, piping raw frames at `analysis_fps` into Python one frame at a time,
so memory stays constant regardless of video length. For every analysed frame we record:
  * visual change vs the previous frame of the same scene (pixel + hue/sat histogram distance, 0..1)
  * mean brightness
  * global optical flow statistics (median dx/dy, mean magnitude, radial divergence) - the raw
    signal behind camera-motion labels and motion-aware sampling.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from video_dataset.config import FrameSamplingConfig
from video_dataset.schemas.scene import ScanSample, Scene, SceneScan
from video_dataset.schemas.video import MediaInfo
from video_dataset.utils.ffmpeg import find_ffmpeg
from video_dataset.utils.logging import get_logger

log = get_logger("frame_sampling.scan")


def _even(x: int) -> int:
    return max(2, x - (x % 2))


def _hist(hsv: np.ndarray) -> np.ndarray:
    h = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(h, h, alpha=1.0, norm_type=cv2.NORM_L1)
    return h


def _radial_unit_vectors(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    rx, ry = xs - cx, ys - cy
    norm = np.sqrt(rx * rx + ry * ry) + 1e-6
    return rx / norm, ry / norm


def compute_change(prev_gray: np.ndarray, gray: np.ndarray, prev_hist: np.ndarray, hist: np.ndarray) -> float:
    pixel = float(cv2.absdiff(gray, prev_gray).mean()) / 255.0
    pixel_term = min(1.0, pixel * 4.0)  # a mean abs diff of 64/255 counts as a full change
    hist_term = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
    return round(max(pixel_term, min(1.0, hist_term)), 4)


def compute_flow_stats(prev_gray: np.ndarray, gray: np.ndarray, rx: np.ndarray, ry: np.ndarray) -> tuple[float, float, float, float]:
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)  # type: ignore[call-overload]
    fx, fy = flow[..., 0], flow[..., 1]
    mag = np.sqrt(fx * fx + fy * fy)
    mean_mag = float(mag.mean())
    dx = float(np.median(fx))
    dy = float(np.median(fy))
    radial = float((fx * rx + fy * ry).mean())
    div = radial / (mean_mag + 1e-6)  # -1..1 : +expanding (zoom in / forward), -contracting
    return round(dx, 4), round(dy, 4), round(mean_mag, 4), round(max(-1.0, min(1.0, div)), 4)


def scan_video(
    video_path: Path,
    info: MediaInfo,
    scenes: Sequence[Scene],
    cfg: FrameSamplingConfig,
    ffmpeg_path: str | None = None,
) -> list[SceneScan]:
    if not scenes:
        return []
    w = int(cfg.analysis_width)
    h = _even(int(round(w * info.height / max(1, info.width))))
    w = _even(w)
    afps = float(cfg.analysis_fps)
    cmd = [
        find_ffmpeg(ffmpeg_path), "-hide_banner", "-loglevel", "error", "-nostdin",
        "-i", str(video_path),
        "-vf", f"fps={afps},scale={w}:{h}:flags=area",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    frame_bytes = w * h * 3
    scans: dict[str, SceneScan] = {s.scene_id: SceneScan(scene_id=s.scene_id, analysis_width=w, analysis_fps=afps) for s in scenes}
    rx, ry = _radial_unit_vectors(h, w)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=frame_bytes * 8)
    assert proc.stdout is not None
    n = 0
    scene_idx = 0
    prev_gray: np.ndarray | None = None
    prev_hist: np.ndarray | None = None
    prev_scene: str | None = None
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            t = n / afps
            n += 1
            while scene_idx < len(scenes) - 1 and t >= scenes[scene_idx].end_time:
                scene_idx += 1
            scene = scenes[scene_idx]
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = _hist(hsv)
            brightness = float(gray.mean())
            change = 0.0
            dx = dy = mag = div = None
            if prev_gray is not None and prev_scene == scene.scene_id and prev_hist is not None:
                change = compute_change(prev_gray, gray, prev_hist, hist)
                if cfg.compute_optical_flow:
                    dx, dy, mag, div = compute_flow_stats(prev_gray, gray, rx, ry)
            scans[scene.scene_id].samples.append(
                ScanSample(
                    timestamp=round(t, 3),
                    frame_index=int(round(t * info.fps)),
                    change=change,
                    brightness=round(brightness, 2),
                    flow_dx=dx,
                    flow_dy=dy,
                    flow_mag=mag,
                    flow_div=div,
                )
            )
            prev_gray, prev_hist, prev_scene = gray, hist, scene.scene_id
    finally:
        try:
            proc.stdout.close()
        except Exception as exc:
            log.debug("closing ffmpeg stdout failed: %s", exc)
        stderr = b""
        if proc.stderr is not None:
            try:
                stderr = proc.stderr.read()
            except Exception as exc:
                log.debug("reading ffmpeg stderr failed: %s", exc)
        proc.wait(timeout=30)
    if n == 0:
        raise RuntimeError(f"scan produced no frames: {stderr.decode(errors='ignore')[:500]}")
    log.debug("scanned %d low-res frames at %.1f fps (%dx%d)", n, afps, w, h)
    return [scans[s.scene_id] for s in scenes]
