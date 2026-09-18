"""Model-free image statistics. Everything here is measured, never inferred."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from video_dataset.schemas.scene import SceneScan
from video_dataset.schemas.vision import Measurements


def _load_small(path: Path, max_side: int = 320) -> np.ndarray | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    m = max(h, w)
    if m > max_side:
        s = max_side / float(m)
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return img


def dominant_colors(img: np.ndarray, k: int = 3) -> list[str]:
    small = cv2.resize(img, (48, 27), interpolation=cv2.INTER_AREA).reshape(-1, 3).astype(np.float32)
    k = min(k, len(small))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _ret, labels, centers = cv2.kmeans(small, k, None, criteria, 2, cv2.KMEANS_PP_CENTERS)  # type: ignore[call-overload]
    counts = np.bincount(labels.flatten(), minlength=k)
    order = np.argsort(-counts)
    out = []
    for i in order:
        b, g, r = (int(round(float(c))) for c in centers[i])
        out.append(f"#{r:02x}{g:02x}{b:02x}")
    return out


def colorfulness(img: np.ndarray) -> float:
    b, g, r = (img[..., i].astype(np.float32) for i in range(3))
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def lighting_level(brightness: float) -> str:
    if brightness < 50:
        return "dark"
    if brightness < 95:
        return "dim"
    if brightness < 170:
        return "normal"
    return "bright"


def color_temperature(img: np.ndarray) -> str:
    b = float(img[..., 0].mean())
    r = float(img[..., 2].mean())
    if r - b > 18:
        return "warm"
    if b - r > 18:
        return "cool"
    return "neutral"


def measure_image(img: np.ndarray) -> dict:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 100, 200)
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    bmean = float(gray.mean())
    bstd = float(gray.std())
    return {
        "brightness_mean": bmean,
        "brightness_std": bstd,
        "contrast": min(1.0, bstd / 128.0),
        "saturation_mean": float(hsv[..., 1].mean()),
        "colorfulness": colorfulness(img),
        "dominant_colors": dominant_colors(img),
        "color_temperature": color_temperature(img),
        "edge_density": float((edges > 0).mean()),
        "sharpness": float(lap),
    }


def measure_frame_file(path: Path) -> Measurements | None:
    img = _load_small(path)
    if img is None:
        return None
    m = measure_image(img)
    return Measurements(
        brightness_mean=round(m["brightness_mean"], 2),
        brightness_std=round(m["brightness_std"], 2),
        contrast=round(m["contrast"], 3),
        saturation_mean=round(m["saturation_mean"], 2),
        colorfulness=round(m["colorfulness"], 2),
        dominant_colors=m["dominant_colors"],
        color_temperature=m["color_temperature"],
        edge_density=round(m["edge_density"], 4),
        sharpness=round(m["sharpness"], 2),
        lighting_level=lighting_level(m["brightness_mean"]),
    )


def brightness_trend(scan: SceneScan | None, min_delta: float = 20.0) -> str | None:
    if scan is None or len(scan.samples) < 3:
        return None
    n = max(1, len(scan.samples) // 4)
    first = float(np.mean([s.brightness for s in scan.samples[:n]]))
    last = float(np.mean([s.brightness for s in scan.samples[-n:]]))
    if last - first > min_delta:
        return "brightening"
    if first - last > min_delta:
        return "darkening"
    return "stable"


def measure_scene(frame_paths: list[Path], scan: SceneScan | None, max_frames: int = 5) -> Measurements:
    paths = [p for p in frame_paths if p.exists()]
    if len(paths) > max_frames:
        step = (len(paths) - 1) / (max_frames - 1)
        paths = [paths[int(round(i * step))] for i in range(max_frames)]
    per: list[dict] = []
    for p in paths:
        img = _load_small(p)
        if img is not None:
            per.append(measure_image(img))
    if not per:
        return Measurements(brightness_trend=brightness_trend(scan))
    avg = {k: float(np.mean([m[k] for m in per])) for k in ("brightness_mean", "brightness_std", "contrast", "saturation_mean", "colorfulness", "edge_density", "sharpness")}
    temps = [m["color_temperature"] for m in per]
    temp = max(set(temps), key=temps.count)
    # dominant colors: take the top colour of each measured frame, most common first
    colors: list[str] = []
    for m in per:
        for c in m["dominant_colors"][:2]:
            if c not in colors:
                colors.append(c)
    motion = None
    if scan is not None:
        mags = [s.flow_mag for s in scan.samples if s.flow_mag is not None]
        motion = round(float(np.mean(mags)), 4) if mags else None
    return Measurements(
        brightness_mean=round(avg["brightness_mean"], 2),
        brightness_std=round(avg["brightness_std"], 2),
        contrast=round(avg["contrast"], 3),
        saturation_mean=round(avg["saturation_mean"], 2),
        colorfulness=round(avg["colorfulness"], 2),
        dominant_colors=colors[:5],
        color_temperature=temp,
        edge_density=round(avg["edge_density"], 4),
        sharpness=round(avg["sharpness"], 2),
        motion_magnitude=motion,
        brightness_trend=brightness_trend(scan),
        lighting_level=lighting_level(avg["brightness_mean"]),
    )
