"""Model-free image statistics. Everything here is measured, never inferred."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from video_dataset.schemas.scene import SceneScan
from video_dataset.schemas.vision import Measurements


def _shrink(img: np.ndarray, max_side: int = 320) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m > max_side:
        s = max_side / float(m)
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return img


def _load_small(path: Path, max_side: int = 320) -> np.ndarray | None:
    img = cv2.imread(str(path))
    return _shrink(img, max_side) if img is not None else None


def blockiness(gray_full: np.ndarray, block: int = 8, max_side: int = 512) -> float:
    """Compression-artifact score on a full-resolution grayscale image: how much stronger the
    pixel-to-pixel differences are on 8x8 block boundaries than inside the blocks (0 = no visible
    macroblock grid; values above ~0.3 show as blocking). Measured on a centre crop so the grid
    alignment of the encoder is preserved (resizing would destroy it)."""
    h, w = gray_full.shape[:2]
    if h < 3 * block or w < 3 * block:
        return 0.0
    y0, x0 = max(0, (h - max_side) // 2), max(0, (w - max_side) // 2)
    g = gray_full[y0 : y0 + max_side, x0 : x0 + max_side].astype(np.float32)
    dx = np.abs(np.diff(g, axis=1))
    dy = np.abs(np.diff(g, axis=0))
    cols = np.arange(dx.shape[1])
    rows = np.arange(dy.shape[0])
    bx = (cols + 1) % block == 0
    by = (rows + 1) % block == 0
    on_boundary = float(np.concatenate([dx[:, bx].ravel(), dy[by, :].ravel()]).mean()) if bx.any() or by.any() else 0.0
    inside = float(np.concatenate([dx[:, ~bx].ravel(), dy[~by, :].ravel()]).mean()) + 1e-6
    if on_boundary < 2.0:
        return 0.0  # steps on the grid are below visibility (flat areas have ~0 inside too; the ratio would explode)
    return round(max(0.0, on_boundary / inside - 1.0), 4)


def exposure_stats(gray: np.ndarray) -> tuple[float, float]:
    """(share of near-black pixels, share of near-white pixels)."""
    n = float(gray.size) or 1.0
    return float((gray < 16).sum()) / n, float((gray > 240).sum()) / n


def exposure_label(dark_fraction: float, bright_fraction: float, max_dark: float = 0.6, max_bright: float = 0.4) -> str:
    if dark_fraction > max_dark:
        return "underexposed"
    if bright_fraction > max_bright:
        return "overexposed"
    return "normal"


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


def measure_image(img: np.ndarray, full: np.ndarray | None = None) -> dict:
    """Statistics of a (downscaled) BGR image. ``full`` is the full-resolution original when available:
    exposure and blockiness are measured on it (resizing hides clipped pixels and the macroblock grid)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 100, 200)
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    bmean = float(gray.mean())
    bstd = float(gray.std())
    gray_full = cv2.cvtColor(full, cv2.COLOR_BGR2GRAY) if full is not None and full.ndim == 3 else (full if full is not None else gray)
    dark, bright = exposure_stats(gray_full)
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
        "dark_fraction": dark,
        "bright_fraction": bright,
        "blockiness": blockiness(gray_full) if full is not None else 0.0,
    }


def _measure_file(path: Path) -> dict | None:
    full = cv2.imread(str(path))
    if full is None:
        return None
    return measure_image(_shrink(full), full)


def measure_frame_file(path: Path) -> Measurements | None:
    m = _measure_file(path)
    if m is None:
        return None
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
        dark_fraction=round(m["dark_fraction"], 4),
        bright_fraction=round(m["bright_fraction"], 4),
        exposure=exposure_label(m["dark_fraction"], m["bright_fraction"]),
        blockiness=round(m["blockiness"], 4),
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
        m = _measure_file(p)
        if m is not None:
            per.append(m)
    if not per:
        return Measurements(brightness_trend=brightness_trend(scan))
    avg = {k: float(np.mean([m[k] for m in per])) for k in ("brightness_mean", "brightness_std", "contrast", "saturation_mean", "colorfulness", "edge_density", "sharpness", "dark_fraction", "bright_fraction", "blockiness")}
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
        dark_fraction=round(avg["dark_fraction"], 4),
        bright_fraction=round(avg["bright_fraction"], 4),
        exposure=exposure_label(avg["dark_fraction"], avg["bright_fraction"]),
        blockiness=round(avg["blockiness"], 4),
    )
