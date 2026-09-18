#!/usr/bin/env python
"""Generate a small synthetic test video with known scene structure.

Layout (24 fps, 640x360, ~14 s):
  scene 1  0.0- 4.0 s  blue background, red circle moving left->right (object motion, static camera)
  scene 2  4.0- 8.0 s  green textured background scrolling right (simulates a camera pan LEFT)
  scene 3  8.0-11.5 s  dark background with white text "HELLO WORLD" (OCR), fading to black at the end
  scene 4 11.5-14.0 s  bright yellow background with a growing orange square (zoom-in-like expansion)
Audio: 440 Hz tone 0-5 s, silence 5-9 s, 880 Hz tone 9-14 s.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import cv2
import numpy as np

FPS = 24
W, H = 640, 360
SCENES = [(0.0, 4.0), (4.0, 8.0), (8.0, 11.5), (11.5, 14.0)]
DURATION = SCENES[-1][1]


def _frame(t: float) -> np.ndarray:
    img = np.zeros((H, W, 3), dtype=np.uint8)
    if t < 4.0:
        img[:] = (160, 90, 40)  # BGR blue-ish
        x = int(60 + (W - 120) * (t / 4.0))
        cv2.circle(img, (x, H // 2), 40, (40, 40, 220), -1)
        cv2.rectangle(img, (0, H - 40), (W, H), (90, 60, 30), -1)
    elif t < 8.0:
        # scrolling checker/grid texture: content shifts +x over time (camera panning left)
        shift = int((t - 4.0) * 60)
        ys, xs = np.mgrid[0:H, 0:W]
        pattern = (((xs - shift) // 40 + ys // 40) % 2).astype(np.uint8)  # texture slides right -> camera pans left
        img[..., 0] = 60 + 40 * pattern
        img[..., 1] = 140 + 80 * pattern
        img[..., 2] = 60 + 30 * pattern
        for k in range(6):
            cx = (k * 160 + shift) % (W + 160) - 80
            cv2.circle(img, (cx, 80 + 40 * (k % 3)), 18, (30, 30, 30), -1)
    elif t < 11.5:
        img[:] = (25, 25, 25)
        cv2.putText(img, "HELLO WORLD", (110, 190), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(img, "NEW YORK", (200, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (200, 200, 255), 3, cv2.LINE_AA)
        if t > 10.5:  # fade to black over the last second
            alpha = max(0.0, 1.0 - (t - 10.5) / 1.0)
            img = (img.astype(np.float32) * alpha).astype(np.uint8)
    else:
        img[:] = (60, 220, 240)  # bright yellow
        s = int(30 + 250 * ((t - 11.5) / 2.5))
        cx, cy = W // 2, H // 2
        cv2.rectangle(img, (cx - s // 2, cy - s // 2), (cx + s // 2, cy + s // 2), (20, 120, 240), -1)
    return img


def _audio(path: Path, sr: int = 16000) -> None:
    n = int(DURATION * sr)
    t = np.arange(n) / sr
    sig = np.zeros(n, dtype=np.float32)
    sig[(t < 5.0)] = 0.4 * np.sin(2 * math.pi * 440 * t[(t < 5.0)])
    late = t >= 9.0
    sig[late] = 0.4 * np.sin(2 * math.pi * 880 * t[late])
    pcm = (sig * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def make_test_video(out: Path, with_audio: bool = True, ffmpeg: str | None = None) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = ffmpeg or shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg  # type: ignore

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("ffmpeg is required to build the test video") from exc
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "raw.avi"
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
        n_frames = int(DURATION * FPS)
        for i in range(n_frames):
            writer.write(_frame(i / FPS))
        writer.release()
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(raw)]
        if with_audio:
            wav = Path(td) / "audio.wav"
            _audio(wav)
            cmd += ["-i", str(wav), "-c:a", "aac", "-b:a", "96k", "-shortest"]
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-g", "12", "-movflags", "+faststart", str(out)]
        subprocess.run(cmd, check=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output", nargs="?", default="data/input/synthetic_test.mp4")
    ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args()
    p = make_test_video(Path(args.output), with_audio=not args.no_audio)
    print(p)
