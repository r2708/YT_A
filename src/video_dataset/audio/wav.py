"""Chunked WAV reading (never loads a whole soundtrack into memory at once)."""

from __future__ import annotations

import wave
from collections.abc import Iterator
from pathlib import Path

import numpy as np


def wav_info(path: Path) -> tuple[int, int, float]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        ch = wf.getnchannels()
    return sr, ch, n / float(sr) if sr else 0.0


def iter_windows(path: Path, window_s: float, hop_s: float) -> Iterator[tuple[float, np.ndarray, int]]:
    """Yield (start_time, mono float32 samples in [-1, 1], sample_rate) for overlapping windows."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        width = wf.getsampwidth()
        total = wf.getnframes()
        win = int(window_s * sr)
        hop = max(1, int(hop_s * sr))
        dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(width, np.int16)
        scale = float(2 ** (8 * width - 1))
        pos = 0
        buf = np.zeros((0,), dtype=np.float32)
        buf_start = 0
        while pos < total:
            need_end = pos + win
            while buf_start + len(buf) < min(need_end, total):
                chunk = wf.readframes(min(hop * 8, total - (buf_start + len(buf))))
                if not chunk:
                    break
                arr = np.frombuffer(chunk, dtype=dtype).astype(np.float32) / scale
                if ch > 1:
                    arr = arr.reshape(-1, ch).mean(axis=1)
                buf = np.concatenate([buf, arr])
            start_idx = pos - buf_start
            seg = buf[start_idx : start_idx + win]
            if len(seg) == 0:
                break
            yield pos / sr, seg, sr
            pos += hop
            # drop consumed samples
            drop = pos - buf_start
            if drop > 0:
                buf = buf[drop:]
                buf_start = pos


def rms_db(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    return 20.0 * np.log10(rms + 1e-9)
