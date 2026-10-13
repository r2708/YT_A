"""Perceptual image hashing without extra dependencies (OpenCV + NumPy only).

``dhash`` (difference hash): the image is reduced to a 9x8 grayscale grid and every bit records
whether a pixel is brighter than its right-hand neighbour. Two frames of the same picture (re-encoded,
rescaled, slightly recompressed) differ in a handful of bits; unrelated frames differ in ~32 of 64.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

HASH_BITS = 64


def dhash(image: np.ndarray, size: int = 8) -> str:
    """64-bit difference hash of a BGR or grayscale image as 16 hex characters."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return f"{value:0{size * size // 4}x}"


def dhash_file(path: str | Path) -> str | None:
    img = cv2.imread(str(path))
    return dhash(img) if img is not None else None


def hamming(a: str, b: str) -> int:
    """Number of differing bits between two hex hashes (64 when either is missing)."""
    if not a or not b or len(a) != len(b):
        return HASH_BITS
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def similar(a: str | None, b: str | None, max_distance: int) -> bool:
    return a is not None and b is not None and hamming(a, b) <= max_distance


def jaccard_hashes(a: set[str], b: set[str]) -> float:
    """Set overlap of two hash sets (exact matches; re-encodes of the same video share most hashes)."""
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))
