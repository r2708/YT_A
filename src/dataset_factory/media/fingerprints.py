"""Exact (SHA-256) and perceptual (dHash of sampled frames) fingerprints."""

from __future__ import annotations

from pathlib import Path

from dataset_factory.media.decode import DecodedFrame
from video_dataset.utils.imagehash import dhash, hamming, jaccard_hashes
from video_dataset.utils.io import file_sha256


def sha256_of(path: Path) -> str:
    return file_sha256(path)


def frame_hashes(frames: list[DecodedFrame]) -> list[str]:
    return [dhash(f.image) for f in frames]


def fingerprint_similarity(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    return jaccard_hashes(set(a), set(b))


def all_identical(hashes: list[str], max_hamming: int) -> bool:
    return len(hashes) > 1 and all(hamming(hashes[0], h) <= max_hamming for h in hashes[1:])


def duplicate_fraction(hashes: list[str], max_hamming: int) -> float:
    """Share of sampled frames that repeat an earlier sampled frame."""
    if len(hashes) < 2:
        return 0.0
    dup = 0
    for i, h in enumerate(hashes):
        if any(hamming(h, hashes[j]) <= max_hamming for j in range(i)):
            dup += 1
    return dup / (len(hashes) - 1)
