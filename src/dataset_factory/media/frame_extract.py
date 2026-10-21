"""Starting-frame extraction for I2V: decode from the configured timestamp, skip black frames
under the ``first_non_black`` policy, save the original pixels losslessly with checksum,
dimensions, the actual timestamp and the clip's SHA-256 so the pair is verifiable later."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from dataset_factory.config import FrameRules, MediaRules
from dataset_factory.media.decode import decode_from, is_black
from video_dataset.utils.imagehash import dhash, hamming


@dataclass
class FrameResult:
    status: str  # passed | failed | not_checked
    path: Path | None = None
    timestamp_seconds: float | None = None
    sha256: str | None = None
    width: int | None = None
    height: int | None = None
    dhash: str | None = None
    reasons: list[str] = field(default_factory=list)
    settings: dict = field(default_factory=dict)  # type: ignore[type-arg]


def extract_start_frame(clip: Path, out_dir: Path, stem: str, rules: FrameRules, media: MediaRules) -> FrameResult:
    settings = {"tool": f"opencv-{cv2.__version__}", "policy": rules.first_usable_policy, "requested_timestamp_seconds": rules.start_timestamp_seconds, "max_search_seconds": rules.max_search_seconds, "format": rules.format, "black_mean_max": media.black_mean_max, "black_std_max": media.black_std_max}
    if not clip.exists():
        return FrameResult("not_checked", reasons=["asset_unavailable"], settings=settings)
    search = rules.max_search_seconds if rules.first_usable_policy == "first_non_black" else 0.0
    chosen = None
    skipped = 0
    for fr in decode_from(clip, rules.start_timestamp_seconds, search):
        if is_black(fr.image, media.black_mean_max, media.black_std_max):
            skipped += 1
            if rules.first_usable_policy == "fixed":
                return FrameResult("failed", reasons=["first_frame_black"], settings=settings)
            continue
        chosen = fr
        break
    if chosen is None:
        return FrameResult("failed", reasons=["no_usable_frame_in_search_window" if skipped else "decode_failed"], settings=settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{stem}_t{chosen.timestamp_seconds:.3f}.{rules.format}"
    params = [cv2.IMWRITE_PNG_COMPRESSION, 3] if rules.format == "png" else []
    if not cv2.imwrite(str(out), chosen.image, params):
        return FrameResult("failed", reasons=["write_failed"], settings=settings)
    h, w = chosen.image.shape[:2]
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    reasons = [f"skipped_black_frames:{skipped}"] if skipped else []
    return FrameResult("passed", out, round(chosen.timestamp_seconds, 3), sha, w, h, dhash(chosen.image), reasons, settings)


def verify_pair(clip: Path, frame_path: Path, timestamp: float, stored_dhash: str | None, max_hamming: int) -> tuple[bool, str]:
    """Re-decode the clip at ``timestamp`` and compare with the stored frame: the image must come
    from this clip at that time."""
    if not clip.exists() or not frame_path.exists():
        return False, "asset_unavailable"
    img = cv2.imread(str(frame_path))
    if img is None:
        return False, "frame_unreadable"
    stored = dhash(img)
    if stored_dhash and hamming(stored, stored_dhash) > max_hamming:
        return False, "frame_file_changed"
    for fr in decode_from(clip, timestamp, 0.0):
        return (hamming(dhash(fr.image), stored) <= max_hamming, "frame_matches_clip" if hamming(dhash(fr.image), stored) <= max_hamming else "frame_differs_from_clip")
    return False, "clip_undecodable_at_timestamp"
