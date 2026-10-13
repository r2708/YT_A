"""Duplicate frames: sampled frames whose perceptual hashes (dHash) are (nearly) identical."""

from __future__ import annotations

from video_dataset.schemas.scene import Frame
from video_dataset.utils.imagehash import hamming


def duplicate_frames(frames: list[Frame], max_distance: int = 4, within_scene: bool = True) -> dict[str, str]:
    """{frame_id: earlier frame_id it duplicates}. Frames are compared in timestamp order; a frame
    that matches any earlier *kept* frame (of the same scene when ``within_scene``) is a duplicate."""
    out: dict[str, str] = {}
    kept: list[Frame] = []
    for f in sorted(frames, key=lambda x: (x.timestamp, x.frame_id)):
        if not f.dhash:
            kept.append(f)
            continue
        match = next((k for k in kept if k.dhash and (not within_scene or k.scene_id == f.scene_id) and hamming(k.dhash, f.dhash) <= max_distance), None)
        if match is not None:
            out[f.frame_id] = match.frame_id
        else:
            kept.append(f)
    return out
