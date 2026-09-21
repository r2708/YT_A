"""Clips must cover exactly their scene: stream copy is verified and re-encoded when it misses the boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from video_dataset.config import load_config
from video_dataset.frame_sampling.clips import extract_clips, media_duration
from video_dataset.preprocessing.probe import probe_media
from video_dataset.schemas.scene import Scene, TransitionType


def _scenes(bounds: list[tuple[float, float]], fps: float = 24.0) -> list[Scene]:
    return [
        Scene(scene_id=f"scene_{i + 1:03d}", video_id="v", index=i, start_time=s, end_time=e, start_frame=int(s * fps), end_frame=int(e * fps), transition_in=TransitionType.START if i == 0 else TransitionType.CUT)
        for i, (s, e) in enumerate(bounds)
    ]


# boundaries deliberately off the synthetic video's keyframes (GOP 12 frames = 0.5 s): stream copy cannot hit them
BOUNDS = [(0.0, 1.3), (1.3, 2.9), (2.9, 4.0), (5.2, 7.7), (9.1, 10.6)]


@pytest.mark.slow
def test_auto_reencodes_clips_that_miss_the_boundary(synthetic_video: Path, tmp_path: Path):
    cfg = load_config(None, {"frame_sampling.clip_codec": "auto", "frame_sampling.clip_tolerance_seconds": "0.1"}).frame_sampling
    info = probe_media(synthetic_video)
    clips, stats = extract_clips(synthetic_video, "v", _scenes(BOUNDS), [], tmp_path / "auto", cfg, info.has_audio)
    assert stats.total == len(BOUNDS) and stats.failed == 0 and stats.inexact == 0
    for c in clips:
        assert c.exact and c.media_duration is not None
        assert abs(c.media_duration - (c.end_time - c.start_time)) <= 0.1, (c.clip_id, c.media_duration, c.end_time - c.start_time)
        assert abs(media_duration(Path(c.clip_path)) - c.media_duration) < 1e-3
    assert stats.reencoded >= 1 and {c.codec for c in clips} <= {"copy", "libx264"}
    # second call reuses the files (no re-cut) and still reports them exact
    clips2, stats2 = extract_clips(synthetic_video, "v", _scenes(BOUNDS), [], tmp_path / "auto", cfg, info.has_audio)
    assert stats2.reused == len(BOUNDS) and stats2.reencoded == 0 and all(c.exact for c in clips2)


@pytest.mark.slow
def test_copy_mode_flags_inexact_clips_instead_of_hiding_them(synthetic_video: Path, tmp_path: Path):
    cfg = load_config(None, {"frame_sampling.clip_codec": "copy", "frame_sampling.clip_tolerance_seconds": "0.05"}).frame_sampling
    info = probe_media(synthetic_video)
    clips, stats = extract_clips(synthetic_video, "v", _scenes(BOUNDS), [], tmp_path / "copy", cfg, info.has_audio)
    assert stats.reencoded == 0 and stats.stream_copied == len(BOUNDS)
    assert all(c.codec == "copy" and c.media_duration is not None for c in clips)
    flagged = [c for c in clips if not c.exact]
    assert stats.inexact == len(flagged)
    for c in flagged:
        assert abs(c.media_duration - (c.end_time - c.start_time)) > 0.05  # the record tells the truth about the file


@pytest.mark.slow
def test_libx264_is_always_exact(synthetic_video: Path, tmp_path: Path):
    cfg = load_config(None, {"frame_sampling.clip_codec": "libx264"}).frame_sampling
    info = probe_media(synthetic_video)
    clips, stats = extract_clips(synthetic_video, "v", _scenes(BOUNDS[:3]), [], tmp_path / "x264", cfg, info.has_audio)
    assert stats.reencoded == 3 and stats.stream_copied == 0 and all(c.exact and c.codec == "libx264" for c in clips)


def test_unknown_codec_rejected(synthetic_video: Path, tmp_path: Path):
    cfg = load_config(None, {"frame_sampling.clip_codec": "prores"}).frame_sampling
    with pytest.raises(ValueError, match="clip_codec"):
        extract_clips(synthetic_video, "v", _scenes(BOUNDS[:1]), [], tmp_path, cfg, True)
