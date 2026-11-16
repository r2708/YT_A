"""preprocess.compress: optional lossy re-encode of every downloaded video."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from video_dataset.config import PreprocessConfig
from video_dataset.preprocessing.normalize import COMPRESSION_MARKER, compress_source, ensure_canonical_video
from video_dataset.preprocessing.probe import probe_media
from video_dataset.utils.ffmpeg import run_ffmpeg


@pytest.fixture()
def big_source(synthetic_video: Path, tmp_path: Path) -> Path:
    """A deliberately high-bitrate download (near-lossless MKV) so there is something to save."""
    src = tmp_path / "vid" / "source.mkv"
    src.parent.mkdir()
    run_ffmpeg(["-i", str(synthetic_video), "-c:v", "libx264", "-preset", "ultrafast", "-crf", "1", "-c:a", "copy", str(src)])
    return src


def _decoded_frames(path: Path) -> int:
    """Frames actually decoded (an MKV container only gives an estimate)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n


def _cfg(**kw) -> PreprocessConfig:
    return PreprocessConfig(max_transcode_height=720, **kw)


def test_off_by_default_leaves_the_download_alone(big_source: Path):
    assert PreprocessConfig().compress is False
    size = big_source.stat().st_size
    assert compress_source(big_source, _cfg()) == (big_source, False)
    assert big_source.stat().st_size == size
    assert not (big_source.parent / COMPRESSION_MARKER).exists()


def test_on_replaces_the_download_with_a_smaller_file(big_source: Path):
    before, frames = probe_media(big_source), _decoded_frames(big_source)
    size = big_source.stat().st_size
    out, compressed = compress_source(big_source, _cfg(compress=True))
    assert compressed and out.name == "source.mp4" and not big_source.exists()
    assert out.stat().st_size < size
    after = probe_media(out)
    assert (after.width, after.height) == (before.width, before.height)
    assert abs(after.duration - before.duration) < 0.2 and _decoded_frames(out) == frames
    marker = json.loads((out.parent / COMPRESSION_MARKER).read_text())
    assert marker["compressed"] is True and marker["original_bytes"] == size and marker["crf"] == 26

    # a second call must not re-encode the already compressed file
    mtime = out.stat().st_mtime_ns
    assert compress_source(out, _cfg(compress=True)) == (out, True)
    assert out.stat().st_mtime_ns == mtime


def test_original_is_kept_when_the_result_is_not_smaller(synthetic_video: Path, tmp_path: Path):
    src = tmp_path / "source.mp4"
    shutil.copy2(synthetic_video, src)
    size = src.stat().st_size
    out, compressed = compress_source(src, _cfg(compress=True, compress_crf=0))  # lossless -> bigger
    assert (out, compressed) == (src, False) and src.stat().st_size == size
    assert not (tmp_path / "compressing.tmp.mp4").exists()


def test_canonical_video_is_the_compressed_file(big_source: Path):
    size = big_source.stat().st_size
    dest = big_source.parent / "video.mp4"
    out, changed = ensure_canonical_video(big_source, dest, probe_media(big_source), _cfg(compress=True))
    assert out == dest and changed and dest.stat().st_size < size
    assert sorted(p.name for p in dest.parent.glob("source.*")) == ["source.mp4"]
    assert dest.samefile(dest.parent / "source.mp4")  # hard link: one copy on disk


def test_crf_out_of_range_is_rejected():
    with pytest.raises(ValueError):
        PreprocessConfig(compress_crf=60)
