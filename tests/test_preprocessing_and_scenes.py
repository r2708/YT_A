from pathlib import Path

import pytest

from video_dataset.config import load_config
from video_dataset.preprocessing.normalize import extract_audio
from video_dataset.preprocessing.probe import probe_media
from video_dataset.scene_detection.detector import PySceneDetector, split_long_scenes
from video_dataset.schemas.scene import Scene, TransitionType


def test_probe_media(synthetic_video: Path):
    info = probe_media(synthetic_video)
    assert info.width == 640 and info.height == 360
    assert abs(info.fps - 24.0) < 0.01
    assert 13.5 <= info.duration <= 14.5
    assert info.frame_count >= 330
    assert info.video_codec == "h264" and info.has_audio and info.audio is not None
    assert info.audio.sample_rate in (16000, 44100, 48000)


def test_extract_audio(synthetic_video: Path, tmp_path: Path):
    wav = tmp_path / "a.wav"
    assert extract_audio(synthetic_video, wav, 16000, 1)
    assert wav.exists() and wav.stat().st_size > 16000 * 2 * 10


@pytest.mark.slow
def test_scene_detection_finds_known_boundaries(synthetic_video: Path, expected_scenes):
    cfg = load_config(None, {"scene_detection.max_scene_duration": "60"})
    info = probe_media(synthetic_video)
    result = PySceneDetector(cfg.scene_detection).detect(synthetic_video, info, "vid_test")
    starts = [s.start_time for s in result.scenes]
    assert result.scenes[0].start_time == 0.0
    assert abs(result.scenes[-1].end_time - info.duration) < 0.2
    for expected_start in (4.0, 8.0):
        assert any(abs(st - expected_start) < 0.35 for st in starts), f"missing cut near {expected_start}: {starts}"
    assert any(abs(st - 11.5) < 0.6 for st in starts), f"missing fade/cut near 11.5: {starts}"
    assert all(s.end_time > s.start_time for s in result.scenes)
    assert result.scenes[0].transition_in == TransitionType.START
    assert all(s.scene_id == f"scene_{i + 1:03d}" for i, s in enumerate(result.scenes))
    # every non-first boundary should be a real transition, not an artificial split
    assert all(s.transition_in in (TransitionType.CUT, TransitionType.FADE) for s in result.scenes[1:])


def test_split_long_scenes_marks_artificial_boundaries():
    sc = Scene(scene_id="scene_001", video_id="v", index=0, start_time=0.0, end_time=150.0, start_frame=0, end_frame=3600, transition_in=TransitionType.START)
    out = split_long_scenes([sc], 60.0, 24.0, "v")
    assert len(out) == 3
    assert out[0].transition_in == TransitionType.START and out[1].transition_in == TransitionType.SPLIT
    assert all(o.split_from == "scene_001" for o in out)
    assert out[-1].end_time == 150.0 and [o.scene_id for o in out] == ["scene_001", "scene_002", "scene_003"]
