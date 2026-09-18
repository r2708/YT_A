from pathlib import Path

import pytest

from video_dataset.config import load_config
from video_dataset.frame_sampling.extractor import FrameExtractor
from video_dataset.frame_sampling.sampler import select_frames
from video_dataset.frame_sampling.scan import scan_video
from video_dataset.preprocessing.probe import probe_media
from video_dataset.schemas.scene import SamplingReason, ScanSample, Scene, SceneScan, TransitionType
from video_dataset.schemas.vision import CameraMovement
from video_dataset.vision.motion import dominant_motion, motion_segments


def _scene(start: float, end: float, fps: float = 24.0, idx: int = 0) -> Scene:
    return Scene(scene_id=f"scene_{idx + 1:03d}", video_id="v", index=idx, start_time=start, end_time=end, start_frame=int(start * fps), end_frame=int(end * fps), transition_in=TransitionType.START)


def _scan(scene: Scene, changes: list[float], fps: float = 4.0) -> SceneScan:
    samples = [ScanSample(timestamp=scene.start_time + i / fps, frame_index=int((scene.start_time + i / fps) * 24), change=c, brightness=100.0, flow_mag=0.1) for i, c in enumerate(changes)]
    return SceneScan(scene_id=scene.scene_id, analysis_width=160, analysis_fps=fps, samples=samples)


def test_select_frames_boundaries_and_change_peaks():
    cfg = load_config(None, {"frame_sampling.min_frames_per_scene": "3", "frame_sampling.max_frames_per_scene": "6", "frame_sampling.change_threshold": "0.3"}).frame_sampling
    scene = _scene(10.0, 20.0)
    changes = [0.0] * 40
    changes[20] = 0.9  # big change at 15.0s
    cands = select_frames(scene, _scan(scene, changes), cfg, 24.0)
    reasons = {c.reason for c in cands}
    assert {SamplingReason.SCENE_START, SamplingReason.SCENE_MIDDLE, SamplingReason.SCENE_END} <= reasons
    assert all(10.0 <= c.timestamp < 20.0 for c in cands)
    assert cands == sorted(cands, key=lambda c: c.timestamp)
    assert len(cands) <= 6
    # the visual change frame at 15.0 sits at the midpoint; ensure a change-driven frame or the midpoint covers it
    assert any(abs(c.timestamp - 15.0) < 0.3 for c in cands)


def test_select_frames_respects_min_and_max():
    cfg = load_config(None, {"frame_sampling.min_frames_per_scene": "5", "frame_sampling.max_frames_per_scene": "5", "frame_sampling.min_frame_gap_seconds": "0.2"}).frame_sampling
    scene = _scene(0.0, 3.0)
    cands = select_frames(scene, None, cfg, 24.0)
    assert len(cands) == 5
    scene2 = _scene(0.0, 0.3)
    cands2 = select_frames(scene2, None, cfg, 24.0)
    assert 1 <= len(cands2) <= 5


@pytest.mark.slow
def test_scan_and_extract_on_synthetic_video(synthetic_video: Path, tmp_path: Path):
    cfg = load_config(None, {"frame_sampling.max_frames_per_scene": "6"}).frame_sampling
    info = probe_media(synthetic_video)
    scenes = [_scene(0.0, 4.0, 24.0, 0), _scene(4.0, 8.0, 24.0, 1), _scene(8.0, 11.5, 24.0, 2), _scene(11.5, info.duration, 24.0, 3)]
    scans = scan_video(synthetic_video, info, scenes, cfg)
    assert len(scans) == 4 and all(len(s.samples) >= 8 for s in scans)
    # scene 2 scrolls content to the right -> camera pans left; scene 1 is static camera with a moving object
    label2, cons2, mag2 = dominant_motion(scans[1], 0.35)
    assert label2 == CameraMovement.PAN_LEFT, (label2, cons2, mag2)
    assert cons2 >= 0.6
    label3, _, _ = dominant_motion(scans[2], 0.35)
    assert label3 == CameraMovement.STATIC
    segs = motion_segments(scans[1], 0.35)
    assert segs and segs[0].label == CameraMovement.PAN_LEFT and segs[0].end_time - segs[0].start_time > 2.0
    # brightness of the dark scene is far lower than the yellow one
    assert sum(s.brightness for s in scans[2].samples) / len(scans[2].samples) < sum(s.brightness for s in scans[3].samples) / len(scans[3].samples)
    plan = [(sc, select_frames(sc, scan, cfg, info.fps)) for sc, scan in zip(scenes, scans)]
    frames = FrameExtractor(synthetic_video, info.fps, 640, 90).extract("vid_test", plan, tmp_path / "frames")
    assert len(frames) >= 12
    for f in frames:
        assert Path(f.frame_path).exists() and f.width == 640 and f.height == 360
        sc = next(s for s in scenes if s.scene_id == f.scene_id)
        assert sc.start_time - 0.05 <= f.timestamp <= sc.end_time + 0.05
