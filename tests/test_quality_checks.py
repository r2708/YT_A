"""Dataset quality work: perceptual hashes + duplicate frames / videos, exposure + blockiness,
quality flags, consistency checks, frame-sequence / clip checks, tracking, sequences, prompt
variants, duplicate-aware curation, licensing, dissolves, media normalisation, media verification,
diversity statistics and the production reset."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from video_dataset.config import load_config
from video_dataset.dataset.curation import annotate_records, video_info_from_manifests
from video_dataset.dataset.sequences import build_sequences
from video_dataset.dataset.variations import fill_prompt_variants, prompt_variants
from video_dataset.deduplication.frames import duplicate_frames
from video_dataset.deduplication.videos import fingerprint_similarity, near_duplicate_videos
from video_dataset.frame_sampling.clips import video_filter
from video_dataset.frame_sampling.stage import remove_duplicate_frames
from video_dataset.scene_detection.detector import PySceneDetector
from video_dataset.schemas.common import ConfidenceSource
from video_dataset.schemas.dataset import SceneRecord, SequenceRecord
from video_dataset.schemas.scene import Clip, Frame, ScanSample, SceneScan
from video_dataset.schemas.video import license_status
from video_dataset.schemas.vision import CameraAnnotation, CameraMovement, Measurements, SceneAnalysis, SubjectMotion
from video_dataset.utils.imagehash import dhash, hamming, jaccard_hashes
from video_dataset.validation.checks import check_clips, check_frame_sequence
from video_dataset.validation.consistency import check_scene_consistency, text_movements
from video_dataset.validation.quality import scene_quality_flags
from video_dataset.vision.measurements import blockiness, exposure_stats, measure_frame_file
from video_dataset.vision.subject_motion import subject_motion_from_track
from video_dataset.vision.tracker import IoUTracker, Track, primary_track, track_scene, trajectory

ROOT = Path(__file__).resolve().parents[1]


def _cfg(tmp_path: Path, **over):
    base = {"project.data_dir": str(tmp_path / "data"), "project.log_level": "ERROR"}
    base.update(over)
    return load_config(None, base, config_dir=ROOT / "config")


def _img(seed: int, w: int = 160, h: int = 90) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


# ------------------------------------------------------------------------------------ hashing / duplicates
def test_dhash_is_stable_under_rescale_and_distinguishes_images():
    a = _img(1, 320, 180)
    blurred = cv2.GaussianBlur(a, (3, 3), 0)
    assert dhash(a) == dhash(cv2.resize(a, (640, 360))) or hamming(dhash(a), dhash(cv2.resize(a, (640, 360)))) <= 6
    assert hamming(dhash(a), dhash(blurred)) <= 10
    assert hamming(dhash(a), dhash(_img(2, 320, 180))) >= 16
    assert hamming("", "ff") == 64 and jaccard_hashes({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)


def _frame(i: int, scene: str, t: float, h: str) -> Frame:
    return Frame(frame_id=f"f{i}", video_id="v", scene_id=scene, timestamp=t, frame_index=int(t * 24), frame_path=f"/x/f{i}.jpg", width=10, height=10, dhash=h)


def test_duplicate_frames_within_scene_and_removal_keeps_minimum(tmp_path: Path):
    h1, h2 = "0" * 16, "f" * 16
    frames = [_frame(0, "s1", 0.0, h1), _frame(1, "s1", 1.0, h1), _frame(2, "s1", 2.0, h2), _frame(3, "s2", 5.0, h1)]
    dups = duplicate_frames(frames, 4)
    assert dups == {"f1": "f0"}  # same hash in another scene is not a duplicate (within_scene)
    for f in frames:
        p = tmp_path / Path(f.frame_path).name
        p.write_bytes(b"x")
        f.frame_path = str(p)
    kept, dropped = remove_duplicate_frames(frames, 4)
    assert dropped == 1 and [f.frame_id for f in kept] == ["f0", "f2", "f3"] and not (tmp_path / "f1.jpg").exists()
    single = [_frame(0, "s1", 0.0, h1), _frame(1, "s1", 1.0, h1)]
    kept, dropped = remove_duplicate_frames(single, 4, min_per_scene=2)
    assert dropped == 0 and kept[1].duplicate_of == "f0"


def test_near_duplicate_videos_from_fingerprints():
    base = [f"{i:016x}" for i in range(100)]
    fps = {"vid_b": base, "vid_a": base[:90] + [f"{i:016x}" for i in range(500, 510)], "vid_c": [f"{i:016x}" for i in range(1000, 1100)], "vid_d": base[:3]}
    assert fingerprint_similarity(fps["vid_a"], fps["vid_b"]) == pytest.approx(90 / 110)
    assert near_duplicate_videos(fps, 0.8) == {"vid_b": "vid_a"}  # smallest id is canonical; vid_d too short to match
    assert near_duplicate_videos(fps, 0.95) == {}


# ------------------------------------------------------------------------------------ measurements / flags
def test_exposure_and_blockiness_measurements(tmp_path: Path):
    dark = np.zeros((64, 64), dtype=np.uint8)
    d, b = exposure_stats(dark)
    assert d == 1.0 and b == 0.0
    smooth = np.full((256, 256), 128, dtype=np.uint8)
    assert blockiness(smooth) == 0.0
    blocky = np.kron(np.random.default_rng(0).integers(0, 255, size=(32, 32)), np.ones((8, 8))).astype(np.uint8)
    assert blockiness(blocky) > 1.0  # all differences sit on the 8x8 grid
    p = tmp_path / "f.jpg"
    cv2.imwrite(str(p), _img(3, 320, 180), [cv2.IMWRITE_JPEG_QUALITY, 90])
    m = measure_frame_file(p)
    assert m and m.exposure == "normal" and m.dark_fraction is not None and m.blockiness is not None and m.sharpness > 0


def test_scene_quality_flags_from_measurements_and_video(tmp_path: Path):
    cfg = _cfg(tmp_path).quality
    m = Measurements(sharpness=5.0, dark_fraction=0.9, bright_fraction=0.0, blockiness=0.5)
    assert scene_quality_flags(m, ["low_resolution", "duration_mismatch"], cfg) == ["blurry", "underexposed", "compression_artifacts", "low_resolution"]
    assert scene_quality_flags(Measurements(sharpness=200.0, dark_fraction=0.1, bright_fraction=0.1, blockiness=0.05), [], cfg) == []
    assert scene_quality_flags(None, ["low_bitrate"], cfg) == ["low_bitrate"]


# ------------------------------------------------------------------------------------ consistency
def _analysis(**over) -> SceneAnalysis:
    base = dict(scene_id="s", video_id="v", start_time=0.0, end_time=4.0, summary="A shot; the camera pans left.", camera=CameraAnnotation(movement="pan_left"), measurements=Measurements(brightness_mean=120.0, lighting_level="normal", camera_motion_label="pan_left", camera_motion_consistency=0.9))
    base.update(over)
    return SceneAnalysis(**base)


def test_consistency_detects_contradictions_and_accepts_consistent_scenes():
    hard, soft = check_scene_consistency(_analysis())
    assert hard == [] and soft == []
    # caption says pan left, field says pan right
    hard, _ = check_scene_consistency(_analysis(camera=CameraAnnotation(movement="pan_right"), measurements=Measurements(brightness_mean=120.0, lighting_level="normal", camera_motion_label="pan_right", camera_motion_consistency=0.9)))
    assert any("caption says" in h for h in hard)
    # static camera but tracked subject, and a lighting label that the brightness contradicts
    sm = SubjectMotion(subject="person", label="tracked_by_camera", camera_relation="tracked")
    hard, soft = check_scene_consistency(_analysis(summary="A shot.", camera=CameraAnnotation(movement="static"), subject_motion=sm, measurements=Measurements(brightness_mean=200.0, lighting_level="dark", camera_motion_label="static", camera_motion_consistency=0.9)))
    assert any("static but the subject" in h for h in hard) and any("lighting labelled 'dark'" in h for h in hard)
    # flow disagrees with the label
    hard, _ = check_scene_consistency(_analysis(summary="A shot.", measurements=Measurements(brightness_mean=120.0, lighting_level="normal", camera_motion_label="pan_right", camera_motion_consistency=0.9)))
    assert any("optical flow measures pan_right" in h for h in hard)
    assert text_movements("The camera pans to the left and then zooms in") == [CameraMovement.PAN_LEFT, CameraMovement.ZOOM_IN]


# ------------------------------------------------------------------------------------ frame sequence / clips
def test_frame_sequence_and_clip_checks(tmp_path: Path):
    good = [_frame(i, "s", float(i), "0" * 16) for i in range(4)]
    assert check_frame_sequence(good, 0.0, 4.0) == (True, None)
    bad = [_frame(0, "s", 0.0, None), _frame(1, "s", 9.0, None)]
    ok, msg = check_frame_sequence(bad, 0.0, 4.0)
    assert not ok and "outside the shot" in msg
    unordered = [Frame(frame_id="a", video_id="v", scene_id="s", timestamp=1.0, frame_index=50, frame_path="/x", width=1, height=1), Frame(frame_id="b", video_id="v", scene_id="s", timestamp=2.0, frame_index=40, frame_path="/x", width=1, height=1)]
    ok, msg = check_frame_sequence(unordered, 0.0, 4.0)
    assert not ok and "not increasing" in msg
    gap = [_frame(0, "s", 0.0, None), _frame(1, "s", 30.0, None), _frame(2, "s", 30.5, None), _frame(3, "s", 31.0, None)]
    ok, msg = check_frame_sequence(gap, 0.0, 31.0)
    assert not ok and "gap of 30.0s" in msg
    present = tmp_path / "c1.mp4"
    present.write_bytes(b"m")
    clips = [
        Clip(clip_id="c1", video_id="v", scene_id="s", clip_path=str(present), start_time=0.0, end_time=4.0, media_duration=4.1, exact=True),
        Clip(clip_id="c2", video_id="v", scene_id="s", clip_path=str(tmp_path / "missing.mp4"), start_time=0.0, end_time=4.0),
        Clip(clip_id="c3", video_id="v", scene_id="s", clip_path=str(present), start_time=0.0, end_time=4.0, media_duration=None),
        Clip(clip_id="c4", video_id="v", scene_id="s", clip_path=str(present), start_time=0.0, end_time=4.0, media_duration=6.0),
    ]
    missing, corrupt, inexact = check_clips(clips, 0.25)
    assert missing == ["c2"] and corrupt == ["c3"] and inexact and inexact[0].startswith("c4")


# ------------------------------------------------------------------------------------ tracking
def _det(name, x, y, w=20, h=40, conf=0.9):
    return {"name": name, "confidence": conf, "bbox": [x, y, x + w, y + h], "frame_size": [160, 90]}


def test_iou_tracker_links_detections_and_subject_motion_from_track():
    tr = IoUTracker(iou_threshold=0.3, max_age=1, min_hits=3)
    for i in range(6):
        dets = [_det("person", 10 + 4 * i, 25), _det("car", 120, 60, 30, 20)]
        if i == 3:
            dets = [_det("car", 120, 60, 30, 20)]  # one missed person detection; the track survives max_age=1
        tr.update(dets, 0.5 * i, (160, 90))
    tracks = tr.tracks()
    assert {t.name for t in tracks} == {"person", "car"} and all(t.hits >= 3 for t in tracks)
    person = next(t for t in tracks if t.name == "person")
    assert person.hits == 5 and len({t.track_id for t in tracks}) == 2
    prim = primary_track(tracks)
    assert prim is not None and prim.name in ("person", "car")
    sm = subject_motion_from_track(person, None, CameraMovement.STATIC, len(tracks))
    assert sm and sm.method == "tracker" and sm.label == "moves_right" and sm.direction == "right" and sm.track_id == person.track_id and sm.n_tracks == 2
    assert len(sm.trajectory) == 5 and sm.trajectory[0][0] == 0.0
    assert len(trajectory(Track(1, "x", [(float(i), [0, 0, 10, 10], 0.9) for i in range(50)], frame_size=(100, 100)))) == 20


def test_track_scene_runs_a_fake_detector_over_a_synthetic_video(synthetic_video: Path):
    class FakeDetector:
        calls = 0

        def detect_arrays(self, images):
            self.calls += 1
            out = []
            for img in images:
                h, w = img.shape[:2]
                out.append([{"name": "circle", "confidence": 0.9, "bbox": [w * 0.2, h * 0.4, w * 0.3, h * 0.6], "frame_size": [w, h]}])
            return out

    det = FakeDetector()
    tracks, n = track_scene(synthetic_video, 0.0, 4.0, det, fps=2.0, width=320, max_seconds=20.0, iou_threshold=0.3, max_age=2, min_hits=3)
    assert n >= 6 and det.calls >= 1 and tracks and tracks[0].name == "circle" and tracks[0].hits == n


# ------------------------------------------------------------------------------------ sequences / variants
def _scene_rec(i: int, start: float, end: float, transition: str = "cut", **over) -> dict:
    base = {
        "record_id": f"scene_v_{i:03d}", "video_id": "vid_v", "scene_id": f"scene_{i:03d}", "start_time": start, "end_time": end, "duration": end - start,
        "transition_in": transition, "summary": f"Shot {i} summary text here.", "environment": {"setting": "outdoor", "time_of_day": "day"},
        "objects": ["car", "road"], "object_details": [{"name": "car", "count": 1, "area_fraction": 0.2}], "actions": [],
        "camera": {"shot_type": "wide", "movement": "pan_left", "speed": "slow"}, "visual_style": {}, "measurements": {"lighting_level": "bright", "color_temperature": "warm", "saturation_mean": 40.0},
        "validation": {"status": "accepted", "issues": [], "checks": {}}, "quality": {"overall": 0.9, "components_used": ["grounding"]}, "confidence": 0.8, "confidence_source": "measurement",
    }
    base.update(over)
    return SceneRecord.model_validate(base).model_dump(mode="json")


def test_sequences_join_split_parts_and_compatible_cuts_only(tmp_path: Path):
    cfg = _cfg(tmp_path).export.sequences
    scenes = [
        _scene_rec(1, 0.0, 60.0, "start"), _scene_rec(2, 60.0, 90.0, "split"),  # one long take split for analysis -> rejoined
        _scene_rec(3, 90.0, 95.0, "cut"), _scene_rec(4, 95.0, 100.0, "cut", objects=["tree"]),  # no shared subject -> break
        _scene_rec(5, 100.0, 103.0, "cut", environment={"setting": "indoor"}),  # setting changes -> break
        _scene_rec(6, 103.0, 104.0, "fade"), _scene_rec(7, 104.0, 105.0, "cut"),
    ]
    records = {"scenes": scenes, "clips": [{"record_id": "c1", "video_id": "vid_v", "scene_id": "scene_001", "clip_id": "clip_001", "frame_ids": []}, {"record_id": "c2", "video_id": "vid_v", "scene_id": "scene_002", "clip_id": "clip_002", "frame_ids": []}]}
    seqs = build_sequences(records, cfg)
    assert len(seqs) == 1
    s = seqs[0]
    assert s["scene_ids"] == ["scene_001", "scene_002", "scene_003"] and s["duration"] == 95.0 and "rejoined_split" in s["coherence"] and "shared_subject" in s["coherence"]
    assert s["clip_ids"] == ["clip_001", "clip_002"] and s["subjects"] == ["car", "road"] and s["setting"] == "outdoor"
    assert s["generation_prompt"] is None  # scene prompts are filled by annotate_records; see below
    SequenceRecord.model_validate(s)
    cfg.max_duration = 70.0
    assert [q["scene_ids"] for q in build_sequences(records, cfg)] == [["scene_002", "scene_003"]]  # 0-90s would exceed 70s, so the run restarts at shot 2
    cfg.max_duration = 120.0
    cfg.min_duration = 200.0
    assert build_sequences(records, cfg) == []


def test_annotate_records_builds_sequences_prompts_variants_and_duplicate_aware_splits(tmp_path: Path):
    cfg = _cfg(tmp_path, **{"export.prompt_variants": "3", "export.enforce_policy": "false"})  # hand-made records carry no check states
    scenes = [_scene_rec(1, 0.0, 10.0, "start"), _scene_rec(2, 10.0, 20.0, "cut")]
    dup = [_scene_rec(1, 0.0, 10.0, "start", video_id="vid_w", record_id="scene_w_001")]
    records = {"scenes": scenes + dup, "clips": [], "frames": [], "events": [], "temporal_qa": [], "long_video_qa": [], "video_descriptions": []}
    info = {"vid_v": {"license_status": "creative_commons", "quality_flags": [], "canonical_video_id": None}, "vid_w": {"license_status": "standard", "quality_flags": ["low_resolution"], "canonical_video_id": "vid_v"}}
    counts = annotate_records(records, cfg.export, info, cfg.licensing, seed=1)
    assert counts["sequences"] == 1 and counts["duplicate_videos"] == 1 and counts["licensed_videos"] == 1
    v = [r for r in records["scenes"] if r["video_id"] == "vid_v"]
    w = [r for r in records["scenes"] if r["video_id"] == "vid_w"][0]
    assert v[0]["generation_prompt"] and len(v[0]["prompt_variants"]) == 3 and v[0]["generation_prompt"] not in v[0]["prompt_variants"]
    assert "licensed" in v[0]["subsets"] and "licensed" not in w["subsets"]
    assert w["canonical_video_id"] == "vid_v" and w["split"] == v[0]["split"] and w["tier"] == "bronze" and "near_duplicate_video" in w["tier_reasons"]
    assert v[0]["tier"] == "gold" and v[0]["canonical_video_id"] is None
    seq = records["sequences"][0]
    assert seq["generation_prompt"].startswith("A continuous 20-second sequence of 2 shots featuring car, road outdoors in daylight. Shot 1:") and len(seq["prompt_variants"]) == 3
    assert seq["split"] == v[0]["split"] and seq["tier"] == "gold"
    # quality flags cap the tier at silver; excluded duplicates disappear
    flagged = _scene_rec(3, 20.0, 30.0, "cut", quality_flags=["blurry"])
    records2 = {"scenes": [flagged, *dup], "clips": [], "frames": [], "events": [], "temporal_qa": [], "long_video_qa": [], "video_descriptions": []}
    cfg.export.exclude_duplicate_videos = True
    counts2 = annotate_records(records2, cfg.export, info, cfg.licensing)
    assert counts2["dropped_duplicate_records"] == 1 and [r["video_id"] for r in records2["scenes"]] == ["vid_v"]
    assert records2["scenes"][0]["tier"] == "silver" and "quality_flags" in records2["scenes"][0]["tier_reasons"]


def test_prompt_variants_keep_the_facts_and_are_deterministic():
    rec = _scene_rec(1, 0.0, 6.5, "cut", subject_motion={"subject": "car", "label": "moves_right", "camera_relation": "independent", "direction": "right"})
    from video_dataset.dataset.prompting import generation_prompt

    rec["generation_prompt"] = generation_prompt(rec)
    vs = prompt_variants(rec, 4, seed=7)
    assert len(vs) == 4 and len(set(vs)) == 4 and rec["generation_prompt"] not in vs
    for v in vs:
        low = v.lower()
        assert "car" in low and ("pan" in low) and "left" in low and ("bright" in low) and "warm" in low
        assert "the car" in low and "right" in low  # subject motion survives in every variant
        assert "mm" not in low and "cinematic" not in low
    assert prompt_variants(rec, 4, seed=7) == vs
    assert prompt_variants({"generation_prompt": None}, 3) == [] and prompt_variants(rec, 0) == []
    records = {"scenes": [rec], "clips": [{"record_id": "c", "video_id": "vid_v", "scene_id": "scene_001", "clip_id": "clip_001"}], "sequences": []}
    assert fill_prompt_variants(records, 2, 1) == 2 and records["clips"][0]["prompt_variants"] == records["scenes"][0]["prompt_variants"]


# ------------------------------------------------------------------------------------ licensing / video info
def test_license_status_normalisation_and_manifest_video_info():
    assert license_status("Creative Commons Attribution license (reuse allowed)", "youtube") == "creative_commons"
    assert license_status(None, "youtube") == "standard" and license_status("Standard YouTube License", "youtube") == "standard"
    assert license_status(None, "local") == "unknown" and license_status("Some other terms", "youtube") == "other"
    base = [f"{i:016x}" for i in range(100)]
    manifests = [
        {"video_id": "vid_a", "license_status": "creative_commons", "fingerprint": base, "quality_flags": []},
        {"video_id": "vid_b", "license_status": "standard", "fingerprint": base[:95], "quality_flags": ["low_resolution"]},
        {"video_id": "vid_c", "license_status": "standard", "fingerprint": [], "near_duplicate_of": "vid_b"},
    ]
    info = video_info_from_manifests(manifests, 0.8)
    assert info["vid_b"]["canonical_video_id"] == "vid_a" and info["vid_a"]["canonical_video_id"] is None
    assert info["vid_c"]["canonical_video_id"] == "vid_a"  # chain b -> a resolved
    assert info["vid_b"]["quality_flags"] == ["low_resolution"] and info["vid_a"]["license_status"] == "creative_commons"


def test_youtube_downloader_rejects_unlicensed_videos_only_when_asked(tmp_path: Path):
    from video_dataset.downloader.youtube import YouTubeDownloader

    cfg = _cfg(tmp_path, **{"licensing.action": "reject"})
    dl = YouTubeDownloader(cfg.download, None, cfg.limits, cfg.licensing)
    assert dl.check_limits({"duration": 10, "license": None}) and "licensing.allowed" in dl.check_limits({"duration": 10})
    assert dl.check_limits({"duration": 10, "license": "Creative Commons Attribution license (reuse allowed)"}) is None
    flag = YouTubeDownloader(cfg.download, None, cfg.limits, _cfg(tmp_path).licensing)
    assert flag.check_limits({"duration": 10, "license": None}) is None
    meta = YouTubeDownloader.metadata_from_info({"id": "x", "title": "t", "license": "Creative Commons Attribution license (reuse allowed)"}, "vid_x", "https://youtu.be/x", None)
    assert meta.license_status == "creative_commons"


# ------------------------------------------------------------------------------------ dissolves / media prep
def test_gradual_boundary_is_a_dissolve():
    class Stats:
        def __init__(self, vals):
            self.vals = vals

        def metrics_exist(self, frame, keys):
            return frame in self.vals

        def get_metrics(self, frame, keys):
            return [self.vals[frame]]

    cut = Stats({100: 60.0, 99: 1.0, 101: 1.0, 98: 0.5, 102: 0.5})
    assert not PySceneDetector._is_gradual(cut, 100, 24.0, 27.0, 3)
    dissolve = Stats({f: 12.0 for f in range(95, 106)})
    assert PySceneDetector._is_gradual(dissolve, 100, 24.0, 27.0, 3)


def test_video_filter_for_normalised_clips(tmp_path: Path):
    cfg = _cfg(tmp_path).frame_sampling
    assert video_filter(cfg, 720) == "scale=-2:'min(ih,720)'"
    cfg.normalize_clips, cfg.target_fps, cfg.target_width, cfg.target_height = True, 24, 854, 480
    assert video_filter(cfg, 720) == "fps=24,scale=854:480:force_original_aspect_ratio=decrease:force_divisible_by=2"
    cfg.aspect_mode = "pad"
    assert video_filter(cfg, 720).endswith("pad=854:480:(ow-iw)/2:(oh-ih)/2")
    cfg.aspect_mode = "crop"
    assert video_filter(cfg, 720).endswith("crop=854:480")
    cfg.target_width = None
    assert video_filter(cfg, 720) == "fps=24,scale=-2:480"


# ------------------------------------------------------------------------------------ verify-media / diversity / reset
def test_verify_media_reports_missing_and_unreadable_files(tmp_path: Path, synthetic_video: Path):
    from video_dataset.dataset.verify_media import verify_media
    from video_dataset.utils.io import write_jsonl

    cfg = _cfg(tmp_path)
    final = cfg.export_dir
    final.mkdir(parents=True)
    clips_dir = cfg.data_dir / "clips" / "vid_a"
    clips_dir.mkdir(parents=True)
    good = clips_dir / "clip_001.mp4"
    good.write_bytes(synthetic_video.read_bytes())
    (clips_dir / "clip_002.mp4").write_bytes(b"not a video")
    frames_dir = cfg.data_dir / "frames" / "vid_a" / "scene_001"
    frames_dir.mkdir(parents=True)
    cv2.imwrite(str(frames_dir / "f0.jpg"), _img(1))
    (frames_dir / "f1.jpg").write_bytes(b"junk")
    write_jsonl(final / "clips.jsonl", [
        {"record_id": "c1", "video_id": "vid_a", "clip_path": "clips/vid_a/clip_001.mp4", "start_time": 0.0, "end_time": 14.0},
        {"record_id": "c2", "video_id": "vid_a", "clip_path": "clips/vid_a/clip_002.mp4", "start_time": 0.0, "end_time": 2.0},
        {"record_id": "c3", "video_id": "vid_b", "clip_path": "clips/vid_b/clip_001.mp4", "start_time": 0.0, "end_time": 2.0},
        {"record_id": "c4", "video_id": "vid_a", "clip_path": "clips/vid_a/clip_001.mp4", "start_time": 0.0, "end_time": 5.0},
    ])
    write_jsonl(final / "frames.jsonl", [
        {"record_id": "f0", "video_id": "vid_a", "frame_path": "frames/vid_a/scene_001/f0.jpg"},
        {"record_id": "f1", "video_id": "vid_a", "frame_path": "frames/vid_a/scene_001/f1.jpg"},
        {"record_id": "f2", "video_id": "vid_b", "frame_path": "frames/vid_b/scene_001/f0.jpg"},
    ])
    rep = verify_media(cfg)
    assert rep["clips"] == {"total": 4, "missing": 1, "unreadable": 1, "length_mismatch": 1, "ok": 1}
    assert rep["frames"]["missing"] == 1 and rep["frames"]["unreadable"] == 1 and rep["frames"]["ok"] == 1
    assert not rep["ok"] and "vid_b" in rep["per_video"] and (final / "media_report.json").exists()


def test_diversity_statistics_list_underrepresented_categories():
    from video_dataset.dataset.stats import compute_diversity, format_diversity

    scenes = [_scene_rec(i, i * 2.0, i * 2.0 + 2.0, "cut", camera={"shot_type": "wide", "movement": "pan_left"}) for i in range(9)]
    scenes.append(_scene_rec(99, 100.0, 140.0, "cut", camera={"shot_type": "close_up", "movement": "static"}, quality_flags=["blurry"]))
    div = compute_diversity({"scenes": scenes}, [{"video_id": "vid_v", "license_status": "standard", "media": {"height": 720}}])
    assert div["dimensions"]["shot_size"] == {"wide": 9, "close_up": 1} and div["dimensions"]["scene_duration"] == {"2-5s": 9, ">=30s": 1}
    assert [u["value"] for u in div["underrepresented"]["shot_size"]] == ["close_up"]
    assert div["subjects"]["car"] == 10 and div["quality_flags"] == {"blurry": 1} and div["videos_by"]["license_status"] == {"standard": 1}
    assert "under-represented" in format_diversity(div)


def test_clean_reset_plans_everything_but_input_and_final(tmp_path: Path):
    from video_dataset.storage import cleanup as cl
    from video_dataset.storage.paths import DataPaths

    cfg = _cfg(tmp_path)
    paths = DataPaths(cfg.data_dir)
    paths.ensure_all()
    (paths.root / "frames" / "x.jpg").write_bytes(b"1")
    (paths.root / "input" / "urls.txt").write_text("u")
    (paths.root / "final" / "dataset.jsonl").write_text("{}")
    cfg.db_path.write_bytes(b"db")
    plan = cl.plan_reset(paths, cfg.db_path)
    names = {p.name for p in plan.paths}
    assert "frames" in names and "state.db" in names and "input" not in names and "final" not in names
    assert "final" in {p.name for p in cl.plan_reset(paths, cfg.db_path, include_final=True).paths}
    from typer.testing import CliRunner

    from video_dataset import cli

    cli.state.config = None
    r = CliRunner().invoke(cli.app, ["--data-dir", str(cfg.data_dir), "clean", "--reset", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "dry run" in r.output and (paths.root / "frames" / "x.jpg").exists()
    r = CliRunner().invoke(cli.app, ["--data-dir", str(cfg.data_dir), "clean", "--reset", "--yes"])
    assert r.exit_code == 0, r.output
    assert not (paths.root / "frames" / "x.jpg").exists() and (paths.root / "input" / "urls.txt").exists() and (paths.root / "final" / "dataset.jsonl").exists()
    assert not cfg.db_path.exists() and (paths.root / "frames").is_dir()  # layout recreated empty


def test_schema_round_trips_new_fields():
    cam = CameraAnnotation(movement="tracking_left")
    sm = SubjectMotion(subject="person", label="moves_left", camera_relation="independent", direction="left", method="tracker", track_id=3, n_tracks=4, trajectory=[[0.0, 0.5, 0.5, 0.1]])
    a = SceneAnalysis(scene_id="s", video_id="v", start_time=0, end_time=1, camera=cam, subject_motion=sm, quality_flags=["blurry"], measurements=Measurements(exposure="normal", blockiness=0.1))
    back = SceneAnalysis.model_validate(json.loads(a.model_dump_json()))
    assert back.quality_flags == ["blurry"] and back.subject_motion.trajectory == [[0.0, 0.5, 0.5, 0.1]] and back.measurements.exposure == "normal"
    f = Frame(frame_id="f", video_id="v", scene_id="s", timestamp=0.0, frame_index=0, frame_path="/x", width=1, height=1, dhash="0" * 16, duplicate_of="g")
    assert Frame.model_validate(f.model_dump()).duplicate_of == "g"
    sc = SceneScan(scene_id="s", analysis_width=160, analysis_fps=4.0, samples=[ScanSample(timestamp=0.0, frame_index=0, change=0.0, brightness=1.0, dhash="f" * 16)])
    assert sc.samples[0].dhash == "f" * 16
    assert ConfidenceSource("derived_min") == ConfidenceSource.DERIVED_MIN
    assert SimpleNamespace(x=1).x == 1
