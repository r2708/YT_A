"""Standardized camera taxonomy, granular camera fields, generation prompts and record links."""

from __future__ import annotations

from pathlib import Path

from video_dataset.dataset.negatives import OPPOSITE_MOVEMENT
from video_dataset.dataset.prompting import (
    camera_phrase,
    fill_frame_clip_ids,
    fill_generation_prompts,
    generation_prompt,
)
from video_dataset.schemas.common import ConfidenceSource
from video_dataset.schemas.dataset import FrameCaptionRecord, SceneRecord
from video_dataset.schemas.vision import (
    MOVEMENT_TAXONOMY,
    CameraAnnotation,
    CameraMovement,
    SceneAnalysis,
    SubjectMotion,
    movement_from_taxonomy,
)
from video_dataset.vision.motion import MOVEMENT_PHRASES
from video_dataset.vision.parsing import coerce_enum, parse_camera
from video_dataset.vision.prompts import _ENUM_MOVEMENT, SCENE_SCHEMA
from video_dataset.vision.subject_motion import apply_subject_motion, tracking_direction


def test_taxonomy_covers_every_movement_and_round_trips():
    for m in CameraMovement:
        assert m in MOVEMENT_TAXONOMY and m in MOVEMENT_PHRASES, m
        family, direction = MOVEMENT_TAXONOMY[m]
        assert movement_from_taxonomy(family, direction) == m
    assert movement_from_taxonomy("pan", "left") == CameraMovement.PAN_LEFT
    assert movement_from_taxonomy("orbit", "counter-clockwise") == CameraMovement.ORBIT_COUNTERCLOCKWISE
    assert movement_from_taxonomy("tracking", None) == CameraMovement.TRACKING
    assert movement_from_taxonomy("pan", None) == CameraMovement.UNKNOWN  # a pan needs a direction
    assert set(_ENUM_MOVEMENT) == {m.value for m in CameraMovement}  # the VLM is offered the same vocabulary
    assert {"lens_type", "focal_length_mm", "depth_of_field"} <= set(SCENE_SCHEMA["properties"]["camera"]["properties"])


def test_camera_annotation_derives_standard_fields_from_primary_ones():
    c = CameraAnnotation(movement="pan_left", shot_type="wide", camera_angle="low", speed="slow", stability="stabilized")
    assert (c.camera_movement, c.movement_direction, c.movement_speed) == ("pan", "left", "slow")
    assert (c.shot_size, c.camera_distance, c.camera_height, c.stabilization) == ("wide", "far", "low", "stabilized")
    assert c.lens_type is None and c.focal_length_mm is None and c.lens_confidence_source == ConfidenceSource.UNAVAILABLE
    # taxonomy form -> fine-grained label, and a generic label refined by a direction
    assert CameraAnnotation(camera_movement="dolly", movement_direction="in").movement == CameraMovement.DOLLY_IN
    t = CameraAnnotation(movement="tracking", movement_direction="left")
    assert t.movement == CameraMovement.TRACKING_LEFT and t.camera_movement == "tracking"
    assert CameraAnnotation(movement="drone").is_aerial is True
    # unknown stays honest
    u = CameraAnnotation()
    assert u.camera_movement is None and u.shot_size is None and u.camera_height is None
    # assignment keeps the derived block in sync
    c.movement = CameraMovement.STATIC
    assert c.camera_movement == "static" and c.movement_direction is None
    # legacy records (no taxonomy keys) validate and get the block
    legacy = CameraAnnotation.model_validate({"shot_type": "medium", "camera_angle": "eye_level", "movement": "zoom_in", "zoom": None, "stability": None, "is_aerial": False})
    assert legacy.camera_movement == "zoom" and legacy.movement_direction == "in" and legacy.camera_height == "eye_level"


def test_parse_camera_reads_optics_and_aliases():
    cam = parse_camera({
        "shot_type": "wide shot", "camera_angle": "low angle", "movement": "truck left", "zoom": None, "stability": "gimbal", "is_aerial": False,
        "lens_type": "Wide-angle", "focal_length_mm": "24mm", "depth_of_field": "shallow (background blurred)", "focus_type": "rack focus", "camera_height": "low", "perspective": "linear",
    })
    assert cam.movement == CameraMovement.TRACKING_LEFT and cam.camera_movement == "tracking" and cam.movement_direction == "left"
    assert cam.lens_type == "wide_angle" and cam.focal_length_mm == 24.0 and cam.depth_of_field == "shallow" and cam.focus_type == "rack focus"
    assert cam.lens_confidence_source == ConfidenceSource.MODEL_SELF_REPORT and cam.stabilization == "gimbal"
    assert coerce_enum("dolly in", CameraMovement, CameraMovement.UNKNOWN) == CameraMovement.DOLLY_IN
    assert coerce_enum("orbit", CameraMovement, CameraMovement.UNKNOWN) == CameraMovement.ORBIT
    assert coerce_enum("boom up", CameraMovement, CameraMovement.UNKNOWN) == CameraMovement.CRANE_UP
    assert coerce_enum("POV", CameraMovement, CameraMovement.UNKNOWN) == CameraMovement.FPV
    plain = parse_camera({"shot_type": "medium", "camera_angle": "eye_level", "movement": "static"})
    assert plain.lens_type is None and plain.focal_length_mm is None and plain.lens_confidence_source == ConfidenceSource.UNAVAILABLE
    assert parse_camera({"focal_length_mm": "unknown"}).focal_length_mm is None


def test_tracking_gets_a_direction_from_pan_or_background_flow():
    tracked = SubjectMotion(subject="person", label="tracked_by_camera", camera_relation="tracked", camera_velocity=[0.3, 0.0], confidence=0.8)
    assert tracking_direction(CameraMovement.PAN_RIGHT, tracked) == CameraMovement.TRACKING_RIGHT
    assert tracking_direction(CameraMovement.HANDHELD, tracked) == CameraMovement.TRACKING_LEFT  # content streams right -> camera moves left
    assert tracking_direction(CameraMovement.TILT_UP, SubjectMotion(subject="car", label="tracked_by_camera", camera_relation="tracked", camera_velocity=[0.0, 0.2])) == CameraMovement.TRACKING
    a = SceneAnalysis(scene_id="s", video_id="v", start_time=0, end_time=2, camera=CameraAnnotation(movement="pan_left", confidence=0.9))
    apply_subject_motion(a, tracked)
    assert a.camera.movement == CameraMovement.TRACKING_LEFT and a.camera.camera_movement == "tracking" and a.camera.movement_direction == "left"
    assert a.camera.tracked_subject == "person" and a.camera.confidence == 0.8
    apply_subject_motion(a, tracked)  # idempotent: already tracking
    assert a.camera.movement == CameraMovement.TRACKING_LEFT


def test_opposite_movements_cover_the_directional_labels():
    for m in CameraMovement:
        if MOVEMENT_TAXONOMY[m][1] is not None:
            assert m in OPPOSITE_MOVEMENT and OPPOSITE_MOVEMENT[m] != m, m
    assert OPPOSITE_MOVEMENT[CameraMovement.DOLLY_IN] == CameraMovement.DOLLY_OUT
    assert OPPOSITE_MOVEMENT[CameraMovement.ORBIT_CLOCKWISE] == CameraMovement.ORBIT_COUNTERCLOCKWISE


def _scene(**over):
    base = {
        "record_id": "scene_v_000001", "video_id": "vid_v", "scene_id": "scene_001", "start_time": 10.0, "end_time": 16.5, "duration": 6.5,
        "summary": "A 6.5-second shot with bright lighting and a desaturated, low-contrast, warm-toned palette; the camera pans left.",
        "environment": {"location": "mountain road", "setting": "outdoor", "weather": None, "lighting": "bright", "time_of_day": "golden hour", "background": None},
        "objects": ["car", "mountain"], "object_details": [{"name": "car", "count": 1, "area_fraction": 0.2}, {"name": "mountain", "count": 1, "area_fraction": 0.5}],
        "people": None, "actions": ["A car drives along the road.", "Camera pans left"],
        "camera": {"shot_type": "wide", "camera_angle": "low", "movement": "tracking_left", "speed": "slow", "stability": "stabilized", "tracked_subject": "car"},
        "subject_motion": None, "visual_style": {"depth_of_field": "shallow"},
        "measurements": {"lighting_level": "bright", "color_temperature": "warm", "saturation_mean": 40.0},
    }
    base.update(over)
    return base


def test_generation_prompt_is_built_from_structured_fields_only():
    rec = SceneRecord.model_validate(_scene()).model_dump(mode="json")
    p = generation_prompt(rec)
    assert p == (
        "A 6-second wide shot of a mountain and a car in a mountain road at golden hour; a car drives along the road, "
        "from a low angle, slow tracking shot moving left following a car, bright lighting, warm tones, desaturated colours, shallow depth of field, stabilized camera."
    ), p
    assert "cinematic" not in p and "mm" not in p  # no adjectives or lens claims the record does not contain
    with_lens = SceneRecord.model_validate(_scene(camera={"shot_type": "close_up", "movement": "static", "lens_type": "portrait", "focal_length_mm": 85})).model_dump(mode="json")
    assert "shot on a 85mm portrait lens" in generation_prompt(with_lens) and "static camera" in generation_prompt(with_lens)
    # measurement-only shot: only what was measured
    bare = SceneRecord.model_validate(_scene(environment={"lighting": "dim"}, objects=[], object_details=[], actions=[], camera={"movement": "static"}, visual_style={}, measurements={"lighting_level": "dim", "color_temperature": "cool", "saturation_mean": 80.0})).model_dump(mode="json")
    assert generation_prompt(bare) == "A 6-second shot, static camera, dim lighting, cool tones."
    nothing = {"record_id": "x", "video_id": "v", "scene_id": "s", "start_time": 0.0, "end_time": 3.0, "camera": {}, "environment": {}}
    assert generation_prompt(nothing) is None
    assert camera_phrase({"camera_movement": "orbit", "movement_direction": "clockwise", "movement_speed": "fast"}) == "fast camera orbiting the subject clockwise"
    assert camera_phrase({"camera_movement": "unknown"}) is None


def test_fill_prompts_and_frame_clip_links_on_record_dicts():
    scene = SceneRecord.model_validate(_scene()).model_dump(mode="json")
    records = {
        "scenes": [scene],
        "clips": [
            {"record_id": "clip_a", "video_id": "vid_v", "scene_id": "scene_001", "clip_id": "clip_001", "frame_ids": ["frame_001_000", "frame_001_001"], "camera": scene["camera"]},
            {"record_id": "clip_b", "video_id": "vid_v", "scene_id": "scene_001", "clip_id": "clip_001_01", "frame_ids": ["frame_001_001"]},
            {"record_id": "clip_c", "video_id": "vid_v", "scene_id": "scene_009", "clip_id": "clip_009", "start_time": 0.0, "end_time": 4.0, "camera": {"movement": "pan_right", "camera_movement": "pan", "movement_direction": "right"}},
        ],
        "frames": [
            {"record_id": "f0", "video_id": "vid_v", "scene_id": "scene_001", "frame_id": "frame_001_000"},
            {"record_id": "f1", "video_id": "vid_v", "scene_id": "scene_001", "frame_id": "frame_001_001"},
            {"record_id": "f2", "video_id": "vid_v", "scene_id": "scene_003", "frame_id": "frame_003_000"},
        ],
    }
    assert fill_frame_clip_ids(records) == 2
    assert [f.get("clip_id") for f in records["frames"]] == ["clip_001", "clip_001", None]  # whole-scene clip wins over the sub-clip
    assert fill_generation_prompts(records) == 4
    assert records["clips"][0]["generation_prompt"] == scene["generation_prompt"] == records["clips"][1]["generation_prompt"]
    assert records["clips"][2]["generation_prompt"] == "A 4-second shot, pan to the right."
    FrameCaptionRecord.model_validate({**records["frames"][0], "frame_path": "frames/vid_v/f.jpg", "timestamp": 1.0, "caption": "c"})


def test_media_metadata_is_staged_next_to_the_media(tmp_path: Path):
    from video_dataset.dataset.upload import write_media_metadata
    from video_dataset.utils.io import read_jsonl, write_jsonl

    final = tmp_path / "final"
    final.mkdir()
    write_jsonl(final / "clips.jsonl", [
        {"record_id": "c1", "video_id": "vid_a", "scene_id": "s1", "clip_id": "clip_001", "clip_path": "clips/vid_a/clip_001.mp4", "start_time": 0.0, "end_time": 2.0, "description": "d", "generation_prompt": "p", "split": "train", "tier": "silver", "subsets": [], "camera": {"shot_size": "wide", "camera_movement": "pan", "movement_direction": "left"}},
        {"record_id": "c2", "video_id": "vid_b", "scene_id": "s1", "clip_id": "clip_001", "clip_path": "clips/vid_b/clip_001.mp4", "description": "other video"},
        {"record_id": "c3", "video_id": "vid_a", "scene_id": "s2", "clip_id": "clip_002", "clip_path": "/abs/elsewhere.mp4", "description": "not under clips/"},
    ])
    write_jsonl(final / "frames.jsonl", [{"record_id": "f1", "video_id": "vid_a", "scene_id": "s1", "clip_id": "clip_001", "frame_id": "fr", "frame_path": "frames/vid_a/s1/f0.jpg", "timestamp": 0.5, "caption": "cap"}])
    staging = tmp_path / "staging"
    assert write_media_metadata(final, staging, {"vid_a"}) == 2
    clips = list(read_jsonl(staging / "clips" / "metadata.jsonl"))
    assert [c["file_name"] for c in clips] == ["vid_a/clip_001.mp4"] and clips[0]["generation_prompt"] == "p" and clips[0]["camera_movement"] == "pan"
    frames = list(read_jsonl(staging / "frames" / "metadata.jsonl"))
    assert frames[0]["file_name"] == "vid_a/s1/f0.jpg" and frames[0]["clip_id"] == "clip_001" and frames[0]["caption"] == "cap"
