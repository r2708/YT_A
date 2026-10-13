"""Split / tier / cinematic subset, hard negatives, typed Parquet export, aesthetic head, subject motion."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from video_dataset.config import load_config
from video_dataset.dataset.arrow import arrow_schema, to_table
from video_dataset.dataset.curation import (
    annotate_records,
    scene_is_cinematic,
    split_for_video,
    split_manifest,
    subset_records,
    tier_for_record,
)
from video_dataset.dataset.export import (
    RECORD_MODELS,
    normalize_record,
    upgrade_record,
    write_typed_parquet_files,
)
from video_dataset.dataset.negatives import caption_hard_negatives, qa_hard_negatives
from video_dataset.schemas.common import ConfidenceSource
from video_dataset.schemas.curation import Split, Tier
from video_dataset.schemas.dataset import SceneRecord
from video_dataset.schemas.events import Event, EventSource, EventType
from video_dataset.schemas.qa import Evidence, QARecord, QAType
from video_dataset.schemas.quality import QualityScore, ValidationInfo, ValidationStatus
from video_dataset.schemas.scene import ScanSample, SceneScan
from video_dataset.schemas.vision import (
    CameraAnnotation,
    CameraMovement,
    Environment,
    Measurements,
    ObjectAnnotation,
    SceneAnalysis,
    VisualStyle,
)
from video_dataset.vision.enrichers.aesthetic import AestheticHead
from video_dataset.vision.motion import camera_speed
from video_dataset.vision.subject_motion import (
    analyze_subject_motion,
    apply_subject_motion,
    position_label,
    scale_class,
)

CFG = load_config().export


# ------------------------------------------------------------------------------------ split
def test_split_is_deterministic_and_per_video():
    ids = [f"vid_{i:012x}" for i in range(2000)]
    first = [split_for_video(v, CFG.split) for v in ids]
    assert first == [split_for_video(v, CFG.split) for v in ids]  # stable across calls
    counts = {s: first.count(s) for s in Split}
    assert 60 <= counts[Split.VALIDATION] <= 140 and 60 <= counts[Split.TEST] <= 140  # ~5 % each of 2000
    other = CFG.split.model_copy(update={"salt": "another"})
    assert [split_for_video(v, other) for v in ids] != first  # the salt reshuffles


def test_annotate_records_keeps_every_record_of_a_video_in_one_split():
    rows = {
        "scenes": [{"record_id": f"s{i}", "video_id": f"vid_{i % 7}", "scene_id": "scene_001", "duration": 3.0, "camera": {"movement": "static"}, "validation": {"status": "accepted"}, "quality": {"overall": 0.9, "components_used": ["grounding"]}, "confidence": 0.9, "confidence_source": "measurement"} for i in range(50)],
        "temporal_qa": [{"record_id": f"q{i}", "video_id": f"vid_{i % 7}", "evidence": {"scene_ids": ["scene_001"]}, "validation": {"status": "review"}, "quality": {"overall": 0.4}} for i in range(20)],
    }
    annotate_records(rows, CFG)
    by_video: dict[str, set[str]] = {}
    for recs in rows.values():
        for r in recs:
            by_video.setdefault(r["video_id"], set()).add(r["split"])
    assert all(len(v) == 1 for v in by_video.values())
    manifest = split_manifest(rows)
    assert sum(len(v) for v in manifest.values()) == 7
    assert all(r["tier"] == "gold" for r in rows["scenes"]) and all(r["tier"] == "bronze" for r in rows["temporal_qa"])
    assert all(r["subsets"] == ["cinematic"] for r in rows["scenes"])  # accepted, known camera, long enough
    assert all(r["subsets"] == ["cinematic"] for r in rows["temporal_qa"])  # inherits from its evidence scene
    assert len(subset_records(rows, "cinematic")["scenes"]) == 50


# ------------------------------------------------------------------------------------ tiers
def _rec(status="accepted", overall=0.9, components=("grounding", "description_quality"), conf=0.9, src="measurement"):
    return {"validation": {"status": status}, "quality": {"overall": overall, "components_used": list(components)}, "confidence": conf, "confidence_source": src}


def test_tier_rules_are_honest_about_missing_signals():
    assert tier_for_record(_rec(), CFG.tiers) == (Tier.GOLD, [])
    tier, why = tier_for_record(_rec(components=("description_quality",)), CFG.tiers)
    assert tier == Tier.SILVER and why == ["no_grounding_component"]  # high score but nothing grounded it
    tier, why = tier_for_record(_rec(conf=None), CFG.tiers)
    assert tier == Tier.SILVER and "confidence_unavailable" in why
    tier, why = tier_for_record(_rec(src="model_self_report"), CFG.tiers)
    assert tier == Tier.SILVER and why == ["confidence_source_model_self_report"]
    assert tier_for_record(_rec(overall=0.7), CFG.tiers) == (Tier.SILVER, ["overall_below_0.8"])
    tier, why = tier_for_record(_rec(overall=0.3), CFG.tiers)
    assert tier == Tier.BRONZE and why == ["overall_below_0.6"]
    tier, why = tier_for_record(_rec(status="review"), CFG.tiers)
    assert tier == Tier.BRONZE and why == ["status_review"]
    tier, why = tier_for_record(_rec(overall=None), CFG.tiers)
    assert tier == Tier.BRONZE and why == ["quality_unscored"]
    assert tier_for_record(_rec(status="rejected"), CFG.tiers) == (None, ["status_rejected"])


def test_cinematic_criteria():
    scene = {"tier": "silver", "duration": 4.0, "camera": {"movement": "pan_left"}, "measurements": {"sharpness": 300.0, "aesthetic_score": 5.6}}
    assert scene_is_cinematic(scene, CFG.cinematic) == (True, [])
    ok, why = scene_is_cinematic({**scene, "tier": "bronze"}, CFG.cinematic)
    assert not ok and why == ["tier_below_silver"]
    ok, why = scene_is_cinematic({**scene, "camera": {"movement": "unknown"}, "duration": 0.8}, CFG.cinematic)
    assert set(why) == {"camera_unknown", "too_short"}
    ok, why = scene_is_cinematic({**scene, "measurements": {"sharpness": 10.0, "aesthetic_score": 3.0}}, CFG.cinematic)
    assert set(why) == {"soft_focus", "aesthetic_below_5.0"}
    strict = CFG.cinematic.model_copy(update={"require_aesthetic": True})
    assert scene_is_cinematic({**scene, "measurements": {"sharpness": 300.0}}, strict) == (False, ["no_aesthetic_score"])
    assert scene_is_cinematic({**scene, "measurements": {"sharpness": 300.0}}, CFG.cinematic) == (True, [])


# ------------------------------------------------------------------------------------ negatives
def _event(i, t, text, etype=EventType.ACTION):
    return Event(event_id=f"event_{i:04d}", video_id="v", event_type=etype, start_time=t, end_time=t + 2.0, event=text, source=EventSource.VISION, confidence=0.9, confidence_source=ConfidenceSource.MEASUREMENT, scene_ids=[f"scene_{i:03d}"])


def _qa(qtype, question, answer, events, template="t"):
    return QARecord(question_id="qa_v_000001", video_id="v", type=qtype, question=question, answer=answer, template_id=template, evidence=Evidence(start_time=events[0].start_time, end_time=events[-1].end_time, event_ids=[e.event_id for e in events], scene_ids=[s for e in events for s in e.scene_ids]))


def test_qa_hard_negatives_by_type():
    events = [_event(1, 0.0, "The car enters the tunnel."), _event(2, 5.0, "The car exits the tunnel."), _event(3, 30.0, "A dog runs across the road."), _event(4, 60.0, "The sun sets behind the hills.")]
    rng = random.Random(0)
    ordering = _qa(QAType.TEMPORAL_ORDERING, "Which happens first?", "First the car enters the tunnel, then the car exits the tunnel.", events[:2])
    negs = qa_hard_negatives(ordering, events, 90.0, rng)
    assert negs[0].kind == "swapped_order" and negs[0].text == "First the car exits the tunnel, then the car enters the tunnel."
    assert all(n.kind == "other_event" for n in negs[1:]) and all(n.source_ids[0] in ("event_0003", "event_0004") for n in negs[1:])
    duration = _qa(QAType.DURATION, "How long?", "About 2 seconds (from 0.0s to 2.0s).", events[:1])
    negs = qa_hard_negatives(duration, events, 90.0, rng)
    assert negs[0].kind == "wrong_duration" and "from 0.0s to 5.0s" in negs[0].text
    loc = _qa(QAType.EVENT_LOCALIZATION, "When?", "Early in the video (starts at 5.0s).", events[1:2])
    negs = qa_hard_negatives(loc, events, 90.0, rng)
    assert negs[0].kind == "shifted_time" and "starts at 23.0s" in negs[0].text  # +max(5, 0.2*90)
    verdict = _qa(QAType.TEMPORAL_ORDERING, "Before or after?", "Before. The car enters the tunnel around 0.0s, and the car exits the tunnel around 5.0s.", events[:2])
    assert qa_hard_negatives(verdict, events, 90.0, rng)[0].text.startswith("After.")
    ts = _qa(QAType.TIMESTAMP, "What happens at 1.0s?", "The car enters the tunnel.", events[:1])
    negs = qa_hard_negatives(ts, events, 90.0, rng, limit=2)
    assert len(negs) == 2 and all(n.kind == "other_event" for n in negs) and all(n.text != ts.answer for n in negs)
    # every negative is distinct from the answer and from each other
    texts = [n.text for n in negs]
    assert len(set(texts)) == len(texts)
    assert qa_hard_negatives(ts, events[:1], 90.0, rng) == []  # nothing else to borrow from


def _analysis(sid, movement, lighting, objects, summary):
    return SceneAnalysis(
        scene_id=sid, video_id="v", start_time=0.0, end_time=4.0, summary=summary,
        environment=Environment(lighting=lighting), objects=[ObjectAnnotation(name=o) for o in objects],
        camera=CameraAnnotation(movement=movement, confidence=0.8, confidence_source=ConfidenceSource.MEASUREMENT),
        visual_style=VisualStyle(), measurements=Measurements(lighting_level=lighting),
    )


def test_caption_hard_negatives_swap_measured_attributes_and_borrow_different_shots():
    a = _analysis("scene_001", CameraMovement.PAN_LEFT, "dim", ["car"], "A 4.0-second shot with dim lighting; the camera pans left. Detected objects: a car.")
    b = _analysis("scene_002", CameraMovement.STATIC, "bright", ["person"], "A 4.0-second shot with bright lighting; the camera is static. Detected objects: a person.")
    c = _analysis("scene_003", CameraMovement.PAN_LEFT, "dim", ["car"], "A 4.0-second shot with dim lighting; the camera pans left. Detected objects: a car and a truck.")
    negs = caption_hard_negatives(a, [a, b, c], random.Random(0))
    kinds = [n.kind for n in negs]
    assert kinds[:2] == ["attribute_swap", "attribute_swap"]
    assert "the camera pans right" in negs[0].text and "pan_right" in (negs[0].note or "")
    assert "bright lighting" in negs[1].text
    assert negs[2].kind == "other_shot" and negs[2].source_ids == ["scene_002"] and "camera pan_left vs static" in (negs[2].note or "")
    assert len(negs) == 3


# ------------------------------------------------------------------------------------ schema / loader
def test_legacy_scene_record_upgrades_to_current_schema():
    legacy = {"record_id": "scene_v_scene_001", "video_id": "vid_v", "scene_id": "scene_001", "start": 1.0, "end": 4.5, "duration": 3.5, "summary": "x y z w", "environment": {}, "camera": {}, "visual_style": {}}
    up = upgrade_record("scenes", legacy)
    assert up["start_time"] == 1.0 and up["end_time"] == 4.5 and "start" not in up
    assert up["record_type"] == "scene" and up["split"] is None and up["tier"] is None and up["hard_negatives"] == []
    qa = upgrade_record("long_video_qa", {"question_id": "qa_v_000001", "video_id": "vid_v", "type": "long_range", "question": "q?", "answer": "a.", "evidence": {"start_time": 0, "end_time": 1, "event_ids": ["event_0001"]}})
    assert qa["record_id"] == "qa_v_000001" and qa["record_type"] == "long_video_qa"
    assert SceneRecord.model_validate(up).start_time == 1.0


def test_typed_parquet_has_stable_schema_across_null_columns(tmp_path: Path):
    """The failure the JSON loader has: a column that is null in one file and filled in another."""
    base = {"video_id": "vid_v", "type": "duration", "question": "How long?", "answer": "About 6 seconds.", "evidence": {"start_time": 1.0, "end_time": 7.0, "event_ids": ["event_0001"]}}
    only_nulls = [QARecord(question_id=f"qa_v_{i:06d}", template_id=None, confidence=None, **base) for i in range(3)]
    filled = [QARecord(question_id=f"qa_v_{i:06d}", template_id="duration_v1", confidence=0.8, quality=QualityScore.from_components(grounding=0.9), validation=ValidationInfo(status=ValidationStatus.ACCEPTED, checks={"a": True, "b": None}), **base) for i in range(3, 6)]
    counts = write_typed_parquet_files({"temporal_qa": only_nulls}, tmp_path / "a")
    counts2 = write_typed_parquet_files({"temporal_qa": filled}, tmp_path / "b")
    assert counts["temporal_qa"] == 3 and counts2["temporal_qa"] == 3
    # a type without records still gets an empty, typed file: the card's `shard_*/<type>.parquet` globs resolve in every shard
    assert counts["frames"] == 0 and pq.read_table(tmp_path / "a" / "frames.parquet").num_rows == 0
    ta, tb = pq.read_table(tmp_path / "a" / "temporal_qa.parquet"), pq.read_table(tmp_path / "b" / "temporal_qa.parquet")
    assert ta.schema.equals(tb.schema)  # identical schemas although one file has only nulls
    assert str(ta.schema.field("template_id").type) == "string" and str(ta.schema.field("confidence").type) == "double"
    merged = __import__("pyarrow").concat_tables([ta, tb])
    assert merged.num_rows == 6
    row = tb.to_pylist()[0]
    assert json.loads(row["validation"]["checks"]) == {"a": True, "b": None}  # free-form dict kept as JSON text
    datasets = pytest.importorskip("datasets")
    ds = datasets.load_dataset("parquet", data_files=[str(tmp_path / "a" / "temporal_qa.parquet"), str(tmp_path / "b" / "temporal_qa.parquet")], split="train")
    assert len(ds) == 6 and ds.features["template_id"].dtype == "string"


def test_every_record_model_has_an_arrow_schema_and_shared_envelope():
    envelope = {"record_id", "record_type", "video_id", "split", "tier", "tier_reasons", "subsets", "confidence", "confidence_source", "quality", "validation"}
    for name, model in RECORD_MODELS.items():
        schema, _ = arrow_schema(model)
        assert envelope <= set(schema.names), name
        # time window: frames carry `timestamp`, QA keeps it inside `evidence`, everything else start_time / end_time
        assert ("timestamp" in schema.names) or ("evidence" in schema.names) or ({"start_time", "end_time"} <= set(schema.names)), name
    rec = SceneRecord(record_id="scene_v_scene_001", video_id="vid_v", scene_id="scene_001", start_time=0.0, end_time=2.0, duration=2.0, summary="a b c d", environment=Environment(), camera=CameraAnnotation(), visual_style=VisualStyle(), split=Split.TEST, tier=Tier.SILVER, measurements=Measurements(aesthetic_score=6.1))
    table = to_table(SceneRecord, [rec])
    assert table.num_rows == 1 and table.column("split").to_pylist() == ["test"]
    n = normalize_record("scene", rec)
    assert n["split"] == "test" and n["tier"] == "silver" and n["aesthetic_score"] == 6.1 and n["start_time"] == 0.0 and n["hard_negatives"] == 0


# ------------------------------------------------------------------------------------ aesthetic head
def test_aesthetic_head_scores_normalised_embeddings():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(1, 8)).astype(np.float32)
    head = AestheticHead.from_state_dict({"weight": w, "bias": np.array([5.0], dtype=np.float32)}, "test-head")
    emb = rng.normal(size=(3, 8))
    scores = head.score(emb)
    unit = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    assert np.allclose(scores, (unit @ w.reshape(-1) + 5.0), atol=1e-3)
    assert head.score(emb * 100.0) == scores  # scale invariant: embeddings are normalised first
    summary = head.summarize(emb)
    assert summary["model"] == "test-head" and summary["min"] <= summary["mean"] <= summary["max"] and len(summary["per_frame"]) == 3
    with pytest.raises(ValueError):
        head.score(np.zeros((1, 7)))


# ------------------------------------------------------------------------------------ subject / camera motion
def _det(name, x, y, w=20, h=40):
    return {"name": name, "confidence": 0.9, "bbox": [x, y, x + w, y + h], "frame_size": [160, 90]}


def _scan(dx: float, n: int = 13) -> SceneScan:
    return SceneScan(scene_id="s", analysis_width=160, analysis_fps=4.0, samples=[ScanSample(timestamp=0.25 * i, frame_index=i, change=0.1, brightness=100.0, flow_dx=dx, flow_dy=0.0, flow_mag=abs(dx), flow_div=0.0) for i in range(n)])


def test_subject_motion_separates_subject_from_camera():
    ts = [0.0, 1.0, 2.0, 3.0]
    tracked = analyze_subject_motion([[_det("person", 70, 25)] for _ in ts], ts, _scan(8.0), CameraMovement.PAN_LEFT)
    assert tracked and tracked.label == "tracked_by_camera" and tracked.camera_relation == "tracked" and tracked.camera_velocity == [0.2, 0.0]
    past = analyze_subject_motion([[_det("person", 10 + 32 * i, 25)] for i in range(4)], ts, _scan(8.0), CameraMovement.PAN_LEFT)
    assert past and past.label == "panned_past"
    walk = analyze_subject_motion([[_det("person", 10 + 20 * i, 25), _det("car", 120, 60, 30, 20)] for i in range(4)], ts, None, CameraMovement.STATIC)
    assert walk and walk.subject == "person" and walk.label == "moves_right" and walk.camera_relation == "camera_static" and walk.box_velocity == [0.125, 0.0]
    grow = analyze_subject_motion([[_det("dog", 70 - 5 * i, 40 - 8 * i, 20 + 10 * i, 20 + 16 * i)] for i in range(4)], ts, None, CameraMovement.STATIC)
    assert grow and grow.label == "approaches" and (grow.scale_change_per_second or 0) > 0
    assert analyze_subject_motion([[_det("dog", 1, 1)]], [0.0], None, CameraMovement.STATIC) is None
    assert analyze_subject_motion([[], [], []], ts[:3], None, CameraMovement.STATIC) is None
    assert position_label(0.1, 0.1) == "upper-left" and position_label(0.5, 0.5) == "center" and position_label(0.9, 0.5) == "right"
    assert scale_class(0.003) == "tiny" and scale_class(0.3) == "large" and scale_class(0.7) == "dominant"


def test_apply_subject_motion_promotes_pan_to_tracking_and_describes_it():
    ts = [0.0, 1.0, 2.0, 3.0]
    sm = analyze_subject_motion([[_det("person", 70, 25)] for _ in ts], ts, _scan(8.0), CameraMovement.PAN_LEFT)
    a = SceneAnalysis(scene_id="s", video_id="v", start_time=0, end_time=3, summary="A shot; the camera pans left.", camera=CameraAnnotation(movement=CameraMovement.PAN_LEFT, confidence=0.9, confidence_source=ConfidenceSource.MEASUREMENT))
    apply_subject_motion(a, sm)
    assert a.camera.movement == CameraMovement.TRACKING_LEFT and a.camera.tracked_subject == "person" and a.subject_motion is sm  # pan left -> tracks moving left
    assert a.camera.camera_movement == "tracking" and a.camera.movement_direction == "left"
    assert a.summary.endswith("The camera tracks the person, which stays in place in the frame while the background moves.")
    assert any("tracks the person" in x for x in a.actions) and "tracks the person" in (a.visual_style.motion or "")
    static = SceneAnalysis(scene_id="s", video_id="v", start_time=0, end_time=3, summary="A shot.", camera=CameraAnnotation(movement=CameraMovement.STATIC))
    walk = analyze_subject_motion([[_det("person", 10 + 20 * i, 25)] for i in range(4)], ts, None, CameraMovement.STATIC)
    apply_subject_motion(static, walk)
    assert static.camera.movement == CameraMovement.STATIC and "moves right while the camera stays still" in static.summary
    type(static).model_validate(static.model_dump(mode="json"))


def test_camera_speed_from_flow():
    assert camera_speed(_scan(8.0), CameraMovement.PAN_LEFT) == ("moderate", 0.2)  # 8 px/frame * 4 fps / 160 px
    assert camera_speed(_scan(1.6), CameraMovement.PAN_LEFT) == ("slow", 0.04)
    assert camera_speed(_scan(16.0), CameraMovement.TILT_UP) == ("fast", 0.4)
    assert camera_speed(_scan(8.0), CameraMovement.STATIC) == (None, None)
    assert camera_speed(None, CameraMovement.PAN_LEFT) == (None, None)
