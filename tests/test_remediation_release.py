"""Permanent fixes for the issue document (2026-11-02): schema 2.3.0 migrations, export-time
checks, grounding methods, camera ambiguity, world / camera / subject motion, rights blocks, the
media asset registry, clip integrity fields, transcript alignment, OCR grounding, description
evidence and the tri-state release status with the new gates."""

from __future__ import annotations

import json
from pathlib import Path

from video_dataset.config import load_config
from video_dataset.dataset.curation import (
    RIGHTS_TERMS,
    annotate_records,
    channel_group_id,
    rights_info_for,
    split_leaks,
)
from video_dataset.dataset.export import (
    DATASET_FILES,
    UpgradeStats,
    upgrade_record,
    upgrade_records,
    write_jsonl_files,
    write_typed_parquet_files,
)
from video_dataset.dataset.inventory import verify_checksums, write_inventory
from video_dataset.dataset.media_manifest import apply_asset_ids, asset_id_for, build_media_manifest
from video_dataset.dataset.release import (
    checksums_gate,
    exclusions_block,
    reproducibility_gate,
    rights_gate,
    run_release_check,
    schema_gate,
    temporal_gate,
)
from video_dataset.dataset.upload import rights_cleared_videos, stage_shard, verify_remote_checksums
from video_dataset.frame_sampling.clips import frame_count_check, stream_integrity
from video_dataset.schemas.common import SCHEMA_VERSION
from video_dataset.schemas.migrations import KNOWN_VERSIONS, MIGRATIONS, migrate_record, migration_path
from video_dataset.schemas.quality import QualityScore
from video_dataset.schemas.transcript import Transcript, TranscriptSegment, Word
from video_dataset.schemas.vision import (
    CameraAnnotation,
    Measurements,
    SceneAnalysis,
    normalize_stabilization,
)
from video_dataset.transcription.align import validate_alignment
from video_dataset.utils.io import write_json_atomic
from video_dataset.validation.export_checks import fill_export_checks
from video_dataset.validation.policy import (
    POLICY_VERSION,
    REQUIRED_CHECKS,
    enforce_exported,
    record_status_change,
)
from video_dataset.validation.quality import event_quality, relation_quality
from video_dataset.validation.stage import camera_motion_agreement
from video_dataset.vision.subject_motion import apply_subject_motion, world_motion

CFG = load_config(None, {"export.min_source_videos": "0", "export.split.min_validation_videos": "0", "export.split.min_test_videos": "0", "export.split.required_splits": "[train]"})


def _full_checks(kind: str) -> dict[str, str]:
    return {name: "passed" for name in REQUIRED_CHECKS[kind]}


def _scene(i: int, vid: str = "vid_a", *, status: str = "accepted", start: float | None = None, end: float | None = None) -> dict:
    s = 2.0 * i if start is None else start
    e = s + 2.0 if end is None else end
    v = {"status": status, "issues": [], "checks": _full_checks("scene"), "reason_codes": [], "policy_version": POLICY_VERSION, "semantic_verification": "template"}
    return {"record_id": f"scene_{vid}_scene_{i:03d}", "video_id": vid, "scene_id": f"scene_{i:03d}", "start_time": s, "end_time": e, "duration": round(e - s, 3), "summary": "A shot with a person walking.", "environment": {"setting": "outdoor", "time_of_day": "day"}, "objects": ["person"], "camera": {"movement": "pan_left"}, "visual_style": {}, "measurements": {"lighting_level": "normal"}, "provider": "heuristic", "confidence": 0.9, "confidence_source": "measurement", "quality": {"grounding": 0.9, "overall": 0.9, "components_used": ["grounding"], "grounding_method": "instrument:measurement"}, "validation": v, "generation_target": "single_shot"}


def _clip(i: int, vid: str = "vid_a", **kw) -> dict:
    s = _scene(i, vid, **kw)
    return {"record_id": f"clip_{vid}_clip_{i:03d}", "video_id": vid, "scene_id": s["scene_id"], "clip_id": f"clip_{i:03d}", "clip_path": f"clips/{vid}/clip_{i:03d}.mp4", "start_time": s["start_time"], "end_time": s["end_time"], "boundary_check": "passed", "description": s["summary"], "camera": s["camera"], "provider": "heuristic", "confidence": 0.9, "confidence_source": "measurement", "quality": s["quality"], "validation": json.loads(json.dumps(s["validation"])), "generation_target": "single_shot", "frame_ids": [f"frame_{i:03d}_000"]}


def _frame(i: int, vid: str = "vid_a") -> dict:
    s = _scene(i, vid)
    return {"record_id": f"frame_{vid}_frame_{i:03d}_000", "video_id": vid, "scene_id": s["scene_id"], "frame_id": f"frame_{i:03d}_000", "frame_path": f"frames/{vid}/scene_{i:03d}/frame_{i:03d}_000.jpg", "timestamp": s["start_time"] + 0.5, "caption": "frame", "validation": json.loads(json.dumps(s["validation"])), "quality": s["quality"]}


def _event(i: int, vid: str = "vid_a", start: float = 0.0, end: float = 1.0) -> dict:
    return {"record_id": f"event_{vid}_event_{i:04d}", "video_id": vid, "event_id": f"event_{i:04d}", "event_type": "camera", "start_time": start, "end_time": end, "event": "The camera pans left.", "source": "motion", "confidence": 0.8, "confidence_source": "measurement", "validation": {"status": "accepted", "issues": [], "checks": _full_checks("event"), "reason_codes": [], "policy_version": POLICY_VERSION, "semantic_verification": "measured"}}


def _qa(i: int, vid: str = "vid_a", ev_start: float = 0.0, ev_end: float = 2.0, event_ids: list[str] | None = None) -> dict:
    checks = _full_checks("qa")
    return {"question_id": f"qa_{vid}_{i:06d}", "record_id": f"qa_{vid}_{i:06d}", "video_id": vid, "type": "timestamp", "question": "When does the camera pan left?", "answer": "At 0.0s.", "evidence": {"start_time": ev_start, "end_time": ev_end, "event_ids": event_ids or ["event_0001"]}, "confidence": 0.8, "confidence_source": "interval_algebra", "validation": {"status": "accepted", "issues": [], "checks": checks, "reason_codes": [], "policy_version": POLICY_VERSION, "semantic_verification": "template"}}


def _video_info(*vids: str, channel: str | None = "Creator One", duration: float = 10.0) -> dict:
    return {v: {"license_status": "creative_commons", "rights_status": "declared_cc_by", "channel": channel, "duration": duration, "url": f"https://youtube.com/watch?v={v}", "retrieved_at": "2026-10-01T00:00:00+00:00"} for v in vids}


# ------------------------------------------------------------------ migrations (P0-06)
def test_migrations_cover_every_known_version_and_keep_unknown_fields():
    assert migration_path("0.0.0")[-1].endswith(SCHEMA_VERSION)
    assert migration_path(SCHEMA_VERSION) == []
    assert all(v in KNOWN_VERSIONS for v in MIGRATIONS)
    legacy = {"record_id": "scene_x", "video_id": "vid_a", "scene_id": "scene_001", "start": 1.0, "end": 3.0, "summary": "A legacy shot of a tree.", "environment": {}, "camera": {"movement": "static"}, "visual_style": {}, "validation": {"status": "accepted"}, "some_old_field": {"nested": 1}}
    migrated, steps = migrate_record("scenes", dict(legacy))
    assert steps[0] == "0.0.0->2.0.0" and migrated["schema_version"] == SCHEMA_VERSION
    assert migrated["start_time"] == 1.0 and migrated["end_time"] == 3.0 and migrated["duration"] == 2.0
    assert migrated["validation"]["policy_version"] is None and migrated["validation"]["history"] == []
    stats = UpgradeStats()
    out = upgrade_record("scenes", legacy, stats)
    assert out["schema_version"] == SCHEMA_VERSION and out["extensions"] == {"some_old_field": {"nested": 1}}
    assert stats.extensions == {"some_old_field": 1} and not stats.failed and stats.steps["0.0.0->2.0.0"] == 1
    # a record newer than the code is never downgraded
    newer = {**_scene(1), "schema_version": "9.9.9"}
    _m, steps2 = migrate_record("scenes", newer)
    assert steps2 == ["newer:9.9.9"]
    # unmigratable rows are kept verbatim and flagged, never dropped
    bad = {"record_id": "scene_bad", "video_id": "vid_a"}
    stats2 = UpgradeStats()
    kept = upgrade_record("scenes", bad, stats2)
    assert kept["record_id"] == "scene_bad" and kept["extensions"]["_schema_error"] and stats2.failed


def test_schema_gate_rejects_mixed_versions_and_unmigrated_rows():
    rows = {"scenes": [upgrade_record("scenes", _scene(0)), {**upgrade_record("scenes", _scene(1)), "schema_version": "2.2.0"}]}
    g = schema_gate(rows)
    assert g["state"] == "failed" and g["other_schema_versions"] == {"2.2.0": 1}
    rows["scenes"][1]["schema_version"] = SCHEMA_VERSION
    assert schema_gate(rows)["state"] == "passed"


# ------------------------------------------------------------------ export-time checks (P1-02, P1-04, P1-06, P1-11)
def test_export_checks_fill_temporal_validation_and_type_checks():
    records = {
        "scenes": [_scene(0), _scene(1), _scene(2, status="review"), _scene(3, start=20.0, end=19.0)],
        "clips": [_clip(0), {**_clip(1), "start_time": 1.0, "end_time": 5.0}],  # second clip pokes out of its scene
        "frames": [_frame(0)],
        "events": [_event(1, start=0.0, end=1.0), _event(2, start=3.0, end=4.0)],
        "temporal_qa": [_qa(1, ev_start=0.0, ev_end=2.0, event_ids=["event_0001"]), _qa(2, ev_start=0.0, ev_end=2.0, event_ids=["event_0002"])],
        "sequences": [{"record_id": "sequence_vid_a_seq_001", "video_id": "vid_a", "sequence_id": "seq_001", "scene_ids": ["scene_000", "scene_001"], "clip_ids": ["clip_000"], "start_time": 0.0, "end_time": 4.0, "duration": 4.0, "n_shots": 2, "summary": "two shots", "validation": {"status": "accepted", "checks": {}, "issues": [], "reason_codes": [], "policy_version": POLICY_VERSION}},
                      {"record_id": "sequence_vid_a_seq_002", "video_id": "vid_a", "sequence_id": "seq_002", "scene_ids": ["scene_001", "scene_000", "scene_099"], "clip_ids": [], "start_time": 0.0, "end_time": 4.0, "duration": 4.0, "n_shots": 3, "summary": "bad", "validation": {"status": "accepted", "checks": {}, "issues": [], "reason_codes": [], "policy_version": POLICY_VERSION}}],
        "video_descriptions": [{"record_id": "videodesc_vid_a_00", "video_id": "vid_a", "start_time": 0.0, "end_time": 4.0, "scene_ids": ["scene_000", "scene_001"], "temporal_progression": "x", "prompt": "p", "shots": [{"scene_id": "scene_000", "start_time": 0.0, "end_time": 2.0}, {"scene_id": "scene_001", "start_time": 2.0, "end_time": 4.0}], "evidence": {"scene_ids": ["scene_000"], "clip_ids": ["clip_000"], "frame_ids": ["frame_000_000"]}, "claims": [{"field": "camera", "status": "supported"}], "validation": {"status": "accepted", "checks": {}, "issues": [], "reason_codes": [], "policy_version": POLICY_VERSION}}],
        "long_video_qa": [],
    }
    counts = fill_export_checks(records, {"vid_a": 10.0})
    tv = {r["record_id"]: r["temporal_validation"] for rows in records.values() for r in rows}
    assert tv["scene_vid_a_scene_000"]["status"] == "passed" and tv["scene_vid_a_scene_000"]["checks"]["within_duration"] == "passed"
    assert tv["scene_vid_a_scene_003"]["status"] == "failed" and tv["scene_vid_a_scene_003"]["checks"]["ordered"] == "failed"
    assert tv["clip_vid_a_clip_001"]["checks"]["inside_parent"] == "failed" and tv["clip_vid_a_clip_000"]["checks"]["inside_parent"] == "passed"
    assert tv["frame_vid_a_frame_000_000"]["checks"]["inside_parent"] == "passed"
    assert records["scenes"][3]["validation"]["checks"]["timestamps"] == "failed"
    seq_ok, seq_bad = records["sequences"]
    assert all(seq_ok["validation"]["checks"][c] in ("passed", "not_applicable") for c in REQUIRED_CHECKS["sequence"])
    assert seq_bad["validation"]["checks"]["scene_membership"] == "failed" and seq_bad["validation"]["checks"]["scene_order"] == "failed"
    desc = records["video_descriptions"][0]["validation"]["checks"]
    assert desc["scenes_exist"] == "passed" and desc["evidence_resolves"] == "passed" and desc["shots_cover_span"] == "passed" and desc["claims_not_contradicted"] == "passed"
    q1, q2 = records["temporal_qa"]
    assert q1["validation"]["checks"]["evidence_covers_events"] == "passed" and q2["validation"]["checks"]["evidence_covers_events"] == "failed"
    assert counts["qa_evidence_failed"] == 1 and counts["sequence_checks_failed"] == 1
    # the policy turns the failed required checks into review at export, with a traced status change
    annotate_records(records, CFG.export, _video_info("vid_a"))
    assert q2["validation"]["status"] == "review" and "failed_check:evidence_covers_events" in q2["validation"]["reason_codes"]
    assert q2["validation"]["history"][-1]["actor"] == "export_policy" and q2["validation"]["history"][-1]["from"] == "accepted"
    # (annotate_records rebuilds the sequence list from the scenes; the hand-made ones are judged directly)
    assert enforce_exported(seq_bad) and seq_bad["validation"]["status"] == "review" and "failed_check:scene_membership" in seq_bad["validation"]["reason_codes"]
    assert enforce_exported(seq_ok) == [] and seq_ok["validation"]["status"] == "accepted"
    assert all(s["validation"]["checks"].get("scene_order") in ("passed", "not_applicable") for s in records["sequences"])
    g = temporal_gate(records)
    assert g["state"] == "passed" and g["failed"] >= 2 and g["accepted_with_failed_timestamps"] == 0


# ------------------------------------------------------------------ grounding methods (P1-01)
def test_grounding_always_names_its_method_and_is_never_copied_from_confidence():
    from video_dataset.schemas.events import Event

    q = QualityScore.from_components(grounding=None, description_quality=0.5)
    assert q.grounding_method == "not_assessed" and q.scoring_version
    e = Event(event_id="event_0001", video_id="vid_a", event_type="text_on_screen", start_time=0.0, end_time=1.0, event="Text 'EXIT' appears.", source="ocr", confidence=0.7, confidence_source="ocr_score")
    assert event_quality(e).grounding == 0.7 and event_quality(e).grounding_method == "instrument:ocr_score"
    e2 = e.model_copy(update={"confidence_source": "model_self_report"})
    assert event_quality(e2).grounding is None and event_quality(e2).grounding_method == "not_assessed"
    rq = relation_quality(0.99, [event_quality(e), event_quality(e2)])
    assert rq.grounding is None  # one event ungrounded -> the relation's grounding is not the interval confidence
    rq2 = relation_quality(0.99, [event_quality(e), event_quality(e)])
    assert rq2.grounding == 0.7 and rq2.grounding_method == "derived_min"


# ------------------------------------------------------------------ camera ambiguity + motion entities (P1-09, P1-10, P2-03)
def _analysis(label: str, flow_label: str, consistency: float) -> SceneAnalysis:
    return SceneAnalysis(scene_id="scene_001", video_id="vid_a", start_time=0.0, end_time=2.0, summary="x", camera=CameraAnnotation(movement=label), measurements=Measurements(camera_motion_label=flow_label, camera_motion_consistency=consistency))


def test_camera_motion_agreement_marks_lookalikes_ambiguous_and_fails_them():
    a = _analysis("zoom_in", "zoom_in", 0.9)
    assert camera_motion_agreement(a)[0] == "passed" and not a.camera.movement_ambiguous
    a = _analysis("dolly_in", "zoom_in", 0.9)
    state, detail = camera_motion_agreement(a)
    assert state == "failed" and a.camera.movement_ambiguous and a.camera.movement_alternatives == ["zoom_in"] and "ambiguous" in (detail or "")
    a = _analysis("pan_left", "pan_left", 0.3)
    assert camera_motion_agreement(a, 0.6)[0] == "failed" and a.camera.movement_ambiguous
    a = _analysis("tracking_left", "pan_left", 0.9)
    assert camera_motion_agreement(a)[0] == "failed" and a.camera.movement_ambiguous
    a = _analysis("unknown", "pan_left", 0.9)
    assert camera_motion_agreement(a)[0] == "not_applicable"


def test_subject_motion_keeps_world_camera_subject_apart_and_never_promotes_from_subject_alone():
    from video_dataset.schemas.vision import SubjectMotion

    assert world_motion((0.2, 0.0))["label"] == "streams_right" and world_motion((0.0, 0.0))["label"] == "static"
    a = _analysis("pan_left", "pan_left", 0.9)
    sm = SubjectMotion(subject="person", label="tracked_by_camera", camera_relation="tracked", camera_velocity=[0.0, 0.0], box_velocity=[0.0, 0.0])
    apply_subject_motion(a, sm)
    assert str(a.camera.movement) == "pan_left"  # no background flow -> no promotion
    a = _analysis("pan_left", "pan_left", 0.9)
    sm = SubjectMotion(subject="person", label="tracked_by_camera", camera_relation="tracked", camera_velocity=[0.3, 0.0], box_velocity=[0.0, 0.0])
    apply_subject_motion(a, sm)
    assert str(a.camera.movement) == "tracking_left" and a.camera.camera_motion_source == "optical_flow+subject_track" and a.camera.movement_alternatives == ["pan_left"]
    assert normalize_stabilization("Steadicam") == "steadicam" and normalize_stabilization("car mount") == "vehicle_mount" and normalize_stabilization("???") == "unknown"
    cam = CameraAnnotation(movement="static", focal_length_mm=35)
    assert cam.stabilization == "static" and cam.stabilization_evidence == "movement_label" and cam.focal_length_source == "model_estimate" and cam.pose is None


# ------------------------------------------------------------------ rights, provenance, channel groups (P0-07, P2-11, P3-03)
def test_annotate_fills_rights_provenance_and_groups_by_channel():
    records = {"scenes": [_scene(0, "vid_a"), _scene(1, "vid_b"), _scene(2, "vid_c")], "clips": [_clip(0, "vid_a")]}
    info = {**_video_info("vid_a", "vid_b", channel="Same Creator"), **_video_info("vid_c", channel="Other")}
    info["vid_c"]["rights_evidence"] = {"status": "verified_cc_by", "verified_at": "2026-11-01", "evidence_url": "https://example.org/licence", "verified_by": "raj"}
    annotate_records(records, CFG.export, info, CFG.licensing, provenance_base={"pipeline_version": "0.1.0", "git_commit": "abc123"})
    a, b, c = records["scenes"]
    assert a["source_group_id"] == b["source_group_id"] == channel_group_id("Same Creator") and a["split"] == b["split"]
    assert c["source_group_id"] == channel_group_id("Other")
    assert a["rights"]["status"] == "declared_cc_by" and a["rights"]["method"] == "platform_declared" and a["rights"]["redistribution"] == "allowed" and a["rights"]["evidence_url"].startswith("https://youtube.com")
    assert c["rights"]["status"] == "verified_cc_by" and c["rights"]["method"] == "human_evidence" and c["rights"]["verified_at"] == "2026-11-01"
    assert a["provenance"]["git_commit"] == "abc123" and a["provenance"]["policy_version"] == POLICY_VERSION and a["provenance"]["template_version"]
    assert a["confidence_calibration"] == "uncalibrated" and a["continuity"]["next_scene_id"] is None
    assert not split_leaks(records)
    assert rights_info_for({"rights_status": "restricted"})["allowed_use"] == RIGHTS_TERMS["restricted"][0]
    cfg_no_channel = CFG.export.model_copy(update={"split": CFG.export.split.model_copy(update={"group_by_channel": False})})
    records2 = {"scenes": [_scene(0, "vid_a"), _scene(1, "vid_b")]}
    annotate_records(records2, cfg_no_channel, _video_info("vid_a", "vid_b", channel="Same Creator"))
    assert records2["scenes"][0]["source_group_id"] == "vid_a"


# ------------------------------------------------------------------ asset registry + rights-gated media (P0-02, P0-03, P0-07)
def test_media_manifest_has_asset_ids_and_media_status(tmp_path: Path):
    data = tmp_path / "data"
    (data / "clips" / "vid_a").mkdir(parents=True)
    (data / "clips" / "vid_a" / "clip_000.mp4").write_bytes(b"x" * 10)
    records = {"clips": [_clip(0), _clip(1)], "frames": [_frame(0)]}
    rows = build_media_manifest(records, data, rights={"vid_a": "declared_cc_by"})
    by = {r["path"]: r for r in rows}
    assert by["clips/vid_a/clip_000.mp4"]["media_status"] == "available" and by["clips/vid_a/clip_001.mp4"]["media_status"] == "missing"
    assert by["clips/vid_a/clip_000.mp4"]["asset_id"] == asset_id_for("vid_a", "clips/vid_a/clip_000.mp4") and by["clips/vid_a/clip_000.mp4"]["rights_status"] == "declared_cc_by"
    assert by["clips/vid_a/clip_000.mp4"]["start_time"] == 0.0 and by["clips/vid_a/clip_000.mp4"]["end_time"] == 2.0
    apply_asset_ids(records, rows)
    assert records["clips"][0]["asset_id"].startswith("asset_") and records["clips"][1]["media_status"] == "missing" and records["frames"][0]["media_status"] == "missing"


def test_stage_shard_ships_media_only_for_rights_cleared_videos(tmp_path: Path):
    final, data = tmp_path / "final", tmp_path / "data"
    final.mkdir()
    for vid in ("vid_cc", "vid_std"):
        (data / "clips" / vid).mkdir(parents=True)
        (data / "clips" / vid / "clip_000.mp4").write_bytes(b"x")
        (data / "frames" / vid).mkdir(parents=True)
        (data / "frames" / vid / "f.jpg").write_bytes(b"x")
    (final / "provenance.jsonl").write_text('{"video_id": "vid_cc", "rights_status": "declared_cc_by"}\n{"video_id": "vid_std", "rights_status": "restricted"}\n')
    (final / "clips.jsonl").write_text("")
    cleared, status = rights_cleared_videos(final, ["vid_cc", "vid_std"], ["declared_cc_by"])
    assert cleared == ["vid_cc"] and status["vid_std"] == "restricted"
    staging = tmp_path / "staging"
    stage_shard(final, data, ["vid_cc", "vid_std"], True, staging, release_rights=["declared_cc_by"], rights_gate=True)
    assert (staging / "clips" / "vid_cc" / "clip_000.mp4").exists() and not (staging / "clips" / "vid_std").exists()
    gate = json.loads((staging / "media_rights.json").read_text())
    assert gate["media_blocked_for"] == {"vid_std": "restricted"}
    staging2 = tmp_path / "staging2"
    stage_shard(final, data, ["vid_cc", "vid_std"], True, staging2, release_rights=["declared_cc_by"], rights_gate=False)
    assert (staging2 / "clips" / "vid_std").exists()


def test_verify_remote_checksums_reports_mismatch_and_missing():
    class Up:
        def remote_checksums(self, path_in_repo):
            return {"a.parquet": "aa", "b.parquet": "bb", "c.json": None}

    res = verify_remote_checksums(Up(), "shard_0001", {"a.parquet": "aa", "b.parquet": "xx", "c.json": "cc", "d.jsonl": "dd"})
    assert res["verified"] == 1 and res["mismatched"] == ["b.parquet"] and res["missing_on_hub"] == ["d.jsonl"] and res["unverifiable"] == 1 and not res["ok"]


# ------------------------------------------------------------------ clip integrity, transcript alignment (P1-13, P2-08)
def test_clip_integrity_checks():
    assert frame_count_check({"frame_count": 60, "duration": 2.0, "fps": 30.0}) == "passed"
    assert frame_count_check({"frame_count": 50, "duration": 2.0, "fps": 30.0}) == "failed"
    assert frame_count_check({"frame_count": None, "duration": 2.0, "fps": 30.0}) == "not_run"
    assert stream_integrity({"has_video": True, "has_audio": True, "duration": 2.0}, True) == "passed"
    assert stream_integrity({"has_video": True, "has_audio": False, "duration": 2.0}, True) == "failed"
    assert stream_integrity({"has_video": None}, False) == "not_run"


def test_transcript_alignment_validation():
    t = Transcript(video_id="vid_a", provider="mock", segments=[
        TranscriptSegment(segment_id="seg_0001", start=0.0, end=1.0, text="hi", words=[Word(start=0.0, end=0.5, word="hi", probability=0.9)]),
        TranscriptSegment(segment_id="seg_0002", start=9.9, end=10.1, text="bye", words=[Word(start=9.9, end=10.1, word="bye", probability=0.5)]),
        TranscriptSegment(segment_id="seg_0003", start=12.0, end=13.0, text="late"),
    ])
    counts = validate_alignment(t, 10.0, 0.25)
    assert counts == {"within_audio": 1, "clamped": 1, "outside_audio": 1, "not_checked": 0}
    assert t.segments[1].end == 10.0 and t.segments[1].words[0].end == 10.0 and t.segments[1].alignment_status == "clamped"
    assert t.segments[0].alignment_confidence == 0.9 and t.segments[2].alignment_confidence is None and t.segments[0].speaker_id is None
    assert t.diarization == "not_run" and t.audio_duration == 10.0


# ------------------------------------------------------------------ status history (P1-03)
def test_record_status_change_keeps_history():
    rec = {"validation": {"status": "accepted"}}
    entry = record_status_change(rec, "review", "policy:missing_check:x", "export_policy")
    assert entry and rec["validation"]["status"] == "review" and rec["validation"]["history"][0]["from"] == "accepted"
    assert record_status_change(rec, "review", "again", "export_policy") is None


# ------------------------------------------------------------------ release status + new gates (section 7)
def _final(tmp_path: Path, records: dict, *, with_inventory: bool = True) -> Path:
    final = tmp_path / "final"
    full = {name: list(records.get(name, [])) for name in DATASET_FILES}
    counts = write_jsonl_files(full, final)
    typed = write_typed_parquet_files(full, final)
    vids = sorted({r["video_id"] for rows in full.values() for r in rows})
    write_json_atomic(final / "manifest.json", {"release_id": "r1", "schema_version": SCHEMA_VERSION, "policy_version": POLICY_VERSION, "videos": [{"video_id": v, "channel": "c", "rights_status": "declared_cc_by"} for v in vids], "counts": counts, "typed_parquet_counts": typed, "splits": {}, "environment": {"code": {"git_commit": "f" * 40}, "models": {"vision": {}}}})
    write_json_atomic(final / "statistics.json", {"jsonl_counts": counts, **{n: counts[n] for n in DATASET_FILES}})
    by_split: dict[str, set[str]] = {}
    for rows in full.values():
        for r in rows:
            if r.get("split"):
                by_split.setdefault(r["split"], set()).add(r["video_id"])
    write_json_atomic(final / "splits.json", {"videos": {k: sorted(v) for k, v in by_split.items()}, "counts": {k: len(v) for k, v in by_split.items()}})
    if with_inventory:
        write_inventory(final, include_media=False)
    return final


def test_release_status_is_tri_state_with_exclusions(tmp_path: Path):
    records = {"scenes": [_scene(i, f"vid_{i}") for i in range(4)], "clips": [_clip(i, f"vid_{i}") for i in range(4)]}
    annotate_records(records, CFG.export, _video_info(*[f"vid_{i}" for i in range(4)]), CFG.licensing, provenance_base={"git_commit": "abc"})
    final = _final(tmp_path, records)
    report = run_release_check(CFG, final, require_media=False)
    names = {g["gate"]: g["state"] for g in report["gates"]}
    assert {"checksums", "timestamps", "rights", "reproducibility"} <= set(names) and all(names[g] == "passed" for g in ("checksums", "timestamps", "rights", "reproducibility"))
    # no media on disk: disclosed as an exclusion (metadata-only release), never hidden and never a failure
    assert report["status"] == "passed_with_exclusions" and report["ok"] and report["release_id"] == "r1"
    assert report["exclusions"]["missing_media_assets"] == 4 and report["exclusions"]["review_records_excluded_from_strict"] == 0
    assert report["metrics"]["semantic_verification_split"]["scenes"]["heuristic"] == 4 and report["metrics"]["grounding_methods"]["scenes"]
    assert report["human_verification"]["status"] == "not_run"
    # a review record is an exclusion, not a failure
    records["scenes"].append(_scene(9, "vid_0", status="review"))
    records["scenes"][-1]["camera"] = {"movement": "zoom_in", "movement_ambiguous": True, "movement_alternatives": ["dolly_in"]}
    annotate_records(records, CFG.export, _video_info(*[f"vid_{i}" for i in range(4)]), CFG.licensing)
    final = _final(tmp_path / "b", records)
    report = run_release_check(CFG, final, require_media=False)
    assert report["status"] == "passed_with_exclusions" and report["exclusions"]["review_by_type"]["scenes"] == 1 and report["exclusions"]["camera_ambiguous"] == 1
    # a tampered file blocks the release
    (final / "scenes.jsonl").write_text((final / "scenes.jsonl").read_text() + "\n")
    report = run_release_check(CFG, final, require_media=False)
    assert report["status"] == "blocked" and "checksums" in report["failed_gates"]
    assert verify_checksums(final)["mismatched"] == ["scenes.jsonl"]


def test_new_gates_fail_closed(tmp_path: Path):
    final = tmp_path / "empty"
    final.mkdir()
    assert checksums_gate(final)["state"] == "failed"
    assert reproducibility_gate(final)["state"] == "failed"
    write_json_atomic(final / "manifest.json", {"release_id": "x", "schema_version": "1.0.0", "policy_version": POLICY_VERSION, "environment": {"code": {}, "models": {}}})
    g = reproducibility_gate(final)
    assert g["state"] == "failed" and any("schema_version" in p for p in g["problems"]) and any("git commit" in p for p in g["problems"])
    records = {"scenes": [{**_scene(0), "rights": None}]}
    g = rights_gate(CFG, records, [{"video_id": "vid_a", "rights_status": "restricted"}], final, media_shipped=False)
    assert g["state"] == "failed" and g["records_without_rights_block"] == 1
    (final / "clips" / "vid_a").mkdir(parents=True)
    records = {"scenes": [{**_scene(0), "rights": rights_info_for({"rights_status": "restricted"})}]}
    g = rights_gate(CFG, records, [{"video_id": "vid_a", "rights_status": "restricted"}], final, media_shipped=True)
    assert g["state"] == "failed" and g["media_rights_violations"] == [{"video_id": "vid_a", "rights_status": "restricted"}]
    accepted_bad = {"scenes": [{**_scene(0), "temporal_validation": {"status": "failed", "checks": {"ordered": "failed"}}}]}
    assert temporal_gate(accepted_bad)["state"] == "failed"
    ex = exclusions_block({"clips": [{**_clip(0), "boundary_check": "failed"}]}, [], {})
    assert ex["residual_boundary_failure_count"] == 1 and ex["any"]


# ------------------------------------------------------------------ descriptions carry evidence (P1-05, P1-15) via the export schema
def test_video_description_record_carries_evidence_claims_and_unassessed_grounding():
    rec = {"record_id": "videodesc_vid_a_00", "video_id": "vid_a", "start_time": 0.0, "end_time": 4.0, "scene_ids": ["scene_000"], "temporal_progression": "x", "prompt": "p", "shots": [], "evidence": {"scene_ids": ["scene_000"], "clip_ids": [], "frame_ids": [], "timestamps": [[0.0, 4.0]]}, "claims": [{"field": "camera", "value": "pan", "source": "optical_flow", "status": "supported"}], "facts": {"observed": ["camera=pan (scene_000)"], "inferred": []}, "quality": {"grounding": None, "grounding_method": "not_assessed"}}
    out = upgrade_records({"video_descriptions": [rec]})["video_descriptions"][0]
    assert out["evidence"]["scene_ids"] == ["scene_000"] and out["grounding_assessed"] is False and out["quality"]["grounding_method"] == "not_assessed" and out["facts"]["observed"]
    # a legacy description gets the slots with explicit "not assessed" values, never a copied score
    legacy = {"record_id": "videodesc_vid_a_01", "video_id": "vid_a", "start_time": 0.0, "end_time": 4.0, "scene_ids": ["scene_000"], "temporal_progression": "x", "prompt": "p", "schema_version": "2.2.0", "quality": {"grounding": 0.8, "overall": 0.8, "components_used": ["grounding"]}}
    out2 = upgrade_records({"video_descriptions": [legacy]})["video_descriptions"][0]
    assert out2["evidence"]["scene_ids"] == ["scene_000"] and out2["schema_version"] == SCHEMA_VERSION and out2["quality"]["grounding_method"] is None


def test_frames_keep_grounded_ocr_detections():
    frame = {**_frame(0), "ocr_detections": [{"detection_id": "det_00001", "text": "EXIT", "bbox": [0.1, 0.2, 0.3, 0.4], "bbox_units": "normalized", "timestamp": 0.5, "confidence": 0.9, "engine": "rapidocr", "engine_version": "3.0.0"}]}
    out = upgrade_records({"frames": [frame]})["frames"][0]
    assert out["ocr_detections"][0]["engine_version"] == "3.0.0" and out["ocr_detections"][0]["bbox_units"] == "normalized"
