"""Regression tests for the 23-issue backlog (YT_A_23_Issues_and_Solutions.md).

Each test names the issue it covers. Nothing here needs a network, an API key or a generative model.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from video_dataset.config import load_config
from video_dataset.dataset.calibration import calibrate
from video_dataset.dataset.coverage import coverage_rates, stage_coverage
from video_dataset.dataset.curation import annotate_records, duration_band, rights_status_for, split_leaks
from video_dataset.dataset.export import write_typed_parquet_files
from video_dataset.dataset.media_manifest import build_media_manifest, summarize
from video_dataset.dataset.negatives import caption_hard_negatives, qa_hard_negatives
from video_dataset.dataset.release import (
    parquet_gate,
    reference_gate,
    run_release_check,
    schema_gate,
    split_gate,
    strict_gate,
)
from video_dataset.dataset.review import (
    apply_review_labels,
    export_review_queue,
    labels_for_calibration,
    read_review_labels,
)
from video_dataset.dataset.stats import compute_statistics
from video_dataset.dataset.takedown import remove_source, removed_sources
from video_dataset.frame_sampling.boundary import check_clip_boundaries
from video_dataset.frame_sampling.clips import _cut, extract_clips
from video_dataset.schemas.common import SCHEMA_VERSION, ConfidenceSource
from video_dataset.schemas.events import Event, EventSource, EventType, RelationType, TemporalRelation
from video_dataset.schemas.qa import Evidence, QARecord, QAType
from video_dataset.schemas.quality import CheckState, ValidationStatus, check_state
from video_dataset.schemas.scene import Clip, Frame, ScanSample, Scene, SceneScan
from video_dataset.schemas.transcript import Transcript, TranscriptSegment, Word
from video_dataset.schemas.vision import (
    CameraAnnotation,
    CameraMovement,
    Environment,
    Measurements,
    ObjectAnnotation,
    SceneAnalysis,
    SubjectMotion,
)
from video_dataset.stages import Stage, StageStatus
from video_dataset.storage.state_db import StateDB
from video_dataset.temporal.events import EventExtractor, motion_span
from video_dataset.transcription.align import segments_in_window
from video_dataset.utils.environment import environment_manifest
from video_dataset.utils.io import read_json, read_jsonl, write_json_atomic, write_jsonl
from video_dataset.validation.claims import claim_summary, scene_claims
from video_dataset.validation.policy import POLICY_VERSION, decide, strict_ok
from video_dataset.validation.qa_checks import recompute_answer
from video_dataset.validation.quality import scene_quality_flags
from video_dataset.validation.relations import before_cycles, relation_holds
from video_dataset.validation.stage import camera_motion_agreement
from video_dataset.validation.text_checks import (
    TextChecker,
    answer_type_for,
    normalize_caption,
    text_contradictions,
)

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config()


def _cfg(tmp_path: Path, **over):
    base = {"project.data_dir": str(tmp_path / "data"), "project.log_level": "ERROR"}
    base.update(over)
    return load_config(None, base, config_dir=ROOT / "config")


# ------------------------------------------------------------------ issue 15 / 2: policy fails closed
def test_policy_missing_or_failed_required_check_is_never_a_silent_pass():
    full = {k: True for k in ("timestamps", "scene_exists", "frames_exist", "frame_files_exist", "summary_present", "frame_sequence", "visual_quality", "consistent", "text_consistent", "single_shot")}
    status, reasons = decide("scene", full, [], "accepted", [], "measured")
    assert status == ValidationStatus.ACCEPTED and reasons == []
    status, reasons = decide("scene", {**full, "frame_sequence": CheckState.NOT_RUN.value}, [], "accepted", [], "measured")
    assert status == ValidationStatus.REVIEW and "missing_check:frame_sequence" in reasons
    status, reasons = decide("scene", {k: v for k, v in full.items() if k != "consistent"}, [], "accepted", [], "measured")
    assert status == ValidationStatus.REVIEW and "missing_check:consistent" in reasons
    status, reasons = decide("scene", {**full, "visual_quality": False}, [], "accepted", [], "measured")
    assert status == ValidationStatus.REVIEW and "failed_check:visual_quality" in reasons
    status, reasons = decide("scene", full, [], "accepted", [], CheckState.NOT_RUN.value)  # generated text nobody verified
    assert status == ValidationStatus.REVIEW and "semantic_verification:not_run" in reasons
    status, reasons = decide("scene", full, ["bad timestamps"], "accepted", [], "measured")
    assert status == ValidationStatus.REJECTED and reasons == ["hard:bad timestamps"]
    # checks that only exist when the record has clips are tolerated when absent, not when not_run
    status, _ = decide("scene", {**full, "clip_boundaries": CheckState.NOT_APPLICABLE.value}, [], "accepted", [], "measured")
    assert status == ValidationStatus.ACCEPTED
    status, reasons = decide("scene", {**full, "clip_boundaries": CheckState.NOT_RUN.value}, [], "accepted", [], "measured")
    assert status == ValidationStatus.REVIEW and "missing_check:clip_boundaries" in reasons
    assert check_state(True) == "passed" and check_state(False) == "failed" and check_state(None) == "not_applicable" and check_state("not_run") == "not_run" and check_state("weird") == "error"


def test_strict_ok_requires_accepted_passed_checks_and_semantic_verification():
    checks = {k: True for k in ("timestamps", "scene_exists", "frames_exist", "frame_files_exist", "summary_present", "frame_sequence", "visual_quality", "consistent", "text_consistent", "single_shot")}
    rec = {"record_type": "scene", "generation_target": "single_shot", "validation": {"status": "accepted", "checks": checks, "semantic_verification": "measured", "policy_version": POLICY_VERSION}}
    assert strict_ok(rec) == (True, [])
    ok, why = strict_ok({**rec, "validation": {**rec["validation"], "policy_version": None}})
    assert not ok and why == ["policy_version:missing"]  # records validated before the policy existed never enter strict
    ok, why = strict_ok({**rec, "validation": {**rec["validation"], "status": "review"}})
    assert not ok and why == ["status_review"]
    ok, why = strict_ok({**rec, "validation": {**rec["validation"], "semantic_verification": "not_run"}})
    assert not ok and why == ["semantic_verification:not_run"]
    ok, why = strict_ok({**rec, "generation_target": "not_for_generation"})
    assert not ok and why == ["not_for_generation"]


# ------------------------------------------------------------------ issue 1: claim-level evidence
def _analysis(provider="cv_models", movement=CameraMovement.PAN_LEFT, src=ConfidenceSource.MEASUREMENT):
    return SceneAnalysis(
        scene_id="scene_001", video_id="v", start_time=0.0, end_time=4.0, summary="A 4.0-second shot with dim lighting; the camera pans left. Detected objects: a car.",
        environment=Environment(lighting="dim", setting="outdoor"), objects=[ObjectAnnotation(name="car", confidence=0.8, confidence_source=ConfidenceSource.DETECTOR_SCORE, frame_fraction=1.0)],
        actions=["Camera pans left"], camera=CameraAnnotation(movement=movement, confidence=0.9, confidence_source=src),
        measurements=Measurements(lighting_level="dim", brightness_mean=60.0, camera_motion_label="pan_left", camera_motion_consistency=0.9),
        frame_ids=[f"frame_001_{i:03d}" for i in range(6)], provider=provider, enrichments={"clip": {"setting": "outdoor"}},
    )


def test_claims_record_provenance_status_and_evidence_frames():
    claims = scene_claims(_analysis())
    by_field = {c["field"]: c for c in claims}
    assert by_field["camera.movement"]["status"] == "supported" and by_field["camera.movement"]["source"] == "optical_flow" and len(by_field["camera.movement"]["evidence_frame_ids"]) == 4
    assert by_field["subject"]["status"] == "supported" and by_field["subject"]["source"] == "detector"
    assert by_field["lighting"]["status"] == "supported" and by_field["environment.setting"]["source"] == "clip_zero_shot"
    gen = scene_claims(_analysis(provider="mock", src=ConfidenceSource.MODEL_SELF_REPORT))
    assert {c["status"] for c in gen if c["field"] == "camera.movement"} == {"not_checked"}
    bad = scene_claims(_analysis(), contradictions=["camera labelled pan_left but optical flow measures pan_right"])
    assert {c["status"] for c in bad if c["field"] == "camera.movement"} == {"contradicted"}
    unknown = scene_claims(_analysis(movement=CameraMovement.UNKNOWN))
    assert {c["status"] for c in unknown if c["field"] == "camera.movement"} == {"unknown"}
    assert claim_summary(claims)["supported"] >= 4


# ------------------------------------------------------------------ issue 3: camera label vs measured motion
def test_camera_motion_agreement_is_an_explicit_check():
    assert camera_motion_agreement(_analysis())[0] == "passed"
    assert camera_motion_agreement(_analysis(movement=CameraMovement.UNKNOWN))[0] == "not_applicable"
    state, detail = camera_motion_agreement(_analysis(movement=CameraMovement.STATIC))
    assert state == "failed" and "pan_left" in (detail or "")
    assert camera_motion_agreement(_analysis(movement=CameraMovement.TRACKING_LEFT))[0] == "passed"  # tracking refines a pan


# ------------------------------------------------------------------ issue 6 / 13: text structure
def test_text_checks_find_contradictions_and_malformed_questions():
    assert text_contradictions(["The camera pans left.", "Later the camera pans right."]) and text_contradictions(["The camera is static while it zooms in."])
    assert text_contradictions(["The camera pans left at a slow pace."]) == []
    assert normalize_caption("A  shot ,with  odd   spacing ...") == "A shot, with odd spacing."
    t = TextChecker()
    assert t.question_form("What happens right after the car enters the tunnel?")[0] == "passed"
    state, problems = t.question_form("the car enters")
    assert state == "failed" and "does not end with '?'" in problems[0]
    assert t.answer_matches_type("About 6 seconds (from 1.0s to 7.0s).", "duration")[0] == "passed"
    assert t.answer_matches_type("The car enters the tunnel.", "duration")[0] == "failed"
    assert t.answer_matches_type("Around 12 seconds (starts at 12.3s).", "time")[0] == "passed"
    assert t.answer_matches_type("First the car enters, then it leaves.", "order")[0] == "passed"
    assert t.answer_matches_type("3 shots (starting around 1s, 30s, 60s).", "count")[0] == "passed"
    assert t.answer_matches_type("", "description")[0] == "failed"
    assert answer_type_for("duration") == "duration" and answer_type_for("long_range", "count_v1") == "count" and answer_type_for("multi_event", "between_v1") == "description"


# ------------------------------------------------------------------ issue 13: independent answer recomputation
def _ev(i, start, end, text):
    return Event(event_id=f"event_{i:04d}", video_id="v", event_type=EventType.ACTION, start_time=start, end_time=end, event=text, source=EventSource.VISION)


def _q(qtype, question, answer, events, template="t"):
    return QARecord(question_id="qa_v_000001", video_id="v", type=qtype, question=question, answer=answer, template_id=template, evidence=Evidence(start_time=events[0].start_time, end_time=events[-1].end_time, event_ids=[e.event_id for e in events]))


def test_recompute_answer_from_evidence_events():
    a, b = _ev(1, 2.0, 5.0, "The car enters the tunnel."), _ev(2, 7.0, 9.0, "The car leaves the tunnel.")
    evs = {a.event_id: a, b.event_id: b}
    assert recompute_answer(_q(QAType.DURATION, "How long?", "About 3 seconds (from 2.0s to 5.0s).", [a]), evs)[0] == "passed"
    assert recompute_answer(_q(QAType.DURATION, "How long?", "About 3 seconds (from 2.0s to 6.0s).", [a]), evs)[0] == "failed"
    assert recompute_answer(_q(QAType.EVENT_LOCALIZATION, "When?", "Around 7 seconds (starts at 7.0s).", [b]), evs)[0] == "passed"
    assert recompute_answer(_q(QAType.EVENT_LOCALIZATION, "When?", "Around 9 seconds (starts at 9.0s).", [b]), evs)[0] == "failed"
    assert recompute_answer(_q(QAType.TEMPORAL_ORDERING, "Which first?", "First the car enters the tunnel, then the car leaves the tunnel.", [a, b]), evs)[0] == "passed"
    assert recompute_answer(_q(QAType.TEMPORAL_ORDERING, "Which first?", "First the car leaves the tunnel, then the car enters the tunnel.", [a, b]), evs)[0] == "failed"
    q = "Does the moment when the car leaves the tunnel come before or after the moment when the car enters the tunnel?"
    assert recompute_answer(_q(QAType.TEMPORAL_ORDERING, q, "After. The car enters the tunnel around 2.0s, and the car leaves the tunnel around 7.0s.", [a, b]), evs)[0] == "passed"
    assert recompute_answer(_q(QAType.TEMPORAL_ORDERING, q, "Before. The car enters the tunnel around 2.0s, and the car leaves the tunnel around 7.0s.", [a, b]), evs)[0] == "failed"
    assert recompute_answer(_q(QAType.LONG_RANGE, "How many?", "2 shots (starting around 2s, 7s).", [a, b], "count_v1"), evs)[0] == "passed"
    assert recompute_answer(_q(QAType.LONG_RANGE, "How many?", "3 shots (starting around 2s, 7s).", [a, b], "count_v1"), evs)[0] == "failed"
    assert recompute_answer(_q(QAType.TIMESTAMP, "What happens?", "The car enters the tunnel.", [a]), evs)[0] == "not_applicable"
    assert recompute_answer(_q(QAType.DURATION, "How long?", "x", [a]), {})[0] == "not_run"


# ------------------------------------------------------------------ issue 11: relation semantics
def test_relation_semantics_and_cycles():
    a, b = _ev(1, 0.0, 2.0, "A"), _ev(2, 2.0, 5.0, "B")
    inner = _ev(3, 2.5, 3.0, "C")

    def rel(kind, x, y):
        return TemporalRelation(relation_id="r", video_id="v", event_a=x.event_id, event_b=y.event_id, relation=kind, event_a_text=x.event, event_b_text=y.event)

    assert relation_holds(rel(RelationType.BEFORE, a, b), a, b)[0]  # boundary equality counts as before
    assert not relation_holds(rel(RelationType.BEFORE, b, a), b, a)[0]
    assert relation_holds(rel(RelationType.AFTER, b, a), b, a)[0]
    assert relation_holds(rel(RelationType.DURING, inner, b), inner, b)[0] and not relation_holds(rel(RelationType.DURING, b, inner), b, inner)[0]
    o1, o2 = _ev(4, 0.0, 3.0, "O1"), _ev(5, 2.0, 6.0, "O2")
    assert relation_holds(rel(RelationType.OVERLAPS, o1, o2), o1, o2)[0] and not relation_holds(rel(RelationType.OVERLAPS, o2, o1), o2, o1)[0]
    s1, s2 = _ev(6, 2.0, 3.0, "S1"), _ev(7, 2.0, 5.0, "S2")
    assert relation_holds(rel(RelationType.STARTS, s1, s2), s1, s2)[0] and relation_holds(rel(RelationType.ENDS, _ev(8, 4.0, 5.0, "E"), s2), _ev(8, 4.0, 5.0, "E"), s2)[0]
    assert not relation_holds(rel(RelationType.INTERRUPTS, a, b), a, b)[0] and relation_holds(rel(RelationType.INTERRUPTS, inner, b), inner, b)[0]
    assert not relation_holds(rel(RelationType.CAUSES, b, a), b, a)[0]  # a cause cannot start after its effect
    cyc = before_cycles([rel(RelationType.BEFORE, a, b), rel(RelationType.BEFORE, b, inner), rel(RelationType.BEFORE, inner, a)])
    assert cyc and set(cyc[0]) >= {a.event_id, b.event_id, inner.event_id}
    assert before_cycles([rel(RelationType.BEFORE, a, b), rel(RelationType.BEFORE, b, inner)]) == []
    assert TemporalRelation.model_validate(rel(RelationType.BEFORE, a, b).model_dump()).evidence_kind == "measured"


# ------------------------------------------------------------------ issue 4: clip boundaries are measured
def test_clip_boundary_check_measures_keyframe_drift(synthetic_video: Path, tmp_path: Path):
    """The synthetic video has a keyframe every 0.5 s. A stream copy requested at 4.3 s starts at
    the keyframe before it; the boundary check must *measure* that, not trust ffmpeg's exit code."""
    exact = tmp_path / "exact.mp4"
    _cut(synthetic_video, exact, 4.3, 6.3, reencode=True, has_audio=True, clip_height=720, ffmpeg_path=None)
    res = check_clip_boundaries(synthetic_video, exact, 4.3, 6.3, 24.0, tolerance=0.1)
    assert res.state == "passed" and res.actual_start is not None and abs(res.actual_start - 4.3) <= 0.1 and abs(res.actual_end - 6.3) <= 0.1, res
    copied = tmp_path / "copy.mp4"
    _cut(synthetic_video, copied, 4.3, 6.3, reencode=False, has_audio=True, clip_height=720, ffmpeg_path=None)
    res = check_clip_boundaries(synthetic_video, copied, 4.3, 6.3, 24.0, tolerance=0.1)
    assert res.state in ("passed", "failed") and res.actual_start is not None
    assert abs(res.error_seconds - max(abs(res.actual_start - 4.3), abs(res.actual_end - 6.3))) < 1e-6
    if res.actual_start < 4.2:  # keyframe-aligned start was measured
        assert res.state == "failed" and "requested 4.300s" in (res.detail or "")
    assert check_clip_boundaries(synthetic_video, tmp_path / "missing.mp4", 0.0, 1.0, 24.0).state == "error"


def test_extract_clips_records_actual_boundaries(synthetic_video: Path, tmp_path: Path):
    cfg = _cfg(tmp_path, **{"frame_sampling.clip_codec": "copy", "frame_sampling.verify_clip_boundaries": "all", "frame_sampling.clip_tolerance_seconds": "0.1"}).frame_sampling
    scenes = [Scene(scene_id="scene_001", video_id="v", index=0, start_time=4.3, end_time=6.3, start_frame=103, end_frame=151)]
    clips, stats = extract_clips(synthetic_video, "v", scenes, [], tmp_path / "clips", cfg, True, workers=1, source_fps=24.0)
    assert len(clips) == 1 and stats.boundary_checked == 1
    c = clips[0]
    assert c.boundary_check in ("passed", "failed") and c.actual_start is not None and c.boundary_error_seconds is not None
    assert Clip.model_validate(c.model_dump()).boundary_check == c.boundary_check
    cfg2 = _cfg(tmp_path, **{"frame_sampling.clip_codec": "libx264", "frame_sampling.verify_clip_boundaries": "copy"}).frame_sampling
    clips2, stats2 = extract_clips(synthetic_video, "v", scenes, [], tmp_path / "clips2", cfg2, True, workers=1, source_fps=24.0)
    assert clips2[0].boundary_check == "not_run" and stats2.boundary_checked == 0  # copy mode skips re-encoded clips on purpose
    cfg3 = _cfg(tmp_path, **{"frame_sampling.clip_codec": "libx264", "frame_sampling.verify_clip_boundaries": "all", "frame_sampling.clip_tolerance_seconds": "0.1"}).frame_sampling
    clips3, _ = extract_clips(synthetic_video, "v", scenes, [], tmp_path / "clips3", cfg3, True, workers=1, source_fps=24.0)
    assert clips3[0].boundary_check == "passed", clips3[0]


# ------------------------------------------------------------------ issue 9: transcript windows
def test_transcript_segments_are_clipped_to_the_window_with_both_time_bases():
    words = [Word(start=10.0 + 0.5 * i, end=10.4 + 0.5 * i, word=w) for i, w in enumerate(["one", "two", "three", "four", "five", "six"])]
    seg = TranscriptSegment(segment_id="seg_0001", start=10.0, end=12.9, text="one two three four five six", words=words, confidence=0.8, confidence_source=ConfidenceSource.ASR_LOGPROB)
    t = Transcript(video_id="v", provider="mock", language="en", segments=[seg, TranscriptSegment(segment_id="seg_0002", start=20.0, end=21.0, text="later")])
    segs, scope = segments_in_window(t, 11.0, 12.0)
    assert scope == "word_clipped" and len(segs) == 1
    s = segs[0]
    assert s["text"] == "three four" and s["start"] == 11.0 and s["end"] == 11.9 and s["rel_start"] == 0.0 and s["rel_end"] == 0.9 and s["partial"] and s["word_clipped"] and s["speaker_visible"] is None
    no_words = Transcript(video_id="v", provider="mock", segments=[TranscriptSegment(segment_id="seg_0001", start=10.0, end=13.0, text="whole segment")])
    segs, scope = segments_in_window(no_words, 11.0, 12.0)
    assert scope == "segment_overlap" and segs[0]["partial"] and segs[0]["start"] == 11.0 and segs[0]["rel_end"] == 1.0
    assert segments_in_window(t, 30.0, 31.0) == ([], "none")
    assert segments_in_window(Transcript(video_id="v", provider="none", has_speech=False, status="no_audio"), 0, 1) == ([], "not_run")
    assert segments_in_window(None, 0, 1) == ([], "not_run")


# ------------------------------------------------------------------ issue 10: event timing + uncertainty
def test_events_carry_uncertainty_and_measured_action_timing():
    assert motion_span([[0.0, 0.5, 0.5, 0.1], [1.0, 0.5, 0.5, 0.1], [2.0, 0.6, 0.5, 0.1], [3.0, 0.7, 0.5, 0.1], [4.0, 0.7, 0.5, 0.1]]) == (1.0, 3.0)
    assert motion_span([[0.0, 0.5, 0.5, 0.1], [1.0, 0.5, 0.5, 0.1], [2.0, 0.5, 0.5, 0.1]]) is None
    scene = Scene(scene_id="scene_001", video_id="v", index=0, start_time=0.0, end_time=6.0, start_frame=0, end_frame=144)
    frames = [Frame(frame_id=f"frame_001_{i:03d}", video_id="v", scene_id="scene_001", timestamp=float(i), frame_index=24 * i, frame_path="/x.jpg", width=10, height=10) for i in range(6)]
    sm = SubjectMotion(subject="person", label="moves_right", camera_relation="camera_static", box_velocity=[0.1, 0], camera_velocity=[0, 0], relative_velocity=[0.1, 0], n_frames=5, confidence=0.8, method="tracker", trajectory=[[0.0, 0.3, 0.5, 0.1], [1.0, 0.3, 0.5, 0.1], [2.0, 0.4, 0.5, 0.1], [3.0, 0.5, 0.5, 0.1], [4.0, 0.5, 0.5, 0.1], [5.0, 0.5, 0.5, 0.1]])
    a = SceneAnalysis(scene_id="scene_001", video_id="v", start_time=0.0, end_time=6.0, summary="A shot.", actions=["The person moves right while the camera stays still"], subject_motion=sm, provider="cv_models", confidence=0.7, confidence_source=ConfidenceSource.DETECTOR_SCORE)
    vision = SimpleNamespace(scenes=[a])
    scan = SceneScan(scene_id="scene_001", analysis_width=160, analysis_fps=4.0, samples=[ScanSample(timestamp=0.25 * i, frame_index=i, change=0.0, brightness=100.0, flow_dx=0.0, flow_dy=0.0, flow_mag=0.0, flow_div=0.0) for i in range(24)])
    events = EventExtractor(CFG.temporal, scene_threshold=27.0, source_fps=24.0, track_fps=3.0).extract("v", 6.0, [scene], frames, vision, None, None, None, [scan])
    action = next(e for e in events if e.event_type == EventType.ACTION)
    assert action.boundary_precision == "measured" and (action.start_time, action.end_time) == (1.0, 3.0)
    assert action.start_uncertainty_ms == pytest.approx(1000 / 3.0, abs=1) and action.attributes["timing_method"] == "tracker"
    assert all(e.start_uncertainty_ms is not None and e.end_uncertainty_ms is not None for e in events)
    scene_bound = SceneAnalysis(scene_id="scene_001", video_id="v", start_time=0.0, end_time=6.0, summary="A shot.", actions=["A dog runs"], provider="mock")
    events = EventExtractor(CFG.temporal, scene_threshold=27.0).extract("v", 6.0, [scene], frames, SimpleNamespace(scenes=[scene_bound]), None, None, None, [])
    action = next(e for e in events if e.event_type == EventType.ACTION)
    assert action.boundary_precision == "scene" and action.start_uncertainty_ms == 6000.0  # never implies more precision than the shot


# ------------------------------------------------------------------ issue 16: technical quality
def test_quality_flags_cover_frozen_and_video_level_defects():
    q = CFG.quality
    still = SceneScan(scene_id="s", analysis_width=160, analysis_fps=4.0, samples=[ScanSample(timestamp=0.25 * i, frame_index=i, change=0.001, brightness=100.0, flow_dx=0.0, flow_dy=0.0, flow_mag=0.0, flow_div=0.0) for i in range(20)])
    assert "frozen" in scene_quality_flags(Measurements(sharpness=200.0), [], q, still, 5.0)
    assert "frozen" not in scene_quality_flags(Measurements(sharpness=200.0), [], q, still, 1.0)  # too short to call
    moving = still.model_copy(update={"samples": [s.model_copy(update={"change": 0.05, "flow_mag": 1.0}) for s in still.samples]})
    assert "frozen" not in scene_quality_flags(Measurements(sharpness=200.0), [], q, moving, 5.0)
    flags = scene_quality_flags(Measurements(sharpness=200.0), ["low_fps", "extreme_aspect_ratio", "low_resolution", "duration_mismatch"], q)
    assert flags == ["low_fps", "extreme_aspect_ratio", "low_resolution"]  # duration mismatch stays a video-level fact


# ------------------------------------------------------------------ issue 7 / 17 / 20: groups, bands, rights, strict subset
def _scene_rec(i, vid, status="accepted", checks_ok=True, semantic="measured", target="single_shot"):
    checks = {k: checks_ok for k in ("timestamps", "scene_exists", "frames_exist", "frame_files_exist", "summary_present", "frame_sequence", "visual_quality", "consistent", "text_consistent", "single_shot")}
    return {"record_id": f"s{i}", "record_type": "scene", "video_id": vid, "scene_id": f"scene_{i:03d}", "start_time": 0.0, "end_time": 5.0, "duration": 5.0, "generation_target": target, "camera": {"movement": "static"}, "validation": {"status": status, "checks": checks, "semantic_verification": semantic, "policy_version": POLICY_VERSION}, "quality": {"overall": 0.9, "components_used": ["grounding"]}, "confidence": 0.9, "confidence_source": "measurement"}


def test_annotate_records_groups_duplicates_bands_rights_and_strict_subset():
    rows = {"scenes": [_scene_rec(1, "vid_a"), _scene_rec(2, "vid_b"), _scene_rec(3, "vid_c", status="review"), _scene_rec(4, "vid_a", checks_ok=False), _scene_rec(5, "vid_a", target="not_for_generation"), _scene_rec(6, "vid_a", semantic="not_run")]}
    info = {"vid_b": {"canonical_video_id": "vid_a", "rights_status": "declared_cc_by", "license_status": "creative_commons", "coverage": {"ocr": "ran"}}, "vid_a": {"rights_status": "restricted"}, "vid_c": {"rights_status": "unknown"}}
    counts = annotate_records(rows, CFG.export, info, CFG.licensing)
    by_id = {r["record_id"]: r for r in rows["scenes"]}
    assert by_id["s2"]["source_group_id"] == "vid_a" and by_id["s2"]["split"] == by_id["s1"]["split"] and by_id["s2"]["canonical_video_id"] == "vid_a"
    assert by_id["s1"]["source_group_id"] == "vid_a" and by_id["s3"]["source_group_id"] == "vid_c"
    assert all(r["schema_version"] == SCHEMA_VERSION for r in rows["scenes"]) and by_id["s2"]["coverage"] == {"ocr": "ran"}
    assert "duration_medium" in by_id["s1"]["subsets"] and "rights_declared_cc_by" in by_id["s2"]["subsets"] and "licensed" in by_id["s2"]["subsets"]
    assert "strict" in by_id["s1"]["subsets"] and "strict" not in by_id["s2"]["subsets"]  # near-duplicate videos are capped at bronze but still accepted ... strict needs all checks
    assert "strict" not in by_id["s3"]["subsets"] and "strict" not in by_id["s4"]["subsets"] and "strict" not in by_id["s5"]["subsets"] and "strict" not in by_id["s6"]["subsets"]
    assert counts["strict"] >= 1 and split_leaks(rows) == {}
    leaked = {"scenes": [{**_scene_rec(7, "vid_z"), "split": "train", "source_group_id": "g"}, {**_scene_rec(8, "vid_z"), "split": "test", "source_group_id": "g"}]}
    assert set(split_leaks(leaked)) == {"vid_z", "group:g"}
    strict_rows = {"scenes": [_scene_rec(9, "vid_a")]}
    strict_cfg = CFG.export.model_copy(update={"require_rights": True})
    counts = annotate_records(strict_rows, strict_cfg, {"vid_a": {"rights_status": "restricted"}}, CFG.licensing)
    assert strict_rows["scenes"] == [] and counts["dropped_rights_records"] == 1
    assert rights_status_for("creative_commons") == "declared_cc_by" and rights_status_for("standard") == "restricted" and rights_status_for(None) == "unknown"
    assert duration_band(2.0, CFG.export.duration_bands) == "short" and duration_band(50.0, CFG.export.duration_bands) == "long"


# ------------------------------------------------------------------ issue 12: hard negatives are typed and provenanced
def test_hard_negatives_carry_verification_status_and_evidence_interval():
    import random

    events = [_ev(1, 0.0, 2.0, "The car enters the tunnel."), _ev(2, 5.0, 7.0, "The car exits the tunnel."), _ev(3, 30.0, 32.0, "A dog runs across the road.")]
    for e in events:
        e.confidence, e.confidence_source, e.scene_ids = 0.9, ConfidenceSource.MEASUREMENT, [f"scene_{e.event_id[-3:]}"]
    q = _q(QAType.DURATION, "How long?", "About 2 seconds (from 0.0s to 2.0s).", events[:1])
    negs = qa_hard_negatives(q, events, 90.0, random.Random(0))
    assert negs and all(n.verification_status in ("measured_difference", "rule_derived") and n.evidence_interval == [0.0, 2.0] for n in negs)
    a = _analysis()
    b = SceneAnalysis(scene_id="scene_002", video_id="v", start_time=4.0, end_time=8.0, summary="A 4.0-second shot with bright lighting; the camera is static. Detected objects: a person.", camera=CameraAnnotation(movement=CameraMovement.STATIC, confidence=0.9, confidence_source=ConfidenceSource.MEASUREMENT), measurements=Measurements(lighting_level="bright"), objects=[ObjectAnnotation(name="person")])
    negs = caption_hard_negatives(a, [a, b], random.Random(0))
    assert negs[0].kind == "attribute_swap" and negs[0].verification_status == "measured_difference" and negs[0].evidence_interval == [0.0, 4.0]
    assert any(n.kind == "other_shot" and n.verification_status == "measured_difference" for n in negs)
    stats = compute_statistics([], {"scenes": [{"record_id": "s", "hard_negatives": [n.model_dump() for n in negs]}], "temporal_qa": [], "long_video_qa": [], "events": [], "frames": [], "clips": [], "video_descriptions": []}, [])
    assert stats["hard_negatives"]["positive_records"] == 1 and stats["hard_negatives"]["negative_texts"] == len(negs) and stats["hard_negatives"]["by_verification_status"]["measured_difference"] >= 2


# ------------------------------------------------------------------ issue 5 / 18 / 23: release gates
def test_release_gates_catch_broken_schema_references_parquet_splits_media_and_strict(tmp_path: Path):
    good = _scene_rec(1, "vid_a")
    good.update({"summary": "a b c d e", "environment": {}, "visual_style": {}, "split": "train", "subsets": ["strict"], "source_group_id": "vid_a", "schema_version": SCHEMA_VERSION})
    bad = {**_scene_rec(2, "vid_a"), "summary": "x y z w", "environment": {}, "visual_style": {}, "start_time": 5.0, "end_time": 1.0, "split": "train"}
    records = {"scenes": [good, bad], "frames": [{"record_id": "f1", "record_type": "frame", "video_id": "vid_a", "scene_id": "scene_009", "clip_id": "clip_009", "frame_id": "x", "frame_path": "frames/vid_a/missing.jpg", "timestamp": 1.0, "caption": "c", "split": "test"}], "clips": [], "events": [], "temporal_qa": [{"record_id": "q1", "record_type": "temporal_qa", "question_id": "q1", "video_id": "vid_a", "type": "duration", "question": "How long?", "answer": "a", "evidence": {"start_time": 0, "end_time": 1, "event_ids": ["event_0001"]}, "split": "train", "hard_negatives": [{"text": "a", "kind": "x"}]}], "long_video_qa": [], "sequences": [], "video_descriptions": []}
    g = schema_gate(records)
    assert g["state"] == "failed" and g["invalid"] == 1 and "bad interval" in g["examples"][0]["error"]
    g = reference_gate(records)
    assert g["state"] == "failed" and g["frames_without_scene"] == 1 and g["frames_without_clip"] == 0 and g["qa_missing_event"] == 0 and g["negatives_equal_positive"] == 1
    records["events"] = [{"record_id": "e1", "record_type": "event", "video_id": "vid_a", "event_id": "event_0002", "event_type": "action", "start_time": 0, "end_time": 1, "event": "x", "source": "vision"}]
    assert reference_gate(records)["qa_missing_event"] == 1
    g = split_gate(tmp_path, records)
    assert g["state"] == "failed" and "vid_a" in g["leaking"]
    write_typed_parquet_files({"scenes": [good]}, tmp_path)
    write_jsonl(tmp_path / "scenes.jsonl", [good, bad])
    g = parquet_gate(tmp_path, {"scenes": [good, bad]})
    assert g["state"] == "failed" and g["mismatches"]["scenes"]["only_jsonl"] == 1
    assert strict_gate({"scenes": [{**good, "validation": {**good["validation"], "status": "review"}}]})["state"] == "failed"
    assert strict_gate({"scenes": [good]})["state"] == "passed"
    rows = build_media_manifest(records, tmp_path)
    assert summarize(rows)["missing"] == 1 and rows[0]["exists"] is False and rows[0]["records"] == 1


def test_upload_release_gate_blocks_broken_data(tmp_path: Path):
    from video_dataset.dataset import upload as up

    cfg = _cfg(tmp_path, **{"upload.provider": "huggingface", "upload.repo_id": "u/d", "upload.threshold_mb": "0.001"})
    final = cfg.export_dir
    final.mkdir(parents=True)
    (final / "per_video" / "vid_a").mkdir(parents=True)
    write_json_atomic(final / "per_video" / "vid_a" / "manifest.json", {"video_id": "vid_a"})
    write_jsonl(final / "scenes.jsonl", [{"record_id": "s1", "video_id": "vid_a", "scene_id": "scene_001", "start_time": 5.0, "end_time": 1.0, "duration": -4, "summary": "a b c d", "environment": {}, "camera": {}, "visual_style": {}}])
    write_json_atomic(final / "manifest.json", {"videos": [{"video_id": "vid_a"}]})

    class Fake:
        def ensure_repo(self):
            return "url"

        def ensure_card(self, readme):
            return True

        def upload_dir(self, *a):
            return "commit/1"

    with pytest.raises(up.UploadError, match="release gate failed"):
        up.check_and_upload(cfg, force=True, uploader=Fake())
    assert (final / "release_report.json").exists() and (final / "scenes.jsonl").exists()  # nothing moved


# ------------------------------------------------------------------ issue 20: takedown
def test_remove_source_strips_every_export_and_leaves_a_tombstone(tmp_path: Path):
    cfg = _cfg(tmp_path)
    final = cfg.export_dir
    for sub in (final, final / "uploaded" / "shard_0001"):
        sub.mkdir(parents=True, exist_ok=True)
        write_jsonl(sub / "scenes.jsonl", [{"record_id": "s1", "video_id": "vid_gone"}, {"record_id": "s2", "video_id": "vid_keep"}])
        write_jsonl(sub / "dataset.jsonl", [{"record_type": "scene", "record_id": "s1", "video_id": "vid_gone"}])
        write_json_atomic(sub / "manifest.json", {"videos": [{"video_id": "vid_gone"}, {"video_id": "vid_keep"}]})
    (final / "per_video" / "vid_gone").mkdir(parents=True)
    write_json_atomic(final / "uploads.json", {"next_shard": 2, "shards": [{"name": "shard_0001", "repo_id": "u/d", "path_in_repo": "shard_0001", "videos": ["vid_gone"]}]})
    frames = cfg.data_dir / "frames" / "vid_gone"
    frames.mkdir(parents=True)
    (frames / "f.jpg").write_bytes(b"x")
    with StateDB(cfg.db_path) as db:
        db.upsert_video("vid_gone", "https://example.com/v")
    plan = remove_source(cfg, "vid_gone", "rights holder request", dry_run=True)
    assert plan["dry_run"] and str(frames) in plan["working_paths"] and plan["hub_shards"][0]["shard"] == "shard_0001" and frames.exists()
    res = remove_source(cfg, "vid_gone", "rights holder request")
    assert not frames.exists() and not (final / "per_video" / "vid_gone").exists()
    assert [r["video_id"] for r in read_jsonl(final / "scenes.jsonl")] == ["vid_keep"] and list(read_jsonl(final / "dataset.jsonl")) == []
    assert [r["video_id"] for r in read_jsonl(final / "uploaded" / "shard_0001" / "scenes.jsonl")] == ["vid_keep"]
    assert read_json(final / "manifest.json")["videos"] == [{"video_id": "vid_keep"}] and res["archived_shards"]["shard_0001"]["scenes"] == 1
    assert "vid_gone" in removed_sources(final) and removed_sources(final)["vid_gone"]["reason"] == "rights holder request"
    with StateDB(cfg.db_path) as db:
        assert db.get_video("vid_gone") is None


# ------------------------------------------------------------------ issue 23 / 14: review loop + calibration
def test_review_queue_labels_and_calibration(tmp_path: Path):
    accepted = _scene_rec(1, "vid_a")
    review = {**_scene_rec(2, "vid_a", status="review"), "claims": [{"field": "lighting", "status": "contradicted"}], "summary": "s", "frame_ids": ["f1"]}
    records = {"scenes": [accepted, review], "temporal_qa": [{"record_id": "q1", "video_id": "vid_a", "question": "Q?", "answer": "A.", "confidence": 0.3, "confidence_source": "derived_min", "validation": {"status": "accepted", "checks": {}, "reason_codes": ["failed_check:caption_alignment"]}}]}
    n = export_review_queue(records, tmp_path)
    queue = list(read_jsonl(tmp_path / "review_queue.jsonl"))
    assert n == 2 and {r["record_id"] for r in queue} == {"s2", "q1"} and "contradicted_claim" in next(r for r in queue if r["record_id"] == "s2")["reasons"]
    (tmp_path / "reviews").mkdir()
    write_jsonl(tmp_path / "reviews" / "labels.jsonl", [{"record_id": "s2", "label": "accept"}, {"record_id": "q1", "label": "reject", "note": "wrong"}, {"record_id": "s1", "label": "maybe"}])
    labels = read_review_labels(tmp_path / "reviews" / "labels.jsonl")
    assert set(labels) == {"s2", "q1"}
    counts = apply_review_labels(records, labels)
    assert counts == {"accepted": 1, "rejected": 1, "corrected": 0}
    assert review["validation"]["status"] == "accepted" and review["validation"]["semantic_verification"] == "human" and "human:accepted" in review["validation"]["reason_codes"]
    assert records["temporal_qa"][0]["validation"]["status"] == "rejected"
    cal = calibrate(records, labels_for_calibration(labels) | {"s1": True}, threshold=0.75, target_far=0.5)
    assert cal["matched_records"] == 3
    key = "derived_min|temporal_qa"
    assert cal["by_source"][key]["count"] == 1 and cal["by_source"][key]["accuracy"] == 0.0 and cal["by_source"][key]["brier"] == pytest.approx(0.09)
    assert "measurement|scenes" in cal["by_source"] and cal["by_source"]["measurement|scenes"]["recommended_threshold"] == 0.9


# ------------------------------------------------------------------ issue 21: coverage
def test_stage_coverage_distinguishes_not_run_from_empty(tmp_path: Path):
    cfg = _cfg(tmp_path, **{"transcription.provider": "none", "ocr.provider": "none"})
    db = StateDB(cfg.db_path)
    db.upsert_video("vid_x", "u")
    db.finish_stage("vid_x", Stage.OCR, status=StageStatus.SKIPPED)
    db.finish_stage("vid_x", Stage.VISION_ANALYSIS, status=StageStatus.DONE)
    db.fail_stage("vid_x", Stage.AUDIO, "boom")
    ctx = SimpleNamespace(config=cfg, db=db, video_id="vid_x", media_info=SimpleNamespace(has_audio=True))
    cov = stage_coverage(ctx, Transcript(video_id="vid_x", provider="none", has_speech=False, status="not_run"), None, SimpleNamespace(summary={"verifier": "heuristic", "caption_alignment": "model_unavailable", "text_checker": "rules"}))
    assert cov["transcription"] == "disabled" and cov["ocr"] == "disabled" and cov["audio_events"] == "failed" and cov["vision"] == "ran:heuristic"
    assert cov["clip_alignment"] == "not_run" and cov["verifier"] == "ran:heuristic" and cov["text_checks"] == "ran" and cov["clip_boundaries"] == "ran:all"
    cov2 = stage_coverage(ctx, Transcript(video_id="vid_x", provider="none", has_speech=False, status="no_audio"), None, SimpleNamespace(summary={}))
    assert cov2["transcription"] == "no_audio"
    rates = coverage_rates([{"coverage": cov}, {"coverage": cov2}])
    assert rates["transcription"] == {"disabled": 1, "no_audio": 1}
    db.close()


# ------------------------------------------------------------------ issue 22: reproducibility manifest + config drift
def test_environment_manifest_is_complete_and_secret_free():
    env = environment_manifest(CFG)
    assert env["schema_version"] == SCHEMA_VERSION and env["policy_version"] == POLICY_VERSION
    assert env["code"]["package_version"] and env["python"] and env["tools"]["ffmpeg"] and "numpy" in env["packages"]
    assert env["models"]["vision"]["provider"] == CFG.vision.provider and env["config_hashes"]["vision"]
    text = json.dumps(env)
    assert "api_key" not in text.lower() and "hf_" not in text.lower() and "token" not in text.lower()
    other = CFG.model_copy(deep=True)
    other.vision.yolo_confidence = 0.9
    assert environment_manifest(other)["config_hashes"]["vision"] != env["config_hashes"]["vision"]


def test_rerun_on_config_change_invalidates_cached_stages(tmp_path: Path, monkeypatch):
    from video_dataset.pipeline import runner as rmod
    from video_dataset.pipeline.context import StageOutput

    calls: list[str] = []
    monkeypatch.setitem(rmod.STAGE_FUNCTIONS, Stage.DOWNLOAD, lambda ctx: calls.append("d") or StageOutput(message="ok"))
    cfg = _cfg(tmp_path, **{"pipeline.stage_retries": "0"})
    runner = rmod.PipelineRunner(cfg)
    try:
        runner.db.upsert_video("vid_x", "u")
        ctx = runner.context("vid_x")
        assert runner.run_stage(ctx, Stage.DOWNLOAD) == (StageStatus.DONE, True)
        assert runner.run_stage(ctx, Stage.DOWNLOAD) == (StageStatus.DONE, False)  # cached
        runner.db.finish_stage("vid_x", Stage.PREPROCESS, status=StageStatus.DONE)
        runner.config.download.retries = 7  # config drift
        ctx = runner.context("vid_x")
        assert runner.run_stage(ctx, Stage.DOWNLOAD) == (StageStatus.DONE, False)  # default: reuse, note in the log
        runner.config.pipeline.rerun_on_config_change = True
        assert runner.run_stage(ctx, Stage.DOWNLOAD) == (StageStatus.DONE, True)
        assert runner.db.stage_status("vid_x", Stage.PREPROCESS) == StageStatus.PENDING  # downstream invalidated too
        assert calls == ["d", "d"]
    finally:
        runner.close()


# ------------------------------------------------------------------ end to end: the exported dataset passes its own release gates
@pytest.mark.integration
@pytest.mark.slow
def test_pipeline_export_passes_release_gates(synthetic_video: Path, test_config):
    from video_dataset.dataset.aggregate import aggregate_exports
    from video_dataset.pipeline.runner import PipelineRunner
    from video_dataset.utils.urls import classify_input

    cfg = test_config.model_copy(deep=True)
    cfg.cleanup.frames_and_clips = False
    cfg.frame_sampling.verify_clip_boundaries = "all"
    cfg.transcription.provider = "mock"
    runner = PipelineRunner(cfg)
    try:
        item = classify_input(str(synthetic_video))
        registered = runner.register_inputs([item])
        video_id = registered[0][0]
        results = runner.run_batch(registered)
        assert results[0].completed, results[0].error
        validated = read_json(runner.paths.validated_file(video_id))
        assert validated["summary"]["policy_version"] == POLICY_VERSION and validated["summary"]["claims"]["supported"] >= 1
        for s in validated["scenes"]:
            v = s["validation"]
            assert v["policy_version"] == POLICY_VERSION and v["semantic_verification"] == "measured"
            assert check_state(v["checks"]["clip_boundaries"]) in ("passed", "failed", "error") and check_state(v["checks"]["text_consistent"]) == "passed" and check_state(v["checks"]["camera_motion_agreement"]) in ("passed", "not_applicable")
            assert s["analysis"]["enrichments"]["claims"]
        for q in validated["qa"]:
            assert q["answer_type"] and check_state(q["validation"]["checks"]["question_form"]) in ("passed", "failed") and check_state(q["validation"]["checks"]["answer_recomputed"]) in ("passed", "not_applicable", "not_run")
        for r in validated["relations"]:
            assert r["validation"]["checks"].get("interval_semantics") is True, r
        aggregate_exports(cfg, runner.db)
        final = cfg.export_dir
        clips = list(read_jsonl(final / "clips.jsonl"))
        assert clips and all(c["boundary_check"] in ("passed", "failed", "error") and c["transcript_scope"] == "none" and c["coverage"]["transcription"] in ("ran", "no_audio") and c["schema_version"] == SCHEMA_VERSION for c in clips)
        scenes = list(read_jsonl(final / "scenes.jsonl"))
        assert all(s["claims"] and s["source_group_id"] == video_id and s["generation_target"] in ("single_shot", "not_for_generation") for s in scenes)
        events = list(read_jsonl(final / "events.jsonl"))
        assert all(e["start_uncertainty_ms"] is not None for e in events)
        assert (final / "media_manifest.jsonl").exists() and (final / "provenance.jsonl").exists()
        prov = list(read_jsonl(final / "provenance.jsonl"))
        assert prov[0]["video_id"] == video_id and prov[0]["rights_status"] == "unknown" and prov[0]["source_sha256"]
        manifest = read_json(final / "manifest.json")
        assert manifest["environment"]["config_hashes"]["validation"] and manifest["media_references"]["missing"] == 0
        stats = read_json(final / "statistics.json")
        assert stats["coverage"]["by_modality"]["transcription"] and stats["clip_boundary_checks"] and stats["calibration"]["status"] == "uncalibrated"
        report = run_release_check(cfg, final, require_media=True)
        assert report["ok"], [g for g in report["gates"] if g["state"] == "failed"]
        assert (final / "release_report.json").exists() and report["metrics"]["claims"]["supported"] >= 1
        if stats["records_in_strict_subset"].get("scenes"):
            assert (final / "strict" / "scenes.jsonl").exists()
        n = export_review_queue({k: list(read_jsonl(final / f"{k}.jsonl")) for k in ("scenes", "clips", "temporal_qa")}, final)
        assert n >= 0 and (final / "review_queue.jsonl").exists()
    finally:
        runner.close()
