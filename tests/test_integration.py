"""End-to-end run on the synthetic video with network-free components."""

from __future__ import annotations

from pathlib import Path

import pytest

from video_dataset.dataset.aggregate import aggregate_exports
from video_dataset.pipeline.runner import PipelineRunner
from video_dataset.schemas.events import Timeline
from video_dataset.schemas.qa import QAResult
from video_dataset.schemas.scene import FrameSamplingResult
from video_dataset.schemas.validated import ValidatedVideo
from video_dataset.schemas.vision import CameraMovement, VisionResult
from video_dataset.stages import STAGE_ORDER, Stage, StageStatus
from video_dataset.utils.io import read_json, read_jsonl
from video_dataset.utils.urls import classify_input


@pytest.mark.integration
@pytest.mark.slow
def test_full_pipeline_on_synthetic_video(synthetic_video: Path, test_config):
    runner = PipelineRunner(test_config)
    try:
        item = classify_input(str(synthetic_video))
        assert item is not None
        registered = runner.register_inputs([item])
        assert len(registered) == 1
        video_id = registered[0][0]
        results = runner.run_batch(registered)
        assert results[0].completed, results[0].error
        stages = runner.db.get_stages(video_id)
        for st in STAGE_ORDER:
            assert stages[st].status in (StageStatus.DONE, StageStatus.SKIPPED), (st, stages[st].error)
        paths = runner.paths

        sampling = FrameSamplingResult.model_validate(read_json(paths.frames_file(video_id)))
        assert len(sampling.frames) >= 12 and all(Path(f.frame_path).exists() for f in sampling.frames)
        assert sampling.clips and all(Path(c.clip_path).exists() for c in sampling.clips)

        vision = VisionResult.model_validate(read_json(paths.vision_file(video_id)))
        assert vision.provider == "heuristic" and len(vision.scenes) >= 3
        pan_scene = [s for s in vision.scenes if s.camera.movement == CameraMovement.PAN_LEFT]
        assert pan_scene, [s.camera.movement for s in vision.scenes]
        assert all(s.measurements is not None and s.measurements.brightness_mean is not None for s in vision.scenes)

        timeline = Timeline.model_validate(read_json(paths.timeline_file(video_id)))
        types = {str(e.event_type) for e in timeline.events}
        assert {"transition", "camera", "sound", "state_change"} <= types, types
        assert timeline.relations
        for e in timeline.events:
            assert 0 <= e.start_time <= e.end_time <= timeline.duration + 0.05
            assert e.frame_ids, e.event_id  # every event is linked to frames
        ocr_events = [e for e in timeline.events if str(e.event_type) == "text_on_screen"]
        if stages[Stage.OCR].status == StageStatus.DONE and stages[Stage.OCR].metrics.get("tracks"):
            assert any("HELLO" in e.event.upper() for e in ocr_events), [e.event for e in ocr_events]

        qa = QAResult.model_validate(read_json(paths.qa_file(video_id)))
        assert len(qa.questions) >= 8
        event_ids = {e.event_id for e in timeline.events}
        for q in qa.questions:
            assert q.evidence.event_ids and set(q.evidence.event_ids) <= event_ids
            assert q.evidence.end_time <= timeline.duration + 0.05
        assert len({q.type for q in qa.questions}) >= 4

        validated = ValidatedVideo.model_validate(read_json(paths.validated_file(video_id)))
        assert validated.summary["qa"]["accepted"] + validated.summary["qa"]["review"] >= 1
        assert all(q.validation is not None and q.quality is not None for q in validated.qa)

        stats = aggregate_exports(runner.config, runner.db)
        final = runner.config.export_dir
        for name in ("frames", "clips", "scenes", "events", "temporal_qa", "video_descriptions"):
            assert (final / f"{name}.jsonl").exists()
        assert (final / "dataset.parquet").exists() and (final / "statistics.json").exists()
        assert stats["temporal_qa"] + stats["long_video_qa"] >= 1 and stats["frames"] >= 12
        frames_rows = list(read_jsonl(final / "frames.jsonl"))
        assert frames_rows and frames_rows[0]["video_id"] == video_id and "caption" in frames_rows[0]
        desc = list(read_jsonl(final / "video_descriptions.jsonl"))
        assert desc and "Temporal progression" in desc[0]["prompt"]

        # resume: a second run must not redo anything
        results2 = runner.run_batch(registered)
        assert results2[0].completed and results2[0].stages_run == []
    finally:
        runner.close()


@pytest.mark.integration
@pytest.mark.slow
def test_pipeline_with_mock_vlm_produces_semantic_events(synthetic_video: Path, test_config):
    """Exercise the generative-vision code path (parsing, verification, appearance/state-change events, long-range QA)."""
    cfg = test_config.model_copy(deep=True)
    cfg.vision.provider = "mock"
    cfg.ocr.provider = "none"
    cfg.audio_events.enabled = False
    runner = PipelineRunner(cfg)
    try:
        registered = runner.register_inputs([classify_input(str(synthetic_video))])  # type: ignore[list-item]
        video_id = registered[0][0]
        results = runner.run_batch(registered)
        assert results[0].completed, results[0].error
        vision = VisionResult.model_validate(read_json(runner.paths.vision_file(video_id)))
        assert vision.provider == "mock" and all(s.objects for s in vision.scenes)
        assert all(s.verification is not None and s.confidence == 0.9 and str(s.confidence_source) == "verifier" for s in vision.scenes)
        assert all(s.measurements is not None for s in vision.scenes)  # measured signals attached even with a VLM
        timeline = Timeline.model_validate(read_json(runner.paths.timeline_file(video_id)))
        types = {str(e.event_type) for e in timeline.events}
        assert {"action", "appearance", "state_change", "transition"} <= types, types
        qa = QAResult.model_validate(read_json(runner.paths.qa_file(video_id)))
        assert {str(q.type) for q in qa.questions} >= {"timestamp", "before_after", "temporal_ordering", "state_change"}
        validated = ValidatedVideo.model_validate(read_json(runner.paths.validated_file(video_id)))
        assert validated.summary["scenes"]["accepted"] >= 3
        accepted_qa = [q for q in validated.qa if q.validation and q.validation.status == "accepted"]
        assert accepted_qa
        final = runner.config.export_dir / "per_video" / video_id
        scenes_rows = list(read_jsonl(final / "scenes.jsonl"))
        assert scenes_rows and scenes_rows[0]["objects"] and scenes_rows[0]["provider"] == "mock"
    finally:
        runner.close()
