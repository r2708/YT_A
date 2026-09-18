from pathlib import Path

from video_dataset.config import load_config
from video_dataset.pipeline.context import StageOutput, VideoContext
from video_dataset.pipeline.runner import STAGE_FUNCTIONS, PipelineRunner
from video_dataset.stages import STAGE_ORDER, Stage, StageStatus
from video_dataset.storage.state_db import StateDB


def test_state_db_stage_lifecycle(tmp_path: Path):
    db = StateDB(tmp_path / "state.db")
    db.upsert_video("vid_a", "https://youtu.be/x", youtube_id="xxxxxxxxxxx")
    assert db.stage_status("vid_a", Stage.DOWNLOAD) == StageStatus.PENDING
    db.start_stage("vid_a", Stage.DOWNLOAD)
    assert db.stage_status("vid_a", Stage.DOWNLOAD) == StageStatus.RUNNING
    db.finish_stage("vid_a", Stage.DOWNLOAD, "meta.json", {"n": 1}, "abc", 1.5)
    rec = db.get_stage("vid_a", Stage.DOWNLOAD)
    assert rec.status == StageStatus.DONE and rec.metrics == {"n": 1} and rec.attempts == 1
    db.start_stage("vid_a", Stage.PREPROCESS)
    db.fail_stage("vid_a", Stage.PREPROCESS, "boom")
    assert db.failed_stages()[0].error == "boom"
    assert db.first_incomplete_stage("vid_a") == Stage.PREPROCESS
    db.start_stage("vid_a", Stage.SCENE_DETECTION)
    assert db.recover_interrupted() == 1
    assert db.stage_status("vid_a", Stage.SCENE_DETECTION) == StageStatus.FAILED
    db.reset_stages("vid_a", Stage.PREPROCESS)
    assert db.stage_status("vid_a", Stage.PREPROCESS) == StageStatus.PENDING
    assert db.stage_status("vid_a", Stage.DOWNLOAD) == StageStatus.DONE
    assert db.find_by_youtube_id("xxxxxxxxxxx")["video_id"] == "vid_a"
    assert db.register_file_hash("h1", "vid_a") is None
    assert db.register_file_hash("h1", "vid_b") == "vid_a"
    db.close()


def test_runner_resumes_and_skips_done_stages(tmp_path: Path, monkeypatch):
    cfg = load_config(None, {"project.data_dir": str(tmp_path / "data"), "project.log_level": "ERROR", "pipeline.stage_retries": "0"})
    calls: list[str] = []
    attempts = {"n": 0}

    def ok(stage: Stage):
        def fn(ctx: VideoContext) -> StageOutput:
            calls.append(stage.value)
            return StageOutput(artifact_path=None, metrics={"ok": True}, message="done")
        return fn

    def flaky(ctx: VideoContext) -> StageOutput:
        attempts["n"] += 1
        calls.append("FRAME_EXTRACTION")
        if attempts["n"] == 1:
            raise RuntimeError("simulated crash")
        return StageOutput(message="recovered")

    for st in STAGE_ORDER:
        monkeypatch.setitem(STAGE_FUNCTIONS, st, ok(st))
    monkeypatch.setitem(STAGE_FUNCTIONS, Stage.FRAME_EXTRACTION, flaky)

    runner = PipelineRunner(cfg)
    runner.db.upsert_video("vid_t", "file:///x.mp4", source_type="local")
    r1 = runner.run_video("vid_t")
    assert not r1.completed and r1.failed_stage == Stage.FRAME_EXTRACTION
    assert calls == ["DOWNLOAD", "PREPROCESS", "SCENE_DETECTION", "FRAME_EXTRACTION"]
    assert runner.db.stage_status("vid_t", Stage.FRAME_EXTRACTION) == StageStatus.FAILED

    calls.clear()
    r2 = runner.run_video("vid_t")  # resume: first three stages are cached, flaky recovers
    assert r2.completed
    assert calls[0] == "FRAME_EXTRACTION" and "DOWNLOAD" not in calls and calls[-1] == "EXPORT"
    assert runner.db.get_video("vid_t")["status"] == "done"

    calls.clear()
    r3 = runner.run_video("vid_t")  # everything cached: no stage function runs
    assert r3.completed and calls == []

    calls.clear()
    runner.run_video("vid_t", force_from=Stage.QA_GENERATION)
    assert calls == ["QA_GENERATION", "VALIDATION", "EXPORT"]
    runner.close()


def test_optional_stage_failure_does_not_block(tmp_path: Path, monkeypatch):
    cfg = load_config(None, {"project.data_dir": str(tmp_path / "data"), "project.log_level": "ERROR", "pipeline.stage_retries": "0"})

    def ok(ctx):
        return StageOutput(message="ok")

    def boom(ctx):
        raise RuntimeError("no OCR engine")

    for st in STAGE_ORDER:
        monkeypatch.setitem(STAGE_FUNCTIONS, st, ok)
    monkeypatch.setitem(STAGE_FUNCTIONS, Stage.OCR, boom)
    runner = PipelineRunner(cfg)
    runner.db.upsert_video("vid_o", "file:///y.mp4", source_type="local")
    r = runner.run_video("vid_o")
    assert r.completed and Stage.OCR in r.stages_skipped
    assert runner.db.stage_status("vid_o", Stage.OCR) == StageStatus.FAILED
    assert runner.db.stage_status("vid_o", Stage.EXPORT) == StageStatus.DONE
    runner.close()
