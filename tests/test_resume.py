"""Resuming a batch after the process died: finished videos cached, interrupted stage re-run,
never-started videos fetched from the URL stored in the database."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from video_dataset.pipeline.runner import PipelineRunner
from video_dataset.stages import Stage, StageStatus
from video_dataset.utils.urls import classify_input, item_from_stored_url


def test_item_from_stored_url(tmp_path: Path, synthetic_video: Path):
    local = classify_input(str(synthetic_video))
    assert local is not None
    back = item_from_stored_url(local.url)  # file:// URI as stored in videos.url
    assert back is not None and back.kind == "local" and back.local_path == local.local_path
    yt = item_from_stored_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert yt is not None and yt.kind == "youtube" and yt.youtube_id == "dQw4w9WgXcQ"
    assert item_from_stored_url("file:///does/not/exist.mp4") is None
    assert item_from_stored_url("") is None


@pytest.mark.integration
@pytest.mark.slow
def test_resume_after_simulated_crash(tmp_path: Path, synthetic_video: Path, test_config):
    cfg = test_config.model_copy(deep=True)
    cfg.ocr.provider = "none"
    cfg.cleanup.after_export = "media"
    copies = []
    for i in range(3):
        dst = tmp_path / f"video_{i}.mp4"
        shutil.copy2(synthetic_video, dst)
        copies.append(dst)
    items = [classify_input(str(p)) for p in copies]
    assert all(items)

    # --- "first run": video 0 finishes, video 1 is interrupted inside AUDIO, video 2 never starts
    runner = PipelineRunner(cfg)
    registered = runner.register_inputs(items)  # type: ignore[arg-type]
    v0, v1, v2 = (vid for vid, _ in registered)
    assert runner.run_batch([registered[0]])[0].completed
    assert runner.run_batch([registered[1]], until=Stage.FRAME_EXTRACTION)[0].completed
    runner.db.start_stage(v1, Stage.AUDIO)  # process dies here: the row stays RUNNING
    runner.close()

    # --- "second run": a fresh process, no URL file
    runner = PipelineRunner(cfg)
    try:
        assert runner.db.get_stage(v1, Stage.AUDIO).status == StageStatus.FAILED  # recovered at start-up
        assert "interrupted" in (runner.db.get_stage(v1, Stage.AUDIO).error or "")
        pending = runner.unfinished_videos()
        assert [vid for vid, _ in pending] == [v1, v2]
        assert all(item is not None and item.kind == "local" for _, item in pending)

        results = runner.resume()
        by_id = {r.video_id: r for r in results}
        assert set(by_id) == {v1, v2} and all(r.completed for r in results), [(r.video_id, r.error) for r in results]
        # v1: early stages reused, AUDIO re-run (attempt 2), the rest ran once
        assert Stage.DOWNLOAD in by_id[v1].stages_cached and Stage.FRAME_EXTRACTION in by_id[v1].stages_cached
        assert Stage.AUDIO in by_id[v1].stages_run and runner.db.get_stage(v1, Stage.AUDIO).attempts == 2
        assert runner.db.get_stage(v1, Stage.AUDIO).error is None
        # v2: everything ran, input came from the database
        assert by_id[v2].stages_cached == [] and Stage.DOWNLOAD in by_id[v2].stages_run
        for vid in (v0, v1, v2):
            assert runner.db.get_video(vid)["status"] == "done"  # type: ignore[index]
            assert (cfg.export_dir / "per_video" / vid / "manifest.json").exists()
        # nothing left; a third resume is a no-op and v0 was never touched
        assert runner.resume() == []
        assert runner.db.get_stage(v0, Stage.EXPORT).attempts == 1
    finally:
        runner.close()


def test_resume_skips_rejected_unless_asked(tmp_path: Path, test_config):
    cfg = test_config.model_copy(deep=True)
    runner = PipelineRunner(cfg)
    try:
        item = classify_input("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert item is not None
        (vid, _), = runner.register_inputs([item])
        runner.db.start_stage(vid, Stage.DOWNLOAD)
        runner.db.fail_stage(vid, Stage.DOWNLOAD, "DownloadRejected: rejected: duration 200.0 min exceeds limits.max_duration_seconds (120 min)")
        assert runner.unfinished_videos() == []
        pending = runner.unfinished_videos(include_rejected=True)
        assert len(pending) == 1 and pending[0][0] == vid and pending[0][1] is not None and pending[0][1].youtube_id == "dQw4w9WgXcQ"
        assert runner.resume() == []
    finally:
        runner.close()


def test_resume_cli_when_nothing_pending(tmp_path: Path):
    from typer.testing import CliRunner

    from video_dataset import cli

    cli.state.config = None
    cli.state.config_path = None
    cli.state.overrides = {}
    cli.state.log_level = None
    r = CliRunner().invoke(cli.app, ["--data-dir", str(tmp_path / "d"), "resume"])
    assert r.exit_code == 0, r.output
    assert "Nothing to resume" in r.output
