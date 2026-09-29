"""A live run must never have its RUNNING stages reset by another process (a `status` check from a
second terminal, or a second `run`). Only a crashed run - dead pid or no lock - is recovered."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from video_dataset.errors import PipelineBusy
from video_dataset.pipeline.runner import PipelineRunner
from video_dataset.stages import Stage, StageStatus
from video_dataset.utils.runlock import LOCK_NAME, RunLock, pid_alive
from video_dataset.utils.urls import classify_input


def _register_running(runner: PipelineRunner) -> str:
    item = classify_input("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert item is not None
    ((vid, _),) = runner.register_inputs([item])
    runner.db.start_stage(vid, Stage.FRAME_EXTRACTION)
    return vid


def test_pid_alive():
    assert pid_alive(os.getpid())
    assert not pid_alive(0)
    assert not pid_alive(2**22 + 12345)  # far above any plausible pid


def test_read_only_runner_does_not_reset_running_stages(test_config):
    cfg = test_config.model_copy(deep=True)
    live = PipelineRunner(cfg)  # the batch process: holds the lock
    try:
        vid = _register_running(live)
        assert (Path(cfg.data_dir) / LOCK_NAME).read_text() == str(os.getpid())
        viewer = PipelineRunner(cfg, recover=False)  # `video-dataset status` in another terminal
        try:
            assert viewer.db.get_stage(vid, Stage.FRAME_EXTRACTION).status == StageStatus.RUNNING
        finally:
            viewer.close()
        assert live.db.get_stage(vid, Stage.FRAME_EXTRACTION).status == StageStatus.RUNNING
        assert (
            Path(cfg.data_dir) / LOCK_NAME
        ).exists()  # the viewer never held the lock, so never removed it
    finally:
        live.close()
    assert not (Path(cfg.data_dir) / LOCK_NAME).exists()


def test_second_run_refused_while_a_live_process_holds_the_lock(test_config):
    cfg = test_config.model_copy(deep=True)
    first = PipelineRunner(cfg)
    try:
        vid = _register_running(first)
    finally:
        first.close()
    # pretend another live process owns the lock (the parent process is alive and is not us)
    other = os.getppid()
    assert pid_alive(other) and other != os.getpid()
    (Path(cfg.data_dir) / LOCK_NAME).write_text(str(other))
    with pytest.raises(PipelineBusy, match=str(other)):
        PipelineRunner(cfg)
    viewer = PipelineRunner(cfg, recover=False)  # status still works
    try:
        assert viewer.db.get_stage(vid, Stage.FRAME_EXTRACTION).status == StageStatus.RUNNING
    finally:
        viewer.close()
    assert (Path(cfg.data_dir) / LOCK_NAME).read_text() == str(other)  # not ours: left alone


def test_second_run_cli_exits_3(test_config, tmp_path: Path):
    from typer.testing import CliRunner

    from video_dataset import cli

    cfg = test_config.model_copy(deep=True)
    PipelineRunner(cfg).close()
    (Path(cfg.data_dir) / LOCK_NAME).write_text(str(os.getppid()))
    cli.state.config = None
    cli.state.config_path = None
    cli.state.overrides = {}
    cli.state.log_level = None
    r = CliRunner().invoke(cli.app, ["--data-dir", str(cfg.data_dir), "resume"])
    assert r.exit_code == 3, r.output
    assert "already running" in r.output


def test_stale_lock_is_recovered(test_config):
    cfg = test_config.model_copy(deep=True)
    first = PipelineRunner(cfg)
    try:
        vid = _register_running(first)
    finally:
        first.close()
    (Path(cfg.data_dir) / LOCK_NAME).write_text(str(2**22 + 12345))  # a crashed run's dead pid
    second = PipelineRunner(cfg)
    try:
        assert second.lock.held and second.lock_owner is None
        rec = second.db.get_stage(vid, Stage.FRAME_EXTRACTION)
        assert rec.status == StageStatus.FAILED and "interrupted" in (rec.error or "")
    finally:
        second.close()


def test_runlock_same_pid_reacquires(tmp_path: Path):
    a = RunLock(tmp_path)
    assert a.acquire()
    b = RunLock(tmp_path)  # same process, e.g. two runners in one script
    assert b.live_owner() is None and b.acquire()
    b.release()
    assert not (tmp_path / LOCK_NAME).exists()
