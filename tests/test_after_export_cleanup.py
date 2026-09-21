"""Automatic cleanup once a video is exported, and merge-on-aggregate so the final dataset only grows."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_dataset.config import load_config
from video_dataset.dataset.aggregate import aggregate_exports, load_existing_records, merge_records
from video_dataset.dataset.export import DATASET_FILES
from video_dataset.pipeline.runner import PipelineRunner
from video_dataset.storage import cleanup as cl
from video_dataset.storage.paths import DataPaths
from video_dataset.storage.state_db import StateDB
from video_dataset.utils.io import read_json, read_jsonl, write_json_atomic, write_jsonl
from video_dataset.utils.urls import classify_input

# --------------------------------------------------------------------------- plan levels


def _touch(p: Path, size: int = 10) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * size)
    return p


def _populate(paths: DataPaths, vid: str) -> dict[str, Path]:
    return {
        "source": _touch(paths.video_dir(vid) / "source.mkv", 100),
        "video": _touch(paths.video_file(vid), 100),
        "meta": _touch(paths.metadata_file(vid), 5),
        "wav": _touch(paths.audio_file(vid), 50),
        "events": _touch(paths.audio_events_file(vid), 5),
        "scenes": _touch(paths.scenes_file(vid), 5),
        "frame": _touch(paths.frames_dir(vid) / "scene_001" / "f.jpg", 5),
        "frames_json": _touch(paths.frames_file(vid), 5),
        "clip": _touch(paths.clips_dir(vid) / "clip_001.mp4", 20),
        "transcript": _touch(paths.transcript_file(vid), 5),
        "ocr": _touch(paths.ocr_file(vid), 5),
        "vision": _touch(paths.vision_file(vid), 5),
        "qa": _touch(paths.qa_file(vid), 5),
        "validated": _touch(paths.validated_file(vid), 5),
        "log": _touch(paths.video_log_file(vid), 5),
    }


def test_plan_after_export_levels(tmp_path: Path):
    paths = DataPaths(tmp_path / "data")
    vid = "vid_aaaaaaaaaaaa"
    f = _populate(paths, vid)

    assert cl.plan_after_export(paths, vid, "none").paths == []

    media = cl.plan_after_export(paths, vid, "media")
    assert set(media.paths) == {f["source"], f["video"], f["wav"]}
    assert media.bytes == 250

    full = cl.plan_after_export(paths, vid, "all")
    assert paths.video_dir(vid) in full.paths and f["source"] not in full.paths  # directory once, not double counted
    assert full.bytes == 100 + 100 + 5 + 50 + 5 + 5 + 5 + 5 + 5 + 5 + 5
    for kept in ("frame", "frames_json", "clip", "log"):
        assert f[kept] not in full.paths and not any(str(f[kept]).startswith(str(p)) for p in full.paths), kept

    with pytest.raises(ValueError):
        cl.plan_after_export(paths, vid, "everything")

    # frames_and_clips adds the whole frames/<id>/ (scene_NNN folders + frames.json) and clips/<id>/ dirs
    assert cl.plan_after_export(paths, vid, "none", frames_and_clips=True).paths == []
    with_media = cl.plan_after_export(paths, vid, "media", frames_and_clips=True)
    assert set(with_media.paths) == {f["source"], f["video"], f["wav"], paths.frames_dir(vid), paths.clips_dir(vid)}
    assert with_media.bytes == 250 + 5 + 5 + 20
    assert f["log"] not in with_media.paths

    removed, freed = cl.apply_plan(full)
    assert removed == len(full.paths) and freed == full.bytes
    assert not paths.video_dir(vid).exists() and not f["wav"].exists() and not f["validated"].exists()
    assert f["frame"].exists() and f["clip"].exists() and f["log"].exists()

    cl.apply_plan(cl.plan_after_export(paths, vid, "media", frames_and_clips=True))
    assert not paths.frames_dir(vid).exists() and not paths.clips_dir(vid).exists()
    assert not (paths.frames_dir(vid) / "scene_001").exists() and f["log"].exists()


# --------------------------------------------------------------------------- runner integration


@pytest.mark.integration
@pytest.mark.slow
def test_runner_cleans_after_export_and_keeps_dataset(synthetic_video: Path, test_config):
    cfg = test_config.model_copy(deep=True)
    cfg.cleanup.after_export = "all"
    cfg.ocr.provider = "none"
    runner = PipelineRunner(cfg)
    try:
        item = classify_input(str(synthetic_video))
        assert item is not None
        registered = runner.register_inputs([item])
        vid = registered[0][0]
        res = runner.run_batch(registered)[0]
        assert res.completed, res.error
        paths = runner.paths

        # working files gone, including the frames/<id>/scene_NNN folders and clips/<id>/ ...
        assert not paths.video_dir(vid).exists()
        assert not paths.audio_file(vid).exists() and not paths.validated_file(vid).exists()
        assert not paths.annotations_dir(vid).exists() and not paths.qa_file(vid).exists()
        assert not paths.frames_dir(vid).exists() and not paths.clips_dir(vid).exists()
        # ... per-video export and log kept
        assert paths.video_log_file(vid).exists()
        per_video = cfg.export_dir / "per_video" / vid
        assert (per_video / "manifest.json").exists() and (per_video / "frames.jsonl").exists()
        assert runner.db.get_video(vid)["status"] == "done"  # type: ignore[index]
        assert any("CLEANUP" in (row.get("stage") or "") for row in runner.db.get_log(vid))

        stats = aggregate_exports(cfg, runner.db)
        assert stats["frames"] >= 12
        frame_rows = list(read_jsonl(cfg.export_dir / "frames.jsonl"))
        assert frame_rows and all(r["frame_path"].startswith("frames/") for r in frame_rows)

        # re-running the same input reuses every checkpoint: nothing re-downloaded, nothing re-run
        res2 = runner.run_batch(registered)[0]
        assert res2.completed and res2.stages_run == [] and not paths.video_dir(vid).exists()
    finally:
        runner.close()


# --------------------------------------------------------------------------- merge on aggregate


def _fake_export(export_dir: Path, vid: str, n: int, tag: str = "a") -> None:
    d = export_dir / "per_video" / vid
    d.mkdir(parents=True, exist_ok=True)
    records = {name: [] for name in DATASET_FILES}
    records["frames"] = [{"record_id": f"frame_{vid}_{i}", "video_id": vid, "scene_id": "scene_001", "frame_id": f"f{i}", "timestamp": float(i), "caption": f"{tag}{i}", "frame_path": "x.jpg"} for i in range(n)]
    records["temporal_qa"] = [{"question_id": f"qa_{vid}_{i}", "video_id": vid, "type": "duration", "question": f"q{tag}{i}", "answer": "1", "evidence": {"start_time": 0, "end_time": 1}} for i in range(n)]
    for name in DATASET_FILES:
        write_jsonl(d / f"{name}.jsonl", records[name])
    write_json_atomic(d / "manifest.json", {"video_id": vid, "title": vid, "url": "u", "duration": 10.0, "counts": {k: len(v) for k, v in records.items()}, "validation_summary": {}})


def test_merge_records_replaces_reexported_video_and_dedupes():
    existing = {"frames": [{"record_id": "f1", "video_id": "A"}, {"record_id": "f2", "video_id": "B"}], "temporal_qa": [{"question_id": "q1", "video_id": "A"}]}
    fresh = {"frames": [{"record_id": "f2", "video_id": "B"}, {"record_id": "f3", "video_id": "B"}, {"record_id": "f3", "video_id": "B"}]}
    merged = merge_records(existing, fresh, {"B"})
    assert [r["record_id"] for r in merged["frames"]] == ["f1", "f2", "f3"]  # A kept, B replaced, duplicate f3 dropped
    assert [r["question_id"] for r in merged["temporal_qa"]] == ["q1"]
    assert merged["events"] == []


def test_aggregate_keeps_previous_videos_and_replaces_reexported(tmp_path: Path):
    cfg = load_config(None, {"project.data_dir": str(tmp_path / "data"), "project.log_level": "ERROR"})
    paths = DataPaths(cfg.data_dir)
    paths.ensure_all()
    db = StateDB(cfg.db_path)
    try:
        for vid in ("vid_a", "vid_b"):
            db.upsert_video(vid, "u", duration=10.0)
        _fake_export(cfg.export_dir, "vid_a", 2)
        _fake_export(cfg.export_dir, "vid_b", 3)
        stats = aggregate_exports(cfg, db)
        assert stats["frames"] == 5 and stats["temporal_qa"] == 5 and stats["videos"] == 2

        # per-video export of vid_a disappears (cleaned / moved): its records must survive the rebuild
        import shutil

        shutil.rmtree(cfg.export_dir / "per_video" / "vid_a")
        stats = aggregate_exports(cfg, db)
        assert stats["frames"] == 5 and stats["videos"] == 2
        ids = {r["video_id"] for r in read_jsonl(cfg.export_dir / "frames.jsonl")}
        assert ids == {"vid_a", "vid_b"}
        manifest = read_json(cfg.export_dir / "manifest.json")
        assert {m["video_id"] for m in manifest["videos"]} == {"vid_a", "vid_b"}

        # a new video is added: everything accumulates, nothing overwritten
        db.upsert_video("vid_c", "u", duration=10.0)
        _fake_export(cfg.export_dir, "vid_c", 1)
        stats = aggregate_exports(cfg, db)
        assert stats["frames"] == 6 and stats["videos"] == 3
        combined = list(read_jsonl(cfg.export_dir / "dataset.jsonl"))
        assert len(combined) == 12 and {r["record_type"] for r in combined} == {"frame", "temporal_qa"}

        # vid_b re-exported with different content: replaced, not duplicated
        _fake_export(cfg.export_dir, "vid_b", 2, tag="new")
        stats = aggregate_exports(cfg, db)
        assert stats["frames"] == 5
        b_rows = [r for r in read_jsonl(cfg.export_dir / "frames.jsonl") if r["video_id"] == "vid_b"]
        assert len(b_rows) == 2 and all(r["caption"].startswith("new") for r in b_rows)

        # parquet mirrors the merged set
        import pandas as pd

        assert len(pd.read_parquet(cfg.export_dir / "dataset.parquet")) == 10

        # merge_existing=false restores the old rebuild-only behaviour
        cfg.export.merge_existing = False
        stats = aggregate_exports(cfg, db)
        assert stats["frames"] == 3 and stats["videos"] == 2  # vid_a has no per_video export any more
    finally:
        db.close()


def test_load_existing_records_falls_back_to_combined_file(tmp_path: Path):
    write_jsonl(tmp_path / "dataset.jsonl", [{"record_type": "frame", "record_id": "f", "video_id": "v"}, {"record_type": "event", "record_id": "e", "video_id": "v"}, {"record_type": "bogus", "record_id": "x"}])
    existing = load_existing_records(tmp_path)
    assert [r["record_id"] for r in existing["frames"]] == ["f"] and [r["record_id"] for r in existing["events"]] == ["e"]
    assert "record_type" not in existing["frames"][0]
    assert json.dumps(existing["temporal_qa"]) == "[]"


@pytest.mark.integration
@pytest.mark.slow
def test_runner_keeps_frames_when_configured_or_uploading_media(synthetic_video: Path, test_config):
    for keep_via in ("flag", "include_media"):
        cfg = test_config.model_copy(deep=True)
        cfg.ocr.provider = "none"
        if keep_via == "flag":
            cfg.cleanup.frames_and_clips = False
        else:
            cfg.upload.include_media = True  # provider stays none: nothing is uploaded, media must survive
        runner = PipelineRunner(cfg)
        try:
            item = classify_input(str(synthetic_video))
            assert item is not None
            registered = runner.register_inputs([item])
            vid = registered[0][0]
            res = runner.run_batch(registered)[0]
            assert res.completed, res.error
            paths = runner.paths
            assert not paths.video_file(vid).exists() and not paths.audio_file(vid).exists()  # level media
            assert any(paths.frames_dir(vid).rglob("*.jpg")) and paths.frames_file(vid).exists(), keep_via
            assert paths.clips_dir(vid).exists(), keep_via
            frame_rows = list(read_jsonl(cfg.export_dir / "per_video" / vid / "frames.jsonl"))
            assert frame_rows and all((cfg.data_dir / r["frame_path"]).exists() for r in frame_rows)
        finally:
            runner.close()
