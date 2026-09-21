"""Threshold-triggered Hugging Face upload in shards (Hub client replaced by a fake)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_dataset.config import load_config
from video_dataset.dataset import upload as up
from video_dataset.dataset.export import DATASET_FILES
from video_dataset.utils.io import read_jsonl, write_json_atomic, write_jsonl


class FakeUploader:
    def __init__(self, fail_times: int = 0):
        self.calls: list[tuple] = []
        self.staged_files: list[str] = []
        self.fail_times = fail_times

    def ensure_repo(self) -> str:
        self.calls.append(("repo",))
        return "https://huggingface.co/datasets/u/d"

    def ensure_card(self, readme: str) -> bool:
        self.calls.append(("card", readme))
        return True

    def upload_dir(self, local_dir: Path, path_in_repo: str, message: str) -> str:
        if self.fail_times:
            self.fail_times -= 1
            raise ConnectionError("flaky")
        self.staged_files = sorted(str(p.relative_to(local_dir)) for p in Path(local_dir).rglob("*") if p.is_file())
        self.calls.append(("upload", path_in_repo, message))
        return f"https://huggingface.co/datasets/u/d/commit/{len(self.calls)}"


def _fake_final(cfg, videos: list[str], rows_per_video: int = 3, media: bool = True) -> None:
    final = cfg.export_dir
    final.mkdir(parents=True, exist_ok=True)
    records = {n: [] for n in DATASET_FILES}
    for vid in videos:
        pv = final / "per_video" / vid
        pv.mkdir(parents=True, exist_ok=True)
        frames = [{"record_id": f"frame_{vid}_{i}", "video_id": vid, "frame_path": f"frames/{vid}/scene_001/f{i}.jpg"} for i in range(rows_per_video)]
        records["frames"] += frames
        write_jsonl(pv / "frames.jsonl", frames)
        write_json_atomic(pv / "manifest.json", {"video_id": vid, "counts": {"frames": rows_per_video}})
        if media:
            for i in range(rows_per_video):
                p = cfg.data_dir / "frames" / vid / "scene_001" / f"f{i}.jpg"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"j" * 1000)
            c = cfg.data_dir / "clips" / vid / "clip_001.mp4"
            c.parent.mkdir(parents=True, exist_ok=True)
            c.write_bytes(b"m" * 5000)
    for n in DATASET_FILES:
        write_jsonl(final / f"{n}.jsonl", records[n])
    write_jsonl(final / "dataset.jsonl", [{"record_type": "frame", **r} for r in records["frames"]])
    (final / "dataset.parquet").write_bytes(b"p" * 100)
    write_json_atomic(final / "statistics.json", {"frames": len(records["frames"])})
    write_json_atomic(final / "manifest.json", {"videos": [{"video_id": v} for v in videos]})


def _cfg(tmp_path: Path, **over):
    base = {"project.data_dir": str(tmp_path / "data"), "project.log_level": "ERROR", "upload.provider": "huggingface", "upload.repo_id": "u/d", "upload.threshold_mb": "0.001"}
    base.update(over)
    return load_config(None, base)


def test_disabled_provider_is_noop(tmp_path: Path):
    cfg = _cfg(tmp_path, **{"upload.provider": "none"})
    _fake_final(cfg, ["vid_a"])
    assert up.check_and_upload(cfg, uploader=FakeUploader()) is None


def test_nothing_exported_is_noop(tmp_path: Path):
    cfg = _cfg(tmp_path)
    assert up.check_and_upload(cfg, uploader=FakeUploader()) is None


def test_below_threshold_waits_and_force_uploads(tmp_path: Path):
    cfg = _cfg(tmp_path, **{"upload.threshold_mb": "1024"})
    _fake_final(cfg, ["vid_a", "vid_b"])
    fake = FakeUploader()
    assert up.check_and_upload(cfg, uploader=fake) is None and fake.calls == []
    plan = up.check_and_upload(cfg, force=True, dry_run=True, uploader=fake)
    assert plan and plan["dry_run"] and plan["shard"] == "shard_0001" and fake.calls == []
    assert (cfg.export_dir / "frames.jsonl").exists()  # dry run touches nothing
    res = up.check_and_upload(cfg, force=True, uploader=fake)
    assert res and res["commit"].endswith("/3") and [c[0] for c in fake.calls] == ["repo", "card", "upload"]


def test_threshold_upload_archives_and_starts_new_shard(tmp_path: Path):
    cfg = _cfg(tmp_path)  # threshold ~1 KB -> immediately reached
    _fake_final(cfg, ["vid_a", "vid_b"])
    fake = FakeUploader()
    res = up.check_and_upload(cfg, uploader=fake)
    assert res is not None and res["shard"] == "shard_0001" and res["path_in_repo"] == "shard_0001"
    assert set(res["videos"]) == {"vid_a", "vid_b"} and res["after_upload"] == "archive"
    # staged content: dataset files + per-video exports, no media by default, no state file
    assert "frames.jsonl" in fake.staged_files and "dataset.parquet" in fake.staged_files and "manifest.json" in fake.staged_files
    assert "per_video/vid_a/frames.jsonl" in fake.staged_files
    assert not any(f.startswith("frames/") or f.startswith("clips/") for f in fake.staged_files)
    assert up.STATE_FILE not in fake.staged_files
    # rotated: final/ is empty again, the archive holds the shard, media untouched
    assert not (cfg.export_dir / "frames.jsonl").exists() and not (cfg.export_dir / "per_video" / "vid_a").exists()
    archive = cfg.export_dir / "uploaded" / "shard_0001"
    assert (archive / "frames.jsonl").exists() and (archive / "per_video" / "vid_b" / "manifest.json").exists()
    assert (cfg.data_dir / "frames" / "vid_a" / "scene_001" / "f0.jpg").exists()
    state = up.UploadState.load(cfg.export_dir)
    assert state.next_shard == 2 and state.shards[0]["videos"] == ["vid_a", "vid_b"] and state.uploaded_videos == {"vid_a", "vid_b"}
    # nothing left -> no second upload; a new video starts shard_0002 (card already there, still checked)
    assert up.check_and_upload(cfg, uploader=fake) is None
    _fake_final(cfg, ["vid_c"])
    res2 = up.check_and_upload(cfg, uploader=fake)
    assert res2 and res2["shard"] == "shard_0002" and res2["videos"] == ["vid_c"]
    assert [c[1] for c in fake.calls if c[0] == "upload"] == ["shard_0001", "shard_0002"]
    assert (cfg.export_dir / "uploaded" / "shard_0002" / "manifest.json").exists()
    rows = list(read_jsonl(cfg.export_dir / "uploaded" / "shard_0001" / "frames.jsonl"))
    assert {r["video_id"] for r in rows} == {"vid_a", "vid_b"}  # earlier shard untouched


def test_include_media_and_delete_mode(tmp_path: Path):
    cfg = _cfg(tmp_path, **{"upload.include_media": "true", "upload.after_upload": "delete", "upload.path_in_repo": "v1/"})
    _fake_final(cfg, ["vid_a"])
    size, files, videos = up.measure_shard(cfg.export_dir, cfg.data_dir, True)
    assert videos == ["vid_a"] and files >= 4 + 2 + 3 + 1  # root files + per_video + 3 frames + clip
    fake = FakeUploader()
    res = up.check_and_upload(cfg, uploader=fake)
    assert res and res["path_in_repo"] == "v1/shard_0001"
    assert "frames/vid_a/scene_001/f0.jpg" in fake.staged_files and "clips/vid_a/clip_001.mp4" in fake.staged_files
    # delete mode: local dataset files, per-video export AND the uploaded media are gone; nothing archived
    assert not (cfg.export_dir / "frames.jsonl").exists() and not (cfg.export_dir / "uploaded").exists()
    assert not (cfg.data_dir / "frames" / "vid_a").exists() and not (cfg.data_dir / "clips" / "vid_a").exists()
    card = next(c[1] for c in fake.calls if c[0] == "card")
    assert 'data_files: "v1/shard_*/temporal_qa.jsonl"' in card and "frames/" in card


def test_upload_retries_then_succeeds_and_staging_is_cleaned(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(up.time, "sleep", lambda s: None)
    import video_dataset.utils.retry as retry_mod

    monkeypatch.setattr(retry_mod.time, "sleep", lambda s: None)
    cfg = _cfg(tmp_path, **{"upload.retries": "2"})
    _fake_final(cfg, ["vid_a"])
    fake = FakeUploader(fail_times=2)
    res = up.check_and_upload(cfg, uploader=fake)
    assert res and res["shard"] == "shard_0001"
    assert not list(cfg.export_dir.glob("vd-upload-*"))  # staging dir removed


def test_upload_failure_leaves_everything_in_place(tmp_path: Path, monkeypatch):
    import video_dataset.utils.retry as retry_mod

    monkeypatch.setattr(retry_mod.time, "sleep", lambda s: None)
    cfg = _cfg(tmp_path, **{"upload.retries": "0"})
    _fake_final(cfg, ["vid_a"])
    with pytest.raises(ConnectionError):
        up.check_and_upload(cfg, uploader=FakeUploader(fail_times=5))
    assert (cfg.export_dir / "frames.jsonl").exists() and (cfg.export_dir / "per_video" / "vid_a").exists()
    assert not (cfg.export_dir / up.STATE_FILE).exists() and not list(cfg.export_dir.glob("vd-upload-*"))


def test_missing_token_and_bad_config(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    cfg = _cfg(tmp_path)
    _fake_final(cfg, ["vid_a"])
    with pytest.raises(up.UploadError, match="HF_TOKEN"):
        up.check_and_upload(cfg)
    with pytest.raises(up.UploadError, match="after_upload"):
        up.check_and_upload(_cfg(tmp_path, **{"upload.after_upload": "shred"}), uploader=FakeUploader())
    with pytest.raises(up.UploadError, match="provider"):
        up.check_and_upload(_cfg(tmp_path, **{"upload.provider": "s3"}), uploader=FakeUploader())
    monkeypatch.setenv("HF_TOKEN", "hf_fake")
    with pytest.raises(up.UploadError, match="repo_id"):
        up.check_and_upload(_cfg(tmp_path, **{"upload.repo_id": "no-slash"}))


def test_cli_upload_status_and_dry_run(tmp_path: Path):
    from typer.testing import CliRunner

    from video_dataset import cli

    cli.state.config = None
    cli.state.config_path = None
    cli.state.overrides = {}
    cli.state.log_level = None
    cfg = _cfg(tmp_path, **{"upload.threshold_mb": "1024"})
    _fake_final(cfg, ["vid_a"])
    runner = CliRunner()
    common = ["--data-dir", str(cfg.data_dir), "--set", "upload.provider=huggingface", "--set", "upload.repo_id=u/d"]
    r = runner.invoke(cli.app, [*common, "upload", "--status"])
    assert r.exit_code == 0, r.output
    assert "current shard" in r.output and "repo=u/d" in r.output
    r = runner.invoke(cli.app, [*common, "upload", "--dry-run", "--force"])
    assert r.exit_code == 0, r.output
    assert "would upload shard_0001" in r.output
    assert json.loads((cfg.export_dir / "manifest.json").read_text())["videos"]  # untouched
    cli.state.config = None  # the CLI caches the config per process; a fresh process would not have the --set values
    r = runner.invoke(cli.app, ["--data-dir", str(cfg.data_dir), "upload"])
    assert r.exit_code == 1 and "upload.provider" in r.output
