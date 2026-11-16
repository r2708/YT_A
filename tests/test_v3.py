"""V3 (youtube_dataset_v3_pipeline.md): resolution knob, device selection, rights gate, the
per-source job, publish + remote verification, the cleanup gate and the unified loader.
Network-free: the source is a local file named in the rights manifest, the Hub is a directory."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from video_dataset.config import ResolutionConfig, load_config
from video_dataset.utils import device as device_mod
from video_dataset.utils.io import file_sha256, read_jsonl, write_jsonl
from video_dataset.v3 import legacy, rights
from video_dataset.v3.cleanup import cleanup_block_reason, run_cleanup
from video_dataset.v3.loader import AssetCache, CacheFull, RepoSource, UnifiedLoader
from video_dataset.v3.pipeline import job_dir, open_store
from video_dataset.v3.pipeline import run as run_v3
from video_dataset.v3.publish import publish
from video_dataset.v3.resolution import choose_resolution

VID_A, VID_B = "vid_aaaaaaaaaaaa", "vid_bbbbbbbbbbbb"


# ----------------------------------------------------------------------------- resolution
def test_resolution_policy_buckets():
    d = choose_resolution(1920, 1080)
    assert (d.bucket, d.action, d.target_width, d.target_height, d.upscaled) == ("720p", "downscaled", 1280, 720, False)
    assert choose_resolution(3840, 2160).target_height == 720
    assert choose_resolution(1280, 720).action == "kept"
    d = choose_resolution(854, 480)
    assert (d.bucket, d.action, d.target_width, d.target_height, d.include_in_primary_training) == ("native_480_576", "keep_native", 854, 480, True)
    d = choose_resolution(640, 360)
    assert d.bucket == "lowres_360" and not d.include_in_primary_training and not d.upscaled
    assert choose_resolution(320, 240).excluded
    d = choose_resolution(1080, 1920)  # portrait: the short edge decides, the aspect ratio stays
    assert (d.target_width, d.target_height) == (720, 1280)
    with pytest.raises(ValueError):
        choose_resolution(0, 720)


def test_resolution_target_is_configurable():
    d = choose_resolution(1920, 1080, ResolutionConfig(target=1080))
    assert (d.bucket, d.action, d.target_height) == ("1080p", "kept", 1080)
    d = choose_resolution(1280, 720, ResolutionConfig(target=1080))  # never upscaled, never called 1080p
    assert (d.bucket, d.target_height, d.upscaled) == ("native_720", 720, False)
    d = choose_resolution(1920, 1080, ResolutionConfig(target=480))
    assert (d.bucket, d.target_width, d.target_height) == ("480p", 854, 480)
    d = choose_resolution(854, 480, ResolutionConfig(target=720, allow_upscale=True))
    assert d.upscaled and d.bucket == "720p_upscaled" and not d.include_in_primary_training


def test_one_resolution_knob_drives_the_pipeline(tmp_path: Path):
    cfg = load_config()
    assert cfg.resolution.target == cfg.download.max_resolution == cfg.preprocess.max_transcode_height == cfg.frame_sampling.clip_max_height
    cfg = load_config(None, {"resolution.target": "1080"})
    assert (cfg.download.max_resolution, cfg.preprocess.max_transcode_height, cfg.frame_sampling.clip_max_height) == (1080, 1080, 1080)
    cfg = load_config(None, {"resolution.target": "720", "resolution.download_max": "1080"})  # download better, downscale clips
    assert (cfg.download.max_resolution, cfg.frame_sampling.clip_max_height) == (1080, 720)
    user = tmp_path / "my.yaml"
    user.write_text("resolution:\n  target: 480\ndownload:\n  max_resolution: 2160\n")
    cfg = load_config(user)
    assert (cfg.download.max_resolution, cfg.frame_sampling.clip_max_height) == (2160, 480)  # an explicit key still wins
    with pytest.raises(ValueError):
        load_config(None, {"resolution.target": "0"})


# ----------------------------------------------------------------------------- device
@pytest.fixture()
def clean_device(monkeypatch):
    monkeypatch.delenv(device_mod.ENV_DEVICE, raising=False)
    monkeypatch.delenv(device_mod.ENV_REQUIRE, raising=False)
    device_mod.configure(False)
    yield monkeypatch
    device_mod.configure(False)


def _fake(monkeypatch, system: str, cuda: bool, mps: bool) -> None:
    monkeypatch.setattr(device_mod.platform, "system", lambda: system)
    monkeypatch.setattr(device_mod, "_cuda_available", lambda torch: cuda)
    monkeypatch.setattr(device_mod, "_mps_available", lambda torch: mps)


def test_device_auto_detection_per_os(clean_device):
    _fake(clean_device, "Windows", cuda=True, mps=False)
    assert device_mod.resolve_device("auto") == "cuda"
    _fake(clean_device, "Windows", cuda=False, mps=False)
    assert device_mod.resolve_device("auto") == "cpu"
    _fake(clean_device, "Darwin", cuda=False, mps=True)
    assert device_mod.resolve_device("auto") == "mps"
    _fake(clean_device, "Darwin", cuda=False, mps=False)
    assert device_mod.resolve_device("auto") == "cpu"
    _fake(clean_device, "Linux", cuda=True, mps=False)
    assert device_mod.resolve_device("auto") == "cuda"


def test_device_overrides_and_fallback(clean_device):
    _fake(clean_device, "Windows", cuda=False, mps=False)
    assert device_mod.resolve_device("cuda") == "cpu"  # unavailable accelerator: warning + CPU
    info = device_mod.runtime_info("cuda")
    assert info["selected_device"] == "cpu" and info["fallback_reason"] and info["device_policy"] == "cuda" and info["torch_version"]
    clean_device.setenv(device_mod.ENV_REQUIRE, "true")
    with pytest.raises(device_mod.AcceleratorUnavailable):
        device_mod.resolve_device("cuda")
    clean_device.delenv(device_mod.ENV_REQUIRE)
    _fake(clean_device, "Linux", cuda=True, mps=False)
    clean_device.setenv(device_mod.ENV_DEVICE, "cpu")  # the env var beats the configured device on every OS
    assert device_mod.resolve_device("cuda") == "cpu"
    assert device_mod.runtime_info("cuda")["device_policy_source"] == "env"
    clean_device.setenv(device_mod.ENV_DEVICE, "gpu")
    with pytest.raises(ValueError):
        device_mod.resolve_device("auto")


def test_runtime_block_in_environment_manifest(clean_device):
    from video_dataset.utils.environment import environment_manifest

    runtime = environment_manifest(load_config())["runtime"]
    assert runtime["selected_device"] in ("cpu", "cuda", "mps") and runtime["os"] and "accelerated_media_decode" in runtime


# ----------------------------------------------------------------------------- fixtures
def _clip(vid: str, n: int, start: float, end: float, status: str = "accepted", target: str = "single_shot", prompt: str | None = None) -> dict:
    return {
        "record_id": f"clip_{vid[4:]}_clip_{n:03d}", "record_type": "clip", "video_id": vid, "clip_id": f"clip_{n:03d}", "scene_id": f"scene_{n:03d}",
        "start_time": start, "end_time": end, "split": "train", "source_group_id": f"channel:{vid}", "generation_target": target, "boundary_check": "passed",
        "generation_prompt": prompt if prompt is not None else f"A {end - start:.0f}-second wide shot, static camera.", "prompt_variants": ["Wide static shot."],
        "description": "A shot.", "camera": {"shot_type": "wide", "camera_angle": "eye_level", "movement": "static", "speed": None}, "validation": {"status": status},
    }


@pytest.fixture()
def v3_env(tmp_path: Path, synthetic_video: Path):
    """A legacy export with two sources (A, B) and a config whose thresholds fit the flat synthetic video."""
    final = tmp_path / "data" / "final"
    final.mkdir(parents=True)
    prov = [
        {"video_id": VID_A, "url": "https://www.youtube.com/watch?v=aaaaaaaaaaa", "youtube_id": "aaaaaaaaaaa", "channel": "Chan A", "title": "A", "duration": 14.0, "license_status": "creative_commons", "rights_status": "declared_cc_by", "source_sha256": None, "source_group_id": f"channel:{VID_A}"},
        {"video_id": VID_B, "url": "https://www.youtube.com/watch?v=bbbbbbbbbbb", "youtube_id": "bbbbbbbbbbb", "channel": "Chan B", "title": "B", "duration": 14.0, "license_status": "standard", "rights_status": "restricted", "source_sha256": None, "source_group_id": f"channel:{VID_B}"},
    ]
    clips = [
        _clip(VID_A, 1, 0.0, 4.0), _clip(VID_A, 2, 4.0, 8.0), _clip(VID_A, 3, 0.0, 4.0),  # 3 = the same span as 1: a duplicate
        _clip(VID_A, 4, 8.0, 11.5, status="review"), _clip(VID_A, 5, 4.0, 8.0, target="not_for_generation"), _clip(VID_A, 6, 8.0, 11.0, prompt=""),
        _clip(VID_A, 7, 2.0, 6.0),  # crosses the cut at 4.0s
        _clip(VID_A, 8, 4.5, 7.5, prompt="Very slowly, then quickly."),  # names nothing
        _clip(VID_B, 1, 0.0, 4.0),
    ]
    write_jsonl(final / "provenance.jsonl", prov)
    write_jsonl(final / "clips.jsonl", clips)
    (final / "manifest.json").write_text(json.dumps({"release_id": "r1", "source_revision": "abc", "videos": [{"video_id": VID_A, "split": "train", "counts": {"clips": 6, "scenes": 4}}, {"video_id": VID_B, "split": "test", "counts": {"clips": 1}}]}))
    cfg = load_config(None, {
        "project.data_dir": str(tmp_path / "data"), "project.log_level": "WARNING", "resolution.target": "360",
        "v3.clips.min_sharpness": "0", "v3.clips.min_motion": "0", "v3.clips.preset": "veryfast", "v3.v2v.pairs_per_clip": "4", "v3.verify_readback_files": "1",
    })
    return cfg, final, synthetic_video


def _approve(cfg, vid: str, video: Path, **extra) -> Path:
    path = rights.default_rights_path(cfg)
    entries = rights.load_rights(path)
    (path.parent / "permission_a.txt").write_text("written permission")
    entries[vid].update(rights_status="approved", download_authorized=True, ml_training_allowed=True, derivative_processing_allowed=True, source_license="written permission 2026-11-01",
                        evidence_uri="permission_a.txt", verified_at="2026-11-10", verified_by="reviewer", local_path=str(video), **extra)
    write_jsonl(path, entries.values())
    return path


class FakeRepo:
    """A directory standing in for a Hub repository. Media files report an LFS sha256, small files none."""

    def __init__(self, repo_id: str, root: Path, corrupt: bool = False):
        self.repo_id, self.root, self.corrupt = repo_id, root, corrupt

    def ensure_repo(self) -> str:
        self.root.mkdir(parents=True, exist_ok=True)
        return str(self.root)

    def upload_dir(self, local_dir: Path, message: str) -> str:
        shutil.copytree(local_dir, self.root, dirs_exist_ok=True)
        if self.corrupt:
            for f in self.root.rglob("*.mp4"):
                f.write_bytes(f.read_bytes()[:-10])
                break
        return "commit"

    def remote_files(self) -> dict[str, str | None]:
        return {str(p.relative_to(self.root)): (file_sha256(p) if p.suffix == ".mp4" else None) for p in self.root.rglob("*") if p.is_file()}

    def download(self, path: str, dest_dir: Path) -> Path:
        dst = dest_dir / path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.root / path, dst)
        return dst


def _prepare(cfg, video: Path, approve: bool = True) -> None:
    legacy.build_inventory(cfg)
    rights.init_rights(legacy.load_source_index(cfg), rights.default_rights_path(cfg))
    if approve:
        _approve(cfg, VID_A, video)


# ----------------------------------------------------------------------------- legacy + rights
def test_inventory_and_rights_gate(v3_env):
    cfg, final, video = v3_env
    before = {p.name: file_sha256(p) for p in final.iterdir() if p.is_file()}
    out = legacy.build_inventory(cfg)
    assert out["source_videos"] == 2 and out["clips"] == 9
    ldir = cfg.v3_dir / "legacy"
    for name in ("legacy_inventory.json", "legacy_schema_report.json", "legacy_source_index.parquet", "legacy_dataset_revision.txt"):
        assert (ldir / name).exists()
    inv = json.loads((ldir / "legacy_inventory.json").read_text())
    clips_file = next(f for f in inv["files"] if f["path"] == "clips.jsonl")
    assert clips_file["rows"] == 9 and clips_file["sha256"] == before["clips.jsonl"]
    assert {p.name: file_sha256(p) for p in final.iterdir() if p.is_file()} == before  # the legacy export is never touched
    index = legacy.load_source_index(cfg)
    assert index[0]["split"] == "train" and len(index[0]["clip_ids"]) == 8

    path = rights.default_rights_path(cfg)
    assert rights.init_rights(index, path) == (2, 2)
    entries = rights.load_rights(path)
    assert all(e["rights_status"] == "metadata_only" and e["download_authorized"] is False for e in entries.values())  # CC label != permission
    out = run_v3(cfg)
    assert out["approved"] == 0 and set(out["results"].values()) == {"RIGHTS_PENDING"}
    assert not job_dir(cfg, VID_A).exists()  # nothing acquired without an approved entry

    entries[VID_A].update(rights_status="approved")  # a bare status is not enough
    write_jsonl(path, entries.values())
    assert not rights.is_approved(rights.load_rights(path)[VID_A], path.parent)
    assert run_v3(cfg)["approved"] == 0
    _approve(cfg, VID_A, video)
    assert rights.is_approved(rights.load_rights(path)[VID_A], path.parent)
    assert rights.init_rights(index, path) == (0, 2)  # re-running init never rewrites a decision
    assert rights.is_approved(rights.load_rights(path)[VID_A], path.parent)
    report = rights.rights_report(index, rights.load_rights(path), path.parent)
    assert report["eligible"] == [VID_A] and report["by_status"]["metadata_only"] == 1
    _approve(cfg, VID_A, video, expires_at="2020-01-01")
    assert any("expired" in p for p in rights.entry_problems(rights.load_rights(path)[VID_A], path.parent))
    e = dict(rights.load_rights(path)[VID_A], expires_at=None, ml_training_allowed="yes")
    assert "ml_training_allowed is not true" in rights.entry_problems(e, path.parent)


# ----------------------------------------------------------------------------- end to end
def test_v3_end_to_end(v3_env, tmp_path: Path):
    cfg, _final, video = v3_env
    _prepare(cfg, video)
    out = run_v3(cfg)
    assert out["results"] == {VID_A: "TASK_ASSETS_CREATED", VID_B: "RIGHTS_PENDING"}
    rec = json.loads((job_dir(cfg, VID_A) / "records.json").read_text())
    assert rec["skipped"] == {"legacy_status_review": 1, "generation_target_not_for_generation": 1, "no_prompt": 1}
    assert rec["runtime"]["selected_device"] and rec["match"]["status"] == "matched"
    t2v = {r["clip_id"]: r for r in rec["t2v"]}
    cut, vague = t2v.pop(f"clip_{VID_A[4:]}_clip_007"), t2v.pop(f"clip_{VID_A[4:]}_clip_008")
    assert cut["quality_status"] == "rejected" and "PySceneDetect found 1 cut" in cut["quality_reasons"][0]
    assert vague["quality_status"] == "needs_human_review" and "no noun" in vague["quality_reasons"][0] and vague["prompt_check"] == "failed"
    assert len(t2v) == 3 and all(r["quality_status"] == "accepted" and r["prompt_check"] == "passed" for r in t2v.values()), [(r["clip_id"], r["quality_reasons"]) for r in t2v.values()]
    first = t2v[f"clip_{VID_A[4:]}_clip_001"]
    assert first["annotation_origin"] == "reused_legacy" and first["resolution_bucket"] == "360p" and first["upscaled"] is False
    assert (first["training_width"], first["training_height"], first["source_width"]) == (640, 360, 640) and abs(first["duration_seconds"] - 4.0) <= 0.1
    clip_file = job_dir(cfg, VID_A) / "clips" / f"{first['clip_id']}.mp4"
    assert clip_file.exists() and file_sha256(clip_file) == first["target_media"]["sha256"]  # a real clip, not a path string
    pairs = rec["v2v"]
    assert len(pairs) == 12 and {p["transform"] for p in pairs} == {"cinematic_grade", "film_grain", "monochrome", "slow_motion"}
    bad = [(p["sample_id"], p["pair_checks"]) for p in pairs if p["quality_status"] != "accepted"]
    assert not bad, bad
    slow = next(p for p in pairs if p["transform"] == "slow_motion" and p["clip_id"] == first["clip_id"])
    ta = slow["temporal_alignment"]
    assert ta["speed_factor"] == 0.5 and abs(ta["target_duration_seconds"] - 8.0) <= 0.2 and abs(ta["target_fps"] - ta["source_fps"] / 2) < 0.1
    assert all(p["target_media"]["sha256"] != p["source_media"]["sha256"] and p["source_media"]["repo_id"] == cfg.v3.repos.t2v for p in pairs)
    mtime = clip_file.stat().st_mtime
    assert run_v3(cfg)["results"][VID_A] == "TASK_ASSETS_CREATED" and clip_file.stat().st_mtime == mtime  # resumed, not redone

    # the cleanup gate: an approved source keeps its media until the upload is verified
    assert cleanup_block_reason(cfg, VID_A) and cleanup_block_reason(cfg, VID_B) is None
    assert run_cleanup(cfg)["cleaned"] == [] and clip_file.exists()

    dry = publish(cfg, dry_run=True)
    t2v_root, v2v_root = Path(dry["staged"]["t2v"]), Path(dry["staged"]["v2v"])
    train = list(read_jsonl(t2v_root / "data" / "train.jsonl"))
    assert len(train) == 2 and dry["t2v_accepted"] == 2  # clip 3 is byte-identical to clip 1 -> rejected as a duplicate
    report = json.loads((t2v_root / "quality" / "t2v_quality_report.json").read_text())
    assert report["by_status"] == {"accepted": 2, "rejected": 2, "needs_human_review": 1} and any("duplicate of" in r["reasons"][0] for r in report["rejections"])
    for rel in ("README.md", "dataset_info.json", "schemas/t2v_schema.json", "schemas/media_schema.json", "manifests/media_manifest.jsonl", "manifests/rights_manifest.jsonl"):
        assert (t2v_root / rel).exists(), rel
    assert all((t2v_root / r["target_media"]["path"]).exists() for r in train)
    v2v_train = list(read_jsonl(v2v_root / "data" / "train.jsonl"))
    assert len(v2v_train) == 8 and (v2v_root / "manifests" / "pair_manifest.jsonl").exists()
    assert all((v2v_root / r["target_media"]["path"]).exists() and not (v2v_root / r["source_media"]["path"]).exists() for r in v2v_train)  # sources referenced, not copied
    pub_rights = list(read_jsonl(t2v_root / "manifests" / "rights_manifest.jsonl"))
    assert [r["source_video_id"] for r in pub_rights] == [VID_A] and "evidence_uri" not in pub_rights[0] and "local_path" not in pub_rights[0]
    assert "local_path" not in next(iter(read_jsonl(t2v_root / "manifests" / "media_manifest.jsonl")))
    assert cleanup_block_reason(cfg, VID_A)  # a dry run verifies nothing

    # a corrupted remote copy: nothing is marked verified, media stays
    bad_clients = {"t2v": FakeRepo(cfg.v3.repos.t2v, tmp_path / "bad_t2v", corrupt=True), "v2v": FakeRepo(cfg.v3.repos.v2v, tmp_path / "bad_v2v")}
    out = publish(cfg, clients=bad_clients)
    assert out["verified"] is False and out["repos"]["t2v"]["verification"]["mismatched"]
    store = open_store(cfg)
    job = store.get(VID_A)
    store.close()
    assert job["state"] == "UPLOADED" and not job["flags"].get("cleanup_allowed") and cleanup_block_reason(cfg, VID_A)
    assert run_cleanup(cfg)["cleaned"] == [] and clip_file.exists()

    hub = {"t2v": FakeRepo(cfg.v3.repos.t2v, tmp_path / "hub_t2v"), "v2v": FakeRepo(cfg.v3.repos.v2v, tmp_path / "hub_v2v")}
    out = publish(cfg, clients=hub)
    assert out["verified"] is True and out["repos"]["t2v"]["verification"]["read_back"] == 1
    store = open_store(cfg)
    job = store.get(VID_A)
    states = [h["state"] for h in store.history(VID_A)]
    store.close()
    assert job["state"] == "CLEANUP_ALLOWED" and job["flags"]["metadata_upload_verified"] and job["flags"]["visual_asset_upload_verified"]
    assert states.index("REMOTE_ASSETS_VERIFIED") > states.index("UPLOADED") > states.index("TASK_ASSETS_CREATED")
    assert cleanup_block_reason(cfg, VID_A) is None

    # one loader over both repositories, cross-repository source clips, bounded cache
    local_dirs = {cfg.v3.repos.t2v: hub["t2v"].root, cfg.v3.repos.v2v: hub["v2v"].root}
    loader = UnifiedLoader.from_config(cfg, local_dirs=local_dirs)
    seen: dict[str, int] = {}
    for batch in loader.batches():
        for s in batch["samples"]:
            assert all(p.exists() for p in s["media"].values())
            if batch["task"] == "video_to_video":
                assert set(s["media"]) == {"source", "target"}
        seen[batch["task"]] = seen.get(batch["task"], 0) + len(batch["samples"])
        loader.release(batch)
    assert seen == {"text_to_video": 2, "video_to_video": 8} and not loader.stats["failed"]
    biggest = max(p.stat().st_size for p in hub["v2v"].root.rglob("*.mp4"))
    small = UnifiedLoader({rid: RepoSource(rid, d) for rid, d in local_dirs.items()}, AssetCache(tmp_path / "small_cache", int(biggest * 2.5)))
    small.task_repo = {"text_to_video": cfg.v3.repos.t2v, "video_to_video": cfg.v3.repos.v2v}
    n = 0
    for batch in small.batches():
        n += len(batch["samples"])
        assert small.cache.bytes <= small.cache.max_bytes
        small.release(batch)
    assert n == 10 and small.cache.peak_bytes <= small.cache.max_bytes  # everything read, the limit never exceeded
    (hub["t2v"].root / train[0]["target_media"]["path"]).write_bytes(b"tampered")
    fresh = UnifiedLoader.from_config(cfg.model_copy(update={"v3": cfg.v3.model_copy(update={"cache": cfg.v3.cache.model_copy(update={"dir": str(tmp_path / "fresh_cache")})})}), local_dirs=local_dirs)
    got = sum(len(b["samples"]) for b in fresh.batches())
    assert got < 10 and fresh.stats["failed"] and "checksum mismatch" in fresh.stats["failed"][0]["error"]  # reported, not silently used

    out = run_cleanup(cfg)
    assert out["cleaned"] == [VID_A] and out["bytes"] > 0 and not clip_file.exists()
    store = open_store(cfg)
    assert store.get(VID_A)["state"] == "COMPLETE"
    store.close()
    assert (job_dir(cfg, VID_A) / "records.json").exists()
    assert publish(cfg, dry_run=True)["t2v_accepted"] == 2  # re-staging after cleanup still works (assets are on the Hub)


def test_cache_refuses_to_exceed_limit_with_pinned_files(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / "media").mkdir(parents=True)
    for name in ("a.mp4", "b.mp4"):
        (repo / "media" / name).write_bytes(b"x" * 1000)
    src = RepoSource("o/r", repo)
    cache = AssetCache(tmp_path / "cache", 1500)
    ref_a, ref_b = ({"repo_id": "o/r", "path": f"media/{n}", "sha256": file_sha256(repo / "media" / n)} for n in ("a.mp4", "b.mp4"))
    fetch = lambda repo_id, path, dest: src.fetch(path, dest)  # noqa: E731
    cache.get(ref_a, fetch)
    with pytest.raises(CacheFull):
        cache.get(ref_b, fetch)  # a is pinned by the active batch
    cache.release(ref_a)
    assert cache.get(ref_b, fetch).exists() and cache.bytes == 1000  # a was evicted
    (tmp_path / "cache" / "junk.part").write_bytes(b"partial")
    assert AssetCache(tmp_path / "cache", 1500).bytes == 1000 and not (tmp_path / "cache" / "junk.part").exists()  # crash recovery


def test_annotation_mismatch_goes_to_review(v3_env):
    cfg, final, video = v3_env
    rows = list(read_jsonl(final / "provenance.jsonl"))
    rows[0]["duration"] = 120.0  # the legacy annotations describe a different edit of the video
    write_jsonl(final / "provenance.jsonl", rows)
    _prepare(cfg, video)
    assert run_v3(cfg)["results"][VID_A] == "FAILED_REVIEW"
    store = open_store(cfg)
    job = store.get(VID_A)
    store.close()
    assert job["failure"] == "FAILED_REVIEW" and "annotation_mismatch" in job["reason"]
    assert run_v3(cfg)["results"][VID_A] == "FAILED_REVIEW"  # not retried until a person looks at it
    assert publish(cfg, dry_run=True)["t2v_accepted"] == 0


# ----------------------------------------------------------------------------- no AI models
def test_default_config_uses_no_models():
    from video_dataset.config import assert_no_models, model_providers_in_use

    cfg = load_config()
    assert cfg.project.no_models and cfg.project.no_models_except == ["transcription", "ocr", "vision.provider=cv_models"]
    assert len(model_providers_in_use(cfg)) == 3  # Whisper, OCR and YOLO + CLIP are the only models, all allowed explicitly
    assert (cfg.transcription.provider, cfg.ocr.provider, cfg.vision.provider, cfg.llm.provider) == ("faster_whisper", "rapidocr", "cv_models", "none")
    assert cfg.vision.aesthetic and cfg.validation.clip_alignment == "auto"
    assert cfg.scene_detection.detector == "content" and cfg.qa.generator == "both" and cfg.validation.nlp_model == "auto"  # PySceneDetect + spaCy
    assert_no_models(cfg)
    for key, value in (("project.no_models_except", "[ocr]"), ("vision.provider", "hf"), ("vision.provider", "anthropic"), ("vision.enrichers", "[clip]"), ("llm.provider", "anthropic")):
        with pytest.raises(ValueError, match="no_models"):
            assert_no_models(load_config(None, {key: value}))
    assert_no_models(load_config(None, {"vision.provider": "cv_models", "project.no_models": "false"}))  # an explicit opt-out still works
