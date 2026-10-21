"""dataset_factory tests on a tiny synthetic fixture: no dataset download, no API, no paid calls.
ffmpeg is needed to cut the fixture clips; the module is skipped with a precise reason without it."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from dataset_factory import pipeline
from dataset_factory.captions import providers as prov
from dataset_factory.captions.cache import ProviderCache, cache_key
from dataset_factory.captions.heuristics import check_caption
from dataset_factory.config import ConfigError, FactoryConfig, load_factory_config
from dataset_factory.curation.deduplicate import group_duplicates
from dataset_factory.curation.eligibility import decide, resolve_rights
from dataset_factory.curation.splits import assign
from dataset_factory.exports.release import ExportConflict
from dataset_factory.media.frame_extract import extract_start_frame, verify_pair
from dataset_factory.media.probe import probe
from dataset_factory.schema.canonical import CanonicalRecord
from dataset_factory.schema.source_mapping import map_source_row
from dataset_factory.state import StateStore

FFMPEG = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg binary not found on PATH: install it (macOS: brew install ffmpeg) to run the dataset_factory fixture tests")
ROOT = Path(__file__).resolve().parents[2]


def _ffmpeg(*args: str) -> None:
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", *args], check=True)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _row(vid: str, clip: str, start: float, end: float, **extra):  # type: ignore[no-untyped-def]
    row = {
        "record_id": f"clip_{vid}_{clip}", "record_type": "clip", "video_id": vid, "scene_id": "scene_001", "clip_id": clip, "clip_path": f"clips/{vid}/{clip}.mp4",
        "start_time": start, "end_time": end, "fps": 24.0, "width": 640, "height": 360, "boundary_check": "passed",
        "description": "A 4.0-second shot with normal lighting (dominant colors #112233); the camera is static.",
        "generation_prompt": "A 4-second medium shot of a coloured panel, static camera, the panel slowly drifts right.",
        "prompt_variants": ["Static camera: a coloured panel drifts right over 4 seconds."],
        "actions": [], "camera": {"movement": "static", "camera_movement": "static", "shot_size": "medium"}, "quality_flags": [], "confidence": 0.9, "tier": "silver",
        "validation": {"status": "accepted", "checks": {"timestamps": True}}, "split": "train",
    }
    row.update(extra)
    return row


@pytest.fixture(scope="module")
def fixture_source(tmp_path_factory: pytest.TempPathFactory, synthetic_video: Path) -> dict:  # type: ignore[type-arg]
    """A local source export + media root with: A (good), B (byte copy of A), C (different shot),
    D (all black), E (corrupt), F (1 s black lead-in then content), G (asset missing)."""
    base = tmp_path_factory.mktemp("factory")
    media = base / "media"
    src = base / "source"
    src.mkdir()
    clips = {}

    def cut(vid: str, clip: str, ss: float, t: float) -> Path:
        out = media / "clips" / vid / f"{clip}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        _ffmpeg("-ss", str(ss), "-t", str(t), "-i", str(synthetic_video), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out))
        return out

    clips["A"] = cut("vid_a", "clip_001", 0.0, 4.0)
    clips["B"] = media / "clips" / "vid_b" / "clip_001.mp4"
    clips["B"].parent.mkdir(parents=True)
    shutil.copyfile(clips["A"], clips["B"])
    clips["C"] = cut("vid_c", "clip_001", 4.0, 4.0)
    clips["D"] = media / "clips" / "vid_d" / "clip_001.mp4"
    clips["D"].parent.mkdir(parents=True)
    _ffmpeg("-f", "lavfi", "-i", "color=black:s=640x360:d=4:r=24", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clips["D"]))
    clips["E"] = media / "clips" / "vid_e" / "clip_001.mp4"
    clips["E"].parent.mkdir(parents=True)
    clips["E"].write_bytes(clips["A"].read_bytes()[:4000])
    clips["F"] = media / "clips" / "vid_f" / "clip_001.mp4"
    clips["F"].parent.mkdir(parents=True)
    black = base / "black1s.mp4"
    _ffmpeg("-f", "lavfi", "-i", "color=black:s=640x360:d=1:r=24", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(black))
    _ffmpeg("-i", str(black), "-i", str(clips["A"]), "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clips["F"]))
    rows = [
        _row("vid_a", "clip_001", 0.0, 4.0),
        _row("vid_b", "clip_001", 0.0, 4.0),
        _row("vid_c", "clip_001", 4.0, 8.0, generation_prompt="A 4-second wide shot of a bright panel, static camera, colours shift slowly."),
        _row("vid_d", "clip_001", 0.0, 4.0, generation_prompt="A 4-second shot of darkness, static camera, nothing moves."),
        _row("vid_e", "clip_001", 0.0, 4.0),
        _row("vid_f", "clip_001", 0.0, 5.0, generation_prompt="A 5-second shot fading in from black to a coloured panel drifting right, static camera."),
        _row("vid_g", "clip_001", 0.0, 4.0),
    ]
    (src / "clips.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (src / "manifest.json").write_text(json.dumps({"videos": [{"video_id": f"vid_{x}", "url": f"https://example.invalid/{x}", "license_status": "standard" if x != "c" else "creative_commons", "rights_status": "restricted" if x != "c" else "declared_cc_by"} for x in "abcdefg"]}))
    (src / "provenance.jsonl").write_text("")
    before = {k: _sha(v) for k, v in clips.items() if v.exists()}
    return {"base": base, "media": media, "source": src, "clips": clips, "rows": rows, "sha_before": before, "source_sha": _sha(src / "clips.jsonl")}


def _config(fixture: dict, data_root: Path, **overrides: str) -> FactoryConfig:  # type: ignore[type-arg]
    ov = {"paths.data_root": str(data_root), "source.kind": "local", "source.local_dir": str(fixture["source"]), "source.media_root": str(fixture["media"]), "exports.version": "test-v1", **overrides}
    return load_factory_config(ROOT / "config" / "factory" / "dataset.yaml", ov)


def _run_all(cfg: FactoryConfig) -> dict:  # type: ignore[type-arg]
    out = {"ingest": pipeline.ingest(cfg), "validate": pipeline.validate(cfg), "frames": pipeline.extract_frames(cfg), "captions": pipeline.review_captions(cfg), "dedup": pipeline.deduplicate(cfg), "curate": pipeline.curate(cfg)}
    return out


@pytest.fixture(scope="module")
def run(fixture_source: dict, tmp_path_factory: pytest.TempPathFactory) -> dict:  # type: ignore[type-arg]
    data_root = tmp_path_factory.mktemp("data_root")
    cfg = _config(fixture_source, data_root)
    counts = _run_all(cfg)
    with StateStore(cfg.path("state_db")) as store:
        recs = {r.original_video_id: r for r in store.iter_records()}
    return {"cfg": cfg, "counts": counts, "recs": recs, "data_root": data_root}


# ------------------------------------------------------------------ 1. schema mapping is tolerant
def test_source_mapping_handles_absent_and_renamed_fields():
    cfg = load_factory_config(None, {"mapping.aliases.video_path": "[media_file, clip_path]"})
    row = {"record_id": "r1", "video_id": "v1", "media_file": "x/y.mp4", "start_time": 1.0, "end_time": 3.5, "description": "d", "extra_field": {"k": 1}}
    rec = map_source_row(row, cfg, source_dataset="t", source_config="clips", video_meta={"url": "u", "license_status": "standard"})
    assert rec.video_path == "x/y.mp4" and rec.duration_seconds == 2.5 and rec.width is None and rec.fps is None and rec.generation_prompt_original is None
    assert rec.source_payload["extra_field"] == {"k": 1} and rec.source_url == "u" and rec.source_record_hash
    assert rec.record_id == "clips:r1" and rec.canonical_video_id == "v1" and rec.caption_quality_status == "not_checked"
    typed = map_source_row({"record_id": "r2", "video_id": "v2", "camera": json.dumps({"movement": "pan"}), "validation": json.dumps({"status": "review"})}, cfg, source_dataset="t", source_config="clips")
    assert typed.camera_motion == {"movement": "pan"} and typed.source_validation == {"status": "review"}  # Parquet JSON strings decoded


def test_config_validation_reports_helpful_errors(tmp_path: Path):
    with pytest.raises(ConfigError, match="dedup.action"):
        load_factory_config(None, {"dedup.action": "delete"})
    with pytest.raises(ConfigError, match="not found"):
        load_factory_config(tmp_path / "missing.yaml")
    with pytest.raises(ConfigError, match="unknown_key"):
        load_factory_config(None, {"media.unknown_key": "1"})


# ------------------------------------------------------------------ 2. sources are never modified
def test_source_records_and_assets_are_untouched_after_a_full_run(fixture_source: dict, run: dict):
    assert {k: _sha(v) for k, v in fixture_source["clips"].items() if v.exists()} == fixture_source["sha_before"]
    assert _sha(fixture_source["source"] / "clips.jsonl") == fixture_source["source_sha"]
    assert run["recs"]["vid_a"].caption_original.startswith("A 4.0-second shot")  # original text preserved next to caption_final


# ------------------------------------------------------------------ 3 + 4. media integrity
def test_missing_and_corrupted_assets_are_reported_precisely(run: dict):
    recs = run["recs"]
    assert recs["vid_g"].media_validation_status == "not_checked" and recs["vid_g"].media_validation_reasons == ["asset_unavailable:asset_missing"]
    assert recs["vid_e"].media_validation_status == "failed" and recs["vid_e"].media_validation_reasons and not recs["vid_e"].t2v_eligible
    assert recs["vid_d"].media_validation_status == "failed" and any(r.startswith("black_frames") for r in recs["vid_d"].media_validation_reasons)
    assert recs["vid_a"].media_validation_status == "passed" and recs["vid_a"].video_sha256 and len(recs["vid_a"].video_fingerprint) >= 2


def test_probe_returns_dimensions_and_duration_within_tolerance(fixture_source: dict):
    pr = probe(fixture_source["clips"]["A"])
    assert pr.ok and (pr.width, pr.height) == (640, 360) and abs(pr.duration - 4.0) < 0.3 and abs(pr.fps - 24.0) < 0.5 and pr.codec
    assert not probe(fixture_source["clips"]["E"]).ok
    assert probe(fixture_source["base"] / "nope.mp4").error == "asset_missing"


def test_no_media_root_means_not_checked_never_passed(fixture_source: dict, tmp_path: Path):
    cfg = _config(fixture_source, tmp_path / "root", **{"source.media_root": "null"})
    pipeline.ingest(cfg)
    counts = pipeline.validate(cfg)
    assert counts["not_checked"] == 7 and counts["passed"] == 0
    pipeline.review_captions(cfg)
    assert pipeline.curate(cfg)["t2v_eligible"] == 0


# ------------------------------------------------------------------ 5 + 6. starting frames and I2V pairs
def test_frame_extraction_records_timestamp_dimensions_and_checksum(run: dict):
    rec = run["recs"]["vid_a"]
    assert rec.start_frame_status == "passed" and rec.start_frame_timestamp_seconds == 0.0
    frame = run["data_root"] / rec.start_frame_path
    assert frame.exists() and _sha(frame) == rec.start_frame_sha256 and (rec.start_frame_width, rec.start_frame_height) == (640, 360)
    assert rec.start_frame_extraction["policy"] == "first_non_black" and rec.start_frame_clip_sha256 == rec.video_sha256
    f = run["recs"]["vid_f"]
    assert f.start_frame_status == "passed" and 0.9 <= f.start_frame_timestamp_seconds <= 1.3 and any(r.startswith("skipped_black_frames") for r in f.start_frame_reasons)


def test_start_frame_references_the_same_clip(run: dict, fixture_source: dict, tmp_path: Path):
    rec = run["recs"]["vid_a"]
    cfg = run["cfg"]
    frame = run["data_root"] / rec.start_frame_path
    ok, why = verify_pair(fixture_source["clips"]["A"], frame, rec.start_frame_timestamp_seconds, rec.start_frame_dhash, cfg.frames.consistency_hamming)
    assert ok and why == "frame_matches_clip"
    ok, why = verify_pair(fixture_source["clips"]["C"], frame, rec.start_frame_timestamp_seconds, rec.start_frame_dhash, cfg.frames.consistency_hamming)
    assert not ok and why == "frame_differs_from_clip"
    res = extract_start_frame(fixture_source["clips"]["D"], tmp_path / "f", "start", cfg.frames, cfg.media)
    assert res.status == "failed" and res.reasons == ["no_usable_frame_in_search_window"]


# ------------------------------------------------------------------ 7. caption heuristics never claim visual truth
def test_caption_heuristics_only_award_rules_status(run: dict):
    cfg = run["cfg"]
    rec = run["recs"]["vid_a"]
    assert rec.caption_final == rec.generation_prompt_original and rec.caption_source == "generation_prompt_original"
    assert rec.caption_review_method == "rules" and "visual" not in rec.caption_quality_score_basis.split("says")[0]
    assert rec.caption_quality_status == "passed"
    assert check_caption("", cfg.captions).status == "failed"
    assert check_caption("tiny", cfg.captions).reasons[0].startswith("too_short")
    residue = check_caption("A 4.0-second shot (dominant colors #112233); the camera pans left over the field.", cfg.captions)
    assert residue.status == "needs_review" and residue.reasons[0].startswith("template_residue")
    still = check_caption("A quiet portrait of a person sitting in a chair by a window.", cfg.captions)
    assert still.status == "needs_review" and "no_motion_described" in still.reasons and not still.motion_described
    assert check_caption("Insert caption {placeholder} here please now.", cfg.captions).status == "failed"


# ------------------------------------------------------------------ 8 + 9 + 16. providers: cache, invalid output, offline
class CountingProvider:
    name = "counting"
    model = "fake-1"
    calls = 0
    raw = json.dumps({"caption": "A coloured panel drifts right across the frame.", "observed": ["panel"], "assumed": [], "issues": [], "verdict": "supported", "confidence": 0.8})

    def __init__(self, cfg):  # type: ignore[no-untyped-def]
        pass

    def review(self, request):  # type: ignore[no-untyped-def]
        CountingProvider.calls += 1
        return prov.ReviewResult(called=True, output=None, raw=CountingProvider.raw)


def test_provider_cache_prevents_duplicate_calls_and_invalid_output_keeps_original(fixture_source: dict, tmp_path: Path):
    prov.register_provider("counting", CountingProvider)
    CountingProvider.calls = 0
    cfg = _config(fixture_source, tmp_path / "root", **{"provider.name": "counting", "provider.model": "fake-1", "offline": "false", "provider.max_calls": "10"})
    pipeline.ingest(cfg)
    pipeline.validate(cfg)
    first = pipeline.review_captions(cfg, select_all=True)
    # vid_b is a byte copy of vid_a with the same caption -> same cache key -> one call serves both
    assert first["provider_calls"] == 6 and CountingProvider.calls == 6 and first["cache_hits"] == 1
    second = pipeline.review_captions(cfg, select_all=True)
    assert second["provider_calls"] == 0 and second["cache_hits"] == 7 and CountingProvider.calls == 6
    with StateStore(cfg.path("state_db")) as store:
        rec = next(iter(store.iter_records()))
    assert rec.caption_review_method == "provider:counting" and rec.caption_model == "fake-1" and rec.caption_prompt_version == "caption-review-v1"
    assert rec.caption_final == "A coloured panel drifts right across the frame." and rec.generation_prompt_original.startswith("A 4-second")  # original preserved
    # invalid JSON / timeout: original caption stays, record goes to review, nothing cached
    CountingProvider.raw = "{not json"
    cfg2 = _config(fixture_source, tmp_path / "root2", **{"provider.name": "counting", "provider.model": "fake-2", "offline": "false", "provider.max_calls": "10"})
    pipeline.ingest(cfg2)
    res = pipeline.review_captions(cfg2, select_all=True)
    assert res["provider_invalid"] == 7 and res["provider_applied"] == 0
    with StateStore(cfg2.path("state_db")) as store:
        rec = next(iter(store.iter_records()))
    assert rec.caption_final == rec.generation_prompt_original and rec.caption_quality_status == "needs_review" and any(r.startswith("provider_output_invalid:invalid_json") for r in rec.caption_quality_reasons)
    assert ProviderCache(cfg2.path("cache_dir") / "captions.sqlite").size() == 0
    out, why = prov.parse_provider_json(json.dumps({"caption": "x", "verdict": "maybe", "confidence": 2}))
    assert out is None and why.startswith("schema:")
    assert cache_key(asset="a", text="t", operation="review", template_version="v1", provider="p", model="m") != cache_key(asset="a", text="t", operation="review", template_version="v2", provider="p", model="m")


def test_offline_mode_makes_no_provider_calls(fixture_source: dict, tmp_path: Path):
    prov.register_provider("counting", CountingProvider)
    CountingProvider.calls = 0
    cfg = _config(fixture_source, tmp_path / "root", **{"provider.name": "counting", "offline": "true"})
    pipeline.ingest(cfg)
    with pytest.raises(RuntimeError, match="offline mode"):
        pipeline.review_captions(cfg)
    assert CountingProvider.calls == 0
    assert pipeline.review_captions(cfg, offline=False, dry_run=True, select_all=True)["would_call"] == 7 and CountingProvider.calls == 0  # dry run never calls either
    assert pipeline.review_captions(_config(fixture_source, tmp_path / "root"))["provider_calls"] == 0  # provider none: nothing to call


# ------------------------------------------------------------------ 10. duplicates
def test_exact_duplicates_are_grouped_and_near_duplicates_only_flagged(run: dict):
    a, b = run["recs"]["vid_a"], run["recs"]["vid_b"]
    assert a.video_sha256 == b.video_sha256 and a.duplicate_group_id == b.duplicate_group_id and a.duplicate_kind == "exact"
    assert a.duplicate_canonical is True and b.duplicate_canonical is False and "duplicate:exact" in b.t2v_reasons and b.validation_status != "failed"
    assert a.split == b.split  # the duplicate link pulled both into one split
    near = [CanonicalRecord(record_id=f"x{i}", source_dataset="t", canonical_video_id=f"v{i}", original_video_id=f"v{i}", video_fingerprint=["aa", "bb", "cc", "dd"][: 4 - (i % 2)]) for i in range(2)]
    g = group_duplicates(near, run["cfg"].dedup.model_copy(update={"fingerprint_similarity": 0.7}))
    assert g["x0"]["kind"] == "near" and g["x0"]["group"] == g["x1"]["group"]
    rec = CanonicalRecord(record_id="x0", source_dataset="t", media_validation_status="passed", caption_quality_status="passed", caption_final="c", source_rights_status="cleared", duplicate_kind="near", duplicate_canonical=False)
    decide(rec, run["cfg"])
    assert rec.t2v_eligible and rec.validation_status == "needs_review"  # flagged, not dropped (dedup.action: review)


# ------------------------------------------------------------------ 11. splits never leak
def test_splits_keep_groups_and_duplicate_links_together():
    rules = load_factory_config().splits
    groups = [f"g{i:03d}" for i in range(40)]
    first, changes = assign(groups, rules)
    assert set(first.values()) == {"train", "validation", "test"} and len(changes) == 40
    again, changes2 = assign(groups + ["g_new"], rules, existing=first)
    assert all(again[g] == first[g] for g in groups) and changes2 and changes2[0]["group"] == "g_new"
    linked, _ = assign(groups, rules, links=[("g000", "g001"), ("g001", "g002")])
    assert linked["g000"] == linked["g001"] == linked["g002"] and set(linked.values()) == {"train", "validation", "test"}
    # a duplicate link never empties an eval split: the whole cluster moves together
    small, _ = assign(["a", "b", "c"], rules, links=[("a", "b")])
    assert sorted(small.values()).count("train") >= 1 and len(set(small.values())) == 2


def test_run_persists_splits_and_reports_changes(run: dict):
    splits = json.loads((run["data_root"] / "curated" / "splits.json").read_text())
    assert set(splits["groups"]) == {f"vid_{x}" for x in "abcdefg"} and splits["counts"]["validation"] >= 1 and splits["counts"]["test"] >= 1
    by_split = {}
    for r in run["recs"].values():
        by_split.setdefault(r.canonical_video_id, set()).add(r.split)
    assert all(len(v) == 1 for v in by_split.values())


# ------------------------------------------------------------------ 12. rights
def test_unknown_or_declared_rights_never_count_as_cleared(run: dict, fixture_source: dict, tmp_path: Path):
    recs = run["recs"]
    assert recs["vid_a"].source_rights_status == "restricted" and recs["vid_c"].source_rights_status == "review_required"
    assert all(not r.t2v_eligible for r in recs.values()) and all(any(x.startswith("rights:") for x in r.t2v_reasons) for r in recs.values() if r.media_validation_status == "passed")
    rec = CanonicalRecord(record_id="x", source_dataset="t", canonical_video_id="v")
    resolve_rights(rec, run["cfg"].rights, {}, {})
    assert rec.source_rights_status == "review_required" and rec.source_rights_evidence == "none"
    resolve_rights(rec, run["cfg"].rights, {"v": {"status": "cleared", "evidence": "written permission 2026-10-01"}}, {})
    assert rec.source_rights_status == "cleared" and "written permission" in rec.source_rights_evidence
    # explicit evidence file clears vid_a; vid_c stays review_required -> only vid_a exports
    ev = tmp_path / "rights.jsonl"
    ev.write_text(json.dumps({"video_id": "vid_a", "status": "cleared", "evidence": "licence agreement #1"}) + "\n")
    cfg = _config(fixture_source, tmp_path / "root", **{"rights.evidence_file": str(ev)})
    _run_all(cfg)
    t2v = pipeline.export(cfg, "wan-t2v")
    assert t2v["counts"]["records"] == 1 and t2v["rights_override_records"] == 0
    rows = [json.loads(line) for line in (Path(t2v["path"]) / "data.jsonl").read_text().splitlines()]
    assert rows[0]["record_id"] == "clips:clip_vid_a_clip_001" and (Path(t2v["path"]) / rows[0]["video"]).exists()
    # override: review_required allowed, loudly
    cfg_o = _config(fixture_source, tmp_path / "root_o", **{"rights.export_override": "[review_required]", "exports.version": "override"})
    _run_all(cfg_o)
    out = pipeline.export(cfg_o, "wan-t2v")
    manifest = json.loads((Path(out["path"]) / "release_manifest.json").read_text())
    assert out["rights_override_records"] == 1 and "WARNING" in manifest and manifest["rights_override_records"] == 1


# ------------------------------------------------------------------ 13. T2V and I2V differ
def test_t2v_and_i2v_eligibility_differ_for_the_same_record(fixture_source: dict, tmp_path: Path):
    ev = tmp_path / "rights.jsonl"
    ev.write_text("".join(json.dumps({"video_id": v, "status": "cleared", "evidence": "test"}) + "\n" for v in ("vid_a", "vid_f")))
    cfg = _config(fixture_source, tmp_path / "root", **{"rights.evidence_file": str(ev), "frames.max_search_seconds": "0.3"})
    _run_all(cfg)
    with StateStore(cfg.path("state_db")) as store:
        recs = {r.original_video_id: r for r in store.iter_records()}
    f = recs["vid_f"]
    assert f.t2v_eligible and not f.i2v_eligible and f.start_frame_status == "failed" and "start_frame:failed" in f.i2v_reasons
    a = recs["vid_a"]
    assert a.t2v_eligible and a.i2v_eligible
    i2v = pipeline.export(cfg, "wan-i2v")
    t2v = pipeline.export(cfg, "wan-t2v")
    assert t2v["counts"]["records"] == 2 and i2v["counts"]["records"] == 1
    row = json.loads((Path(i2v["path"]) / "data.jsonl").read_text().splitlines()[0])
    assert (Path(i2v["path"]) / row["image"]).exists() and (Path(i2v["path"]) / row["video"]).exists() and row["image_sha256"] == a.start_frame_sha256


# ------------------------------------------------------------------ 14 + 15. exports resolve, are deterministic and versioned
def test_exports_resolve_paths_are_deterministic_and_refuse_to_clobber(fixture_source: dict, tmp_path: Path):
    ev = tmp_path / "rights.jsonl"
    ev.write_text(json.dumps({"video_id": "vid_a", "status": "cleared", "evidence": "test"}) + "\n")
    cfg = _config(fixture_source, tmp_path / "root", **{"rights.evidence_file": str(ev), "exports.version": "v-det"})
    _run_all(cfg)
    one = pipeline.export(cfg, "wan-t2v")
    data_one = (Path(one["path"]) / "data.jsonl").read_bytes()
    manifest_one = json.loads((Path(one["path"]) / "release_manifest.json").read_text())
    for row in (json.loads(x) for x in data_one.decode().splitlines()):
        assert (Path(one["path"]) / row["video"]).read_bytes()[:4]
    with pytest.raises(ExportConflict):
        pipeline.export(cfg, "wan-t2v")
    two = pipeline.export(cfg, "wan-t2v", overwrite=True)
    manifest_two = json.loads((Path(two["path"]) / "release_manifest.json").read_text())
    assert (Path(two["path"]) / "data.jsonl").read_bytes() == data_one and manifest_two["checksums"] == manifest_one["checksums"] and manifest_two["manifest_checksum"] == manifest_one["manifest_checksum"]
    for key in ("schema_version", "dataset_version", "processing_config_hash", "pipeline_version", "counts", "source_dataset", "created_at_utc", "adapter"):
        assert key in manifest_two
    assert not list(Path(one["path"]).parent.glob(".*.tmp-*"))  # no staging left behind
    canon = pipeline.export(cfg, "canonical")
    assert set(canon["counts"]) == {"approved", "review", "rejected"} and canon["counts"]["approved"] == 1 and (Path(canon["path"]) / "approved.parquet").exists()
    rep = pipeline.report(cfg)
    assert Path(rep["markdown"]).exists() and rep["summary"]["t2v_eligible"] == 1 and rep["summary"]["media_validation"]["not_checked"] == 1


# ------------------------------------------------------------------ 17. incremental reruns
def test_incremental_rerun_skips_unchanged_and_reprocesses_on_change(fixture_source: dict, tmp_path: Path):
    cfg = _config(fixture_source, tmp_path / "root")
    pipeline.ingest(cfg)
    first = pipeline.validate(cfg)
    assert first["checked"] == 7 and first["skipped_current"] == 0
    again = pipeline.validate(cfg)
    assert again["checked"] == 0 and again["skipped_current"] == 7
    assert pipeline.ingest(cfg)["unchanged"] == 7
    changed = _config(fixture_source, tmp_path / "root", **{"media.sample_frames": "4"})
    assert pipeline.validate(changed)["checked"] == 7  # a relevant config change re-runs the step
    assert pipeline.validate(cfg)["checked"] == 7 and pipeline.validate(cfg)["checked"] == 0  # back to the original rules, then current again
    # a changed source row is replaced and re-validated
    src2 = tmp_path / "src2"
    shutil.copytree(fixture_source["source"], src2)
    rows = [json.loads(x) for x in (src2 / "clips.jsonl").read_text().splitlines()]
    rows[0]["description"] = "edited"
    (src2 / "clips.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    cfg3 = _config(fixture_source, tmp_path / "root", **{"source.local_dir": str(src2)})
    assert pipeline.ingest(cfg3)["changed"] == 1
    assert pipeline.validate(cfg3)["checked"] == 1


def test_cli_help_and_show_config_work_without_credentials(tmp_path: Path):
    for args in (["--help"], ["captions", "--help"], ["show-config", "--set", f"paths.data_root={tmp_path}"]):
        res = subprocess.run([sys.executable, "-m", "dataset_factory", *args], capture_output=True, text=True, cwd=str(ROOT))
        assert res.returncode == 0, res.stderr
    assert "config_hash" in subprocess.run([sys.executable, "-m", "dataset_factory", "show-config"], capture_output=True, text=True, cwd=str(ROOT)).stdout
