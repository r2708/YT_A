"""Fixes for the Hugging Face dataset review (2026-10-20): non-empty eval splits, reconciled counts,
accepted == verified, consistent long-video QA files, portable manifests, separated quality /
verification, a minimum source pool and licence scopes."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from video_dataset.config import load_config
from video_dataset.dataset.aggregate import licenses_block, tier_policy_block
from video_dataset.dataset.curation import annotate_records, assign_splits, split_for_video, split_manifest
from video_dataset.dataset.export import DATASET_FILES, write_jsonl_files, write_typed_parquet_files
from video_dataset.dataset.release import (
    accepted_means_verified_gate,
    counts_gate,
    portable_paths_gate,
    required_splits_gate,
    run_release_check,
    source_pool_gate,
)
from video_dataset.dataset.stats import validation_report
from video_dataset.schemas.curation import Split
from video_dataset.schemas.quality import QualityScore
from video_dataset.utils.io import read_jsonl, write_json_atomic
from video_dataset.validation.policy import (
    POLICY_VERSION,
    REQUIRED_CHECKS,
    enforce_exported,
    verification_summary,
)

CFG = load_config(None, {"export.min_source_videos": "0"})


def _full_checks(kind: str) -> dict[str, str]:
    return {name: "passed" for name in REQUIRED_CHECKS[kind]}


def _scene(i: int, vid: str = "vid_a", *, verified: bool = True, status: str = "accepted") -> dict:
    v = {"status": status, "issues": [], "checks": _full_checks("scene") if verified else {}, "reason_codes": [], "policy_version": POLICY_VERSION if verified else None, "semantic_verification": "measured" if verified else None}
    return {"record_id": f"scene_{vid}_scene_{i:03d}", "video_id": vid, "scene_id": f"scene_{i:03d}", "start_time": 2.0 * i, "end_time": 2.0 * i + 2.0, "duration": 2.0, "summary": "A shot with a person walking.", "environment": {}, "camera": {"movement": "static"}, "visual_style": {}, "provider": "heuristic", "confidence": 0.9, "confidence_source": "measurement", "quality": {"grounding": 0.9, "overall": 0.9, "components_used": ["grounding"]}, "validation": v, "generation_target": "single_shot"}


def _clip(i: int, vid: str = "vid_a", *, boundary: str = "passed", verified: bool = True) -> dict:
    s = _scene(i, vid, verified=verified)
    return {"record_id": f"clip_{vid}_clip_{i:03d}", "video_id": vid, "scene_id": s["scene_id"], "clip_id": f"clip_{i:03d}", "clip_path": f"clips/{vid}/clip_{i:03d}.mp4", "start_time": s["start_time"], "end_time": s["end_time"], "boundary_check": boundary, "description": s["summary"], "camera": s["camera"], "provider": "heuristic", "confidence": 0.9, "confidence_source": "measurement", "quality": s["quality"], "validation": json.loads(json.dumps(s["validation"])), "generation_target": "single_shot"}


# ------------------------------------------------------------------ 1. validation split is never empty
def test_assign_splits_tops_up_eval_splits_deterministically():
    sp = CFG.export.split
    ids = [f"vid_{i:03d}" for i in range(12)]
    pure = {v: split_for_video(v, sp) for v in ids}
    assert sum(1 for s in pure.values() if s == Split.VALIDATION) <= 2  # at 5 % a dozen ids rarely fill validation by hash alone
    out = assign_splits(ids, sp)
    assert out == assign_splits(reversed(ids), sp)  # a function of the set, not of the order
    counts = {s: sum(1 for v in out.values() if v == s) for s in Split}
    assert counts[Split.VALIDATION] >= 1 and counts[Split.TEST] >= 1 and counts[Split.TRAIN] >= 1
    # hash-based members keep their split; only train groups were moved
    assert all(out[v] == pure[v] for v in ids if pure[v] != Split.TRAIN)
    # minimums scale: 3 of each on the same ids
    big = sp.model_copy(update={"min_validation_videos": 3, "min_test_videos": 3})
    out3 = assign_splits(ids, big)
    assert sum(1 for v in out3.values() if v == Split.VALIDATION) >= 3 and sum(1 for v in out3.values() if v == Split.TEST) >= 3
    # tiny pools: one group is train, two give train + test, three give one of each
    assert assign_splits(["only"], sp) == {"only": Split.TRAIN}
    two = assign_splits(["a", "b"], sp)
    assert sorted(s.value for s in two.values()) == ["test", "train"]
    three = assign_splits(["a", "b", "c"], sp)
    assert sorted(s.value for s in three.values()) == ["test", "train", "validation"]


def test_annotate_records_assigns_per_group_and_fills_every_split():
    rows = {"scenes": [_scene(i, f"vid_{i % 9}") for i in range(45)], "clips": [_clip(i, f"vid_{i % 9}") for i in range(45)]}
    annotate_records(rows, CFG.export, {"vid_8": {"canonical_video_id": "vid_1"}})
    manifest = split_manifest(rows)
    assert all(len(v) >= 1 for v in manifest.values()), manifest
    by_video = {}
    for recs in rows.values():
        for r in recs:
            by_video.setdefault(r["video_id"], set()).add(r["split"])
    assert all(len(v) == 1 for v in by_video.values())
    assert by_video["vid_8"] == by_video["vid_1"]  # a near-duplicate follows its canonical video


def test_required_splits_gate_fails_on_empty_validation():
    rows = {"scenes": [_scene(i, f"vid_{i}") for i in range(4)]}
    for r in rows["scenes"]:
        r["split"] = "train" if r["video_id"] != "vid_3" else "test"
        r["source_group_id"] = r["video_id"]
    g = required_splits_gate(CFG, rows)
    assert g["state"] == "failed" and set(g["below_minimum"]) == {"validation"}
    rows["scenes"][2]["split"] = "validation"
    assert required_splits_gate(CFG, rows)["state"] == "passed"
    off = CFG.model_copy(deep=True)
    off.export.split.required_splits = ["train"]
    rows["scenes"][2]["split"] = "train"
    assert required_splits_gate(off, rows)["state"] == "passed"


# ------------------------------------------------------------------ 2 + 4. every published count reconciles
def _final(tmp_path: Path, records: dict) -> Path:
    final = tmp_path / "final"
    full = {name: list(records.get(name, [])) for name in DATASET_FILES}
    counts = write_jsonl_files(full, final)
    typed = write_typed_parquet_files(full, final)
    vids = sorted({r["video_id"] for rows in full.values() for r in rows})
    write_json_atomic(final / "manifest.json", {"videos": [{"video_id": v, "channel": "c", "files": {"scenes": f"per_video/{v}/scenes.jsonl"}} for v in vids], "counts": counts, "typed_parquet_counts": typed, "splits": {}})
    write_json_atomic(final / "statistics.json", {"jsonl_counts": counts, **{n: counts[n] for n in DATASET_FILES}})
    by_split: dict[str, set[str]] = {}
    for rows in full.values():
        for r in rows:
            if r.get("split"):
                by_split.setdefault(r["split"], set()).add(r["video_id"])
    write_json_atomic(final / "splits.json", {"videos": {k: sorted(v) for k, v in by_split.items()}, "counts": {k: len(v) for k, v in by_split.items()}})
    return final


def test_counts_gate_catches_every_kind_of_disagreement(tmp_path: Path):
    records = {"scenes": [_scene(i) for i in range(3)], "clips": [_clip(i) for i in range(3)]}
    annotate_records(records, CFG.export)
    final = _final(tmp_path, records)
    loaded = {name: list(read_jsonl(final / f"{name}.jsonl")) for name in DATASET_FILES}
    g = counts_gate(final, loaded)
    assert g["state"] == "passed" and "long_video_qa" in g["empty_record_types"], g
    # (a) a Parquet file that does not match its JSONL (the long_video_qa 0-byte / 62 kB case, inverted)
    write_typed_parquet_files({"long_video_qa": [{"record_id": "qa_x", "video_id": "vid_a", "type": "long_range", "question": "q?", "answer": "a", "evidence": {"start_time": 0.0, "end_time": 1.0}}]}, final)
    g = counts_gate(final, loaded)
    assert g["state"] == "failed" and "parquet:long_video_qa" in g["mismatches"]
    write_typed_parquet_files({"long_video_qa": []}, final)
    # (b) a manifest that keeps its own totals
    m = json.loads((final / "manifest.json").read_text())
    m["counts"]["clips"] = 4390
    write_json_atomic(final / "manifest.json", m)
    g = counts_gate(final, loaded)
    assert g["state"] == "failed" and g["mismatches"]["manifest:clips"] == {"jsonl": 3, "manifest": 4390}
    m["counts"]["clips"] = 3
    m["videos"].append({"video_id": "vid_ghost"})
    write_json_atomic(final / "manifest.json", m)
    assert "manifest:videos" in counts_gate(final, loaded)["mismatches"]
    m["videos"].pop()
    write_json_atomic(final / "manifest.json", m)
    # (c) statistics and splits that drift
    st = json.loads((final / "statistics.json").read_text())
    st["scenes"] = 99
    write_json_atomic(final / "statistics.json", st)
    assert "statistics:scenes" in counts_gate(final, loaded)["mismatches"]
    st["scenes"] = 3
    write_json_atomic(final / "statistics.json", st)
    sp = json.loads((final / "splits.json").read_text())
    sp["counts"] = {k: v + 1 for k, v in sp["counts"].items()}
    write_json_atomic(final / "splits.json", sp)
    assert any(k.startswith("splits.counts:") for k in counts_gate(final, loaded)["mismatches"])


def test_release_check_runs_all_new_gates_and_fails_on_a_small_pool(tmp_path: Path):
    records = {"scenes": [_scene(i, f"vid_{i}") for i in range(6)], "clips": [_clip(i, f"vid_{i}") for i in range(6)]}
    annotate_records(records, CFG.export)
    final = _final(tmp_path, records)
    report = run_release_check(CFG, final, require_media=False)
    names = {g["gate"]: g["state"] for g in report["gates"]}
    assert {"counts_consistent", "required_splits", "source_pool", "portable_paths", "accepted_means_verified"} <= set(names)
    assert report["ok"], report["failed_gates"]
    assert report["final_dir"] == "final" and "/" not in report["final_dir"]
    m = report["metrics"]
    assert m["status_pct_by_type"]["clips"]["accepted"] == 100.0 and m["verification_by_type"]["clips"]["temporal"] == {"passed": 6}
    assert m["quality_missing_by_type"]["scenes"]["temporal_accuracy_missing_pct"] == 100.0
    strict = CFG.model_copy(deep=True)
    strict.export.min_source_videos = 50
    report = run_release_check(strict, final, require_media=False, write_report=False)
    assert "source_pool" in report["failed_gates"]
    g = source_pool_gate(strict, records, [])
    assert g["source_videos"] == 6 and g["minimum"] == 50
    # an upload shard is a slice of the released dataset: earlier shards count toward the minimum
    g = source_pool_gate(strict, records, [], already_released=44)
    assert g["state"] == "passed" and g["released_total"] == 50
    write_json_atomic(final / "uploads.json", {"next_shard": 3, "shards": [{"name": "shard_0001", "videos": ["v1", "v2"]}, {"name": "shard_0002", "videos": 42}]})
    report = run_release_check(strict, final, require_media=False, write_report=False)  # reads uploads.json: 2 + 42 + 6 = 50
    assert report["ok"], report["failed_gates"]
    assert run_release_check(strict, final, require_media=False, write_report=False, already_released=0)["failed_gates"] == ["source_pool"]


# ------------------------------------------------------------------ 3 + 6. accepted means verified
def test_enforce_exported_downgrades_unverified_accepted_records_but_not_human_accepts():
    legacy = _clip(0, verified=False, boundary="not_run")
    why = enforce_exported(legacy)
    v = legacy["validation"]
    assert v["status"] == "review" and v["reevaluated"] == "export" and v["policy_version"] == POLICY_VERSION
    assert "policy_version:missing" in why and "missing_check:clip_boundaries" in why
    assert enforce_exported(legacy) == []  # idempotent: already review
    failed = _clip(1, boundary="failed")
    assert "failed_check:clip_boundaries" in enforce_exported(failed) and failed["validation"]["status"] == "review"
    error = _clip(2, boundary="error")
    assert "missing_check:clip_boundaries" in enforce_exported(error)
    good = _clip(3, boundary="passed")
    assert enforce_exported(good) == [] and good["validation"]["status"] == "accepted"
    one_missing = _scene(4)
    one_missing["validation"]["checks"]["clip_boundaries"] = "not_run"
    assert enforce_exported(one_missing) == ["missing_check:clip_boundaries"]
    human = _clip(5, verified=False, boundary="not_run")
    human["validation"]["reason_codes"] = ["human:accepted"]
    human["validation"]["semantic_verification"] = "human"
    assert enforce_exported(human) == [] and human["validation"]["status"] == "accepted"
    qa = {"record_id": "qa_1", "record_type": "temporal_qa", "video_id": "vid_a", "validation": {"status": "accepted", "checks": {}}}
    assert "missing_check:answer_grounded" in enforce_exported(qa) and qa["validation"]["status"] == "review"


def test_annotate_records_enforces_policy_and_fills_verification(tmp_path: Path):
    rows = {"clips": [_clip(0, boundary="passed"), _clip(1, boundary="not_run"), _clip(2, verified=False, boundary="not_run")], "scenes": [_scene(0), _scene(1, verified=False)]}
    counts = annotate_records(rows, CFG.export)
    assert counts["policy_downgraded"] == 3
    c0, c1, c2 = rows["clips"]
    assert c0["validation"]["status"] == "accepted" and c0["tier"] == "gold" and "strict" in c0["subsets"]
    assert c1["validation"]["status"] == "review" and c1["tier"] == "bronze" and "strict" not in c1["subsets"] and c1["tier_reasons"] == ["status_review"]
    assert c2["validation"]["status"] == "review" and "policy_version:missing" in c2["validation"]["reason_codes"]
    assert c0["validation"]["verification"] == {"temporal": "passed", "measurement": "passed", "semantic": "measured", "human": "not_run"}
    assert c1["validation"]["verification"]["temporal"] == "not_run" and c2["validation"]["verification"]["semantic"] == "not_run"
    assert c0["quality"]["missing_components"] == ["temporal_accuracy", "description_quality"]
    g = accepted_means_verified_gate(rows)
    assert g["state"] == "passed" and g["accepted"] == 2
    c1["validation"]["status"] = "accepted"  # someone flips the status by hand
    g = accepted_means_verified_gate(rows)
    assert g["state"] == "failed" and g["unverified_accepted"] == 1 and "boundary_check:not_run" in g["examples"][0]["why"]
    off = CFG.export.model_copy(update={"enforce_policy": False})
    rows2 = {"clips": [_clip(0, verified=False, boundary="not_run")]}
    annotate_records(rows2, off)
    assert rows2["clips"][0]["validation"]["status"] == "accepted"  # opt-out keeps the old behaviour (and the gate will fail)


def test_quality_score_lists_missing_components_and_report_has_percentages():
    q = QualityScore.from_components(description_quality=0.76)
    assert q.overall == 0.76 and q.components_used == ["description_quality"] and q.missing_components == ["grounding", "temporal_accuracy"]
    rows = {"clips": [_clip(0), _clip(1, boundary="not_run")]}
    annotate_records(rows, CFG.export)
    rep = validation_report(rows)["clips"]
    assert rep["status"] == {"accepted": 1, "review": 1, "rejected": 0, "duplicate": 0} and rep["status_pct"]["accepted"] == 50.0 and rep["status_pct"]["review"] == 50.0
    assert rep["quality_missing_pct"]["temporal_accuracy"] == 100.0 and rep["policy_downgraded_at_export"] == 1 and rep["verification"]["temporal"] == {"passed": 1, "not_run": 1}
    assert verification_summary(rows["clips"][1])["temporal"] == "not_run"


# ------------------------------------------------------------------ 5. no machine paths in anything published
def test_portable_paths_gate_flags_local_prefixes(tmp_path: Path):
    records = {"scenes": [_scene(0)], "clips": [_clip(0)]}
    annotate_records(records, CFG.export)
    final = _final(tmp_path, records)
    loaded = {name: list(read_jsonl(final / f"{name}.jsonl")) for name in DATASET_FILES}
    assert portable_paths_gate(final, loaded)["state"] == "passed"
    m = json.loads((final / "manifest.json").read_text())
    m["videos"][0]["files"]["scenes"] = "/Users/someone/Desktop/YT_A/data/final/per_video/vid_a/scenes.jsonl"
    write_json_atomic(final / "manifest.json", m)
    g = portable_paths_gate(final, loaded)
    assert g["state"] == "failed" and g["examples"][0]["where"].startswith("manifest.json.videos[0].files.scenes")
    m["videos"][0]["files"]["scenes"] = "per_video/vid_a/scenes.jsonl"
    write_json_atomic(final / "manifest.json", m)
    (final / "per_video" / "vid_a").mkdir(parents=True)
    write_json_atomic(final / "per_video" / "vid_a" / "manifest.json", {"video_id": "vid_a", "files": {"clips": "C:\\\\Users\\\\x\\\\clips.jsonl"}})
    assert portable_paths_gate(final, loaded)["state"] == "failed"
    write_json_atomic(final / "per_video" / "vid_a" / "manifest.json", {"video_id": "vid_a", "files": {"clips": "per_video/vid_a/clips.jsonl"}})
    loaded["clips"][0]["clip_path"] = "/home/user/data/clips/vid_a/clip_000.mp4"
    g = portable_paths_gate(final, loaded)
    assert g["state"] == "failed" and g["records_with_machine_paths"] == 1


def test_per_video_manifest_and_statistics_use_relative_files(synthetic_video: Path, test_config):
    from video_dataset.dataset.aggregate import aggregate_exports
    from video_dataset.pipeline.runner import PipelineRunner
    from video_dataset.utils.urls import classify_input

    cfg = test_config.model_copy(deep=True)
    cfg.transcription.provider = "mock"
    runner = PipelineRunner(cfg)
    try:
        registered = runner.register_inputs([classify_input(str(synthetic_video))])
        video_id = registered[0][0]
        results = runner.run_batch(registered)
        assert results[0].completed, results[0].error
        per_video = json.loads((cfg.export_dir / "per_video" / video_id / "manifest.json").read_text())
        assert per_video["files"]["clips"] == f"per_video/{video_id}/clips.jsonl"
        aggregate_exports(cfg, runner.db)
        final = cfg.export_dir
        stats = json.loads((final / "statistics.json").read_text())
        assert stats["files"]["clips"] == "clips.jsonl" and stats["files"]["provenance"] == "provenance.jsonl" and stats["licenses"]["code"] == "MIT"
        assert stats["validation_report"]["clips"]["status_pct"] and "tier_policy" in stats
        manifest = json.loads((final / "manifest.json").read_text())
        assert manifest["videos"][0]["files"]["scenes"] == f"per_video/{video_id}/scenes.jsonl" and manifest["licenses"]["source_videos"]
        assert manifest["typed_parquet_counts"]["clips"] == manifest["counts"]["clips"] == pq.read_metadata(final / "clips.parquet").num_rows
        splits = json.loads((final / "splits.json").read_text())
        assert splits["method"].startswith("sha1") and splits["minimums"] == {"validation": 0, "test": 0}
        report = run_release_check(cfg, final, require_media=False)
        assert report["ok"], [g for g in report["gates"] if g["state"] == "failed"]
        for c in read_jsonl(final / "clips.jsonl"):  # every clip was boundary-checked (verify_clip_boundaries: all) ...
            assert c["boundary_check"] in ("passed", "failed", "error")
            if c["validation"]["status"] == "accepted":  # ... and only passed ones are accepted
                assert c["boundary_check"] == "passed" and c["validation"]["verification"]["temporal"] == "passed"
    finally:
        runner.db.close()


# ------------------------------------------------------------------ 8. licence scopes are explicit
def test_license_and_tier_policy_blocks():
    lic = licenses_block(CFG)
    assert lic["code"] == "MIT" and lic["annotations_and_metadata"] == "MIT" and "provenance.jsonl" in lic["source_videos"] and lic["release_rights"] == ["declared_cc_by"]
    assert "no rights" in lic["note"]
    tp = tier_policy_block(CFG)
    assert tp["gold"]["quality.overall_min"] == 0.8 and POLICY_VERSION in tp["accepted"] and "boundary_check passed" in tp["accepted"]
