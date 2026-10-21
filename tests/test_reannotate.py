"""Re-annotating an already exported shard: schema upgrade, negatives from exports, split / tier, Parquet."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq

from video_dataset.config import load_config
from video_dataset.dataset.reannotate import load_shard, reannotate_shard, rebuild_hard_negatives
from video_dataset.utils.io import read_jsonl, write_json_atomic, write_jsonl


def _legacy_shard(root: Path) -> Path:
    """A shard in the pre-release layout: root jsonl files with `start`/`end` scenes, no record_type / split / tier."""
    shard = root / "shard_0001"
    scenes = [
        {"record_id": f"scene_v_scene_{i:03d}", "video_id": "vid_v", "scene_id": f"scene_{i:03d}", "start": 10.0 * i, "end": 10.0 * i + 8.0, "duration": 8.0,
         "summary": f"A 8.0-second shot with {'dim' if i % 2 else 'bright'} lighting; {'the camera pans left' if i % 2 else 'the camera is static'}, with moderate on-screen motion.",
         "environment": {"lighting": "dim" if i % 2 else "bright"}, "objects": ["car"] if i % 2 else ["person"],
         "object_details": [{"name": "car" if i % 2 else "person", "count": 1, "confidence": 0.9, "confidence_source": "detector_score"}],
         "camera": {"movement": "pan_left" if i % 2 else "static", "confidence": 0.8, "confidence_source": "measurement"}, "visual_style": {},
         "measurements": {"lighting_level": "dim" if i % 2 else "bright", "sharpness": 400.0}, "provider": "cv_models", "confidence": 0.9, "confidence_source": "detector_score",
         "quality": {"description_quality": 0.7, "overall": 0.7, "components_used": ["description_quality"]}, "validation": {"status": "accepted", "issues": []}}
        for i in range(4)
    ]
    clips = [{"record_id": f"clip_v_clip_{i:03d}", "video_id": "vid_v", "scene_id": f"scene_{i:03d}", "clip_id": f"clip_{i:03d}", "clip_path": f"clips/vid_v/clip_{i:03d}.mp4", "start_time": 10.0 * i, "end_time": 10.0 * i + 8.0,
              "description": s["summary"], "camera": s["camera"], "provider": "cv_models", "confidence": 0.9, "confidence_source": "detector_score", "quality": s["quality"], "validation": s["validation"]} for i, s in enumerate(scenes)]
    events = [
        {"record_id": f"event_v_event_{i:04d}", "video_id": "vid_v", "event_id": f"event_{i:04d}", "event_type": "camera", "start_time": 10.0 * i + 1.0, "end_time": 10.0 * i + 5.0,
         "event": f"The camera pans left ({i}).", "entities": ["camera"], "scene_ids": [f"scene_{i:03d}"], "source": "motion", "confidence": 0.8, "confidence_source": "measurement",
         "quality": {"grounding": 0.8, "overall": 0.8, "components_used": ["grounding"]}, "validation": {"status": "accepted"}}
        for i in range(4)
    ]
    qa = [
        {"question_id": "qa_v_000001", "video_id": "vid_v", "type": "duration", "question": "Approximately how long does it last while the camera pans left (0)?", "answer": "About 4 seconds (from 1.0s to 5.0s).",
         "evidence": {"start_time": 1.0, "end_time": 5.0, "event_ids": ["event_0000"], "scene_ids": ["scene_000"]}, "confidence": 0.8, "confidence_source": "derived_min", "template_id": None,
         "quality": {"grounding": 0.9, "temporal_accuracy": 0.8, "description_quality": 0.6, "overall": 0.767, "components_used": ["grounding", "temporal_accuracy", "description_quality"]}, "validation": {"status": "accepted", "issues": []}},
        {"question_id": "qa_v_000002", "video_id": "vid_v", "type": "timestamp", "question": "What is happening at 13.0 seconds?", "answer": "The camera pans left (1).",
         "evidence": {"start_time": 11.0, "end_time": 15.0, "timestamps": [13.0], "event_ids": ["event_0001"], "scene_ids": ["scene_001"]}, "confidence": None, "confidence_source": "unavailable",
         "quality": {"grounding": 1.0, "overall": 1.0, "components_used": ["grounding"]}, "validation": {"status": "review", "issues": ["evidence confidence unavailable"]}},
    ]
    for name, rows in {"scenes": scenes, "clips": clips, "events": events, "temporal_qa": qa}.items():
        write_jsonl(shard / f"{name}.jsonl", rows)
    for name in ("frames", "long_video_qa", "video_descriptions"):
        write_jsonl(shard / f"{name}.jsonl", [])
    write_json_atomic(shard / "manifest.json", {"videos": [{"video_id": "vid_v", "duration": 40.0, "validation_summary": {"qa": {"accepted": 1, "review": 1}}}]})
    (shard / "per_video" / "vid_v").mkdir(parents=True)
    write_json_atomic(shard / "per_video" / "vid_v" / "manifest.json", {"video_id": "vid_v", "duration": 40.0})
    return shard


def test_rebuild_hard_negatives_from_exported_records(tmp_path: Path):
    shard = _legacy_shard(tmp_path)
    records, manifests = load_shard(shard)
    assert manifests and manifests[0]["video_id"] == "vid_v" and len(records["scenes"]) == 4
    from video_dataset.dataset.export import upgrade_records

    records = upgrade_records(records)
    counts = rebuild_hard_negatives(records, limit=3, seed=1)
    assert counts == {"temporal_qa": 2, "long_video_qa": 0, "scenes": 4, "clips": 4}
    dur = records["temporal_qa"][0]["hard_negatives"]
    assert dur[0]["kind"] == "wrong_duration" and any(n["kind"] == "other_event" for n in dur)
    ts = records["temporal_qa"][1]["hard_negatives"]
    assert all(n["kind"] == "other_event" for n in ts) and all("(1)" not in n["text"] for n in ts)
    scene = records["scenes"][1]  # dim + pan_left -> both measured attributes can be swapped, and a static/bright shot differs
    kinds = [n["kind"] for n in scene["hard_negatives"]]
    assert kinds[:2] == ["attribute_swap", "attribute_swap"] and "other_shot" in kinds
    assert "the camera pans right" in scene["hard_negatives"][0]["text"]
    # clips borrow their scene's negatives
    assert records["clips"][1]["hard_negatives"] == scene["hard_negatives"]
    # a second pass with only_missing leaves them untouched
    assert rebuild_hard_negatives(records, limit=3, seed=1, only_missing=True) == {"temporal_qa": 0, "long_video_qa": 0, "scenes": 0, "clips": 0}


def test_reannotate_shard_rewrites_everything_into_the_output_dir(tmp_path: Path):
    shard = _legacy_shard(tmp_path)
    cfg = load_config(None, {"project.data_dir": str(tmp_path / "data"), "export.enforce_policy": "false"})
    out = tmp_path / "out"
    summary = reannotate_shard(shard, cfg, out)
    assert summary["videos"] == 1 and summary["records_in"] == 14 and summary["records_out"] == 14
    assert summary["hard_negatives"]["scenes"] == 4 and summary["splits"][next(iter(k for k, v in summary["splits"].items() if v))] == 1
    names = {p.name for p in out.iterdir()}
    assert {"scenes.jsonl", "scenes.parquet", "temporal_qa.parquet", "dataset.jsonl", "dataset.parquet", "splits.json", "schemas.json", "statistics.json", "manifest.json", "cinematic", "per_video"} <= names
    rows = list(read_jsonl(out / "scenes.jsonl"))
    assert all("start" not in r and r["start_time"] == 10.0 * i and r["record_type"] == "scene" for i, r in enumerate(rows))
    assert all(r["split"] and r["tier"] == "silver" and r["tier_reasons"] == ["overall_below_0.8"] for r in rows)  # overall 0.7: silver, not gold
    assert all("cinematic" in r["subsets"] for r in rows)  # silver, 8 s, known camera, sharp
    qa = list(read_jsonl(out / "temporal_qa.jsonl"))
    assert qa[0]["record_id"] == "qa_v_000001" and qa[0]["tier"] == "silver" and qa[1]["tier"] == "bronze" and qa[1]["tier_reasons"] == ["status_review"]
    assert pq.read_table(out / "temporal_qa.parquet").num_rows == 2 and str(pq.read_table(out / "temporal_qa.parquet").schema.field("template_id").type) == "string"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["reannotated"] and manifest["videos"][0]["split"] == rows[0]["split"]
    per_video = list(read_jsonl(out / "per_video" / "vid_v" / "scenes.jsonl"))
    assert per_video == rows  # per-video copies hold the same upgraded records
    assert (shard / "scenes.jsonl").read_text()  # the source shard is untouched when an output dir is given
    assert "start" in next(read_jsonl(shard / "scenes.jsonl"))
    # in place: a second run on the output is idempotent
    again = reannotate_shard(out, cfg)
    assert again["records_out"] == 14 and list(read_jsonl(out / "scenes.jsonl")) == rows


def test_reannotate_fails_closed_on_legacy_accepted_records(tmp_path: Path):
    """Legacy records were `accepted` without check states or a boundary check: under the export-time
    policy they are review (bronze) with the missing checks named, never silently accepted."""
    shard = _legacy_shard(tmp_path)
    cfg = load_config(None, {"project.data_dir": str(tmp_path / "data")})  # enforce_policy defaults to true
    out = tmp_path / "out"
    summary = reannotate_shard(shard, cfg, out)
    assert summary["records_out"] == 14
    for name in ("scenes", "clips", "events", "temporal_qa"):
        for r in read_jsonl(out / f"{name}.jsonl"):
            v = r["validation"]
            assert v["status"] in ("review", "rejected"), (name, r["record_id"], v)
            assert r["tier"] == "bronze" and "strict" not in r["subsets"]
            if v.get("reevaluated"):
                assert v["reevaluated"] == "export" and v["policy_version"] and "policy_version:missing" in v["reason_codes"]
                assert v["verification"]["human"] == "not_run"
    clips = list(read_jsonl(out / "clips.jsonl"))
    assert all("missing_check:clip_boundaries" in c["validation"]["reason_codes"] and c["validation"]["verification"]["temporal"] == "not_run" for c in clips)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["policy_downgraded"] == 13 and manifest["licenses"]["annotations_and_metadata"] and manifest["tier_policy"]["gold"]
    assert manifest["videos"][0]["files"]["scenes"] == "per_video/vid_v/scenes.jsonl"
    assert manifest["empty_record_types"] == ["frames", "sequences", "long_video_qa", "video_descriptions"]


def test_push_shard_uploads_in_small_commits_and_retries(tmp_path: Path, monkeypatch):
    from video_dataset.dataset import reannotate as mod

    shard = _legacy_shard(tmp_path)
    cfg = load_config(None, {"project.data_dir": str(tmp_path / "data"), "upload.repo_id": "user/ds"})
    reannotate_shard(shard, cfg)
    files = mod.shard_files(shard)
    assert files and all("frames" not in p.parts for p in files)
    assert len(files[0].relative_to(shard).parts) == 1  # root files first
    monkeypatch.setattr(mod, "log", mod.log)
    monkeypatch.setattr("video_dataset.utils.retry.time.sleep", lambda s: None)

    class FakeApi:
        def __init__(self):
            self.commits: list[list[str]] = []
            self.failures = 1

        def create_commit(self, repo_id, repo_type, operations, commit_message):
            if self.failures:
                self.failures -= 1
                raise RuntimeError("408 Request Timeout")
            self.commits.append([op.path_in_repo for op in operations])
            return f"https://hf.co/{repo_id}/commit/{len(self.commits)}"

    api = FakeApi()
    url = mod.push_shard(cfg, shard, "shard_0001", "tok", files_per_commit=5, api=api)
    assert url.endswith(f"/commit/{len(api.commits)}")
    assert len(api.commits) == -(-len(files) // 5) and all(len(c) <= 5 for c in api.commits)
    uploaded = [p for c in api.commits for p in c]
    assert len(uploaded) == len(files) and all(p.startswith("shard_0001/") for p in uploaded)
    assert "shard_0001/scenes.parquet" in uploaded and any(p.startswith("shard_0001/per_video/vid_v/") for p in uploaded)


def test_hf_reannotate_all_worker_rewrites_gates_and_reports(tmp_path: Path, monkeypatch):
    """The per-shard child of scripts/hf_reannotate_all.py: download (patched to a local copy),
    re-annotate to the current schema, gate, write the status file the parent reads."""
    import importlib.util
    import shutil

    from video_dataset.dataset import reannotate as mod

    spec = importlib.util.spec_from_file_location("hf_reannotate_all", Path(__file__).resolve().parents[1] / "scripts" / "hf_reannotate_all.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    src = _legacy_shard(tmp_path / "src")
    work = tmp_path / "work"

    def fake_download(config, path_in_repo, dest, token):
        local = dest / path_in_repo
        shutil.copytree(src, local, dirs_exist_ok=True)
        return local

    monkeypatch.setattr(mod, "download_shard", fake_download)
    overrides = {"project.data_dir": str(tmp_path / "data"), "export.min_source_videos": "0", "export.split.required_splits": "[train]"}
    rc = script.worker("shard_0001", work, overrides, rebuild_negatives=True)
    status = json.loads((work / "status" / "shard_0001.json").read_text())
    assert rc == 0 and status["ok"] and status["failed_gates"] == [] and status["schema_mismatch"] == []
    out = Path(status["rewritten"])
    assert status["videos"] == 1 and status["records"] == 14 and (out / "release_report.json").exists()
    rows = list(read_jsonl(out / "clips.jsonl"))
    assert all(r["schema_version"] == status["schema_version"] and r["validation"]["status"] == "review" and "missing_check:clip_boundaries" in r["validation"]["reason_codes"] for r in rows)
    manifest = json.loads((out / "per_video" / "vid_v" / "manifest.json").read_text())
    assert manifest["files"]["clips"] == "per_video/vid_v/clips.jsonl"
    # the status file lives outside the shard so it is never pushed
    assert not any(p.name.endswith("shard_0001.json") for p in mod.shard_files(out))
    # a gate failure is reported, not hidden: the 50-video pool minimum
    rc = script.worker("shard_0001", work, {"project.data_dir": str(tmp_path / "data")}, rebuild_negatives=False)
    status = json.loads((work / "status" / "shard_0001.json").read_text())
    assert rc == 2 and not status["ok"] and "source_pool" in status["failed_gates"]


def test_hf_reannotate_all_parent_runs_child_pushes_marks_done_and_cleans(tmp_path: Path, monkeypatch, capsys):
    """Parent flow with a fake Hub: child process per shard, push, done marker, local cleanup, resume."""
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "hf_reannotate_all.py"
    spec = importlib.util.spec_from_file_location("hf_reannotate_all", script_path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    import hf_replace_shards as base

    source = tmp_path / "src"
    _legacy_shard(source)  # shard_0001
    shutil.copytree(source / "shard_0001", source / "shard_0002")
    work = tmp_path / "work"
    pushed: list[tuple[str, Path]] = []
    monkeypatch.setattr(base, "_live_run_pid", lambda data_dir: None)
    monkeypatch.setattr(base, "_stale_root_files", lambda api, cfg: ["clips.jsonl", "manifest.json"])
    monkeypatch.setattr(base, "_push_replace", lambda api, cfg, local, path, delete_media: (pushed.append((path, local)) or ("https://hf.co/c/" + path, 3)))
    import video_dataset.dataset.reannotate as mod

    monkeypatch.setattr(mod, "list_hub_shards", lambda cfg, token: ["shard_0001", "shard_0002"])
    monkeypatch.setattr(mod, "update_dataset_card", lambda cfg, token: "card-updated")
    monkeypatch.setenv("HF_TOKEN", "tok")
    argv = ["hf_reannotate_all.py", "--source-dir", str(source), "--work-dir", str(work), "--no-negatives",
            "--set", f"project.data_dir={tmp_path / 'data'}", "--set", "export.min_source_videos=0", "--set", "export.split.required_splits=[train]"]
    monkeypatch.setattr(sys, "argv", argv)
    assert script.main() == 0
    out = capsys.readouterr().out
    assert [p for p, _ in pushed] == ["shard_0001", "shard_0002"] and "card-updated" in out and "pass --clean-root" in out
    assert {p.stem for p in (work / "done").glob("*.json")} == {"shard_0001", "shard_0002"}
    done = json.loads((work / "done" / "shard_0001.json").read_text())
    assert done["ok"] and done["commit"] == "https://hf.co/c/shard_0001" and done["superseded_files_removed"] == 3 and done["records"] == 14
    assert not (work / "rewritten" / "shard_0001").exists() and not (source / "shard_0001").exists()  # local copies deleted after the push
    # resume: done shards are skipped, nothing pushed again
    pushed.clear()
    assert script.main() == 0 and pushed == [] and "already done" in capsys.readouterr().out
    # a failing gate stops the run before any push of that shard
    _legacy_shard(tmp_path / "src2")
    monkeypatch.setattr(mod, "list_hub_shards", lambda cfg, token: ["shard_0001"])
    argv[2] = str(tmp_path / "src2")
    argv[-1] = "export.min_source_videos=50"
    monkeypatch.setattr(sys, "argv", argv + ["--redo"])
    assert script.main() == 2 and pushed == [] and (work / "rewritten" / "shard_0001").exists()
    assert "source_pool" in capsys.readouterr().out
