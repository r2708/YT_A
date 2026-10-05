"""Re-annotating an already exported shard: schema upgrade, negatives from exports, split / tier, Parquet."""

from __future__ import annotations

import json
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
    cfg = load_config(None, {"project.data_dir": str(tmp_path / "data")})
    out = tmp_path / "out"
    summary = reannotate_shard(shard, cfg, out)
    assert summary["videos"] == 1 and summary["records_in"] == 14 and summary["records_out"] == 14
    assert summary["hard_negatives"]["scenes"] == 4 and summary["splits"][next(iter(k for k, v in summary["splits"].items() if v))] == 1
    names = {p.name for p in out.iterdir()}
    assert {"scenes.jsonl", "scenes.parquet", "temporal_qa.parquet", "dataset.jsonl", "dataset.parquet", "splits.json", "schemas.json", "statistics.json", "manifest.json", "cinematic", "per_video"} <= names
    rows = list(read_jsonl(out / "scenes.jsonl"))
    assert all("start" not in r and r["start_time"] == 10.0 * i and r["record_type"] == "scene" for i, r in enumerate(rows))
    assert all(r["split"] and r["tier"] == "silver" and r["tier_reasons"] == ["overall_below_0.8"] for r in rows)  # overall 0.7: silver, not gold
    assert all(r["subsets"] == ["cinematic"] for r in rows)  # silver, 8 s, known camera, sharp
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
