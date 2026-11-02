"""Task-specific exports, held-out benchmark and continuity links (issue doc P3-04/05/06/07/10)."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from video_dataset.dataset.benchmark import BENCHMARK_DIR, BENCHMARK_VERSION, build_benchmark
from video_dataset.dataset.continuity import CONTINUITY_METHOD, check_continuity, fill_continuity
from video_dataset.dataset.export import DATASET_FILES, RECORD_MODELS
from video_dataset.dataset.task_exports import EXPORTS_DIR, write_task_exports
from video_dataset.dataset.tasks import (
    TASK_POLICY_VERSION,
    TASKS,
    eligible,
    index_ids,
    policy_document,
    select_task_records,
)
from video_dataset.utils.io import read_jsonl
from video_dataset.validation.policy import POLICY_VERSION, REQUIRED_CHECKS

RELEASE_RIGHTS = ["declared_cc_by"]
RIGHTS = {"vid_a": "declared_cc_by", "vid_b": "declared_cc_by", "vid_c": "restricted"}
SPLITS = {"vid_a": "train", "vid_b": "test", "vid_c": "validation"}


def _validation(kind: str, status: str = "accepted", *, human: bool = False) -> dict:
    return {
        "status": status,
        "issues": [],
        "checks": {name: "passed" for name in REQUIRED_CHECKS[kind]},
        "reason_codes": ["human:accepted"] if human else [],
        "policy_version": POLICY_VERSION,
        "semantic_verification": "human" if human else "measured",
    }


def _scene(vid: str, i: int, *, status: str = "accepted", movement: str = "pan_left", subsets: tuple[str, ...] = ("strict", "cinematic"), objects: tuple[str, ...] = ("person",), setting: str = "outdoor", tod: str = "day", transition: str = "cut", human: bool = False, prompt: str | None = "A person walks along a beach, camera pans left.") -> dict:
    return {
        "record_id": f"scene_{vid}_scene_{i:03d}",
        "record_type": "scene",
        "video_id": vid,
        "scene_id": f"scene_{i:03d}",
        "start_time": 2.0 * i,
        "end_time": 2.0 * i + 2.0,
        "duration": 2.0,
        "transition_in": "start" if i == 0 else transition,
        "summary": "A person walks along a beach.",
        "generation_prompt": prompt,
        "environment": {"setting": setting, "time_of_day": tod},
        "objects": list(objects),
        "camera": {"movement": movement, "shot_type": "wide"},
        "visual_style": {},
        "measurements": {"lighting_level": "bright"},
        "frame_ids": [f"frame_{i:03d}_000"],
        "clip_id": f"clip_{i:03d}",
        "claims": [{"field": "camera_movement", "value": movement, "status": "supported"}],
        "provider": "heuristic",
        "confidence": 0.9,
        "confidence_source": "measurement",
        "quality": {"grounding": 0.9, "overall": 0.9, "components_used": ["grounding"]},
        "validation": _validation("scene", status, human=human),
        "split": SPLITS[vid],
        "tier": "gold",
        "subsets": list(subsets),
        "generation_target": "single_shot",
    }


def _clip(scene: dict, *, human: bool = False) -> dict:
    vid, sid = scene["video_id"], scene["scene_id"]
    return {
        "record_id": f"clip_{vid}_{scene['clip_id']}",
        "record_type": "clip",
        "video_id": vid,
        "scene_id": sid,
        "clip_id": scene["clip_id"],
        "clip_path": f"clips/{vid}/{scene['clip_id']}.mp4",
        "start_time": scene["start_time"],
        "end_time": scene["end_time"],
        "boundary_check": "passed",
        "fps": 25.0,
        "width": 1280,
        "height": 720,
        "description": scene["summary"],
        "generation_prompt": scene["generation_prompt"],
        "prompt_variants": ["Camera pans left past a walking person on a beach."] if scene["generation_prompt"] else [],
        "camera": dict(scene["camera"]),
        "subject_motion": {"subject": "person", "label": "moves_left", "camera_relation": "tracked", "direction": "left"},
        "frame_ids": list(scene["frame_ids"]),
        "provider": "heuristic",
        "confidence": 0.9,
        "confidence_source": "measurement",
        "quality": dict(scene["quality"]),
        "validation": json.loads(json.dumps({**scene["validation"], "semantic_verification": "human" if human else scene["validation"]["semantic_verification"], "reason_codes": ["human:accepted"] if human else []})),
        "split": scene["split"],
        "tier": scene["tier"],
        "subsets": list(scene["subsets"]),
        "generation_target": "single_shot",
    }


def _frame(scene: dict) -> dict:
    vid = scene["video_id"]
    return {"record_id": f"frame_{vid}_{scene['frame_ids'][0]}", "record_type": "frame", "video_id": vid, "scene_id": scene["scene_id"], "clip_id": scene["clip_id"], "frame_id": scene["frame_ids"][0], "frame_path": f"frames/{vid}/{scene['frame_ids'][0]}.jpg", "timestamp": scene["start_time"] + 1.0, "caption": scene["summary"], "camera": dict(scene["camera"]), "validation": _validation("scene"), "split": scene["split"], "subsets": list(scene["subsets"]), "objects": list(scene["objects"])}


def _event(vid: str, n: int, scene: dict, *, status: str = "accepted") -> dict:
    return {"record_id": f"event_{vid}_event_{n:04d}", "record_type": "event", "video_id": vid, "event_id": f"event_{n:04d}", "event_type": "action", "start_time": scene["start_time"], "end_time": scene["end_time"], "event": "a person walks", "scene_ids": [scene["scene_id"]], "frame_ids": list(scene["frame_ids"]), "source": "scene", "validation": _validation("event", status), "split": scene["split"]}


def _qa(vid: str, n: int, event: dict, *, event_id: str | None = None, human: bool = False, status: str = "accepted") -> dict:
    return {"question_id": f"qa_{vid}_{n:06d}", "record_id": f"qa_{vid}_{n:06d}", "record_type": "temporal_qa", "video_id": vid, "type": "timestamp", "question": "When does a person walk?", "answer": f"{event['start_time']:.1f} s", "evidence": {"start_time": event["start_time"], "end_time": event["end_time"], "event_ids": [event_id or event["event_id"]], "scene_ids": list(event["scene_ids"])}, "hard_negatives": [{"text": "99.0 s", "kind": "shifted_time"}], "validation": _validation("qa", status, human=human), "split": SPLITS[vid]}


def _sequence(vid: str, scenes: list[dict], clips: list[dict], *, human: bool = False) -> dict:
    return {"record_id": f"sequence_{vid}_seq_001", "record_type": "sequence", "video_id": vid, "sequence_id": "seq_001", "scene_ids": [s["scene_id"] for s in scenes], "clip_ids": [c["clip_id"] for c in clips], "start_time": scenes[0]["start_time"], "end_time": scenes[-1]["end_time"], "duration": scenes[-1]["end_time"] - scenes[0]["start_time"], "n_shots": len(scenes), "coherence": ["same_setting"], "camera_movements": ["pan_left"] * len(scenes), "summary": "two shots", "generation_prompt": "A continuous sequence of two shots.", "validation": {**_validation("scene", human=human), "checks": {"all_shots_accepted": True}}, "split": SPLITS[vid], "subsets": ["strict"], "generation_target": "multi_shot"}


def _description(vid: str, scenes: list[dict]) -> dict:
    return {"record_id": f"video_description_{vid}_full", "record_type": "video_description", "video_id": vid, "start_time": scenes[0]["start_time"], "end_time": scenes[-1]["end_time"], "scene_ids": [s["scene_id"] for s in scenes], "temporal_progression": "a walk", "prompt": "A person walks on a beach.", "validation": _validation("scene"), "split": SPLITS[vid]}


@pytest.fixture()
def records() -> dict[str, list[dict]]:
    """Three videos. vid_a: 3 clean scenes (one in review, one with unknown camera). vid_b: 2
    human-verified scenes in the test split with a sequence and a QA record. vid_c: restricted
    rights, scene not in the strict subset, QA with dangling evidence."""
    a = [_scene("vid_a", 0), _scene("vid_a", 1, status="review"), _scene("vid_a", 2, movement="unknown", transition="fade")]
    b = [_scene("vid_b", 0, human=True), _scene("vid_b", 1, human=True)]
    c = [_scene("vid_c", 0, subsets=("cinematic",), prompt=None, objects=(), setting="unknown")]
    scenes = a + b + c
    clips = [_clip(s, human=s["video_id"] == "vid_b") for s in scenes]
    frames = [_frame(s) for s in scenes]
    events = [_event("vid_a", 1, a[0]), _event("vid_b", 1, b[0]), _event("vid_c", 1, c[0])]
    qa = [_qa("vid_a", 1, events[0]), _qa("vid_b", 1, events[1], human=True), _qa("vid_c", 1, events[2], event_id="event_9999")]
    sequences = [_sequence("vid_b", b, clips[3:5], human=True)]
    descriptions = [_description("vid_a", a), _description("vid_b", b)]
    recs = {"frames": frames, "clips": clips, "scenes": scenes, "sequences": sequences, "events": events, "temporal_qa": qa, "long_video_qa": [], "video_descriptions": descriptions}
    # exactly what the exporter writes: validated by the record model (which also derives the
    # standardized camera fields) and dumped back to JSON-shaped dicts
    return {kind: [{**RECORD_MODELS[kind].model_validate({k: v for k, v in r.items() if k != "record_type"}).model_dump(mode="json"), "record_type": r["record_type"]} for r in recs[kind]] for kind in DATASET_FILES}


def _media_on_disk(rec: dict) -> bool:
    return rec.get("video_id") in ("vid_a", "vid_b")


# ------------------------------------------------------------------ eligibility
def test_eligibility_reasons_per_task(records):
    ids = index_ids(records)
    scenes = {(s["video_id"], s["scene_id"]): s for s in records["scenes"]}
    clean, review, unknown_cam = scenes[("vid_a", "scene_000")], scenes[("vid_a", "scene_001")], scenes[("vid_a", "scene_002")]
    restricted = scenes[("vid_c", "scene_000")]
    assert eligible("understanding", "scenes", clean) == (True, [])
    assert eligible("understanding", "scenes", review) == (False, ["status:review"])
    assert eligible("understanding", "temporal_qa", records["temporal_qa"][0])[1] == ["record_type:temporal_qa"]
    assert eligible("cinematic", "scenes", clean) == (True, [])
    assert eligible("cinematic", "scenes", unknown_cam) == (False, ["camera:unknown"])
    ambiguous = {**clean, "camera": {**clean["camera"], "camera_movement": "pan", "movement_ambiguous": True}}
    assert eligible("cinematic", "scenes", ambiguous)[1] == ["camera:ambiguous"]
    ok, why = eligible("temporal_reasoning", "temporal_qa", records["temporal_qa"][0], ids_by_video=ids)
    assert ok and not why
    ok, why = eligible("temporal_reasoning", "temporal_qa", records["temporal_qa"][2], ids_by_video=ids)
    assert not ok and why == ["evidence:unresolved"]
    ctx = {"media_available": _media_on_disk, "rights": RIGHTS, "release_rights": RELEASE_RIGHTS}
    assert eligible("generation_ready", "scenes", clean, **ctx) == (True, [])
    assert eligible("generation_ready", "scenes", review, **ctx)[1] == ["status:review"]
    assert set(eligible("generation_ready", "scenes", restricted, **ctx)[1]) == {"not_in_subset:strict", "prompt:missing", "rights:restricted", "media:missing"}
    assert eligible("generation_ready", "clips", {**_clip(clean), "generation_target": "not_for_generation"}, **ctx)[1] == ["generation_target:not_for_generation"]
    assert eligible("generation_ready", "scenes", clean, rights=RIGHTS, release_rights=RELEASE_RIGHTS)[1] == ["media:missing"]  # no media check -> no media
    assert eligible("generation_ready", "scenes", clean, media_available=_media_on_disk)[1] == ["rights:unknown"]


def test_select_task_records_reports_counts_and_reasons(records):
    selected, filters = select_task_records("generation_ready", records, media_available=_media_on_disk, rights=RIGHTS, release_rights=RELEASE_RIGHTS)
    assert filters["task"] == "generation_ready" and filters["policy_version"] == TASK_POLICY_VERSION
    assert filters["record_types"] == ["clips", "sequences", "scenes"]
    assert filters["requirements"]["require_media"] and filters["requirements"]["release_rights"] == RELEASE_RIGHTS
    # vid_a scene_001 (review) and vid_c (restricted, no strict, no prompt, no media) are out; an
    # unknown camera is no obstacle for generation (the prompt says nothing about movement then)
    assert filters["counts"]["scenes"] == {"input": 6, "eligible": 4, "excluded": 2}
    assert filters["counts"]["clips"]["eligible"] == 4 and filters["counts"]["sequences"]["eligible"] == 1
    assert filters["excluded_by_reason"]["scenes"]["status:review"] == 1
    assert filters["excluded_by_reason"]["scenes"]["rights:restricted"] == 1
    assert filters["excluded_by_status"]["scenes"] == {"accepted": 1, "review": 1}
    assert filters["splits"] == {"train": 1, "validation": 0, "test": 1}
    assert {r["video_id"] for r in selected["scenes"]} == {"vid_a", "vid_b"}
    sel_t, f_t = select_task_records("temporal_reasoning", records)
    assert f_t["counts"]["temporal_qa"] == {"input": 3, "eligible": 2, "excluded": 1}
    assert f_t["excluded_by_reason"]["temporal_qa"] == {"evidence:unresolved": 1}
    assert f_t["counts"]["sequences"]["eligible"] == 1 and f_t["counts"]["events"]["eligible"] == 3
    with pytest.raises(KeyError):
        select_task_records("nonsense", records)


def test_policy_document_names_requirements():
    for task in TASKS:
        doc = policy_document(task)
        assert f"`{task}`" in doc and TASK_POLICY_VERSION in doc
        for kind in TASKS[task]["record_types"]:
            assert f"`{kind}`" in doc
    gen = policy_document("generation_ready")
    assert "not a substitute for the media" in gen and "`strict`" in gen and "media:missing" in gen
    assert "resolves" in policy_document("temporal_reasoning")


# ------------------------------------------------------------------ exports
def test_write_task_exports_writes_every_task(tmp_path: Path, records):
    final = tmp_path / "final"
    out = write_task_exports(records, final, None, media_available=_media_on_disk, rights=RIGHTS, release_rights=RELEASE_RIGHTS)
    assert set(out) == set(TASKS)
    for task, spec in TASKS.items():
        d = final / EXPORTS_DIR / task
        for kind in spec["record_types"]:
            rows = list(read_jsonl(d / f"{kind}.jsonl"))
            assert len(rows) == out[task]["counts"][kind]["eligible"]
            assert pq.read_table(d / f"{kind}.parquet").num_rows == len(rows)
        filters = json.loads((d / "filters.json").read_text())
        assert filters["policy_version"] == TASK_POLICY_VERSION and filters["generated_at"]
        manifest = json.loads((d / "manifest.json").read_text())
        assert manifest["eligible_total"] == out[task]["eligible_total"] and manifest["schema_version"]
        assert set(manifest["files"]) == set(out[task]["files"]) and all((d / f).exists() for f in manifest["files"])
        readme = (d / "README.md").read_text()
        assert "| record type | input | eligible | excluded |" in readme and f"`{task}`" in readme
    assert out["understanding"]["counts"]["scenes"]["eligible"] == 5
    assert out["cinematic"]["counts"]["scenes"]["eligible"] == 4  # review + unknown camera out; vid_c (cinematic, no strict) in
    assert out["cinematic"]["counts"]["frames"]["eligible"] == 5  # frames are all accepted; only scene_002's camera is unknown
    assert out["cinematic"]["counts"]["clips"]["eligible"] == 4
    pairs = list(read_jsonl(final / EXPORTS_DIR / "generation_ready" / "pairs.jsonl"))
    assert len(pairs) == 5 and {p["record_type"] for p in pairs} == {"clip", "sequence"}  # 4 clips + 1 sequence
    clip_pair = next(p for p in pairs if p["record_id"] == "clip_vid_b_clip_000")
    assert clip_pair["target"] == "clips/vid_b/clip_000.mp4" and clip_pair["prompt_verification"] == "human"
    assert clip_pair["camera"] == {"camera_movement": "pan", "movement_direction": "left", "movement_speed": None, "shot_size": "wide"}
    assert clip_pair["subject_motion"] == {"label": "moves_left", "direction": "left", "camera_relation": "tracked"}
    assert clip_pair["fps"] == 25.0 and clip_pair["duration"] == 2.0 and clip_pair["task"] == "text_to_video" and clip_pair["split"] == "test"
    seq_pair = next(p for p in pairs if p["record_type"] == "sequence")
    assert seq_pair["target"] == ["clip_000", "clip_001"] and seq_pair["prompt"].startswith("A continuous")
    a_pair = next(p for p in pairs if p["record_id"] == "clip_vid_a_clip_000")
    assert a_pair["prompt_verification"] == "template"


def test_generation_ready_is_empty_without_media(tmp_path: Path, records):
    final = tmp_path / "final"
    out = write_task_exports(records, final, None, media_available=lambda rec: False, rights=RIGHTS, release_rights=RELEASE_RIGHTS, write_parquet=False, tasks=["generation_ready"])
    assert set(out) == {"generation_ready"} and out["generation_ready"]["eligible_total"] == 0
    d = final / EXPORTS_DIR / "generation_ready"
    assert not (d / "clips.parquet").exists() and (d / "clips.jsonl").read_text() == ""
    filters = json.loads((d / "filters.json").read_text())
    assert filters["excluded_by_reason"]["clips"]["media:missing"] == 6 and filters["pairs"] == 0
    assert filters["counts"]["clips"] == {"input": 6, "eligible": 0, "excluded": 6}
    readme = (d / "README.md").read_text()
    assert "This export is empty" in readme and "`media:missing`" in readme
    assert (d / "pairs.jsonl").read_text() == ""


# ------------------------------------------------------------------ benchmark
def test_benchmark_from_human_labels(tmp_path: Path, records):
    final = tmp_path / "final"
    m = build_benchmark(records, final)
    assert m["version"] == BENCHMARK_VERSION and m["split"] == "test" and m["require_human"]
    assert m["counts"] == {"captioning": 4, "temporal_qa": 1, "camera_motion": 4, "cinematic": 2}
    assert m["note"] is None and m["videos"] == ["vid_b"] and len(m["frozen_hash"]) == 64
    d = final / BENCHMARK_DIR
    cap = list(read_jsonl(d / "captioning.jsonl"))
    assert all(len(i["item_id"]) == 16 for i in cap) and len({i["item_id"] for i in cap}) == 4
    clip_item = next(i for i in cap if i["record_type"] == "clip")
    assert clip_item["clip_path"].startswith("clips/vid_b/") and clip_item["reference"].startswith("A person walks") and clip_item["claims"] == []
    scene_item = next(i for i in cap if i["record_type"] == "scene")
    assert scene_item["frame_ids"] == ["frame_000_000"] or scene_item["frame_ids"] == ["frame_001_000"]
    qa = list(read_jsonl(d / "temporal_qa.jsonl"))[0]
    assert qa["question"] and qa["answer"] == "0.0 s" and qa["evidence"]["event_ids"] == ["event_0001"] and qa["hard_negatives"] == ["99.0 s"]
    cam = list(read_jsonl(d / "camera_motion.jsonl"))
    assert {(i["camera_movement"], i["movement_direction"], i["shot_size"]) for i in cam} == {("pan", "left", "wide")}
    cin = list(read_jsonl(d / "cinematic.jsonl"))
    assert all(i["lighting_level"] == "bright" and i["setting"] == "outdoor" and i["camera"]["camera_movement"] == "pan" for i in cin)
    metrics = json.loads((d / "metrics.json").read_text())
    assert set(metrics["tasks"]) == set(m["counts"]) and "claim_precision" in metrics["tasks"]["captioning"]["metrics"]
    assert "macro_f1_family" in metrics["tasks"]["camera_motion"]["metrics"]
    guide = (d / "guidelines.md").read_text()
    assert "pan" in guide and "0.5 s" in guide
    manifest = json.loads((d / "manifest.json").read_text())
    assert manifest["frozen_hash"] == m["frozen_hash"]
    # the item set is frozen: the same records give the same hash, a changed set a different one
    assert build_benchmark(records, tmp_path / "again")["frozen_hash"] == m["frozen_hash"]
    fewer = {k: [r for r in v if r.get("record_type") != "clip"] for k, v in records.items()}
    assert build_benchmark(fewer, tmp_path / "fewer")["frozen_hash"] != m["frozen_hash"]


def test_benchmark_without_human_labels_is_empty_and_says_so(tmp_path: Path, records):
    for rows in records.values():
        for r in rows:
            v = r.get("validation") or {}
            v["semantic_verification"] = "measured"
            v["reason_codes"] = []
    m = build_benchmark(records, tmp_path / "final")
    assert m["total"] == 0 and all(n == 0 for n in m["counts"].values())
    assert m["human_verified_in_split"] == 0 and m["records_in_split"] > 0
    assert m["note"] and "no human-verified" in m["note"] and "does not fall back" in m["note"]
    assert (tmp_path / "final" / BENCHMARK_DIR / "captioning.jsonl").read_text() == ""
    loose = build_benchmark(records, tmp_path / "loose", require_human=False)
    assert loose["note"] is None and loose["counts"]["captioning"] == 4 and loose["counts"]["temporal_qa"] == 1
    train = build_benchmark(records, tmp_path / "train", split="train", require_human=False)
    assert train["counts"]["camera_motion"] == 2  # vid_a: scene_001 in review, scene_002 unknown camera (skipped)
    assert train["camera_motion_skipped_unknown"] == 2 and train["counts"]["captioning"] == 4  # 2 scenes + 2 clips


# ------------------------------------------------------------------ continuity
def test_fill_and_check_continuity(records):
    counts = fill_continuity(records)
    assert counts == {"scenes": 6, "clips": 6, "frames": 6}
    assert check_continuity(records) == []
    by = {(s["video_id"], s["scene_id"]): s["continuity"] for s in records["scenes"]}
    first = by[("vid_a", "scene_000")]
    assert first["prev_scene_id"] is None and first["next_scene_id"] == "scene_001" and first["same_sequence_as_prev"] is None
    assert first["transition_in"] == "start" and first["transition_out"] == "cut" and first["temporal_adjacency"] is True
    assert first["subject_continuity"] is True and first["location_continuity"] is True and first["sequence_id"] is None
    assert first["continuity_method"] == CONTINUITY_METHOD
    mid = by[("vid_a", "scene_001")]
    assert mid["prev_scene_id"] == "scene_000" and mid["next_scene_id"] == "scene_002" and mid["transition_out"] == "fade"
    assert mid["same_sequence_as_prev"] is False
    last = by[("vid_a", "scene_002")]
    assert last["next_scene_id"] is None and last["temporal_adjacency"] is False and last["subject_continuity"] is None and last["location_continuity"] is None
    b0, b1 = by[("vid_b", "scene_000")], by[("vid_b", "scene_001")]
    assert b0["sequence_id"] == "seq_001" and b1["sequence_id"] == "seq_001" and b1["same_sequence_as_prev"] is True
    only = by[("vid_c", "scene_000")]
    assert only["prev_scene_id"] is None and only["next_scene_id"] is None
    # clips and frames copy their scene's block
    for kind in ("clips", "frames"):
        for r in records[kind]:
            assert r["continuity"] == by[(r["video_id"], r["scene_id"])]
    # unknown location / empty objects give None, not False
    records["scenes"][0]["environment"]["time_of_day"] = "unknown"
    records["scenes"][0]["objects"] = []
    fill_continuity(records)
    assert records["scenes"][0]["continuity"]["location_continuity"] is None and records["scenes"][0]["continuity"]["subject_continuity"] is None
    records["scenes"][0]["environment"]["time_of_day"] = "night"
    fill_continuity(records)
    assert records["scenes"][0]["continuity"]["location_continuity"] is False


def test_check_continuity_finds_broken_links(records):
    fill_continuity(records)
    scenes = {(s["video_id"], s["scene_id"]): s for s in records["scenes"]}
    scenes[("vid_a", "scene_000")]["continuity"]["next_scene_id"] = "scene_777"
    scenes[("vid_a", "scene_001")]["continuity"]["temporal_adjacency"] = False
    scenes[("vid_b", "scene_000")]["continuity"]["sequence_id"] = "seq_999"
    records["clips"][0]["continuity"]["prev_scene_id"] = "scene_005"
    problems = check_continuity(records)
    assert any("scene_777 does not resolve" in p for p in problems)
    assert any("vid_a/scene_001: temporal_adjacency False disagrees" in p for p in problems)
    assert any("seq_999 does not resolve" in p for p in problems)
    assert any("clips/clip_vid_a_clip_000: continuity differs" in p for p in problems)
    assert any("does not link back" in p for p in problems)  # scene_001.prev -> scene_000 whose next is now scene_777
    # a gap in time without the flag is consistent; the flag without the gap is not
    records["scenes"][1]["start_time"] = 10.0
    fill_continuity(records)
    assert check_continuity(records) == []
    scenes[("vid_a", "scene_000")]["continuity"]["temporal_adjacency"] = True
    assert any("temporal_adjacency True disagrees" in p for p in check_continuity(records))
