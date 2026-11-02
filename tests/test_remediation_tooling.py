"""Inventory / checksums, release comparison, repair queues and lineage on a tiny fake release.
No network, no ffmpeg, no models: everything is JSONL / Parquet / JSON under tmp_path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from video_dataset.dataset.compare import compare_releases, format_comparison
from video_dataset.dataset.inventory import (
    CHECKSUMS_FILE,
    INVENTORY_FILE,
    build_inventory,
    inventory_file,
    iter_files,
    read_checksums,
    verify_checksums,
    write_inventory,
)
from video_dataset.dataset.repair import (
    ACTIONS,
    LINEAGE_FILE,
    QUEUES,
    append_lineage,
    build_repair_queues,
    lineage_row,
    lineage_summary,
    read_lineage,
)
from video_dataset.schemas.common import SCHEMA_VERSION
from video_dataset.utils.io import read_json, read_jsonl, write_json_atomic, write_jsonl

V1, V2 = "vid_aaaaaaaaaaaa", "vid_bbbbbbbbbbbb"


def _val(status: str = "accepted", **extra: Any) -> dict[str, Any]:
    return {"status": status, "reason_codes": [], "checks": {}, **extra}


def _cam(movement: str = "static", confidence: float | None = 0.9, ambiguous: bool | None = None) -> dict[str, Any]:
    cam: dict[str, Any] = {"movement": movement, "confidence": confidence}
    if ambiguous is not None:
        cam["movement_ambiguous"] = ambiguous
    return cam


def _rec(kind: str, rid: str, vid: str, split: str = "train", **fields: Any) -> dict[str, Any]:
    base = {"record_id": rid, "record_type": kind, "video_id": vid, "split": split, "schema_version": SCHEMA_VERSION, "validation": _val()}
    base.update(fields)
    return base


def fake_records() -> dict[str, list[dict[str, Any]]]:
    """2 videos; one record per failure class the repair queues must catch."""
    frames = [
        _rec("frame", f"frame_{V1}_001_000", V1, scene_id="scene_001", frame_id="frame_001_000", timestamp=0.5, frame_path=f"frames/{V1}/scene_001/f0.jpg", camera=_cam()),
        _rec("frame", f"frame_{V1}_001_001", V1, scene_id="scene_001", frame_id="frame_001_001", timestamp=1.0, frame_path=f"frames/{V1}/scene_001/f1.jpg", camera=_cam()),
        _rec("frame", f"frame_{V2}_001_000", V2, split="validation", scene_id="scene_001", frame_id="frame_001_000", timestamp=0.2, frame_path=f"frames/{V2}/scene_001/f0.jpg", camera=_cam()),
    ]
    clips = [
        _rec("clip", f"clip_{V1}_001", V1, scene_id="scene_001", clip_id="clip_001", start_time=0.0, end_time=2.0, clip_path=f"clips/{V1}/clip_001.mp4", camera=_cam()),
        _rec("clip", f"clip_{V1}_002", V1, scene_id="scene_002", clip_id="clip_002", start_time=2.0, end_time=4.0, clip_path=f"clips/{V1}/clip_002.mp4", camera=_cam("pan", 0.3)),  # missing media + low camera
        _rec("clip", f"clip_{V2}_001", V2, split="validation", scene_id="scene_001", clip_id="clip_001", start_time=0.0, end_time=1.0, clip_path=f"clips/{V2}/clip_001.mp4", camera=_cam("unknown", None)),
    ]
    scenes = [
        _rec("scene", f"scene_{V1}_001", V1, scene_id="scene_001", start_time=0.0, end_time=2.0, camera=_cam()),
        _rec("scene", f"scene_{V1}_002", V1, scene_id="scene_002", start_time=4.0, end_time=2.0, camera=_cam("tilt", 0.8, ambiguous=True)),  # end < start + ambiguous
        _rec("scene", f"scene_{V2}_001", V2, split="validation", scene_id="scene_001", start_time=0.0, end_time=1.0, camera=_cam()),
    ]
    scenes[2].pop("schema_version")  # schema failure
    events = [
        _rec("event", f"event_{V1}_0001", V1, event_id="event_0001", start_time=0.0, end_time=1.0),
        _rec("event", f"event_{V1}_0001", V1, event_id="event_0002", start_time=1.0, end_time=2.0),  # duplicate record_id
        _rec("event", f"event_{V2}_0001", V2, split="validation", event_id="event_0001", start_time=0.0, end_time=1.0),
    ]
    temporal_qa = [
        _rec("temporal_qa", f"qa_{V1}_000001", V1, question="What happens first?", answer="x",
             evidence={"start_time": 0.0, "end_time": 2.0, "event_ids": ["event_0001"], "scene_ids": ["scene_001"], "frame_ids": ["frame_001_000"]}),
        _rec("temporal_qa", f"qa_{V1}_000002", V1, question="And then?", answer="y",
             evidence={"start_time": 0.0, "end_time": 2.0, "event_ids": ["event_9999"], "scene_ids": ["scene_001"], "frame_ids": []}),  # broken event id
        _rec("temporal_qa", f"qa_{V2}_000001", V2, split="validation", question="?", answer="z",
             evidence={"start_time": 0.0, "end_time": 1.0, "event_ids": ["event_0001"], "scene_ids": ["scene_001"], "frame_ids": ["frame_001_000"]}),
    ]
    descriptions = [
        _rec("video_description", f"videodesc_{V1}_00", V1, start_time=0.0, end_time=4.0, validation=_val("accepted", semantic_verification="measured")),
        _rec("video_description", f"videodesc_{V2}_00", V2, split="validation", start_time=0.0, end_time=1.0, validation=_val("review", semantic_verification="not_run")),
    ]
    return {
        "frames": frames, "clips": clips, "scenes": scenes, "sequences": [], "events": events,
        "temporal_qa": temporal_qa, "long_video_qa": [], "video_descriptions": descriptions,
    }


def write_release(root: Path, records: dict[str, list[dict[str, Any]]], *, with_media_manifest: bool = True) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for name, rows in records.items():
        counts[name] = write_jsonl(root / f"{name}.jsonl", rows)
        table = pa.table({
            "record_id": pa.array([r["record_id"] for r in rows], pa.string()),
            "schema_version": pa.array([r.get("schema_version") for r in rows], pa.string()),
        })
        pq.write_table(table, root / f"{name}.parquet")
    write_jsonl(root / "dataset.jsonl", (r for rows in records.values() for r in rows))
    videos = sorted({r["video_id"] for rows in records.values() for r in rows})
    write_json_atomic(root / "manifest.json", {"videos": videos, "counts": counts, "schema_version": SCHEMA_VERSION, "policy_version": "test", "splits": {}})
    write_json_atomic(root / "statistics.json", {"records": sum(counts.values())})
    write_json_atomic(root / "splits.json", {"train": [V1], "validation": [V2]})
    write_json_atomic(root / "schemas.json", {})
    write_jsonl(root / "provenance.jsonl", [
        {"video_id": V1, "url": "https://example.invalid/1", "rights_status": "declared_cc_by"},
        {"video_id": V2, "url": "https://example.invalid/2", "rights_status": "restricted"},
    ])
    if with_media_manifest:
        rows = []
        for r in records["frames"]:
            rows.append({"path": r["frame_path"], "kind": "frame", "video_id": r["video_id"], "exists": True})
        for r in records["clips"]:
            rows.append({"path": r["clip_path"], "kind": "clip", "video_id": r["video_id"], "exists": r["clip_id"] != "clip_002"})
        write_jsonl(root / "media_manifest.jsonl", rows)
    for vid in videos:
        pv = root / "per_video" / vid
        for name, rows in records.items():
            write_jsonl(pv / f"{name}.jsonl", [r for r in rows if r["video_id"] == vid])
        write_json_atomic(pv / "manifest.json", {"video_id": vid})
    (root / "frames" / V1 / "scene_001").mkdir(parents=True)
    (root / "frames" / V1 / "scene_001" / "f0.jpg").write_bytes(b"\xff\xd8fake-jpeg")
    (root / ".cache").mkdir()
    (root / ".cache" / "junk.bin").write_bytes(b"x")
    (root / "scratch.tmp").write_text("x")
    return root


@pytest.fixture
def release(tmp_path: Path) -> Path:
    return write_release(tmp_path / "shard_0001", fake_records())


# ------------------------------------------------------------------ inventory


def test_iter_files_is_sorted_and_skips_cache_tmp_and_media(release: Path):
    rels = [p.relative_to(release).as_posix() for p in iter_files(release)]
    assert rels == sorted(rels)
    assert "frames/vid_aaaaaaaaaaaa/scene_001/f0.jpg" in rels
    assert not any(r.startswith(".cache") or r.endswith(".tmp") for r in rels)
    no_media = [p.relative_to(release).as_posix() for p in iter_files(release, include_media=False)]
    assert not any(r.endswith(".jpg") for r in no_media)
    assert "per_video/vid_aaaaaaaaaaaa/scenes.jsonl" in no_media


def test_inventory_file_describes_jsonl_and_parquet(release: Path):
    scenes = inventory_file(release, release / "scenes.jsonl")
    assert scenes["path"] == "scenes.jsonl" and scenes["format"] == "jsonl" and scenes["record_type"] == "scene"
    assert scenes["rows"] == 3 and scenes["shard"] == "shard_0001" and scenes["extension"] == ".jsonl"
    assert scenes["schema_versions"] == {SCHEMA_VERSION: 2, "missing": 1}
    assert len(scenes["sha256"]) == 64 and scenes["size_bytes"] > 0
    pq_entry = inventory_file(release, release / "scenes.parquet")
    assert pq_entry["format"] == "parquet" and pq_entry["rows"] == 3 and pq_entry["record_type"] == "scene"
    assert pq_entry["schema_versions"] == {SCHEMA_VERSION: 2, "missing": 1}
    combined = inventory_file(release, release / "dataset.jsonl")
    assert combined["record_type"] == "mixed" and combined["rows"] == 17
    jpg = inventory_file(release, release / "frames" / V1 / "scene_001" / "f0.jpg")
    assert jpg["format"] == "jpg" and jpg["rows"] is None and jpg["record_type"] is None and jpg["schema_versions"] is None
    nested = inventory_file(release, release / "per_video" / V1 / "scenes.jsonl")
    assert nested["shard"] == "shard_0001" and nested["path"] == f"per_video/{V1}/scenes.jsonl"


def test_build_and_write_inventory_then_verify(release: Path):
    inv = write_inventory(release, workers=2)
    assert inv["root"] == "shard_0001" and inv["schema_version"] == SCHEMA_VERSION
    paths = [f["path"] for f in inv["files"]]
    assert paths == sorted(paths)
    assert INVENTORY_FILE not in paths and CHECKSUMS_FILE not in paths
    assert not any("/Users/" in p or p.startswith("/") for p in paths)
    assert inv["totals"]["files"] == len(paths)
    assert inv["totals"]["rows_by_type"]["scene"] == 6  # root + per_video copies, JSONL only
    assert inv["totals"]["schema_versions"]["missing"] == 3  # scenes.jsonl, per_video copy, dataset.jsonl
    assert inv["totals"]["schema_versions"][SCHEMA_VERSION] == 16 * 3
    assert inv["totals"]["bytes"] == sum(f["size_bytes"] for f in inv["files"])
    on_disk = read_json(release / INVENTORY_FILE)
    assert on_disk["totals"] == inv["totals"]

    text = (release / CHECKSUMS_FILE).read_text()
    first = text.splitlines()[0]
    digest, sep, rel = first.partition("  ")
    assert sep == "  " and len(digest) == 64 and rel == paths[0]
    listed = read_checksums(release / CHECKSUMS_FILE)
    assert set(listed) == set(paths)

    ok = verify_checksums(release)
    assert ok["ok"] and ok["checked"] == len(paths) and ok["mismatched"] == [] and ok["missing"] == [] and ok["unlisted"] == []

    (release / "scenes.jsonl").write_text((release / "scenes.jsonl").read_text() + "\n")
    bad = verify_checksums(release)
    assert not bad["ok"] and bad["mismatched"] == ["scenes.jsonl"]

    (release / "events.parquet").unlink()
    gone = verify_checksums(release)
    assert "events.parquet" in gone["missing"] and not gone["ok"]

    (release / "extra.json").write_text("{}")
    extra = verify_checksums(release)
    assert extra["unlisted"] == ["extra.json"]
    assert build_inventory(release, workers=1)["totals"]["files"] == len(paths)  # inventory files still skipped

    # same inventory and checksums from a rebuilt directory -> identical hashes (determinism)
    again = build_inventory(release, workers=3)
    assert [f["path"] for f in again["files"]] == sorted(f["path"] for f in again["files"])


# ------------------------------------------------------------------ compare


def test_compare_releases_detects_loss_moves_and_explains_removed_sources(tmp_path: Path):
    old = write_release(tmp_path / "release_old", fake_records())
    write_inventory(old)
    new_records = fake_records()
    new_records["temporal_qa"] = [r for r in new_records["temporal_qa"] if r["record_id"] != f"qa_{V1}_000001"]  # unexplained loss
    for rows in new_records.values():
        rows[:] = [r for r in rows if r["video_id"] != V2]  # V2 removed entirely (explained)
    new = write_release(tmp_path / "release_new", new_records)
    write_json_atomic(new / "removed_sources.json", {V2: {"reason": "takedown"}})
    write_inventory(new)

    rep = compare_releases(old, new)
    assert not rep["ok"]
    qa = rep["by_type"]["temporal_qa"]
    assert qa["counts"] == {"old": 3, "new": 1, "delta": -2}
    assert qa["unexplained_lost"] == {"count": 1, "allowed": 0, "examples": [f"qa_{V1}_000001"]}
    assert qa["videos"]["removed"] == [V2] and qa["record_ids"] == {"added": 0, "removed": 2, "kept": 1}
    assert rep["by_type"]["scenes" if "scenes" in rep["by_type"] else "scene"]["unexplained_lost"]["count"] == 0
    assert rep["removed_sources"] == [V2]
    assert rep["overall"]["unexplained_lost"]["count"] == 1
    assert rep["overall"]["duplicate_ids"]["old"] == 1 and rep["overall"]["duplicate_ids"]["new_examples"] == []
    assert rep["overall"]["temporal_failures"] == {"old": 1, "new": 1, "new_examples": []}
    assert rep["overall"]["schema_versions"]["old"] == {SCHEMA_VERSION: 16, "missing": 1}
    assert rep["missing_media"] == {"old": 1, "new": 1}
    assert rep["checksums"] is not None and "temporal_qa.jsonl" in rep["checksums"]["changed"]
    assert any(p.startswith(f"per_video/{V2}/") for p in rep["checksums"]["removed"])
    assert rep["split_changes"] == []
    assert any("UNEXPLAINED LOST 1" in s for s in rep["summary"]) and rep["summary"][-1].startswith("NOT OK")
    text = format_comparison(rep)
    assert "temporal_qa" in text and "NOT OK" in text

    # the loss is explained once the record's video is declared removed, tolerance also covers it
    assert compare_releases(old, new, removed_sources={V1})["ok"]
    assert compare_releases(old, new, tolerance=0.5)["ok"]


def test_compare_releases_split_move_and_regressions(tmp_path: Path):
    old = write_release(tmp_path / "a", fake_records())
    moved = fake_records()
    for rows in moved.values():
        for r in rows:
            if r["video_id"] == V2:
                r["split"] = "train"
    moved["events"][1]["record_id"] = f"event_{V1}_0002"  # old duplicate fixed
    moved["scenes"][1]["end_time"] = 6.0  # old temporal failure fixed
    moved["clips"][0]["start_time"] = -1.0  # new temporal failure
    moved["frames"][2]["record_id"] = moved["frames"][0]["record_id"]  # new duplicate
    new = write_release(tmp_path / "b", moved)

    rep = compare_releases(old, new)
    assert rep["split_changes"] == [{"video_id": V2, "old": "validation", "new": "train"}]
    assert not rep["ok"]
    assert rep["by_type"]["clip"]["temporal_failures"]["new_examples"] == [f"clip_{V1}_001"]
    assert rep["by_type"]["frame"]["duplicate_ids"]["new_examples"] == [f"frame_{V1}_001_000"]
    assert rep["by_type"]["event"]["duplicate_ids"] == {"old": 1, "new": 0, "new_examples": []}
    assert rep["by_type"]["scene"]["temporal_failures"]["new"] == 0
    assert rep["checksums"] is None and rep["missing_media"] == {"old": 1, "new": 1}
    assert any("split changes: 1" in s for s in rep["summary"])

    clean = fake_records()
    for rows in clean.values():
        for r in rows:
            if r["video_id"] == V2:
                r["split"] = "train"
    c = write_release(tmp_path / "c", clean)
    only_split = compare_releases(old, c)
    assert not only_split["ok"] and len(only_split["split_changes"]) == 1
    assert compare_releases(old, c, allow_split_changes=True)["ok"]
    assert compare_releases(old, old)["ok"]


def test_compare_falls_back_to_per_video(tmp_path: Path):
    old = write_release(tmp_path / "x", fake_records())
    new = write_release(tmp_path / "y", fake_records())
    for f in new.glob("*.jsonl"):
        if f.name not in ("media_manifest.jsonl", "provenance.jsonl"):
            f.unlink()
    rep = compare_releases(old, new)
    assert rep["ok"] and rep["overall"]["counts"] == {"old": 17, "new": 17, "delta": 0}


# ------------------------------------------------------------------ repair queues


def _queue(release: Path, name: str) -> list[dict[str, Any]]:
    return list(read_jsonl(release / "repairs" / f"{name}.jsonl"))


def test_build_repair_queues(release: Path):
    counts = build_repair_queues(release, camera_confidence_threshold=0.6)
    assert set(counts) == set(QUEUES)
    for q in QUEUES:
        assert (release / "repairs" / f"{q}.jsonl").exists()
        for row in _queue(release, q):
            assert row["proposed_action"] == ACTIONS[q] and row["shard"] == "shard_0001"
            assert set(row) == {"record_id", "record_type", "video_id", "shard", "reason", "detail", "proposed_action"}
    assert not list((release / "repairs").glob("*.tmp"))

    media = _queue(release, "missing_media")
    assert [(r["record_id"], r["reason"], r["detail"]) for r in media] == [(f"clip_{V1}_002", "clip_path_missing", f"clips/{V1}/clip_002.mp4")]
    assert media[0]["proposed_action"].startswith("re-extract from source")
    assert counts["missing_frames"] == 0

    ts = _queue(release, "invalid_timestamps")
    assert [(r["record_id"], r["reason"]) for r in ts] == [(f"scene_{V1}_002", "end_before_start")]
    assert ts[0]["record_type"] == "scene" and ts[0]["proposed_action"] == "re-evaluate the source segment; reject if it cannot be fixed"

    ev = _queue(release, "broken_evidence")
    assert [(r["record_id"], r["reason"]) for r in ev] == [(f"qa_{V1}_000002", "evidence_event_not_found:event_9999")]
    assert ev[0]["proposed_action"] == "relink evidence to existing ids or move the QA item to review"

    schema = _queue(release, "schema_failures")
    assert [(r["record_id"], r["reason"]) for r in schema] == [(f"scene_{V2}_001", "schema_version_missing")]
    assert schema[0]["proposed_action"] == "apply migration to the current schema_version and re-validate"

    dup = _queue(release, "duplicate_records")
    assert [(r["record_id"], r["reason"], r["video_id"]) for r in dup] == [(f"event_{V1}_0001", "duplicate_record_id", V1)]
    assert dup[0]["proposed_action"] == "keep the first occurrence; drop the rest or re-id"

    cam = _queue(release, "low_confidence_camera")
    assert sorted((r["record_id"], r["reason"]) for r in cam) == [
        (f"clip_{V1}_002", "camera_confidence_below_0.6"),
        (f"scene_{V1}_002", "camera_movement_ambiguous"),
    ]  # clip_vid_b (movement unknown, confidence None) is NOT queued
    assert cam[0]["proposed_action"] == "mark camera movement unknown or send to human review; never guess"

    desc = _queue(release, "unverified_descriptions")
    assert [(r["record_id"], r["reason"]) for r in desc] == [(f"videodesc_{V2}_00", "semantic_verification_not_run")]
    assert desc[0]["proposed_action"] == "human review or verifier pass; keep review status until then"

    rights = _queue(release, "rights_unverified")
    assert {r["video_id"] for r in rights} == {V2}
    assert {r["reason"] for r in rights} == {"rights_status_not_releasable:restricted"}
    assert sorted(r["record_type"] for r in rights) == ["clip", "frame"]
    assert rights[0]["proposed_action"] == "keep annotations; block media redistribution until rights evidence is recorded"

    summary = read_json(release / "repairs" / "summary.json")
    assert summary["counts"] == counts and summary["shard"] == "shard_0001" and summary["generated_at"]
    assert summary["records_scanned"] == 17 and summary["media_source"] == "media_manifest"
    assert summary["total"] == sum(counts.values())


def test_repair_queues_without_manifest_use_data_dir_or_skip(tmp_path: Path):
    rel = write_release(tmp_path / "final", fake_records(), with_media_manifest=False)
    (rel / "provenance.jsonl").unlink()
    counts = build_repair_queues(rel, shard="custom_shard")
    assert counts["missing_media"] == 0 and counts["missing_frames"] == 0  # unknown -> skipped
    assert read_json(rel / "repairs" / "summary.json")["media_unknown"] == 6
    rows = _queue(rel, "rights_unverified")
    assert rows and {r["reason"] for r in rows} == {"rights_status_unknown"} and rows[0]["shard"] == "custom_shard"

    data_dir = tmp_path / "data"
    (data_dir / "frames" / V1 / "scene_001").mkdir(parents=True)
    (data_dir / "frames" / V1 / "scene_001" / "f0.jpg").write_bytes(b"x")
    counts = build_repair_queues(rel, data_dir)
    assert counts["missing_frames"] == 2 and counts["missing_media"] == 3
    assert {r["reason"] for r in _queue(rel, "missing_frames")} == {"frame_path_missing"}

    # a tombstoned source is skipped entirely
    write_json_atomic(rel / "removed_sources.json", {V1: {"reason": "takedown"}})
    counts = build_repair_queues(rel, data_dir)
    assert all(r["video_id"] != V1 for q in QUEUES for r in _queue(rel, q))
    assert read_json(rel / "repairs" / "summary.json")["skipped_removed_sources"] == 11


def test_repair_queues_per_video_fallback_and_outdated_schema(tmp_path: Path):
    recs = fake_records()
    recs["frames"][0]["schema_version"] = "1.0.0"
    rel = write_release(tmp_path / "final", recs)
    for f in rel.glob("*.jsonl"):
        if f.name not in ("media_manifest.jsonl", "provenance.jsonl"):
            f.unlink()
    counts = build_repair_queues(rel)
    assert counts["schema_failures"] == 2
    assert sorted(r["reason"] for r in _queue(rel, "schema_failures")) == ["schema_version_missing", "schema_version_outdated:1.0.0"]
    assert counts["broken_evidence"] == 1 and counts["duplicate_records"] == 1


# ------------------------------------------------------------------ lineage


def test_lineage_append_read_summary(tmp_path: Path):
    final = tmp_path / "final"
    row = lineage_row("scene_x_001", "scene", V1, "status", "accepted", "review", "camera ambiguous", "repair_queue", "pipeline", policy_version="2026.10.19", timestamp="2026-11-01T00:00:00+00:00")
    assert row["timestamp"] == "2026-11-01T00:00:00+00:00" and row["previous_checksum"] is None
    assert lineage_row("a", "clip", V1, "media", None, "clips/x.mp4", "re-extracted", "ffmpeg", "raj")["timestamp"].endswith("+00:00")
    with pytest.raises(ValueError):
        lineage_row("a", "clip", V1, "bogus", None, None, "r", "m", "a")

    assert append_lineage(final, [row]) == 1
    assert append_lineage(final, [
        {"record_id": "clip_y", "record_type": "clip", "video_id": V2, "change": "media", "previous": None, "new": "clips/y.mp4", "reason": "missing", "method": "re-extract", "actor": "raj", "new_checksum": "abc"},
        {"record_id": "qa_z", "record_type": "temporal_qa", "video_id": V2, "change": "removed", "previous": "accepted", "new": None, "reason": "takedown", "method": "takedown", "actor": "pipeline"},
    ]) == 2
    rows = list(read_lineage(final))
    assert [r["record_id"] for r in rows] == ["scene_x_001", "clip_y", "qa_z"]  # appended, never overwritten
    assert rows[1]["new_checksum"] == "abc" and rows[1]["timestamp"] and set(rows[1]) == set(row)
    with (final / LINEAGE_FILE).open("r", encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.strip()]
    assert len(lines) == 3 and all(json.loads(ln) for ln in lines)

    summary = lineage_summary(final)
    assert summary["rows"] == 3
    assert summary["by_change"] == {"media": 1, "removed": 1, "status": 1}
    assert summary["by_actor"] == {"pipeline": 2, "raj": 1}
    assert summary["first"] == "2026-11-01T00:00:00+00:00" and summary["last"] >= summary["first"]
    assert lineage_summary(tmp_path / "nowhere") == {"rows": 0, "by_change": {}, "by_actor": {}, "first": None, "last": None}
