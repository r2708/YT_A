"""Tests for dataset.human_review (stratified sampling, agreement, adjudication) and dataset.camera_eval."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_dataset.dataset import camera_eval
from video_dataset.dataset import human_review as hr
from video_dataset.dataset.review import read_review_labels


def _rec(rid: str, rtype: str, vid: str, tier: str | None = "silver", status: str = "accepted", codes: list[str] | None = None, **extra) -> dict:
    rec = {
        "record_id": rid, "record_type": rtype, "video_id": vid, "tier": tier, "split": "train",
        "validation": {"status": status, "reason_codes": codes or []},
        "start_time": 1.0, "end_time": 3.0, "clip_path": f"clips/{rid}.mp4", "frame_ids": [f"f{i}" for i in range(12)],
        "summary": f"summary of {rid}",
    }
    rec.update(extra)
    return rec


def _population() -> dict[str, list[dict]]:
    scenes, clips, events = [], [], []
    for i in range(40):
        vid = f"v{i % 8}"
        tier = ("gold", "silver", "bronze")[i % 3]
        if i % 5 == 0:
            codes, status = ["failed_check:camera_motion_agreement"], "review"
        elif i % 7 == 0:
            codes, status = ["missing_check:clip_boundaries"], "review"
        else:
            codes, status = [], "accepted"
        cam = {"camera_movement": ("pan", "static", "zoom")[i % 3], "movement_direction": "left" if i % 3 == 0 else None}
        scenes.append(_rec(f"s{i}", "scene", vid, tier, status, codes, camera=cam))
    for i in range(20):
        clips.append(_rec(f"c{i}", "clip", f"v{i % 8}", "silver", "accepted", [], claims=[{"status": "contradicted"}] if i % 4 == 0 else []))
    for i in range(10):
        events.append(_rec(f"e{i}", "event", f"v{i % 8}", None, "accepted", [], boundary_precision="measured"))
    return {"scenes": scenes, "clips": clips, "events": events}


# ---------------------------------------------------------------- failure_category


def test_failure_category_precedence():
    assert hr.failure_category({"validation": {"status": "accepted", "reason_codes": []}}) == "accepted"
    assert hr.failure_category({"validation": {"status": "rejected", "reason_codes": ["human:rejected", "failed_check:clip_boundaries"]}}) == "human_labelled"
    assert hr.failure_category({"validation": {"reason_codes": ["failed_check:clip_boundaries"]}, "claims": [{"status": "contradicted"}]}) == "contradicted_claim"
    assert hr.failure_category({"validation": {"reason_codes": ["failed_check:timestamps"]}}) == "boundary"
    assert hr.failure_category({"validation": {"reason_codes": []}, "camera": {"movement_ambiguous": True}}) == "camera"
    assert hr.failure_category({"validation": {"reason_codes": ["semantic_verification:not_run"]}}) == "semantic"
    assert hr.failure_category({"validation": {"reason_codes": ["missing_check:single_shot"]}}) == "missing_check"
    assert hr.failure_category({"validation": {"reason_codes": ["confidence:review"]}}) == "confidence"
    assert hr.failure_category({"validation": {"status": "review", "reason_codes": ["something_else"]}}) == "other"


# ---------------------------------------------------------------- sampling


def test_stratified_sample_allocation_sums_to_n_and_is_deterministic():
    pop = _population()
    rows = hr.stratified_sample(pop, 25, seed=7)
    assert len(rows) == 25
    assert len({r["record_id"] for r in rows}) == 25
    # every non-empty stratum is represented at least once
    strata_in_pop = {hr._stratum_key(hr.stratum_of(r, hr.DEFAULT_STRATA, r["record_type"])) for rs in pop.values() for r in rs}
    strata_in_sample = {hr._stratum_key(r["stratum"]) for r in rows}
    assert strata_in_sample == strata_in_pop
    again = hr.stratified_sample(pop, 25, seed=7)
    assert [r["record_id"] for r in again] == [r["record_id"] for r in rows]
    assert [r["assigned_to"] for r in again] == [r["assigned_to"] for r in rows]
    assert hr.stratified_sample(pop, 25, seed=8) != rows
    for r in rows:
        assert r["protocol_version"] == hr.PROTOCOL_VERSION
        assert r["label"] is None and r["reviewer"] is None
        assert len(r["evidence"]["frame_ids"]) <= 8
        assert r["evidence"]["media"].endswith(".mp4")
        assert r["stratum_size"] >= 1


def test_stratified_sample_evidence_and_split_filter():
    pop = _population()
    pop["scenes"][0]["split"] = "test"
    rows = hr.stratified_sample(pop, 70, split="test")
    assert [r["record_id"] for r in rows] == ["s0"]
    assert rows[0]["evidence"]["camera_movement"] == "pan_left"
    ev_rows = [r for r in hr.stratified_sample(pop, 70) if r["record_type"] == "event"]
    assert ev_rows and all(r["evidence"]["boundary_precision"] == "measured" for r in ev_rows)
    assert len(hr.stratified_sample(pop, 1000)) == 70  # capped by the population


def test_per_video_cap():
    pop = _population()
    rows = hr.stratified_sample(pop, 30, per_video_cap=2, seed=1)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["video_id"]] = counts.get(r["video_id"], 0) + 1
    assert max(counts.values()) <= 2
    assert len(rows) == 16  # 8 videos x cap 2: the cap, not n, bounds the sample


def test_double_fraction_and_round_robin():
    pop = _population()
    rows = hr.stratified_sample(pop, 40, double_fraction=0.25, reviewers=("ann", "bob", "cy"), seed=3)
    doubles = [r for r in rows if r["double_annotated"]]
    singles = [r for r in rows if not r["double_annotated"]]
    assert len(doubles) == 10
    assert all(r["assigned_to"] == ["ann", "bob", "cy"] for r in doubles)
    assert all(len(r["assigned_to"]) == 1 for r in singles)
    per_reviewer = {n: sum(1 for r in singles if r["assigned_to"] == [n]) for n in ("ann", "bob", "cy")}
    assert max(per_reviewer.values()) - min(per_reviewer.values()) <= 1
    solo = hr.stratified_sample(pop, 10, double_fraction=0.5, reviewers=("ann",), seed=3)
    assert not any(r["double_annotated"] for r in solo)


def test_sample_size_cochran():
    assert hr.sample_size(0) == 0
    # n0 = 1.96^2 * 0.25 / 0.05^2 = 384.16, then the finite population correction
    assert hr.sample_size(100000) == 383
    assert hr.sample_size(500) == 218
    assert hr.sample_size(50) == 45
    assert hr.sample_size(10) == 10  # never more than the population
    assert hr.sample_size(100000, confidence=0.99) > hr.sample_size(100000) > hr.sample_size(100000, confidence=0.90)
    assert hr.sample_size(100000, margin=0.03) > hr.sample_size(100000, margin=0.05)
    with pytest.raises(ValueError):
        hr.sample_size(100, confidence=0.87)


# ---------------------------------------------------------------- labelled rows + agreement


def _labelled(record_id: str, reviewer: str, label: str, **extra) -> dict:
    return {"record_id": record_id, "record_type": "clip", "stratum": {"failure_category": "accepted"}, "reviewer": reviewer, "label": label, **extra}


def test_read_labelled_handles_sample_and_legacy_rows(tmp_path: Path):
    pop = _population()
    rows = hr.stratified_sample(pop, 6, seed=2)
    path = hr.write_sample(rows, tmp_path)
    assert path == tmp_path / "reviews" / "sample.jsonl"
    assert hr.read_labelled(path) == []  # nothing filled in yet
    filled = list(hr.read_sample(path))
    filled[0]["label"], filled[0]["reviewer"] = "Accept", "ann"
    filled[1]["label"] = "reject"  # no reviewer -> unknown
    filled[2]["label"] = "maybe"  # invalid -> dropped
    with path.open("w") as fh:
        for r in filled:
            fh.write(json.dumps(r) + "\n")
    got = hr.read_labelled(path)
    assert [(r["label"], r["reviewer"]) for r in got] == [("accept", "ann"), ("reject", "unknown")]


def test_kappa_on_known_contingency_table():
    # 2x2 table accept/not: a=20 both accept, b=5 (A accept, B reject), c=10 (A reject, B accept), d=15 both reject
    # po = 35/50 = 0.7, pe = 0.5*0.6 + 0.5*0.4 = 0.5 -> kappa = 0.4
    rows: list[dict] = []
    cells = [("accept", "accept", 20), ("accept", "reject", 5), ("reject", "accept", 10), ("reject", "reject", 15)]
    i = 0
    for la, lb, count in cells:
        for _ in range(count):
            rows.append(_labelled(f"r{i}", "reviewer_a", la))
            rows.append(_labelled(f"r{i}", "reviewer_b", lb))
            i += 1
    agg = hr.agreement(rows)
    assert agg["records"] == 50 and agg["records_multi_reviewed"] == 50
    assert agg["percent_agreement"] == 0.7
    assert agg["kappa_binary"] == 0.4
    assert agg["kappa_multiclass"] == 0.4  # only two classes used
    assert len(agg["disagreements"]) == 15 and agg["disagreement_rate"] == 0.3
    assert agg["by_record_type"]["clip"]["kappa_binary"] == 0.4
    assert agg["by_stratum"]["failure_category"]["accepted"]["percent_agreement"] == 0.7
    assert agg["camera_agreement"]["kappa"] is None
    assert agg["boundary_audit"]["n"] == 0 and agg["boundary_audit"]["mean_error_seconds"] is None


def test_agreement_camera_boundary_and_no_nan():
    rows = [
        _labelled("x1", "a", "accept", camera_movement_label="pan_left", boundary_ok=True, boundary_error_seconds=0.1),
        _labelled("x1", "b", "accept", camera_movement_label="pan_right", boundary_ok=True, boundary_error_seconds=0.3),
        _labelled("x2", "a", "accept", camera_movement_label="static", boundary_ok=False, boundary_error_seconds=1.0),
        _labelled("x2", "b", "accept", camera_movement_label="static"),
        _labelled("x3", "a", "correct"),  # single reviewer: not in the agreement stats
    ]
    agg = hr.agreement(rows)
    assert agg["records"] == 3 and agg["records_multi_reviewed"] == 2
    assert agg["percent_agreement"] == 1.0
    assert agg["kappa_binary"] == 1.0  # all accept: pe == 1 -> defined as 1.0, not NaN
    cam = agg["camera_agreement"]
    assert cam["n_pairs"] == 2 and cam["percent_agreement"] == 0.5 and cam["family_percent_agreement"] == 1.0
    ba = agg["boundary_audit"]
    assert ba["n"] == 3 and ba["share_ok"] == 0.6667
    assert ba["mean_error_seconds"] == 0.4667 and ba["median_error_seconds"] == 0.3
    assert json.dumps(agg, allow_nan=False)
    assert hr.agreement([])["percent_agreement"] is None


# ---------------------------------------------------------------- adjudication


def test_adjudicate_and_write(tmp_path: Path):
    rows = [
        _labelled("u1", "a", "accept"), _labelled("u1", "b", "accept", note="fine"),
        _labelled("d1", "a", "accept"), _labelled("d1", "b", "reject", note="blurry"),
        _labelled("d2", "a", "correct", corrections={"summary": "new"}), _labelled("d2", "b", "reject"),
        _labelled("s1", "a", "correct", corrections={"summary": "fixed"}),
    ]
    final = hr.adjudicate(rows, {"d2": "correct"})
    assert final["u1"]["label"] == "accept" and final["u1"]["reviewer"] == "consensus" and final["u1"]["reviewers"] == ["a", "b"]
    assert final["d1"]["label"] is None and final["d1"]["needs_adjudication"] and final["d1"]["votes"] == {"a": "accept", "b": "reject"}
    assert final["d2"]["label"] == "correct" and final["d2"]["reviewer"] == "adjudicated" and final["d2"]["corrections"] == {"summary": "new"}
    assert final["s1"]["reviewer"] == "a" and final["s1"]["corrections"] == {"summary": "fixed"}

    labels_path = tmp_path / "reviews" / "labels.jsonl"
    labels_path.parent.mkdir()
    labels_path.write_text(json.dumps({"record_id": "old", "label": "reject"}) + "\n" + json.dumps({"record_id": "u1", "label": "reject"}) + "\n")
    out = hr.write_adjudicated(tmp_path, final)
    assert out == labels_path
    consumed = read_review_labels(out)
    assert consumed["old"]["label"] == "reject"
    assert consumed["u1"]["label"] == "accept"  # new wins
    assert "d1" not in consumed  # unresolved rows are not written
    assert consumed["s1"]["corrections"] == {"summary": "fixed"}


def test_verification_report(tmp_path: Path):
    pop = _population()
    rows = hr.stratified_sample(pop, 20, double_fraction=0.5, seed=5)
    labelled = []
    for r in rows[:8]:
        for reviewer in r["assigned_to"]:
            labelled.append({**r, "label": "accept", "reviewer": reviewer})
    labelled[0]["label"] = "reject"
    rep = hr.verification_report(rows, labelled)
    assert rep["protocol_version"] == hr.PROTOCOL_VERSION
    assert rep["sample_size"] == 20 and rep["double_annotated"] == 10 and rep["labelled"] == 8
    assert rep["population_hint"] == 70
    assert sum(s["sampled"] for s in rep["coverage_by_stratum"].values()) == 20
    assert sum(s["labelled"] for s in rep["coverage_by_stratum"].values()) == 8
    assert rep["unresolved_disagreements"] == len(rep["agreement"]["disagreements"])
    assert json.dumps(rep, allow_nan=False)


# ---------------------------------------------------------------- camera_eval


def test_confusion_matrix_on_hand_built_records(tmp_path: Path):
    def scene(rid: str, family: str | None, ambiguous: bool = False, movement: str | None = None) -> dict:
        cam: dict = {"camera_movement": family, "movement_ambiguous": ambiguous}
        if movement:
            cam["movement"] = movement
        return {"record_id": rid, "record_type": "scene", "camera": cam}

    records = {
        "scenes": [
            scene("a", "pan"), scene("b", "pan"), scene("c", "static", ambiguous=True),
            scene("d", "zoom"), scene("e", None, movement="tilt_up"), scene("f", None),
        ],
        "frames": [{"record_id": "g", "record_type": "frame", "camera": {"camera_movement": "pan"}}],
    }
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text("record_id,camera_movement\na,pan_left\nb,static\nc,static\nd,dolly_in\ne,tilt\nf,unknown\ng,pan\nzz,pan\n")
    labels = camera_eval.read_camera_labels(csv_path)
    assert labels["a"] == "pan" and labels["d"] == "dolly" and labels["f"] == "unknown" and len(labels) == 8

    rep = camera_eval.confusion_matrix(records, labels)
    assert rep["n"] == 6 and rep["labelled"] == 8  # frames and unmatched ids are not scored
    assert rep["labels"] == ["dolly", "pan", "static", "tilt", "unknown", "zoom"]
    assert rep["accuracy"] == round(4 / 6, 4)
    assert rep["ambiguous_predicted"] == 1
    i = {c: k for k, c in enumerate(rep["labels"])}
    assert rep["matrix"][i["pan"]][i["pan"]] == 1 and rep["matrix"][i["static"]][i["pan"]] == 1
    assert rep["matrix"][i["dolly"]][i["zoom"]] == 1 and rep["matrix"][i["unknown"]][i["unknown"]] == 1
    assert rep["per_class"]["pan"] == {"precision": 0.5, "recall": 1.0, "f1": 0.6667, "support": 1, "predicted": 2}
    assert rep["per_class"]["dolly"]["recall"] == 0.0 and rep["per_class"]["dolly"]["precision"] is None
    assert rep["most_confused_pairs"] == [{"human": "dolly", "predicted": "zoom", "count": 1}, {"human": "static", "predicted": "pan", "count": 1}]
    text = camera_eval.format_confusion(rep)
    assert "human \\ predicted" in text and "dolly->zoom x1" in text

    jsonl = tmp_path / "sample.jsonl"
    jsonl.write_text(json.dumps({"record_id": "a", "camera_movement_label": "Pan Right"}) + "\n" + json.dumps({"record_id": "b"}) + "\n")
    assert camera_eval.read_camera_labels(jsonl) == {"a": "pan"}
    assert camera_eval.confusion_matrix({}, {})["accuracy"] is None
