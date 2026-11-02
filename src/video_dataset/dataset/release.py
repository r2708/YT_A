"""Release gates and the machine-generated release report.

``run_release_check`` validates a final dataset directory the way a consumer would load it:

* schema: every JSONL record validates against the current record model, timestamps are ordered
  and inside the video, confidences are in [0, 1];
* references: frames / clips / events / QA point at parents that exist in the export (scene ids,
  clip ids, event ids), QA evidence intervals contain their events;
* JSONL vs Parquet: identical record ids and counts per type;
* splits: no video or near-duplicate group crosses a split; ``splits.json`` matches the records;
* media: with ``require_media`` every referenced asset resolves (always reported);
* strict subset: the ``strict`` subset only holds records that pass the policy, hard negatives are
  never positives, every negative is typed;
* counts: JSONL line counts, typed Parquet row counts, ``manifest.json``, ``statistics.json`` and
  ``splits.json`` all describe the same records (one authoritative export, no separately kept totals);
* required splits: every split in ``export.split.required_splits`` holds at least its minimum
  number of source groups (an empty validation split is a failed release, not a footnote);
* source pool: at least ``export.min_source_videos`` source videos;
* portable paths: no metadata file or record carries a machine-specific path (``/Users/...``,
  ``/home/...``, drive letters);
* accepted means verified: no ``accepted`` record lacks a policy version, a passed boundary check
  (clips) or a required check;
* checksums: ``checksums.sha256`` lists every dataset file and still verifies;
* timestamps: no accepted record failed the export-time temporal validation;
* rights: every record carries its rights block; shipped media only belongs to release-rights sources;
* reproducibility: the manifest names the pipeline commit, release id, schema and policy versions.

Each gate yields ``passed`` / ``failed`` / ``skipped`` with counts. The report's ``status`` is
``blocked`` (a gate failed), ``passed_with_exclusions`` (every gate passed but records were
downgraded, left in review, kept unmigrated, had missing media or ambiguous camera labels: the
``exclusions`` block discloses them) or ``passed``. It also carries the human-verification summary
(sample, agreement, unresolved disagreements), the repair-queue counts and the quality metrics a
release should publish: status / reason-code / semantic-verification split / grounding methods /
coverage / duration / split / tier / rights distributions.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from video_dataset.config import PipelineConfig
from video_dataset.dataset.curation import STRICT_SUBSET, split_leaks
from video_dataset.dataset.export import DATASET_FILES, RECORD_MODELS, SPLITS_FILE
from video_dataset.dataset.media_manifest import build_media_manifest, summarize
from video_dataset.schemas.common import SCHEMA_VERSION
from video_dataset.schemas.quality import ValidationStatus
from video_dataset.utils.io import read_json, read_jsonl, write_json_atomic
from video_dataset.utils.logging import get_logger
from video_dataset.validation.policy import (
    _OPTIONAL_WHEN_ABSENT,
    HUMAN_ACCEPT,
    POLICY_VERSION,
    REQUIRED_CHECKS,
    _record_kind,
    normalize_checks,
    record_type_of,
    strict_ok,
)

log = get_logger("dataset.release")
REPORT_FILE = "release_report.json"
METADATA_FILES = ("manifest.json", "statistics.json", SPLITS_FILE, "schemas.json", "provenance.jsonl", "media_manifest.jsonl", "review_queue.jsonl")
# Prefixes that only make sense on the producer's machine. Anything matching them inside a published
# file would break on every other computer and leaks the local directory layout.
_MACHINE_PATH = re.compile(r"(?:^|[\s\"'=:(\[,])(?:/Users/|/home/|/private/|/tmp/|/var/folders/|/mnt/[a-z]/|[A-Za-z]:[\\/])")


def load_final(final_dir: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for name in DATASET_FILES:
        f = final_dir / f"{name}.jsonl"
        out[name] = list(read_jsonl(f)) if f.exists() else []
    return out


def _gate(name: str, ok: bool | None, **details: Any) -> dict[str, Any]:
    return {"gate": name, "state": "skipped" if ok is None else ("passed" if ok else "failed"), **details}


def schema_gate(records: dict[str, list[dict[str, Any]]], max_examples: int = 20) -> dict[str, Any]:
    bad: list[dict[str, Any]] = []
    checked = 0
    n_unmigrated = 0
    versions: dict[str, int] = {}
    for name, rows in records.items():
        model = RECORD_MODELS.get(name)
        if model is None:
            continue
        for rec in rows:
            checked += 1
            versions[str(rec.get("schema_version") or "missing")] = versions.get(str(rec.get("schema_version") or "missing"), 0) + 1
            d = dict(rec)
            d.pop("record_type", None)
            try:
                obj = model.model_validate(d)
            except Exception as exc:
                if len(bad) < max_examples:
                    bad.append({"type": name, "record_id": rec.get("record_id"), "error": str(exc)[:200]})
                continue
            start = getattr(obj, "start_time", None)
            end = getattr(obj, "end_time", None)
            ev = getattr(obj, "evidence", None)
            if ev is not None and hasattr(ev, "start_time"):  # QA evidence (descriptions carry an evidence *dict* of ids)
                start, end = ev.start_time, ev.end_time
            if start is not None and end is not None and (float(start) < 0 or float(end) < float(start)):
                if len(bad) < max_examples:
                    bad.append({"type": name, "record_id": rec.get("record_id"), "error": f"bad interval {start}-{end}"})
            conf = rec.get("confidence")
            if conf is not None and not (0.0 <= float(conf) <= 1.0):
                if len(bad) < max_examples:
                    bad.append({"type": name, "record_id": rec.get("record_id"), "error": f"confidence {conf} outside [0, 1]"})
            if isinstance(rec.get("extensions"), dict) and rec["extensions"].get("_schema_error"):
                n_unmigrated += 1
                if len(bad) < max_examples:
                    bad.append({"type": name, "record_id": rec.get("record_id"), "error": "kept verbatim after a failed migration: " + str(rec["extensions"]["_schema_error"])[:120]})
    n_bad = len(bad)
    # one supported schema version per release: a record of another version (or none) is a failed migration
    other_versions = {v: n for v, n in versions.items() if v != SCHEMA_VERSION}
    ok = n_bad == 0 and not other_versions
    return _gate("schema", ok, records_checked=checked, invalid=n_bad, unmigrated=n_unmigrated, examples=bad, schema_versions=versions, expected_schema_version=SCHEMA_VERSION, other_schema_versions=other_versions)


def reference_gate(records: dict[str, list[dict[str, Any]]], max_examples: int = 20) -> dict[str, Any]:
    scenes = {(str(r.get("video_id")), str(r.get("scene_id"))) for r in records.get("scenes", [])}
    clips = {(str(r.get("video_id")), str(r.get("clip_id"))) for r in records.get("clips", [])}
    events = {(str(r.get("video_id")), str(r.get("event_id"))): r for r in records.get("events", [])}
    problems: list[dict[str, Any]] = []
    counts = {"frames_without_scene": 0, "frames_without_clip": 0, "clips_without_scene": 0, "qa_missing_event": 0, "qa_event_outside_evidence": 0, "sequences_missing_scene": 0, "duplicate_record_ids": 0, "negatives_untyped": 0, "negatives_equal_positive": 0}

    def note(kind: str, rec: dict[str, Any], detail: str) -> None:
        counts[kind] += 1
        if len(problems) < max_examples:
            problems.append({"problem": kind, "record_id": rec.get("record_id"), "detail": detail})

    for r in records.get("frames", []):
        if (str(r.get("video_id")), str(r.get("scene_id"))) not in scenes and scenes:
            note("frames_without_scene", r, str(r.get("scene_id")))
        if r.get("clip_id") and (str(r.get("video_id")), str(r.get("clip_id"))) not in clips and clips:
            note("frames_without_clip", r, str(r.get("clip_id")))
    for r in records.get("clips", []):
        if (str(r.get("video_id")), str(r.get("scene_id"))) not in scenes and scenes:
            note("clips_without_scene", r, str(r.get("scene_id")))
    for name in ("temporal_qa", "long_video_qa"):
        for r in records.get(name, []):
            ev = r.get("evidence") or {}
            for eid in ev.get("event_ids") or []:
                e = events.get((str(r.get("video_id")), str(eid)))
                if e is None:
                    if events:
                        note("qa_missing_event", r, str(eid))
                    continue
                if float(e.get("start_time", 0)) < float(ev.get("start_time", 0)) - 0.051 or float(e.get("end_time", 0)) > float(ev.get("end_time", 0)) + 0.051:
                    note("qa_event_outside_evidence", r, f"{eid} {e.get('start_time')}-{e.get('end_time')} vs {ev.get('start_time')}-{ev.get('end_time')}")
    for r in records.get("sequences", []):
        for sid in r.get("scene_ids") or []:
            if (str(r.get("video_id")), str(sid)) not in scenes:
                note("sequences_missing_scene", r, str(sid))
    for name, rows in records.items():
        seen: set[str] = set()
        for r in rows:
            rid = str(r.get("record_id"))
            if rid in seen:
                note("duplicate_record_ids", r, name)
            seen.add(rid)
            positive = r.get("answer") or r.get("summary") or r.get("description")
            for neg in r.get("hard_negatives") or []:
                if not isinstance(neg, dict) or not neg.get("kind"):
                    note("negatives_untyped", r, name)
                elif positive and str(neg.get("text", "")).strip().lower() == str(positive).strip().lower():
                    note("negatives_equal_positive", r, name)
    ok = all(v == 0 for v in counts.values())
    return _gate("references", ok, **counts, examples=problems)


def parquet_gate(final_dir: Path, records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except Exception:
        return _gate("parquet", None, reason="pyarrow unavailable")
    mismatches: dict[str, Any] = {}
    checked = 0
    for name in DATASET_FILES:
        p = final_dir / f"{name}.parquet"
        j = final_dir / f"{name}.jsonl"
        if not p.exists() and not j.exists():
            continue
        if not p.exists() or not j.exists():
            mismatches[name] = {"jsonl": j.exists(), "parquet": p.exists()}
            continue
        checked += 1
        ids_json = [str(r.get("record_id")) for r in records.get(name, [])]
        try:
            table = pq.read_table(p, columns=["record_id"])
            ids_pq = [str(x) for x in table.column("record_id").to_pylist()]
        except Exception as exc:
            mismatches[name] = {"error": str(exc)[:160]}
            continue
        if sorted(ids_json) != sorted(ids_pq):
            mismatches[name] = {"jsonl_rows": len(ids_json), "parquet_rows": len(ids_pq), "only_jsonl": len(set(ids_json) - set(ids_pq)), "only_parquet": len(set(ids_pq) - set(ids_json))}
    return _gate("jsonl_parquet_ids", not mismatches, types_checked=checked, mismatches=mismatches)


def split_gate(final_dir: Path, records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    leaks = split_leaks(records)
    manifest_ok = True
    splits_file = final_dir / SPLITS_FILE
    if splits_file.exists():
        try:
            listed = read_json(splits_file).get("videos", {})
            seen: dict[str, str] = {}
            for sp, vids in listed.items():
                for v in vids:
                    if v in seen and seen[v] != sp:
                        manifest_ok = False
                    seen[v] = sp
            for rows in records.values():
                for r in rows:
                    v, sp = str(r.get("video_id")), r.get("split")
                    if sp and v in seen and seen[v] != sp:
                        manifest_ok = False
        except (ValueError, AttributeError):
            manifest_ok = False
    counts: dict[str, int] = {}
    for rows in records.values():
        for r in rows:
            counts[str(r.get("split") or "unassigned")] = counts.get(str(r.get("split") or "unassigned"), 0) + 1
    return _gate("split_leakage", not leaks and manifest_ok, leaking=dict(list(leaks.items())[:20]), leaking_count=len(leaks), splits_manifest_consistent=manifest_ok, records_by_split=counts)


def media_gate(records: dict[str, list[dict[str, Any]]], data_dir: Path, final_dir: Path, required: bool) -> dict[str, Any]:
    rows = build_media_manifest(records, data_dir, final_dir, hash_files=False)
    summary = summarize(rows)
    all_present = bool(summary.pop("ok"))
    return _gate("media_references", (all_present if required else None), required=required, all_present=all_present, **summary)


def strict_gate(records: dict[str, list[dict[str, Any]]], max_examples: int = 20) -> dict[str, Any]:
    wrong: list[dict[str, Any]] = []
    n = 0
    for name, rows in records.items():
        for r in rows:
            if STRICT_SUBSET not in (r.get("subsets") or []):
                continue
            n += 1
            ok, why = strict_ok(r)
            if not ok and len(wrong) < max_examples:
                wrong.append({"type": name, "record_id": r.get("record_id"), "why": why})
            if (r.get("validation") or {}).get("status") != ValidationStatus.ACCEPTED.value and len(wrong) < max_examples:
                wrong.append({"type": name, "record_id": r.get("record_id"), "why": ["status_not_accepted"]})
    return _gate("strict_subset", not wrong, strict_records=n, violations=len(wrong), examples=wrong)


def counts_gate(final_dir: Path, records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Every published total describes the same records: per-type JSONL counts (what was loaded),
    typed Parquet rows, ``manifest.json["counts"]``, ``statistics.json`` per-type counts and
    ``splits.json`` video counts versus the videos in the records."""
    problems: dict[str, Any] = {}
    jsonl = {name: len(rows) for name, rows in records.items()}
    try:
        import pyarrow.parquet as pq

        for name in DATASET_FILES:
            p = final_dir / f"{name}.parquet"
            if p.exists():
                rows = pq.read_metadata(p).num_rows
                if rows != jsonl.get(name, 0):
                    problems[f"parquet:{name}"] = {"jsonl": jsonl.get(name, 0), "parquet": rows}
    except Exception as exc:  # pragma: no cover - pyarrow missing
        problems["parquet"] = {"error": str(exc)[:120]}
    manifest_path = final_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = read_json(manifest_path)
        except (ValueError, AttributeError):
            problems["manifest"] = "unreadable"
        for name, n in (manifest.get("counts") or {}).items():
            if name in jsonl and int(n) != jsonl[name]:
                problems[f"manifest:{name}"] = {"jsonl": jsonl[name], "manifest": int(n)}
        listed = {str(m.get("video_id")) for m in manifest.get("videos") or [] if m.get("video_id")}
        in_records = {str(r.get("video_id")) for rows in records.values() for r in rows if r.get("video_id")}
        if listed and listed != in_records:
            problems["manifest:videos"] = {"only_manifest": sorted(listed - in_records)[:10], "only_records": sorted(in_records - listed)[:10], "manifest": len(listed), "records": len(in_records)}
    stats_path = final_dir / "statistics.json"
    if stats_path.exists():
        try:
            stats = read_json(stats_path)
            for name in DATASET_FILES:
                n = stats.get(name)
                if isinstance(n, int) and name in jsonl and n != jsonl[name]:
                    problems[f"statistics:{name}"] = {"jsonl": jsonl[name], "statistics": n}
            for name, n in (stats.get("jsonl_counts") or {}).items():
                if name in jsonl and int(n) != jsonl[name]:
                    problems[f"statistics.jsonl_counts:{name}"] = {"jsonl": jsonl[name], "statistics": int(n)}
        except (ValueError, AttributeError):
            problems["statistics"] = "unreadable"
    splits_path = final_dir / SPLITS_FILE
    if splits_path.exists():
        try:
            sp = read_json(splits_path)
            by_split: dict[str, set[str]] = {}
            for rows in records.values():
                for r in rows:
                    if r.get("split") and r.get("video_id"):
                        by_split.setdefault(str(r["split"]), set()).add(str(r["video_id"]))
            for name, vids in (sp.get("videos") or {}).items():
                if set(vids) != by_split.get(name, set()):
                    problems[f"splits:{name}"] = {"splits_json": len(vids), "records": len(by_split.get(name, set()))}
            for name, n in (sp.get("counts") or {}).items():
                if int(n) != len(sp.get("videos", {}).get(name, [])):
                    problems[f"splits.counts:{name}"] = {"counts": int(n), "videos": len(sp.get("videos", {}).get(name, []))}
        except (ValueError, AttributeError):
            problems["splits"] = "unreadable"
    return _gate("counts_consistent", not problems, jsonl_counts=jsonl, mismatches=problems, empty_record_types=[n for n in DATASET_FILES if not jsonl.get(n)])


def required_splits_gate(config: PipelineConfig, records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    cfg = config.export.split
    groups: dict[str, set[str]] = {}
    for rows in records.values():
        for r in rows:
            if r.get("split"):
                groups.setdefault(str(r["split"]), set()).add(str(r.get("source_group_id") or r.get("video_id")))
    counts = {k: len(v) for k, v in groups.items()}
    minimum = {"train": 1, "validation": int(cfg.min_validation_videos), "test": int(cfg.min_test_videos)}
    required = [str(x) for x in (cfg.required_splits or [])] if cfg.enabled else []
    short = {name: {"groups": counts.get(name, 0), "minimum": max(1, minimum.get(name, 1))} for name in required if counts.get(name, 0) < max(1, minimum.get(name, 1))}
    return _gate("required_splits", (not short) if required else None, required=required, groups_by_split=counts, below_minimum=short)


def source_pool_gate(config: PipelineConfig, records: dict[str, list[dict[str, Any]]], manifests: list[dict[str, Any]], already_released: int = 0) -> dict[str, Any]:
    """The minimum applies to the *released dataset*: the videos in this export plus
    ``already_released`` (videos in shards uploaded earlier), so an upload shard of 15 videos passes
    once the dataset as a whole is large enough, while a 15-video standalone release does not."""
    minimum = int(getattr(config.export, "min_source_videos", 0) or 0)
    videos = {str(r.get("video_id")) for rows in records.values() for r in rows if r.get("video_id")}
    groups = {str(r.get("source_group_id") or r.get("video_id")) for rows in records.values() for r in rows if r.get("video_id")}
    channels = {str(m.get("channel")) for m in manifests if m.get("channel")}
    total = len(videos) + max(0, int(already_released))
    return _gate("source_pool", (total >= minimum) if minimum > 0 else None, minimum=minimum, source_videos=len(videos), already_released=int(already_released), released_total=total, source_groups=len(groups), channels=len(channels))


def _scan_for_machine_paths(obj: Any, where: str, hits: list[dict[str, str]], limit: int) -> None:
    if len(hits) >= limit:
        return
    if isinstance(obj, str):
        if _MACHINE_PATH.search(obj):
            hits.append({"where": where, "value": obj[:160]})
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _scan_for_machine_paths(v, f"{where}.{k}", hits, limit)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _scan_for_machine_paths(v, f"{where}[{i}]", hits, limit)


def portable_paths_gate(final_dir: Path, records: dict[str, list[dict[str, Any]]], max_examples: int = 20) -> dict[str, Any]:
    """No published file may carry a machine-specific path."""
    hits: list[dict[str, str]] = []
    files_checked = 0
    for name in METADATA_FILES:
        p = final_dir / name
        if not p.exists():
            continue
        files_checked += 1
        try:
            if p.suffix == ".jsonl":
                for i, row in enumerate(read_jsonl(p)):
                    _scan_for_machine_paths(row, f"{name}:{i}", hits, max_examples)
                    if len(hits) >= max_examples:
                        break
            else:
                _scan_for_machine_paths(read_json(p), name, hits, max_examples)
        except (ValueError, AttributeError):
            hits.append({"where": name, "value": "unreadable"})
    for sub in ("per_video",):
        d = final_dir / sub
        if d.is_dir():
            for mp in sorted(d.glob("*/manifest.json")):
                files_checked += 1
                try:
                    _scan_for_machine_paths(read_json(mp), f"{sub}/{mp.parent.name}/manifest.json", hits, max_examples)
                except (ValueError, AttributeError):
                    hits.append({"where": str(mp.relative_to(final_dir)), "value": "unreadable"})
                if len(hits) >= max_examples:
                    break
    record_hits = 0
    for name, rows in records.items():
        for r in rows:
            for key in ("frame_path", "clip_path", "media_path", "source_path"):
                val = r.get(key)
                if isinstance(val, str) and _MACHINE_PATH.search(val):
                    record_hits += 1
                    if len(hits) < max_examples:
                        hits.append({"where": f"{name}:{r.get('record_id')}:{key}", "value": val[:160]})
    return _gate("portable_paths", not hits and record_hits == 0, files_checked=files_checked, records_with_machine_paths=record_hits, examples=hits)


def accepted_means_verified_gate(records: dict[str, list[dict[str, Any]]], max_examples: int = 20) -> dict[str, Any]:
    """Every ``accepted`` record carries a policy version, all required checks passed (or are
    explicitly not applicable) and - for clips - a passed boundary check. Human accepts are exempt
    (they are the verification)."""
    bad: list[dict[str, Any]] = []
    n_bad = 0
    n_acc = 0
    for name, rows in records.items():
        for r in rows:
            v = r.get("validation") or {}
            if v.get("status") != ValidationStatus.ACCEPTED.value:
                continue
            n_acc += 1
            if HUMAN_ACCEPT in (v.get("reason_codes") or []) or v.get("semantic_verification") == "human":
                continue
            why: list[str] = []
            if not v.get("policy_version"):
                why.append("policy_version:missing")
            states = normalize_checks(v.get("checks") or {})
            for check in REQUIRED_CHECKS.get(_record_kind(r), ()):
                st = states.get(check)
                if st is None and check in _OPTIONAL_WHEN_ABSENT:
                    continue
                if st is None or st not in ("passed", "not_applicable"):
                    why.append(f"{st or 'missing'}:{check}")
            if record_type_of(r) == "clip" and str(r.get("boundary_check") or "not_run") != "passed":
                why.append(f"boundary_check:{r.get('boundary_check') or 'not_run'}")
            if why:
                n_bad += 1
                if len(bad) < max_examples:
                    bad.append({"type": name, "record_id": r.get("record_id"), "why": why[:6]})
    return _gate("accepted_means_verified", n_bad == 0, accepted=n_acc, unverified_accepted=n_bad, examples=bad)


def checksums_gate(final_dir: Path) -> dict[str, Any]:
    """``checksums.sha256`` exists and every listed dataset file still hashes to it. Files the
    release tooling writes *after* the inventory (this report, media_report.json, uploads.json,
    the review queue) are expected to be unlisted; any other unlisted dataset file is a drift."""
    from video_dataset.dataset.inventory import CHECKSUMS_FILE, verify_checksums

    p = final_dir / CHECKSUMS_FILE
    if not p.exists():
        return _gate("checksums", False, reason="checksums.sha256 missing: run `video-dataset export` (export.inventory) or `video-dataset inventory`")
    res = verify_checksums(final_dir, p)
    unlisted = [u for u in res.get("unlisted", []) if u not in _TRANSIENT_FILES and not u.startswith(("frames/", "clips/", "repairs/", "reviews/")) and u != "lineage.jsonl"]
    ok = bool(res.get("ok")) and not unlisted
    return _gate("checksums", ok, checked=res.get("checked"), mismatched=res.get("mismatched", [])[:20], missing=res.get("missing", [])[:20], unlisted=unlisted[:20], mismatched_count=len(res.get("mismatched", [])), missing_count=len(res.get("missing", [])))


_TRANSIENT_FILES = {REPORT_FILE, "media_report.json", "uploads.json", "review_queue.jsonl", "inventory.json", "checksums.sha256", "review_history.jsonl", "calibration.json", "removed_sources.json", "rights_evidence.jsonl"}


def temporal_gate(records: dict[str, list[dict[str, Any]]], max_examples: int = 20) -> dict[str, Any]:
    """No accepted record has an invalid timestamp (export-time ``temporal_validation``), and
    every record carries the block (not_run only when the source duration was unknown)."""
    bad: list[dict[str, Any]] = []
    n_failed = 0
    n_missing = 0
    n_accepted_failed = 0
    for name, rows in records.items():
        for r in rows:
            tv = r.get("temporal_validation")
            if not isinstance(tv, dict):
                n_missing += 1
                continue
            if tv.get("status") == "failed":
                n_failed += 1
                if (r.get("validation") or {}).get("status") == ValidationStatus.ACCEPTED.value:
                    n_accepted_failed += 1
                    if len(bad) < max_examples:
                        bad.append({"type": name, "record_id": r.get("record_id"), "checks": tv.get("checks"), "error_seconds": tv.get("boundary_error_seconds")})
    return _gate("timestamps", n_accepted_failed == 0 and n_missing == 0, failed=n_failed, accepted_with_failed_timestamps=n_accepted_failed, without_temporal_validation=n_missing, examples=bad)


def rights_gate(config: PipelineConfig, records: dict[str, list[dict[str, Any]]], manifests: list[dict[str, Any]], final_dir: Path, media_shipped: bool) -> dict[str, Any]:
    """Every record carries a ``rights`` block; when media is shipped (``include_media`` /
    ``package --include-media``), every frame / clip of the release belongs to a source whose
    rights status is in ``licensing.release_rights`` (or the media files are not under final/)."""
    release_rights = {str(x) for x in (config.licensing.release_rights or [])}
    without_block = sum(1 for rows in records.values() for r in rows if not isinstance(r.get("rights"), dict))
    by_status: dict[str, int] = {}
    for m in manifests:
        st = str(m.get("rights_status") or "unknown")
        by_status[st] = by_status.get(st, 0) + 1
    violations: list[dict[str, Any]] = []
    if media_shipped:
        shipped_videos = {p.name for sub in ("frames", "clips") if (final_dir / sub).is_dir() for p in (final_dir / sub).iterdir() if p.is_dir()}
        rights_by_video = {str(m.get("video_id")): str(m.get("rights_status") or "unknown") for m in manifests}
        for vid in sorted(shipped_videos):
            if rights_by_video.get(vid, "unknown") not in release_rights:
                violations.append({"video_id": vid, "rights_status": rights_by_video.get(vid, "unknown")})
    enforced = bool(getattr(config.licensing, "media_rights_gate", True))
    ok = without_block == 0 and (not violations or not enforced)
    return _gate("rights", ok, release_rights=sorted(release_rights), videos_by_rights_status=by_status, records_without_rights_block=without_block, media_shipped=media_shipped, media_rights_violations=violations[:20], enforced=enforced)


def reproducibility_gate(final_dir: Path) -> dict[str, Any]:
    """manifest.json records the exact pipeline commit, schema and policy versions and a release id,
    and they agree with the code that runs this check."""
    p = final_dir / "manifest.json"
    if not p.exists():
        return _gate("reproducibility", False, reason="manifest.json missing")
    try:
        m = read_json(p)
    except (ValueError, AttributeError):
        return _gate("reproducibility", False, reason="manifest.json unreadable")
    env = m.get("environment") or {}
    code = env.get("code") or {}
    problems: list[str] = []
    if not m.get("release_id"):
        problems.append("release_id missing")
    if not code.get("git_commit"):
        problems.append("git commit missing")
    if m.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"manifest schema_version {m.get('schema_version')} != {SCHEMA_VERSION}")
    if m.get("policy_version") != POLICY_VERSION:
        problems.append(f"manifest policy_version {m.get('policy_version')} != {POLICY_VERSION}")
    if not env.get("models"):
        problems.append("model identifiers missing")
    return _gate("reproducibility", not problems, problems=problems, release_id=m.get("release_id"), git_commit=code.get("git_commit"), git_dirty=code.get("git_dirty"), schema_version=m.get("schema_version"), policy_version=m.get("policy_version"))


def human_verification_block(final_dir: Path) -> dict[str, Any]:
    """What the human review protocol produced so far (reported, not gated): sample size,
    labelled rows, agreement and unresolved disagreements, from reviews/sample.jsonl and labels."""
    try:
        from video_dataset.dataset.human_review import (
            LABELS_FILE,
            SAMPLE_FILE,
            read_labelled,
            read_sample,
            verification_report,
        )
    except Exception:  # pragma: no cover
        return {"status": "unavailable"}
    sample_path, labels_path = final_dir / SAMPLE_FILE, final_dir / LABELS_FILE
    if not sample_path.exists() and not labels_path.exists():
        return {"status": "not_run", "note": "no stratified sample drawn yet: run `video-dataset review-sample`"}
    sample = read_sample(sample_path) if sample_path.exists() else []
    labelled = read_labelled(labels_path) if labels_path.exists() else []
    if sample_path.exists():
        labelled += read_labelled(sample_path)
    rep = verification_report(sample, labelled)
    rep["status"] = "labelled" if labelled else "sampled"
    return rep


def exclusions_block(records: dict[str, list[dict[str, Any]]], gates: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    """Everything a consumer must know was left out, downgraded or kept verbatim: what makes a
    release ``passed_with_exclusions`` instead of ``passed``."""
    by_gate = {g["gate"]: g for g in gates}
    review = {name: sum(1 for r in rows if (r.get("validation") or {}).get("status") == ValidationStatus.REVIEW.value) for name, rows in records.items() if rows}
    out = {
        "policy_downgraded": sum(1 for rows in records.values() for r in rows if (r.get("validation") or {}).get("reevaluated")),
        "review_records_excluded_from_strict": sum(review.values()),
        "review_by_type": review,
        "missing_media_assets": by_gate.get("media_references", {}).get("missing", 0),
        "schema_unmigrated": by_gate.get("schema", {}).get("unmigrated", 0),
        "removed_sources": len(manifest.get("removed_sources") or []),
        "dropped_rights_records": ((manifest.get("curation") or {}).get("dropped_rights_records") if isinstance(manifest.get("curation"), dict) else None),
        "camera_ambiguous": sum(1 for name in ("scenes", "clips") for r in records.get(name, []) if (r.get("camera") or {}).get("movement_ambiguous")),
        "residual_boundary_failures": [r.get("record_id") for r in records.get("clips", []) if str(r.get("boundary_check")) == "failed"][:50],
        "residual_boundary_failure_count": sum(1 for r in records.get("clips", []) if str(r.get("boundary_check")) == "failed"),
    }
    out["any"] = any(bool(v) for k, v in out.items() if k not in ("review_by_type", "residual_boundary_failures"))
    return out


def _dist(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        k = str(v if v is not None else "unknown")
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def release_metrics(records: dict[str, list[dict[str, Any]]], manifests: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: dict[str, dict[str, int]] = {}
    statuses: dict[str, dict[str, int]] = {}
    for name, rows in records.items():
        statuses[name] = _dist((r.get("validation") or {}).get("status") for r in rows)
        rc: dict[str, int] = {}
        for r in rows:
            for code in (r.get("validation") or {}).get("reason_codes") or []:
                key = code.split(" ")[0][:60]
                rc[key] = rc.get(key, 0) + 1
        reasons[name] = dict(sorted(rc.items(), key=lambda kv: -kv[1])[:30])
    scenes = records.get("scenes", [])
    claims = {"supported": 0, "contradicted": 0, "not_checked": 0, "unknown": 0}
    for r in scenes:
        for c in r.get("claims") or []:
            claims[str(c.get("status"))] = claims.get(str(c.get("status")), 0) + 1
    verification = {name: {dim: _dist(((r.get("validation") or {}).get("verification") or {}).get(dim) for r in rows) for dim in ("temporal", "measurement", "semantic", "human")} for name, rows in records.items() if rows}
    # structural (heuristic / template / measured) versus independent (verifier / human) semantic verification, apart
    semantic_split = {}
    for name, rows in records.items():
        if not rows:
            continue
        sem = [str((r.get("validation") or {}).get("semantic_verification") or "not_run") for r in rows]
        semantic_split[name] = {
            "heuristic": sum(1 for s in sem if s in ("measured", "template")),
            "independent": sum(1 for s in sem if s == "human" or s.startswith("verifier:")),
            "not_run": sum(1 for s in sem if s == "not_run"),
        }
    quality_missing: dict[str, dict[str, Any]] = {}
    for name, rows in records.items():
        if not rows:
            continue
        n = len(rows)
        q = {comp: sum(1 for r in rows if (r.get("quality") or {}).get(comp) is None) for comp in ("grounding", "temporal_accuracy", "description_quality")}
        quality_missing[name] = {"records": n, **{f"{k}_missing": v for k, v in q.items()}, **{f"{k}_missing_pct": round(100.0 * v / n, 1) for k, v in q.items()}, "unscored": sum(1 for r in rows if (r.get("quality") or {}).get("overall") is None)}
    status_pct = {name: {st: round(100.0 * n / max(1, len(records[name])), 1) for st, n in dist.items()} for name, dist in statuses.items() if records.get(name)}
    return {
        "records": {name: len(rows) for name, rows in records.items()},
        "status_by_type": statuses,
        "status_pct_by_type": status_pct,
        "verification_by_type": verification,
        "semantic_verification_split": semantic_split,
        "grounding_methods": {name: _dist((r.get("quality") or {}).get("grounding_method") for r in rows) for name, rows in records.items() if rows},
        "quality_missing_by_type": quality_missing,
        "policy_downgraded": {name: sum(1 for r in rows if (r.get("validation") or {}).get("reevaluated")) for name, rows in records.items() if rows},
        "reason_codes_by_type": reasons,
        "tiers": {name: _dist(r.get("tier") for r in rows) for name, rows in records.items()},
        "subsets": {name: _dist(s for r in rows for s in (r.get("subsets") or [])) for name, rows in records.items()},
        "generation_targets": _dist(r.get("generation_target") for r in scenes + records.get("clips", []) + records.get("sequences", [])),
        "claims": claims,
        "boundary_checks": _dist(r.get("boundary_check") for r in records.get("clips", [])),
        "transcript_scope": _dist(r.get("transcript_scope") for r in records.get("clips", [])),
        "event_precision": _dist(r.get("boundary_precision") for r in records.get("events", [])),
        "rights_status": _dist(m.get("rights_status") for m in manifests),
        "coverage": _dist(f"{k}={str(v).split(':')[0]}" for m in manifests for k, v in (m.get("coverage") or {}).items()),
        "videos": len(manifests),
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
    }


def released_video_count(final_dir: Path) -> int:
    """Videos in the shards already uploaded from this export directory (``uploads.json``)."""
    p = final_dir / "uploads.json"
    if not p.exists():
        return 0
    try:
        shards = read_json(p).get("shards") or []
    except (ValueError, AttributeError):
        return 0
    return sum(len(v) if isinstance(v, list) else int(v or 0) for v in (s.get("videos") for s in shards))


def run_release_check(config: PipelineConfig, final_dir: Path | None = None, *, require_media: bool | None = None, write_report: bool = True, already_released: int | None = None) -> dict[str, Any]:
    """``already_released`` (videos in shards uploaded earlier) counts toward the source-pool
    minimum; None reads it from ``<final>/uploads.json``."""
    final = Path(final_dir) if final_dir else config.export_dir
    if already_released is None:
        already_released = released_video_count(final)
    records = load_final(final)
    if not any(records.values()):
        raise FileNotFoundError(f"no exported records under {final}; run `video-dataset export` first")
    manifest = final / "manifest.json"
    manifests = []
    if manifest.exists():
        try:
            manifests = list(read_json(manifest).get("videos", []))
        except (ValueError, AttributeError):
            manifests = []
    media_required = bool(config.upload.include_media) if require_media is None else bool(require_media)
    manifest_data: dict[str, Any] = {}
    if manifest.exists():
        try:
            manifest_data = read_json(manifest)
        except (ValueError, AttributeError):
            manifest_data = {}
    media_shipped = media_required or (final / "frames").is_dir() or (final / "clips").is_dir()
    gates = [
        schema_gate(records),
        reference_gate(records),
        parquet_gate(final, records),
        counts_gate(final, records),
        checksums_gate(final),
        split_gate(final, records),
        required_splits_gate(config, records),
        source_pool_gate(config, records, manifests, already_released),
        media_gate(records, config.data_dir, final, media_required),
        temporal_gate(records),
        rights_gate(config, records, manifests, final, media_shipped),
        strict_gate(records),
        portable_paths_gate(final, records),
        accepted_means_verified_gate(records),
        reproducibility_gate(final),
    ]
    failed = [g["gate"] for g in gates if g["state"] == "failed"]
    exclusions = exclusions_block(records, gates, manifest_data)
    status = "blocked" if failed else ("passed_with_exclusions" if exclusions["any"] else "passed")
    repairs_summary: dict[str, Any] | None = None
    rp = final / "repairs" / "summary.json"
    if rp.exists():
        try:
            repairs_summary = read_json(rp)
        except (ValueError, AttributeError):
            repairs_summary = None
    report = {
        "final_dir": final.name,  # the report is published with the data: no machine path
        "release_id": manifest_data.get("release_id"),
        "source_revision": manifest_data.get("source_revision"),
        "status": status,  # passed | passed_with_exclusions | blocked (issue doc section 7)
        "ok": not failed,
        "failed_gates": failed,
        "gates": gates,
        "exclusions": exclusions,
        "human_verification": human_verification_block(final),
        "repair_queues": (repairs_summary or {}).get("counts") if repairs_summary else None,
        "metrics": release_metrics(records, manifests),
    }
    if write_report:
        write_json_atomic(final / REPORT_FILE, report)
    log.info("release check of %s: %s%s", final, status.upper(), f" ({', '.join(failed)})" if failed else "")
    return report


def format_report(report: dict[str, Any]) -> str:
    lines = [f"release check: {str(report.get('status') or ('PASSED' if report['ok'] else 'FAILED')).upper()}  ({report['final_dir']}, release {report.get('release_id') or 'n/a'})"]
    for g in report["gates"]:
        extra = {k: v for k, v in g.items() if k not in ("gate", "state", "examples", "leaking", "mismatches") and not isinstance(v, (dict, list))}
        if g["state"] == "failed" and isinstance(g.get("mismatches"), dict) and g["mismatches"]:
            extra["mismatches"] = ",".join(list(g["mismatches"])[:6])
        if g["state"] == "failed" and isinstance(g.get("below_minimum"), dict) and g["below_minimum"]:
            extra["below_minimum"] = ",".join(f"{k}={v['groups']}/{v['minimum']}" for k, v in g["below_minimum"].items())
        lines.append(f"  {g['gate']:<20} {g['state']:<8} " + ", ".join(f"{k}={v}" for k, v in extra.items()))
    m = report.get("metrics") or {}
    lines.append(f"records: {json.dumps(m.get('records'))}")
    lines.append(f"strict subset: {json.dumps({k: v.get('strict', 0) for k, v in (m.get('subsets') or {}).items()})}")
    ex = report.get("exclusions") or {}
    if ex.get("any"):
        lines.append("exclusions: " + ", ".join(f"{k}={v}" for k, v in ex.items() if k not in ("any", "review_by_type", "residual_boundary_failures") and v))
    hv = report.get("human_verification") or {}
    if hv.get("status") not in (None, "not_run", "unavailable"):
        ag = hv.get("agreement") or {}
        lines.append(f"human verification: {hv.get('status')} sample={hv.get('sample_size')} labelled={hv.get('labelled')} agreement={ag.get('percent_agreement')} kappa={ag.get('kappa_binary')} unresolved={hv.get('unresolved_disagreements')}")
    return "\n".join(lines)
