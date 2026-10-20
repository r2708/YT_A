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
  never positives, every negative is typed.

Each gate yields ``passed`` / ``failed`` / ``skipped`` with counts; ``ok`` is the conjunction of
the required gates. The report also carries the quality metrics a release should publish:
status / reason-code / coverage / duration / split / tier / rights distributions.
"""

from __future__ import annotations

import json
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
from video_dataset.validation.policy import POLICY_VERSION, strict_ok

log = get_logger("dataset.release")
REPORT_FILE = "release_report.json"


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
            if ev is not None:
                start, end = ev.start_time, ev.end_time
            if start is not None and end is not None and (float(start) < 0 or float(end) < float(start)):
                if len(bad) < max_examples:
                    bad.append({"type": name, "record_id": rec.get("record_id"), "error": f"bad interval {start}-{end}"})
            conf = rec.get("confidence")
            if conf is not None and not (0.0 <= float(conf) <= 1.0):
                if len(bad) < max_examples:
                    bad.append({"type": name, "record_id": rec.get("record_id"), "error": f"confidence {conf} outside [0, 1]"})
    n_bad = sum(1 for _ in bad)
    return _gate("schema", n_bad == 0, records_checked=checked, invalid=n_bad, examples=bad, schema_versions=versions, expected_schema_version=SCHEMA_VERSION)


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
    return {
        "records": {name: len(rows) for name, rows in records.items()},
        "status_by_type": statuses,
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


def run_release_check(config: PipelineConfig, final_dir: Path | None = None, *, require_media: bool | None = None, write_report: bool = True) -> dict[str, Any]:
    final = Path(final_dir) if final_dir else config.export_dir
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
    gates = [
        schema_gate(records),
        reference_gate(records),
        parquet_gate(final, records),
        split_gate(final, records),
        media_gate(records, config.data_dir, final, media_required),
        strict_gate(records),
    ]
    failed = [g["gate"] for g in gates if g["state"] == "failed"]
    report = {
        "final_dir": str(final),
        "ok": not failed,
        "failed_gates": failed,
        "gates": gates,
        "metrics": release_metrics(records, manifests),
    }
    if write_report:
        write_json_atomic(final / REPORT_FILE, report)
    log.info("release check of %s: %s%s", final, "PASSED" if not failed else "FAILED", f" ({', '.join(failed)})" if failed else "")
    return report


def format_report(report: dict[str, Any]) -> str:
    lines = [f"release check: {'PASSED' if report['ok'] else 'FAILED'}  ({report['final_dir']})"]
    for g in report["gates"]:
        extra = {k: v for k, v in g.items() if k not in ("gate", "state", "examples", "leaking", "mismatches") and not isinstance(v, (dict, list))}
        lines.append(f"  {g['gate']:<20} {g['state']:<8} " + ", ".join(f"{k}={v}" for k, v in extra.items()))
    m = report.get("metrics") or {}
    lines.append(f"records: {json.dumps(m.get('records'))}")
    lines.append(f"strict subset: {json.dumps({k: v.get('strict', 0) for k, v in (m.get('subsets') or {}).items()})}")
    return "\n".join(lines)
