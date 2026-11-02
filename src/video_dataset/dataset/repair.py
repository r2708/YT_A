"""Repair queues and lineage for a release / shard (issue doc §5 A, §5 C).

``build_repair_queues`` streams the ``<type>.jsonl`` files of a final directory (root level, or
``per_video/*/<type>.jsonl``) and writes one JSONL queue per failure class into
``<final_dir>/repairs/``::

    repairs/missing_media.jsonl          clip records whose clip_path does not exist
    repairs/missing_frames.jsonl         frame records whose frame_path does not exist
    repairs/invalid_timestamps.jsonl     negative start, end < start (QA: evidence interval)
    repairs/broken_evidence.jsonl        QA evidence naming event / scene / frame ids absent from the video
    repairs/schema_failures.jsonl        missing or outdated schema_version
    repairs/duplicate_records.jsonl      a record_id seen more than once in the release
    repairs/low_confidence_camera.jsonl  scene / clip camera confidence missing or below threshold, or ambiguous
    repairs/unverified_descriptions.jsonl video descriptions without a semantic verification pass
    repairs/rights_unverified.jsonl      media-bearing records of sources whose rights_status is unknown or not releasable
    repairs/summary.json                 {"generated_at", "shard", "counts": {queue: n}, "records_scanned", ...}

Every queue row is ``{"record_id", "record_type", "video_id", "shard", "reason", "detail",
"proposed_action"}``; ``reason`` is a stable code. Media existence comes from
``media_manifest.jsonl`` when present, else from resolving paths against ``data_dir``; with
neither, media queues are skipped. Records of sources listed in ``removed_sources.json`` are not
queued (they are to be removed, not repaired). Nothing in the pipeline guesses a repair: camera
movement ``unknown`` is the honest state and is never queued.

Lineage (§5 C) lives in ``<final_dir>/lineage.jsonl``, append-only, one row per change::

    {"record_id", "record_type", "video_id", "change": status|field|media|removed|migrated|repaired,
     "previous", "new", "reason", "method", "actor", "policy_version", "pipeline_version",
     "previous_checksum", "new_checksum", "timestamp"}
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from video_dataset.dataset.export import DATASET_FILES, RECORD_TYPES
from video_dataset.schemas.common import SCHEMA_VERSION
from video_dataset.utils.io import ensure_dir, read_json, read_jsonl, write_json_atomic
from video_dataset.utils.logging import get_logger

log = get_logger("dataset.repair")

REPAIRS_DIR = "repairs"
SUMMARY_FILE = "summary.json"
LINEAGE_FILE = "lineage.jsonl"
QUEUES = (
    "missing_media",
    "missing_frames",
    "invalid_timestamps",
    "broken_evidence",
    "schema_failures",
    "duplicate_records",
    "low_confidence_camera",
    "unverified_descriptions",
    "rights_unverified",
)
ACTIONS = {
    "missing_media": "re-extract from source when rights and source availability permit, else media_status=missing and exclude from media-dependent exports",
    "missing_frames": "re-extract from source when rights and source availability permit, else media_status=missing and exclude from media-dependent exports",
    "invalid_timestamps": "re-evaluate the source segment; reject if it cannot be fixed",
    "broken_evidence": "relink evidence to existing ids or move the QA item to review",
    "schema_failures": "apply migration to the current schema_version and re-validate",
    "duplicate_records": "keep the first occurrence; drop the rest or re-id",
    "low_confidence_camera": "mark camera movement unknown or send to human review; never guess",
    "unverified_descriptions": "human review or verifier pass; keep review status until then",
    "rights_unverified": "keep annotations; block media redistribution until rights evidence is recorded",
}
LINEAGE_CHANGES = ("status", "field", "media", "removed", "migrated", "repaired")
_MEDIA_MANIFEST = "media_manifest.jsonl"
_PROVENANCE = "provenance.jsonl"
_REMOVED_SOURCES = "removed_sources.json"
_ID_FIELDS = {"events": "event_id", "scenes": "scene_id", "frames": "frame_id"}
_EVIDENCE_FIELDS = (("event_ids", "events"), ("scene_ids", "scenes"), ("frame_ids", "frames"))
_QA_TYPES = ("temporal_qa", "long_video_qa")


def iter_type_records(final_dir: Path, name: str) -> Iterator[dict[str, Any]]:
    """Stream ``<name>.jsonl`` from the root, falling back to ``per_video/*/<name>.jsonl``."""
    root_file = Path(final_dir) / f"{name}.jsonl"
    if root_file.exists():
        yield from read_jsonl(root_file)
        return
    per_video = Path(final_dir) / "per_video"
    if not per_video.is_dir():
        return
    for vdir in sorted(p for p in per_video.iterdir() if p.is_dir()):
        f = vdir / f"{name}.jsonl"
        if f.exists():
            yield from read_jsonl(f)


class _QueueWriter:
    """Nine queue files written line by line to ``.tmp`` and renamed atomically on close."""

    def __init__(self, directory: Path) -> None:
        self.directory = ensure_dir(directory)
        self.counts: dict[str, int] = dict.fromkeys(QUEUES, 0)
        self._handles = {q: (self.directory / f"{q}.jsonl.tmp").open("w", encoding="utf-8") for q in QUEUES}

    def add(self, queue: str, rec: dict[str, Any], shard: str, reason: str, detail: str | None = None) -> None:
        row = {
            "record_id": rec.get("record_id"),
            "record_type": rec.get("record_type"),
            "video_id": rec.get("video_id"),
            "shard": shard,
            "reason": reason,
            "detail": detail,
            "proposed_action": ACTIONS[queue],
        }
        self._handles[queue].write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self.counts[queue] += 1

    def close(self, keep: bool = True) -> None:
        for q, fh in self._handles.items():
            fh.close()
            tmp = self.directory / f"{q}.jsonl.tmp"
            if keep:
                os.replace(tmp, self.directory / f"{q}.jsonl")
            elif tmp.exists():
                tmp.unlink()


def _dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _num(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _media_lookup(final_dir: Path, data_dir: Path | None) -> tuple[dict[str, bool] | None, Path | None]:
    manifest = final_dir / _MEDIA_MANIFEST
    if manifest.exists():
        return {str(r.get("path")): bool(r.get("exists")) for r in read_jsonl(manifest) if r.get("path")}, None
    return None, data_dir


def _media_exists(path: str, lookup: dict[str, bool] | None, data_dir: Path | None, final_dir: Path) -> bool | None:
    if lookup is not None:
        return lookup.get(path)
    if data_dir is None:
        return None
    p = Path(path)
    if p.is_absolute():
        return p.is_file()
    return (data_dir / p).is_file() or (final_dir / p).is_file()


def _rights(final_dir: Path) -> dict[str, str | None]:
    f = final_dir / _PROVENANCE
    if not f.exists():
        return {}
    return {str(r.get("video_id")): r.get("rights_status") for r in read_jsonl(f) if r.get("video_id")}


def _removed(final_dir: Path) -> set[str]:
    f = final_dir / _REMOVED_SOURCES
    if not f.exists():
        return set()
    try:
        data = read_json(f)
    except ValueError:
        return set()
    return {str(k) for k in data} if isinstance(data, dict) else set()


def _check_timestamps(rec: dict[str, Any], kind: str) -> tuple[str, str] | None:
    if kind in _QA_TYPES:
        ev = _dict(rec.get("evidence"))
        start, end = _num(ev.get("start_time")), _num(ev.get("end_time"))
    elif kind == "frames":
        start, end = _num(rec.get("timestamp")), None
    else:
        start, end = _num(rec.get("start_time")), _num(rec.get("end_time"))
    if start is not None and start < 0:
        return "negative_start", f"start={start}"
    if start is not None and end is not None and end < start:
        return "end_before_start", f"start={start} end={end}"
    return None


def _check_camera(rec: dict[str, Any], threshold: float) -> tuple[str, str] | None:
    cam = rec.get("camera")
    if not isinstance(cam, dict):
        return None
    if cam.get("movement") == "unknown":
        return None  # the honest state; nothing to repair
    if cam.get("movement_ambiguous") is True:
        return "camera_movement_ambiguous", f"movement={cam.get('movement')}"
    conf = _num(cam.get("confidence"))
    if conf is None:
        return "camera_confidence_missing", f"movement={cam.get('movement')}"
    if conf < threshold:
        return f"camera_confidence_below_{threshold:g}", f"movement={cam.get('movement')} confidence={conf}"
    return None


def _check_description(rec: dict[str, Any]) -> tuple[str, str | None] | None:
    val = _dict(rec.get("validation"))
    sem = val.get("semantic_verification")
    if sem is None or sem == "not_run":
        return "semantic_verification_not_run", f"status={val.get('status')}"
    if val.get("status") == "review":
        return "description_status_review", "; ".join(str(r) for r in (val.get("reason_codes") or [])) or None
    return None


def build_repair_queues(
    final_dir: Path,
    data_dir: Path | None = None,
    *,
    shard: str | None = None,
    camera_confidence_threshold: float = 0.6,
    release_rights: tuple[str, ...] = ("declared_cc_by",),
) -> dict[str, int]:
    """Write ``repairs/<queue>.jsonl`` + ``repairs/summary.json`` under ``final_dir``; return counts."""
    final_dir = Path(final_dir)
    shard_name = shard or final_dir.name
    media_lookup, media_dir = _media_lookup(final_dir, Path(data_dir) if data_dir else None)
    rights = _rights(final_dir)
    removed = _removed(final_dir)

    # Pass 1: ids that QA evidence may point at, per video (small: one string per record).
    known: dict[str, dict[str, set[str]]] = {k: {} for k in _ID_FIELDS}
    for name, field in _ID_FIELDS.items():
        for rec in iter_type_records(final_dir, name):
            vid = str(rec.get("video_id") or "")
            value = rec.get(field)
            if vid and value:
                known[name].setdefault(vid, set()).add(str(value))

    # Pass 2: every type once, routing each record to its queues.
    seen_ids: set[str] = set()
    scanned = 0
    skipped_removed = 0
    media_unknown = 0
    writer = _QueueWriter(final_dir / REPAIRS_DIR)
    try:
        for name in DATASET_FILES:
            kind = RECORD_TYPES.get(name, name)
            for rec in iter_type_records(final_dir, name):
                scanned += 1
                vid = str(rec.get("video_id") or "")
                if vid in removed:
                    skipped_removed += 1
                    continue
                rec.setdefault("record_type", kind)
                rid = rec.get("record_id")
                if rid is not None:
                    if rid in seen_ids:
                        writer.add("duplicate_records", rec, shard_name, "duplicate_record_id", f"type={kind}")
                    else:
                        seen_ids.add(str(rid))
                ver = rec.get("schema_version")
                if not ver:
                    writer.add("schema_failures", rec, shard_name, "schema_version_missing", f"expected={SCHEMA_VERSION}")
                elif str(ver) != SCHEMA_VERSION:
                    writer.add("schema_failures", rec, shard_name, f"schema_version_outdated:{ver}", f"expected={SCHEMA_VERSION}")
                ts = _check_timestamps(rec, name)
                if ts is not None:
                    writer.add("invalid_timestamps", rec, shard_name, ts[0], ts[1])
                if name in ("clips", "frames"):
                    key = "clip_path" if name == "clips" else "frame_path"
                    queue = "missing_media" if name == "clips" else "missing_frames"
                    path = rec.get(key)
                    if path:
                        exists = _media_exists(str(path), media_lookup, media_dir, final_dir)
                        if exists is None:
                            media_unknown += 1
                        elif not exists:
                            writer.add(queue, rec, shard_name, f"{key}_missing", str(path))
                    if vid:
                        status = rights.get(vid)
                        if status is None:
                            writer.add("rights_unverified", rec, shard_name, "rights_status_unknown", f"{key}={path}")
                        elif status not in release_rights:
                            writer.add("rights_unverified", rec, shard_name, f"rights_status_not_releasable:{status}", f"{key}={path}")
                if name in ("scenes", "clips"):
                    cam = _check_camera(rec, camera_confidence_threshold)
                    if cam is not None:
                        writer.add("low_confidence_camera", rec, shard_name, cam[0], cam[1])
                if name in _QA_TYPES:
                    ev = _dict(rec.get("evidence"))
                    for field, parent in _EVIDENCE_FIELDS:
                        have = known[parent].get(vid, set())
                        for ref in ev.get(field) or []:
                            if str(ref) not in have:
                                noun = field[:-4]  # event_ids -> event
                                writer.add("broken_evidence", rec, shard_name, f"evidence_{noun}_not_found:{ref}", f"{field} of {vid}")
                if name == "video_descriptions":
                    desc = _check_description(rec)
                    if desc is not None:
                        writer.add("unverified_descriptions", rec, shard_name, desc[0], desc[1])
    except BaseException:
        writer.close(keep=False)
        raise
    writer.close(keep=True)

    summary = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "shard": shard_name,
        "schema_version": SCHEMA_VERSION,
        "camera_confidence_threshold": camera_confidence_threshold,
        "release_rights": list(release_rights),
        "media_source": "media_manifest" if media_lookup is not None else ("data_dir" if media_dir else "unknown"),
        "records_scanned": scanned,
        "skipped_removed_sources": skipped_removed,
        "media_unknown": media_unknown,
        "counts": dict(writer.counts),
        "total": sum(writer.counts.values()),
    }
    write_json_atomic(final_dir / REPAIRS_DIR / SUMMARY_FILE, summary)
    log.info("repair queues for %s: %s", shard_name, json.dumps(writer.counts, sort_keys=True))
    return dict(writer.counts)


# --------------------------------------------------------------------------- lineage (§5 C)


def lineage_row(
    record_id: str,
    record_type: str,
    video_id: str,
    change: str,
    previous: Any,
    new: Any,
    reason: str,
    method: str,
    actor: str,
    policy_version: str | None = None,
    pipeline_version: str | None = None,
    previous_checksum: str | None = None,
    new_checksum: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    if change not in LINEAGE_CHANGES:
        raise ValueError(f"lineage change must be one of {LINEAGE_CHANGES}, got {change!r}")
    return {
        "record_id": record_id,
        "record_type": record_type,
        "video_id": video_id,
        "change": change,
        "previous": previous,
        "new": new,
        "reason": reason,
        "method": method,
        "actor": actor,
        "policy_version": policy_version,
        "pipeline_version": pipeline_version,
        "previous_checksum": previous_checksum,
        "new_checksum": new_checksum,
        "timestamp": timestamp or datetime.now(UTC).isoformat(timespec="seconds"),
    }


def append_lineage(final_dir: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Append rows (normalised through ``lineage_row``) to ``lineage.jsonl``; returns rows written."""
    path = ensure_dir(Path(final_dir)) / LINEAGE_FILE
    n = 0
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            norm = lineage_row(
                str(row.get("record_id")),
                str(row.get("record_type")),
                str(row.get("video_id")),
                str(row.get("change")),
                row.get("previous"),
                row.get("new"),
                str(row.get("reason") or ""),
                str(row.get("method") or ""),
                str(row.get("actor") or ""),
                row.get("policy_version"),
                row.get("pipeline_version"),
                row.get("previous_checksum"),
                row.get("new_checksum"),
                row.get("timestamp"),
            )
            fh.write(json.dumps(norm, ensure_ascii=False, default=str) + "\n")
            n += 1
        fh.flush()
        os.fsync(fh.fileno())
    return n


def read_lineage(final_dir: Path) -> Iterator[dict[str, Any]]:
    path = Path(final_dir) / LINEAGE_FILE
    if path.exists():
        yield from read_jsonl(path)


def lineage_summary(final_dir: Path) -> dict[str, Any]:
    by_change: dict[str, int] = {}
    by_actor: dict[str, int] = {}
    total = 0
    first: str | None = None
    last: str | None = None
    for row in read_lineage(final_dir):
        total += 1
        c, a = str(row.get("change")), str(row.get("actor"))
        by_change[c] = by_change.get(c, 0) + 1
        by_actor[a] = by_actor.get(a, 0) + 1
        ts = row.get("timestamp")
        if ts:
            first = ts if first is None or ts < first else first
            last = ts if last is None or ts > last else last
    return {"rows": total, "by_change": dict(sorted(by_change.items())), "by_actor": dict(sorted(by_actor.items())), "first": first, "last": last}
