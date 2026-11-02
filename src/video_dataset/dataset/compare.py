"""Compare two releases / shards record by record without loading either (issue doc §5 D).

``compare_releases(old_dir, new_dir)`` streams the ``<type>.jsonl`` files of both directories
(root level, or ``per_video/*/<type>.jsonl`` when the root files are absent) and keeps only ids
and small per-record tuples, never the records. The report is a plain dict::

    {"old": "<dir name>", "new": "<dir name>", "ok": bool, "allow_split_changes": bool,
     "tolerance": 0.0, "removed_sources": [...],
     "by_type": {"<record_type>": {"counts": {"old", "new", "delta"},
                                   "videos": {"old", "new", "added", "removed"},
                                   "record_ids": {"added", "removed", "kept"},
                                   "status": {"old": {...}, "new": {...}},
                                   "unexplained_lost": {"count", "examples": [...<=50]},
                                   "duplicate_ids": {"old", "new", "new_examples": [...]},
                                   "temporal_failures": {"old", "new", "new_examples": [...]},
                                   "schema_versions": {"old": {...}, "new": {...}}}},
     "overall": {same keys aggregated over every type},
     "split_changes": [{"video_id", "old", "new"}],
     "missing_media": {"old", "new"},
     "checksums": {"changed": [...], "added": [...], "removed": [...]} | None,
     "summary": ["...human readable lines..."]}

``ok`` is false when a record id disappeared although its video is not in ``removed_sources``
(beyond ``tolerance`` as a fraction of the old count), when the new release has duplicate ids or
temporal failures the old one did not, or when a video moved between splits (unless
``allow_split_changes``). ``removed_sources`` is the union of the argument and the keys of
``new_dir/removed_sources.json`` when that tombstone file exists.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from video_dataset.dataset.export import DATASET_FILES, RECORD_TYPES
from video_dataset.utils.io import read_json, read_jsonl
from video_dataset.utils.logging import get_logger

log = get_logger("dataset.compare")

MAX_EXAMPLES = 50
_MEDIA_MANIFEST = "media_manifest.jsonl"
_CHECKSUMS = "checksums.sha256"
_REMOVED_SOURCES = "removed_sources.json"


def iter_type_records(directory: Path, name: str) -> Iterator[dict[str, Any]]:
    """Stream ``<name>.jsonl`` from the root, falling back to ``per_video/*/<name>.jsonl``."""
    directory = Path(directory)
    root_file = directory / f"{name}.jsonl"
    if root_file.exists():
        yield from read_jsonl(root_file)
        return
    per_video = directory / "per_video"
    if not per_video.is_dir():
        return
    for vdir in sorted(p for p in per_video.iterdir() if p.is_dir()):
        f = vdir / f"{name}.jsonl"
        if f.exists():
            yield from read_jsonl(f)


def _interval(rec: dict[str, Any]) -> tuple[float | None, float | None]:
    ev = rec.get("evidence")
    if isinstance(ev, dict):
        return _num(ev.get("start_time")), _num(ev.get("end_time"))
    if "timestamp" in rec and "start_time" not in rec:
        return _num(rec.get("timestamp")), None
    return _num(rec.get("start_time")), _num(rec.get("end_time"))


def _num(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def temporal_failure(rec: dict[str, Any]) -> bool:
    start, end = _interval(rec)
    if start is not None and start < 0:
        return True
    return start is not None and end is not None and end < start


class _TypeScan:
    """Everything we keep about one record type of one release: ids, videos, counts."""

    __slots__ = ("count", "duplicate_ids", "id_video", "schema_versions", "status", "temporal", "video_splits")

    def __init__(self) -> None:
        self.count = 0
        self.id_video: dict[str, str] = {}
        self.duplicate_ids: list[str] = []
        self.status: dict[str, int] = {}
        self.temporal: list[str] = []
        self.schema_versions: dict[str, int] = {}
        self.video_splits: dict[str, set[str]] = {}

    def add(self, rec: dict[str, Any]) -> None:
        self.count += 1
        rid = str(rec.get("record_id") or f"<no id #{self.count}>")
        vid = str(rec.get("video_id") or "")
        if rid in self.id_video:
            self.duplicate_ids.append(rid)
        else:
            self.id_video[rid] = vid
        val = rec.get("validation")
        status = str((val.get("status") if isinstance(val, dict) else None) or "missing")
        self.status[status] = self.status.get(status, 0) + 1
        ver = rec.get("schema_version")
        key = str(ver) if ver else "missing"
        self.schema_versions[key] = self.schema_versions.get(key, 0) + 1
        if temporal_failure(rec):
            self.temporal.append(rid)
        split = rec.get("split")
        if vid and split:
            self.video_splits.setdefault(vid, set()).add(str(split))

    @property
    def videos(self) -> set[str]:
        return {v for v in self.id_video.values() if v}


def _scan(directory: Path, name: str) -> _TypeScan:
    scan = _TypeScan()
    for rec in iter_type_records(directory, name):
        scan.add(rec)
    return scan


def _missing_media(directory: Path) -> int | None:
    f = Path(directory) / _MEDIA_MANIFEST
    if not f.exists():
        return None
    return sum(1 for row in read_jsonl(f) if not row.get("exists"))


def _checksum_diff(old_dir: Path, new_dir: Path) -> dict[str, Any] | None:
    a, b = Path(old_dir) / _CHECKSUMS, Path(new_dir) / _CHECKSUMS
    if not (a.exists() and b.exists()):
        return None
    from video_dataset.dataset.inventory import read_checksums

    old, new = read_checksums(a), read_checksums(b)
    return {
        "changed": sorted(p for p in old if p in new and old[p] != new[p]),
        "added": sorted(set(new) - set(old)),
        "removed": sorted(set(old) - set(new)),
    }


def _tombstones(new_dir: Path) -> set[str]:
    f = Path(new_dir) / _REMOVED_SOURCES
    if not f.exists():
        return set()
    try:
        data = read_json(f)
    except ValueError:
        return set()
    return {str(k) for k in data} if isinstance(data, dict) else set()


def _dict_delta(old: dict[str, int], new: dict[str, int]) -> dict[str, dict[str, int]]:
    return {"old": dict(sorted(old.items())), "new": dict(sorted(new.items()))}


def _compare_type(old: _TypeScan, new: _TypeScan, removed_sources: set[str], tolerance: float) -> dict[str, Any]:
    old_ids, new_ids = set(old.id_video), set(new.id_video)
    lost = old_ids - new_ids
    unexplained = sorted(rid for rid in lost if old.id_video[rid] not in removed_sources)
    allowed = int(tolerance * old.count) if tolerance > 0 else 0
    new_dups = sorted(set(new.duplicate_ids) - set(old.duplicate_ids))
    old_temporal = set(old.temporal)
    new_temporal = sorted(rid for rid in new.temporal if rid not in old_temporal)
    return {
        "counts": {"old": old.count, "new": new.count, "delta": new.count - old.count},
        "videos": {
            "old": len(old.videos), "new": len(new.videos),
            "added": sorted(new.videos - old.videos), "removed": sorted(old.videos - new.videos),
        },
        "record_ids": {"added": len(new_ids - old_ids), "removed": len(lost), "kept": len(old_ids & new_ids)},
        "status": _dict_delta(old.status, new.status),
        "unexplained_lost": {"count": len(unexplained), "allowed": allowed, "examples": unexplained[:MAX_EXAMPLES]},
        "duplicate_ids": {"old": len(old.duplicate_ids), "new": len(new.duplicate_ids), "new_examples": new_dups[:MAX_EXAMPLES]},
        "temporal_failures": {"old": len(old.temporal), "new": len(new.temporal), "new_examples": new_temporal[:MAX_EXAMPLES]},
        "schema_versions": _dict_delta(old.schema_versions, new.schema_versions),
        "ok": len(unexplained) <= allowed and not new_dups and not new_temporal,
    }


def _split_of(splits: dict[str, set[str]], vid: str) -> str | None:
    s = splits.get(vid)
    if not s:
        return None
    return next(iter(s)) if len(s) == 1 else "mixed:" + "+".join(sorted(s))


def compare_releases(
    old_dir: Path,
    new_dir: Path,
    removed_sources: set[str] | None = None,
    tolerance: float = 0.0,
    *,
    allow_split_changes: bool = False,
) -> dict[str, Any]:
    old_dir, new_dir = Path(old_dir), Path(new_dir)
    removed = set(removed_sources or set()) | _tombstones(new_dir)
    by_type: dict[str, dict[str, Any]] = {}
    overall_old, overall_new = _TypeScan(), _TypeScan()
    old_splits: dict[str, set[str]] = {}
    new_splits: dict[str, set[str]] = {}
    for name in DATASET_FILES:
        kind = RECORD_TYPES.get(name, name)
        old, new = _scan(old_dir, name), _scan(new_dir, name)
        if old.count == 0 and new.count == 0:
            continue
        by_type[kind] = _compare_type(old, new, removed, tolerance)
        for scan, acc, splits in ((old, overall_old, old_splits), (new, overall_new, new_splits)):
            acc.count += scan.count
            for rid, vid in scan.id_video.items():
                if rid in acc.id_video:
                    acc.duplicate_ids.append(rid)
                else:
                    acc.id_video[rid] = vid
            acc.duplicate_ids.extend(scan.duplicate_ids)
            for k, n in scan.status.items():
                acc.status[k] = acc.status.get(k, 0) + n
            for k, n in scan.schema_versions.items():
                acc.schema_versions[k] = acc.schema_versions.get(k, 0) + n
            acc.temporal.extend(scan.temporal)
            for vid, s in scan.video_splits.items():
                splits.setdefault(vid, set()).update(s)
    overall = _compare_type(overall_old, overall_new, removed, tolerance)

    split_changes: list[dict[str, Any]] = []
    for vid in sorted(set(old_splits) & set(new_splits)):
        a, b = _split_of(old_splits, vid), _split_of(new_splits, vid)
        if a != b:
            split_changes.append({"video_id": vid, "old": a, "new": b})

    missing_media = {"old": _missing_media(old_dir), "new": _missing_media(new_dir)}
    checksums = _checksum_diff(old_dir, new_dir)
    ok = all(t["ok"] for t in by_type.values()) and overall["ok"] and (allow_split_changes or not split_changes)

    summary: list[str] = []
    for kind, t in by_type.items():
        c = t["counts"]
        line = f"{kind}: {c['old']} -> {c['new']} ({c['delta']:+d}); videos {t['videos']['old']} -> {t['videos']['new']}"
        if t["unexplained_lost"]["count"]:
            line += f"; UNEXPLAINED LOST {t['unexplained_lost']['count']}"
        if t["duplicate_ids"]["new_examples"]:
            line += f"; NEW DUPLICATES {len(t['duplicate_ids']['new_examples'])}"
        if t["temporal_failures"]["new_examples"]:
            line += f"; NEW TEMPORAL FAILURES {len(t['temporal_failures']['new_examples'])}"
        summary.append(line)
    summary.append(
        f"overall: {overall['counts']['old']} -> {overall['counts']['new']} records, "
        f"{overall['videos']['old']} -> {overall['videos']['new']} videos, "
        f"{overall['unexplained_lost']['count']} unexplained lost ids, removed sources {len(removed)}"
    )
    if split_changes:
        summary.append(f"split changes: {len(split_changes)} video(s) moved between splits" + (" (allowed)" if allow_split_changes else ""))
    if missing_media["old"] is not None or missing_media["new"] is not None:
        summary.append(f"missing media: {missing_media['old']} -> {missing_media['new']}")
    if checksums is not None:
        summary.append(f"checksums: {len(checksums['changed'])} changed, {len(checksums['added'])} added, {len(checksums['removed'])} removed")
    summary.append("OK: no unexplained loss, duplicates, temporal regressions or split moves" if ok else "NOT OK: do not publish")

    return {
        "old": old_dir.name,
        "new": new_dir.name,
        "ok": ok,
        "allow_split_changes": allow_split_changes,
        "tolerance": tolerance,
        "removed_sources": sorted(removed),
        "by_type": by_type,
        "overall": overall,
        "split_changes": split_changes,
        "missing_media": missing_media,
        "checksums": checksums,
        "summary": summary,
    }


def format_comparison(report: dict[str, Any]) -> str:
    lines = [f"compare {report.get('old')} -> {report.get('new')}: {'OK' if report.get('ok') else 'NOT OK'}"]
    for kind, t in (report.get("by_type") or {}).items():
        c, v, ids = t["counts"], t["videos"], t["record_ids"]
        st = t["status"]
        lines.append(
            f"  {kind:<18} {c['old']:>7} -> {c['new']:>7} ({c['delta']:+d}) | videos {v['old']}->{v['new']} "
            f"| ids +{ids['added']} -{ids['removed']} ={ids['kept']} | status old {json.dumps(st['old'], sort_keys=True)} "
            f"new {json.dumps(st['new'], sort_keys=True)} | lost {t['unexplained_lost']['count']} "
            f"dup {t['duplicate_ids']['new']} temporal {t['temporal_failures']['new']}"
        )
    for sc in report.get("split_changes") or []:
        lines.append(f"  split change: {sc['video_id']} {sc['old']} -> {sc['new']}")
    lines.extend(f"  {s}" for s in report.get("summary") or [])
    return "\n".join(lines)
