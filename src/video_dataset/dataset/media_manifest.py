"""Referential integrity of the media the records point at.

``final/media_manifest.jsonl`` holds one row per distinct referenced asset: the path as written in
the records, whether it resolves against the data directory, its size, optionally its sha256, the
record types / count that reference it, and the video it belongs to. ``summarize`` reports how
many references are broken so a release can refuse to ship when any *required* media is missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from video_dataset.utils.io import file_sha256, write_jsonl
from video_dataset.utils.logging import get_logger

log = get_logger("dataset.media_manifest")
MANIFEST_FILE = "media_manifest.jsonl"
_MEDIA_KEYS = {"frames": "frame_path", "clips": "clip_path"}


def resolve_media(path: str, data_dir: Path, final_dir: Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    for base in (data_dir, final_dir):
        if base is not None and (base / p).exists():
            return base / p
    return data_dir / p


def build_media_manifest(records: dict[str, list[dict[str, Any]]], data_dir: Path, final_dir: Path | None = None, hash_files: bool = False) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for kind, key in _MEDIA_KEYS.items():
        for rec in records.get(kind, []):
            path = rec.get(key)
            if not path:
                continue
            row = rows.get(str(path))
            if row is None:
                p = resolve_media(str(path), data_dir, final_dir)
                exists = p.exists() and p.is_file()
                row = {
                    "path": str(path), "kind": kind[:-1], "video_id": rec.get("video_id"), "exists": bool(exists),
                    "size_bytes": p.stat().st_size if exists else None, "sha256": None, "records": 0, "record_types": [],
                }
                if hash_files and exists:
                    try:
                        row["sha256"] = file_sha256(p)
                    except OSError:
                        row["sha256"] = None
                rows[str(path)] = row
            row["records"] += 1
            rt = str(rec.get("record_type") or kind[:-1])
            if rt not in row["record_types"]:
                row["record_types"].append(rt)
    return list(rows.values())


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    missing = [r for r in rows if not r.get("exists")]
    by_kind: dict[str, dict[str, int]] = {}
    for r in rows:
        d = by_kind.setdefault(str(r.get("kind")), {"total": 0, "missing": 0})
        d["total"] += 1
        d["missing"] += int(not r.get("exists"))
    videos_missing = sorted({str(r.get("video_id")) for r in missing})
    return {
        "assets": total,
        "present": total - len(missing),
        "missing": len(missing),
        "by_kind": by_kind,
        "videos_with_missing_media": len(videos_missing),
        "bytes_present": sum(int(r.get("size_bytes") or 0) for r in rows if r.get("exists")),
        "ok": not missing,
    }


def write_media_manifest(records: dict[str, list[dict[str, Any]]], final_dir: Path, data_dir: Path, hash_files: bool = False) -> dict[str, Any]:
    rows = build_media_manifest(records, data_dir, final_dir, hash_files)
    write_jsonl(final_dir / MANIFEST_FILE, rows)
    summary = summarize(rows)
    if summary["missing"]:
        log.warning("media manifest: %d of %d referenced assets are missing on disk (%d videos); records keep their paths", summary["missing"], summary["assets"], summary["videos_with_missing_media"])
    return summary
