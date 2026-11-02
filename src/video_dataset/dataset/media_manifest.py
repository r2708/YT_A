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


MEDIA_STATUSES = ("available", "missing", "unreadable", "not_shipped")


def asset_id_for(video_id: Any, path: str) -> str:
    """Stable asset id: ``asset_<16 hex>`` from the video id and the record's relative path, so the
    same file gets the same id on every machine and in every re-export."""
    import hashlib

    key = f"{video_id}|{str(path).replace(chr(92), '/')}"
    return "asset_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def build_media_manifest(
    records: dict[str, list[dict[str, Any]]], data_dir: Path, final_dir: Path | None = None, hash_files: bool = False,
    rights: dict[str, str] | None = None, spans: bool = True,
) -> list[dict[str, Any]]:
    """One row per distinct referenced asset (the asset registry): ``asset_id``, ``path``, ``kind``,
    ``video_id``, ``exists`` / ``media_status``, ``size_bytes``, optional ``sha256``, the source
    time span (``start_time`` / ``end_time`` or ``timestamp``), measured ``duration`` / ``width`` /
    ``height`` / ``fps`` when the record carries them, ``rights_status`` of the source, and the
    record types / count that reference it."""
    rows: dict[str, dict[str, Any]] = {}
    rights = rights or {}
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
                    "asset_id": asset_id_for(rec.get("video_id"), str(path)), "path": str(path), "kind": kind[:-1], "video_id": rec.get("video_id"),
                    "exists": bool(exists), "media_status": "available" if exists else "missing",
                    "size_bytes": p.stat().st_size if exists else None, "sha256": None,
                    "start_time": rec.get("start_time") if kind == "clips" else rec.get("timestamp"),
                    "end_time": rec.get("end_time") if kind == "clips" else rec.get("timestamp"),
                    "duration": rec.get("media_duration") if kind == "clips" else None,
                    "width": rec.get("width"), "height": rec.get("height"), "fps": rec.get("fps") if kind == "clips" else None,
                    "rights_status": rights.get(str(rec.get("video_id"))),
                    "records": 0, "record_types": [],
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


def apply_asset_ids(records: dict[str, list[dict[str, Any]]], rows: list[dict[str, Any]]) -> int:
    """Set ``asset_id`` and ``media_status`` on every frame / clip record from the manifest rows
    (in place). Returns the number of records annotated."""
    by_path = {str(r["path"]): r for r in rows}
    n = 0
    for kind, key in _MEDIA_KEYS.items():
        for rec in records.get(kind, []):
            row = by_path.get(str(rec.get(key) or ""))
            if row is None:
                continue
            rec["asset_id"] = row["asset_id"]
            rec["media_status"] = row.get("media_status")
            n += 1
    return n


def duplicate_assets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assets with identical sha256 (content duplicates across paths); empty unless files were hashed."""
    by_hash: dict[str, list[str]] = {}
    for r in rows:
        if r.get("sha256"):
            by_hash.setdefault(str(r["sha256"]), []).append(str(r["path"]))
    return [{"sha256": h, "paths": ps} for h, ps in by_hash.items() if len(ps) > 1]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    missing = [r for r in rows if not r.get("exists")]
    by_kind: dict[str, dict[str, int]] = {}
    for r in rows:
        d = by_kind.setdefault(str(r.get("kind")), {"total": 0, "missing": 0})
        d["total"] += 1
        d["missing"] += int(not r.get("exists"))
    videos_missing = sorted({str(r.get("video_id")) for r in missing})
    dups = duplicate_assets(rows)
    return {
        "assets": total,
        "present": total - len(missing),
        "missing": len(missing),
        "by_kind": by_kind,
        "videos_with_missing_media": len(videos_missing),
        "bytes_present": sum(int(r.get("size_bytes") or 0) for r in rows if r.get("exists")),
        "hashed": sum(1 for r in rows if r.get("sha256")),
        "duplicate_assets": len(dups),
        "duplicate_asset_examples": dups[:10],
        "by_rights_status": _count(str(r.get("rights_status")) for r in rows),
        "ok": not missing,
    }


def _count(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


def write_media_manifest(records: dict[str, list[dict[str, Any]]], final_dir: Path, data_dir: Path, hash_files: bool = False, rights: dict[str, str] | None = None) -> dict[str, Any]:
    rows = build_media_manifest(records, data_dir, final_dir, hash_files, rights=rights)
    apply_asset_ids(records, rows)
    write_jsonl(final_dir / MANIFEST_FILE, rows)
    summary = summarize(rows)
    if summary["missing"]:
        log.warning("media manifest: %d of %d referenced assets are missing on disk (%d videos); records keep their paths", summary["missing"], summary["assets"], summary["videos_with_missing_media"])
    return summary
