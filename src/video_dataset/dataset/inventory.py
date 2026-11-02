"""File inventory and checksums for a release / shard directory (issue doc §3.3, §3.4).

Two files are written next to the data, never into it:

* ``inventory.json``::

      {"root": "<dir name>", "generated_at": "...", "schema_version": "2.2.0",
       "files": [{"path": "scenes.jsonl", "size_bytes": 123, "sha256": "...", "extension": ".jsonl",
                  "shard": "shard_0001" | null, "format": "jsonl", "rows": 42, "record_type": "scene",
                  "schema_versions": {"2.2.0": 40, "missing": 2}}, ...],
       "totals": {"files": n, "bytes": n, "rows_by_type": {"scene": 42, ...},
                  "schema_versions": {"2.2.0": n, ...}}}

  Paths are POSIX and relative to the directory being described, so the file is portable.
  ``rows_by_type`` / ``schema_versions`` totals are taken from JSONL files with a record type only
  (the Parquet files mirror them; metadata JSONL such as ``provenance.jsonl`` carries no records).
  They describe the files as they are, so ``per_video/`` copies and ``dataset.jsonl`` count too.

* ``checksums.sha256`` in ``sha256sum`` format (``<hex>  <relative posix path>``, two spaces),
  sorted by path, so ``sha256sum -c checksums.sha256`` works from inside the directory.

Everything streams: JSONL rows are counted line by line, Parquet row counts come from the footer,
hashes are computed in chunks, and hashing runs in a bounded thread pool.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from video_dataset.dataset.export import RECORD_TYPES
from video_dataset.schemas.common import SCHEMA_VERSION
from video_dataset.utils.io import file_sha256, write_json_atomic
from video_dataset.utils.logging import get_logger

log = get_logger("dataset.inventory")

INVENTORY_FILE = "inventory.json"
CHECKSUMS_FILE = "checksums.sha256"
_SKIP = {INVENTORY_FILE, CHECKSUMS_FILE}
_DEFAULT_EXCLUDE_DIRS: tuple[str, ...] = (".cache", "uploaded", "vd-upload-*")
_MEDIA_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".mp4", ".webm", ".mkv", ".mov"}
_FORMATS = {
    ".jsonl": "jsonl", ".parquet": "parquet", ".json": "json", ".jpg": "jpg", ".jpeg": "jpg", ".png": "png", ".mp4": "mp4",
}
_SHARD_RE = re.compile(r"^shard_\d+$")


def _excluded(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in patterns)


def iter_files(
    root: Path, include_media: bool = True, exclude_dirs: Iterable[str] = _DEFAULT_EXCLUDE_DIRS
) -> Iterator[Path]:
    """Every regular file under ``root`` in a deterministic (sorted relative POSIX path) order.
    Skips the inventory files themselves, ``*.tmp`` scratch files, excluded directories and, with
    ``include_media=False``, image / video assets."""
    root = Path(root)
    patterns = tuple(exclude_dirs)
    found: list[tuple[str, Path]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not _excluded(d, patterns))
        for fname in filenames:
            if fname in _SKIP or fname.endswith(".tmp") or fname.startswith("."):
                continue
            p = Path(dirpath) / fname
            if not include_media and p.suffix.lower() in _MEDIA_EXTENSIONS:
                continue
            if not p.is_file():
                continue
            found.append((p.relative_to(root).as_posix(), p))
    found.sort(key=lambda t: t[0])
    for _, p in found:
        yield p


def _shard_of(root: Path, rel_parts: tuple[str, ...]) -> str | None:
    if len(rel_parts) > 1 and _SHARD_RE.match(rel_parts[0]):
        return rel_parts[0]
    if _SHARD_RE.match(root.name):
        return root.name
    return None


def _record_type_of(stem: str) -> str | None:
    if stem == "dataset":
        return "mixed"
    return RECORD_TYPES.get(stem)


def _jsonl_rows_and_versions(path: Path) -> tuple[int, dict[str, int]]:
    rows = 0
    versions: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows += 1
            try:
                rec = json.loads(line)
            except ValueError:
                versions["invalid_json"] = versions.get("invalid_json", 0) + 1
                continue
            ver = rec.get("schema_version") if isinstance(rec, dict) else None
            key = str(ver) if ver else "missing"
            versions[key] = versions.get(key, 0) + 1
    return rows, versions


def _parquet_rows_and_versions(path: Path) -> tuple[int | None, dict[str, int] | None]:
    try:
        import pyarrow.parquet as pq
    except ImportError:  # pragma: no cover - pyarrow is a hard dependency of the exporter
        return None, None
    try:
        meta = pq.read_metadata(path)
    except Exception as exc:
        log.warning("cannot read parquet footer %s: %s", path.name, exc)
        return None, None
    rows = int(meta.num_rows)
    names = set(meta.schema.names)
    if "schema_version" not in names:
        return rows, None
    versions: dict[str, int] = {}
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(columns=["schema_version"], batch_size=65536):
        for value in batch.column(0).to_pylist():
            key = str(value) if value else "missing"
            versions[key] = versions.get(key, 0) + 1
    return rows, versions


def inventory_file(root: Path, path: Path) -> dict[str, Any]:
    """Describe one file relative to ``root``: size, sha256, format, shard, rows, schema versions."""
    root = Path(root)
    path = Path(path)
    rel = path.relative_to(root)
    ext = path.suffix.lower()
    fmt = _FORMATS.get(ext, "other")
    rows: int | None = None
    versions: dict[str, int] | None = None
    record_type: str | None = None
    if fmt == "jsonl":
        rows, versions = _jsonl_rows_and_versions(path)
        record_type = _record_type_of(path.stem)
    elif fmt == "parquet":
        rows, versions = _parquet_rows_and_versions(path)
        record_type = _record_type_of(path.stem)
    return {
        "path": rel.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "extension": ext,
        "shard": _shard_of(root, rel.parts),
        "format": fmt,
        "rows": rows,
        "record_type": record_type,
        "schema_versions": versions,
    }


def build_inventory(root: Path, include_media: bool = True, workers: int = 2) -> dict[str, Any]:
    """Inventory every file under ``root`` (hashing in a bounded thread pool) without reading any
    file fully into memory."""
    root = Path(root)
    files = list(iter_files(root, include_media=include_media))
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        entries = list(pool.map(lambda p: inventory_file(root, p), files))
    rows_by_type: dict[str, int] = {}
    schema_versions: dict[str, int] = {}
    total_bytes = 0
    for e in entries:
        total_bytes += int(e["size_bytes"])
        if e["format"] != "jsonl" or not e["record_type"]:
            continue  # totals describe record files only (metadata JSONL carries no schema_version)
        if e["rows"] is not None:
            rows_by_type[e["record_type"]] = rows_by_type.get(e["record_type"], 0) + int(e["rows"])
        for ver, n in (e["schema_versions"] or {}).items():
            schema_versions[ver] = schema_versions.get(ver, 0) + int(n)
    return {
        "root": root.name,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "schema_version": SCHEMA_VERSION,
        "files": entries,
        "totals": {
            "files": len(entries),
            "bytes": total_bytes,
            "rows_by_type": dict(sorted(rows_by_type.items())),
            "schema_versions": dict(sorted(schema_versions.items())),
        },
    }


def write_checksums(path: Path, entries: Iterable[dict[str, Any]]) -> Path:
    path = Path(path)
    lines = sorted((str(e["path"]), str(e["sha256"])) for e in entries)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rel, digest in lines:
            fh.write(f"{digest}  {rel}\n")
    os.replace(tmp, path)
    return path


def write_inventory(root: Path, include_media: bool = True, workers: int = 2) -> dict[str, Any]:
    """Write ``inventory.json`` and ``checksums.sha256`` into ``root``; returns the inventory."""
    root = Path(root)
    inv = build_inventory(root, include_media=include_media, workers=workers)
    write_json_atomic(root / INVENTORY_FILE, inv)
    write_checksums(root / CHECKSUMS_FILE, inv["files"])
    log.info("inventory: %d files, %d bytes in %s", inv["totals"]["files"], inv["totals"]["bytes"], root.name)
    return inv


def read_checksums(path: Path) -> dict[str, str]:
    """Parse a ``sha256sum``-style file into ``{relative posix path: hex digest}``."""
    out: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            digest, _, rel = line.partition("  ")
            rel = rel.lstrip("*")  # sha256sum binary-mode marker
            if digest and rel:
                out[rel] = digest.strip().lower()
    return out


def verify_checksums(root: Path, checksums_path: Path | None = None) -> dict[str, Any]:
    """Re-hash every listed file. ``unlisted`` are files present under ``root`` (media included)
    that the checksum file does not mention."""
    root = Path(root)
    cpath = Path(checksums_path) if checksums_path else root / CHECKSUMS_FILE
    listed = read_checksums(cpath)
    mismatched: list[str] = []
    missing: list[str] = []
    checked = 0
    for rel in sorted(listed):
        p = root / rel
        if not p.is_file():
            missing.append(rel)
            continue
        checked += 1
        if file_sha256(p) != listed[rel]:
            mismatched.append(rel)
    present = {p.relative_to(root).as_posix() for p in iter_files(root, include_media=True)}
    unlisted = sorted(present - set(listed))
    return {
        "ok": not mismatched and not missing,
        "checked": checked,
        "mismatched": mismatched,
        "missing": missing,
        "unlisted": unlisted,
    }
