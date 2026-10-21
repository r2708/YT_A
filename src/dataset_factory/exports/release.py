"""Versioned, atomic export directories and the release manifest every target writes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dataset_factory import PIPELINE_VERSION
from dataset_factory.config import FactoryConfig
from dataset_factory.schema.canonical import SCHEMA_VERSION


class ExportConflict(RuntimeError):
    pass


def export_version(cfg: FactoryConfig) -> str:
    return cfg.exports.version or f"{datetime.now(UTC).strftime('%Y%m%d')}-{cfg.config_hash}"


def staging_dir(final: Path) -> Path:
    return final.parent / f".{final.name}.tmp-{os.getpid()}"


def promote(staging: Path, final: Path, overwrite: bool = False) -> Path:
    """Atomically move the staging directory into place; refuse to clobber an existing release."""
    if final.exists():
        if not overwrite:
            raise ExportConflict(f"{final} already exists; pass --overwrite to replace it or set exports.version")
        shutil.rmtree(final)
    staging.rename(final)
    return final


def file_checksums(root: Path, limit: int = 100000) -> dict[str, str]:
    out: dict[str, str] = {}
    for n, p in enumerate(sorted(x for x in root.rglob("*") if x.is_file())):
        if n >= limit:
            break
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        out[p.relative_to(root).as_posix()] = h.hexdigest()
    return out


def write_release_manifest(root: Path, cfg: FactoryConfig, *, target: str, version: str, counts: dict[str, Any], extra: dict[str, Any] | None = None) -> Path:
    checksums = file_checksums(root)
    manifest = {
        "target": target, "schema_version": SCHEMA_VERSION, "dataset_version": version, "pipeline_version": PIPELINE_VERSION,
        "processing_config_hash": cfg.config_hash, "source_dataset": cfg.source.repo_id if cfg.source.kind == "huggingface" else cfg.source.local_dir,
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"), "counts": counts, "files": len(checksums),
        "manifest_checksum": hashlib.sha256(json.dumps(checksums, sort_keys=True).encode()).hexdigest(), "checksums": checksums,
        **(extra or {}),
    }
    p = root / "release_manifest.json"
    p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return p
