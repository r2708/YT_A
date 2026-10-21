"""Text-to-Video export adapter (generic JSONL, version 1).

One line per record: ``{"record_id", "video", "caption", "duration_seconds", "width", "height",
"fps", "split", "source_record_id"}`` with ``video`` relative to the export root. This is a
documented, generic layout - it is **not** claimed to be the native format of any particular Wan
trainer; write a sibling adapter once the trainer's loader has been inspected.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from dataset_factory.config import ExportTarget
from dataset_factory.schema.canonical import CanonicalRecord

ADAPTER_VERSION = "generic-t2v-jsonl-v1"


def place_asset(src: Path, dest: Path, packaging: str, root: Path) -> str:
    """Put ``src`` at ``dest`` (hardlink | copy) or leave it where it is (reference); return the
    path to write into the data file, relative to ``root``."""
    if packaging == "reference":
        try:
            return os.path.relpath(src, root)
        except ValueError:
            return str(src)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    if packaging == "hardlink":
        try:
            os.link(src, dest)
        except OSError:
            shutil.copy2(src, dest)
    else:
        shutil.copy2(src, dest)
    return dest.relative_to(root).as_posix()


def fits_target(rec: CanonicalRecord, target: ExportTarget) -> list[str]:
    why: list[str] = []
    d = rec.duration_seconds
    if target.min_duration_seconds is not None and (d is None or d < target.min_duration_seconds):
        why.append(f"duration<{target.min_duration_seconds}")
    if target.max_duration_seconds is not None and (d is None or d > target.max_duration_seconds):
        why.append(f"duration>{target.max_duration_seconds}")
    if target.fps_min is not None and (rec.fps is None or rec.fps < target.fps_min):
        why.append(f"fps<{target.fps_min}")
    r = target.resolution
    if r.mode != "any":
        if not rec.width or not rec.height:
            why.append("dimensions_unknown")
        else:
            if r.mode == "min" and r.width and r.height and (rec.width < r.width or rec.height < r.height):
                why.append(f"resolution<{r.width}x{r.height}")
            if r.width and r.height:
                want = r.width / r.height
                have = rec.width / rec.height
                if abs(have - want) / want > r.aspect_tolerance:
                    why.append(f"aspect_ratio:{have:.3f}!={want:.3f}")
            if r.mode == "exact" and r.width and r.height and (rec.width, rec.height) != (r.width, r.height):
                why.append(f"resolution!={r.width}x{r.height}")
    if rec.split and rec.split not in target.include_splits:
        why.append(f"split_excluded:{rec.split}")
    return why


def t2v_row(rec: CanonicalRecord, video_rel: str, target: ExportTarget) -> dict[str, Any]:
    return {"record_id": rec.record_id, "video": video_rel, "caption": getattr(rec, target.caption_field), "duration_seconds": rec.duration_seconds, "width": rec.width, "height": rec.height, "fps": rec.fps, "split": rec.split, "source_record_id": rec.source_record_id, "canonical_video_id": rec.canonical_video_id}


def write_t2v(records: Iterable[CanonicalRecord], root: Path, target: ExportTarget, resolve: Any) -> dict[str, Any]:
    """``resolve(rec) -> Path | None`` gives the local clip. Returns counts + skipped reasons."""
    root.mkdir(parents=True, exist_ok=True)
    skipped: dict[str, int] = {}
    n = 0
    with (root / target.data_file).open("w", encoding="utf-8") as fh:
        for rec in sorted(records, key=lambda r: r.record_id):
            why = fits_target(rec, target)
            src = resolve(rec)
            if src is None or not src.exists():
                why.append("asset_unavailable")
            if why:
                for w in why:
                    skipped[w] = skipped.get(w, 0) + 1
                continue
            rel = place_asset(src, root / target.video_dir / f"{rec.record_id.replace(':', '_')}{src.suffix}", target.packaging, root)
            fh.write(json.dumps(t2v_row(rec, rel, target), ensure_ascii=False) + "\n")
            n += 1
    return {"records": n, "skipped": skipped, "adapter": ADAPTER_VERSION}
