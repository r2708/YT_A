"""Image-to-Video export adapter (generic JSONL, version 1): the T2V row plus ``image`` (the
verified starting frame, relative to the export root), its timestamp, dimensions and checksum.
Generic layout, not a verified trainer-native format (see wan_t2v)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from dataset_factory.config import ExportTarget
from dataset_factory.exports.wan_t2v import fits_target, place_asset, t2v_row
from dataset_factory.schema.canonical import CanonicalRecord

ADAPTER_VERSION = "generic-i2v-jsonl-v1"


def write_i2v(records: Iterable[CanonicalRecord], root: Path, target: ExportTarget, resolve: Any, resolve_frame: Any) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    skipped: dict[str, int] = {}
    n = 0
    with (root / target.data_file).open("w", encoding="utf-8") as fh:
        for rec in sorted(records, key=lambda r: r.record_id):
            why = fits_target(rec, target)
            src, frame = resolve(rec), resolve_frame(rec)
            if src is None or not src.exists():
                why.append("asset_unavailable")
            if frame is None or not frame.exists():
                why.append("start_frame_unavailable")
            if why:
                for w in why:
                    skipped[w] = skipped.get(w, 0) + 1
                continue
            stem = rec.record_id.replace(":", "_")
            vrel = place_asset(src, root / target.video_dir / f"{stem}{src.suffix}", target.packaging, root)
            irel = place_asset(frame, root / target.image_dir / f"{stem}{frame.suffix}", target.packaging, root)
            row = t2v_row(rec, vrel, target)
            row.update({"image": irel, "image_timestamp_seconds": rec.start_frame_timestamp_seconds, "image_width": rec.start_frame_width, "image_height": rec.start_frame_height, "image_sha256": rec.start_frame_sha256})
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return {"records": n, "skipped": skipped, "adapter": ADAPTER_VERSION}
