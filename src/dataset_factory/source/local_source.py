"""Local source: a ``final/`` export or a downloaded shard directory (``<type>.parquet`` or
``<type>.jsonl`` + ``manifest.json`` + optional ``provenance.jsonl``). Rows are streamed."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from dataset_factory.config import FactoryConfig


def iter_parquet_rows(path: Path, batch_size: int = 64) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def iter_jsonl_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_video_meta(shard_dir: Path, cfg: FactoryConfig) -> dict[str, dict[str, Any]]:
    """video_id -> {url, license, license_status, rights_status, channel, title, ...} from the
    producer's manifest, enriched by provenance.jsonl when present."""
    out: dict[str, dict[str, Any]] = {}
    mp = shard_dir / cfg.source.manifest_file
    if mp.exists():
        try:
            for v in json.loads(mp.read_text(encoding="utf-8")).get("videos") or []:
                if v.get("video_id"):
                    out[str(v["video_id"])] = {k: v.get(k) for k in ("video_id", "url", "youtube_id", "title", "channel", "license", "license_status", "rights_status", "duration", "upload_date", "retrieved_at", "canonical_video_id", "near_duplicate_of")}
        except ValueError:
            pass
    pp = shard_dir / cfg.source.provenance_file
    if pp.exists():
        for row in iter_jsonl_rows(pp):
            vid = str(row.get("video_id") or "")
            if vid:
                out.setdefault(vid, {}).update({k: row.get(k) for k in ("url", "youtube_id", "channel", "title", "license", "license_status", "rights_status", "canonical_video_id", "retrieved_at") if row.get(k) is not None})
    return out


def iter_local_rows(cfg: FactoryConfig, root: Path) -> Iterator[tuple[str, dict[str, Any], dict[str, Any]]]:
    """``root`` is one export directory or a directory of shard directories."""
    dirs = [root] if any((root / f"{t}.parquet").exists() or (root / f"{t}.jsonl").exists() for t in cfg.source.record_types) else sorted(p for p in root.glob(cfg.source.shard_glob) if p.is_dir())
    for d in dirs:
        meta = load_video_meta(d, cfg)
        for rtype in cfg.source.record_types:
            pq_path, jl_path = d / f"{rtype}.parquet", d / f"{rtype}.jsonl"
            rows = iter_parquet_rows(pq_path, cfg.batch_size) if pq_path.exists() else (iter_jsonl_rows(jl_path) if jl_path.exists() else iter(()))
            for row in rows:
                yield rtype, row, meta.get(str(row.get("video_id")), {})
