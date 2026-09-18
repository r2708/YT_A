"""Bundle the exported dataset into one zip file (optionally with every referenced frame and clip)."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

from video_dataset.config import PipelineConfig
from video_dataset.dataset.export import COMBINED_FILE, DATASET_FILES
from video_dataset.utils.io import read_jsonl
from video_dataset.utils.logging import get_logger

log = get_logger("dataset.package")

_MEDIA_KEYS = ("frame_path", "clip_path")


def referenced_media(final_dir: Path) -> list[str]:
    """Distinct media paths referenced by the combined dataset (as written, usually relative to data/)."""
    combined = final_dir / COMBINED_FILE
    sources = [combined] if combined.exists() else [final_dir / f"{n}.jsonl" for n in DATASET_FILES if (final_dir / f"{n}.jsonl").exists()]
    seen: set[str] = set()
    for src in sources:
        for rec in read_jsonl(src):
            for key in _MEDIA_KEYS:
                p = rec.get(key)
                if p:
                    seen.add(str(p))
    return sorted(seen)


def package_dataset(config: PipelineConfig, output: Path, include_media: bool = False) -> dict[str, Any]:
    final = config.export_dir
    metadata_files = [final / COMBINED_FILE, final / "dataset.parquet", final / "statistics.json", final / "manifest.json"]
    metadata_files += [final / f"{n}.jsonl" for n in DATASET_FILES]
    present = [f for f in metadata_files if f.exists()]
    if not (final / COMBINED_FILE).exists() and not (final / "dataset.parquet").exists():
        raise FileNotFoundError(f"No exported dataset in {final}. Run `video-dataset export` first.")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    n_media = 0
    missing = 0
    media_bytes = 0
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in present:
            zf.write(f, arcname=f.name)
        if include_media:
            for rel in referenced_media(final):
                p = Path(rel)
                src = p if p.is_absolute() else config.data_dir / p
                if not src.exists():
                    missing += 1
                    continue
                arcname = rel if not p.is_absolute() else f"media/{p.name}"
                zf.write(src, arcname=arcname, compress_type=zipfile.ZIP_STORED)  # JPEG/MP4 are already compressed
                n_media += 1
                media_bytes += src.stat().st_size
    tmp.replace(output)
    size_bytes = output.stat().st_size
    summary: dict[str, Any] = {
        "output": str(output),
        "dataset_files": [f.name for f in present],
        "media_files": n_media,
        "media_missing": missing,
        "media_bytes": media_bytes,
        "size_bytes": size_bytes,
    }
    log.info("packaged %d dataset files + %d media files -> %s (%.1f MB)", len(present), n_media, output, size_bytes / 1e6)
    return summary
