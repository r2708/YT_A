"""Automatic Hugging Face Hub upload of the exported dataset in size-bounded shards.

The local `final/` directory is the *current shard*. After every aggregation the pipeline measures it
(dataset files, plus the referenced frames and clips when `upload.include_media` is on). Once it reaches
`upload.threshold_mb` the shard is uploaded to `repo_id` under `<path_in_repo>/shard_NNNN/` and then
rotated locally: with `after_upload: archive` (default) the uploaded files move to
`final/uploaded/shard_NNNN/` so nothing is deleted; with `after_upload: delete` they are removed to free
disk. Either way `final/` starts empty and the next videos begin shard NNNN+1. The final dataset on the
Hub therefore only ever grows, and no shard is uploaded twice.

Tokens are read from the environment (`upload.token_env`, default HF_TOKEN) and never logged.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from video_dataset.config import PipelineConfig, UploadConfig
from video_dataset.dataset.export import CINEMATIC_DIR, COMBINED_FILE, DATASET_FILES, SCHEMA_FILE, SPLITS_FILE
from video_dataset.utils.io import read_json, read_jsonl, write_json_atomic, write_jsonl
from video_dataset.utils.logging import get_logger
from video_dataset.utils.retry import retry_call

log = get_logger("dataset.upload")

# Everything at the root of final/ that belongs to the shard. The per-type Parquet files are what the
# dataset card's loader configs point at, so a shard uploaded without them is invisible to loaders.
DATASET_ROOT_FILES = (
    [f"{n}.jsonl" for n in DATASET_FILES]
    + [f"{n}.parquet" for n in DATASET_FILES]
    + [COMBINED_FILE, "dataset.parquet", SCHEMA_FILE, SPLITS_FILE, "statistics.json", "manifest.json"]
)
DATASET_ROOT_DIRS = [CINEMATIC_DIR]  # subset directories with the same layout, shipped as a whole
STATE_FILE = "uploads.json"
UPLOADED_DIR = "uploaded"
AFTER_UPLOAD_MODES = ("archive", "delete")


class UploadError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- state
@dataclass
class UploadState:
    next_shard: int = 1
    shards: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, final_dir: Path) -> UploadState:
        p = final_dir / STATE_FILE
        if not p.exists():
            return cls()
        data = read_json(p)
        return cls(next_shard=int(data.get("next_shard", 1)), shards=list(data.get("shards", [])))

    def save(self, final_dir: Path) -> None:
        write_json_atomic(final_dir / STATE_FILE, {"next_shard": self.next_shard, "shards": self.shards})

    @property
    def uploaded_videos(self) -> set[str]:
        return {v for s in self.shards for v in s.get("videos", [])}


def shard_name(index: int) -> str:
    return f"shard_{index:04d}"


# ----------------------------------------------------------------------------- measuring / staging
def shard_videos(final_dir: Path) -> list[str]:
    manifest = final_dir / "manifest.json"
    if not manifest.exists():
        return []
    try:
        return [m["video_id"] for m in read_json(manifest).get("videos", []) if m.get("video_id")]
    except (ValueError, AttributeError, KeyError):
        return []


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


def media_dirs(data_dir: Path, videos: list[str]) -> list[tuple[Path, str]]:
    """(local dir, path inside the shard) for the frames and clips of the given videos."""
    out: list[tuple[Path, str]] = []
    for vid in videos:
        for sub in ("frames", "clips"):
            d = data_dir / sub / vid
            if d.exists():
                out.append((d, f"{sub}/{vid}"))
    return out


def measure_shard(final_dir: Path, data_dir: Path, include_media: bool) -> tuple[int, int, list[str]]:
    """Returns (bytes, files, video_ids) of the current shard."""
    total = 0
    files = 0
    for name in DATASET_ROOT_FILES:
        f = final_dir / name
        if f.exists():
            total += f.stat().st_size
            files += 1
    for name in DATASET_ROOT_DIRS:
        d = final_dir / name
        if d.is_dir():
            total += _dir_size(d)
            files += sum(1 for p in d.rglob("*") if p.is_file())
    videos = shard_videos(final_dir)
    for vid in videos:
        d = final_dir / "per_video" / vid
        if d.exists():
            total += _dir_size(d)
            files += sum(1 for p in d.rglob("*") if p.is_file())
    if include_media:
        for d, _ in media_dirs(data_dir, videos):
            total += _dir_size(d)
            files += sum(1 for p in d.rglob("*") if p.is_file())
    return total, files, videos


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _link_tree(src_dir: Path, dst_dir: Path) -> int:
    n = 0
    for p in src_dir.rglob("*"):
        if p.is_file():
            _link_or_copy(p, dst_dir / p.relative_to(src_dir))
            n += 1
    return n


def stage_shard(final_dir: Path, data_dir: Path, videos: list[str], include_media: bool, staging: Path) -> int:
    """Hard-link (or copy) everything that belongs to the current shard into `staging`. Returns file count."""
    staging.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in DATASET_ROOT_FILES:
        f = final_dir / name
        if f.exists():
            _link_or_copy(f, staging / name)
            n += 1
    for name in DATASET_ROOT_DIRS:
        d = final_dir / name
        if d.is_dir():
            n += _link_tree(d, staging / name)
    for vid in videos:
        d = final_dir / "per_video" / vid
        if d.exists():
            n += _link_tree(d, staging / "per_video" / vid)
    if include_media:
        for d, rel in media_dirs(data_dir, videos):
            n += _link_tree(d, staging / rel)
        n += write_media_metadata(final_dir, staging, set(videos))
    return n


_CLIP_META = ("record_id", "video_id", "scene_id", "clip_id", "start_time", "end_time", "description", "generation_prompt", "transcript", "split", "tier", "subsets")
_FRAME_META = ("record_id", "video_id", "scene_id", "clip_id", "frame_id", "timestamp", "caption", "split", "tier", "subsets")
_CAMERA_META = ("shot_size", "camera_angle", "camera_movement", "movement_direction", "movement_speed", "lens_type", "focal_length_mm", "depth_of_field")


def write_media_metadata(final_dir: Path, staging: Path, videos: set[str]) -> int:
    """``clips/metadata.jsonl`` and ``frames/metadata.jsonl`` next to the media, in the layout the
    Hub's videofolder / imagefolder loaders expect: one row per file with ``file_name`` relative to
    that folder plus the record's caption / prompt / camera fields. Returns the number of files written."""
    written = 0
    for kind, sub, path_key, keys in (("clips", "clips", "clip_path", _CLIP_META), ("frames", "frames", "frame_path", _FRAME_META)):
        src = final_dir / f"{kind}.jsonl"
        if not src.exists():
            continue
        rows = []
        for rec in read_jsonl(src):
            if str(rec.get("video_id")) not in videos:
                continue
            p = str(rec.get(path_key) or "").replace("\\", "/")
            if not p.startswith(f"{sub}/"):
                continue  # not under data/<sub>/: the loader could not find it anyway
            row: dict[str, Any] = {"file_name": p[len(sub) + 1:]}
            row.update({k: rec.get(k) for k in keys})
            cam = rec.get("camera") if isinstance(rec.get("camera"), dict) else {}
            row.update({k: cam.get(k) for k in _CAMERA_META})
            rows.append(row)
        if rows:
            write_jsonl(staging / sub / "metadata.jsonl", rows)
            written += 1
    return written


# ----------------------------------------------------------------------------- Hub client
class Uploader(Protocol):
    def ensure_repo(self) -> str: ...
    def ensure_card(self, readme: str) -> bool: ...
    def upload_dir(self, local_dir: Path, path_in_repo: str, message: str) -> str: ...


class HuggingFaceUploader:
    def __init__(self, cfg: UploadConfig, token: str):
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:  # pragma: no cover
            raise UploadError("huggingface_hub is not installed: pip install 'video-dataset-pipeline[upload]'") from exc
        if not cfg.repo_id or "/" not in cfg.repo_id:
            raise UploadError("upload.repo_id must be '<user-or-org>/<dataset-name>'")
        self.cfg = cfg
        self.repo_id: str = cfg.repo_id
        self.api = HfApi(token=token)

    def ensure_repo(self) -> str:
        url = self.api.create_repo(self.repo_id, repo_type=self.cfg.repo_type, private=bool(self.cfg.private), exist_ok=True)
        return str(url)

    def ensure_card(self, readme: str) -> bool:
        """Create the dataset card, or refresh the loader `configs:` block of an existing one. Without
        that block the Hub viewer globs every JSONL/Parquet file in the repo into one table, and the
        per-type files (frames, clips, scenes, ...) have different columns -> CastError. Returns True
        when README.md was written."""
        if self.api.file_exists(self.repo_id, "README.md", repo_type=self.cfg.repo_type):
            current = Path(self.api.hf_hub_download(self.repo_id, "README.md", repo_type=self.cfg.repo_type)).read_text(encoding="utf-8")
            merged = merge_card_configs(current, self.cfg)
            if merged == current:
                return False
            body, message = merged, "Update dataset card: loader configs point at the per-type Parquet files"
        else:
            body, message = readme, "Add dataset card"
        self.api.upload_file(
            path_or_fileobj=body.encode("utf-8"), path_in_repo="README.md", repo_id=self.repo_id,
            repo_type=self.cfg.repo_type, commit_message=message,
        )
        return True

    def upload_dir(self, local_dir: Path, path_in_repo: str, message: str) -> str:
        info = self.api.upload_folder(
            repo_id=self.repo_id, folder_path=str(local_dir), path_in_repo=path_in_repo,
            repo_type=self.cfg.repo_type, commit_message=message,
        )
        return str(getattr(info, "commit_url", None) or info)


def card_configs(cfg: UploadConfig) -> str:
    """The `configs:` entries of the dataset card: one loader config per record type, reading the
    typed Parquet file of every shard. Parquet carries an explicit schema, so nullable columns load
    the same way in every shard; JSON type inference fails on columns that are null in one shard and
    filled in another. `shard_*/` (single star) matches the shard roots only, not per_video/ or
    cinematic/, so each config holds exactly one record type."""
    prefix = (cfg.path_in_repo.strip("/") + "/") if cfg.path_in_repo else ""
    return "\n".join(f"- config_name: {n}\n  data_files: \"{prefix}shard_*/{n}.parquet\"" for n in DATASET_FILES)


_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---[ \t]*\n?", re.S)
_CONFIGS_BLOCK = re.compile(r"^configs:[ \t]*\n(?:[ \t-].*\n?)*", re.M)


def merge_card_configs(readme: str, cfg: UploadConfig) -> str:
    """Return `readme` with its YAML `configs:` block replaced by (or extended with) the current loader
    configs; everything else in the card is preserved. A card without front matter gets one."""
    configs = "configs:\n" + card_configs(cfg) + "\n"
    m = _FRONT_MATTER.match(readme)
    if not m:
        return f"---\n{configs}---\n\n{readme.lstrip()}"
    fm = _CONFIGS_BLOCK.sub("", m.group(1) + "\n").rstrip("\n")
    fm = (fm + "\n" if fm else "") + configs
    return f"---\n{fm}---\n{readme[m.end():]}"


def dataset_card(cfg: UploadConfig, config: PipelineConfig) -> str:
    name = (cfg.repo_id or "dataset").split("/")[-1]
    return f"""---
pretty_name: {name}
configs:
{card_configs(cfg)}
---

# {name}

Temporally grounded multimodal video dataset produced by
[video-dataset-pipeline](https://github.com/) (vision provider: `{config.vision.provider}`,
ASR: `{config.transcription.provider}`, OCR: `{config.ocr.provider}`).

The dataset is uploaded in size-bounded shards. Every `shard_NNNN/` directory holds the same files:

| file | content |
|---|---|
| `frames.parquet` / `.jsonl` | one record per sampled frame (caption, measurements, OCR text) |
| `clips.parquet` / `.jsonl` | one record per scene clip (+ subject motion, hard-negative captions) |
| `scenes.parquet` / `.jsonl` | scene-level analysis (objects with position/scale, camera + speed, subject motion, aesthetic score) |
| `events.parquet` / `.jsonl` | timeline events with interval relations |
| `temporal_qa`, `long_video_qa` (`.parquet` / `.jsonl`) | evidence-linked temporal question/answer pairs with rule-built hard negatives |
| `video_descriptions.parquet` / `.jsonl` | whole-video descriptions |
| `dataset.jsonl`, `dataset.parquet` | every record of the shard (tagged with `record_type`) |
| `cinematic/` | the cinematic subset in the same layout |
| `splits.json`, `schemas.json` | train/validation/test video ids; JSON Schema of every record type |
| `statistics.json`, `manifest.json` | shard statistics and the list of source videos |

Every record carries `split` (train / validation / test, assigned per source video so no video
leaks across splits), `tier` (gold / silver / bronze with `tier_reasons`) and `subsets`. Filter on
them rather than re-splitting by record.
| `per_video/<video_id>/` | the same records grouped by source video |
{"| `frames/`, `clips/` | the JPEG frames and MP4 clips the records reference (paths are relative to the shard root); each folder has a `metadata.jsonl` (`file_name` + caption / prompt / camera) for the imagefolder / videofolder loaders |" if cfg.include_media else "| *(media)* | frames and clips are not included; `frame_path` / `clip_path` are relative to the producer's data directory |"}

Records are linked top-down: a video has clips (`clip_id`), a clip lists its sampled frames
(`frame_ids`) and every frame names its clip (`clip_id`). Scene and clip records carry both the raw
analysis text (`summary` / `description`) and a `generation_prompt` built from the structured
fields only. Camera annotations use one vocabulary for every provider: `camera_movement`
(static | pan | tilt | dolly | tracking | orbit | crane | zoom | handheld | fpv | drone | complex) with
`movement_direction` and measured `movement_speed`, plus `shot_size`, `camera_height`,
`camera_distance`, `stabilization`; `lens_type`, `focal_length_mm`, `depth_of_field` and
`focus_type` are VLM estimates (`lens_confidence_source`) and stay null for measurement-only providers.

Only public video metadata (title, channel, duration, license) is stored. Make sure you have the
right to redistribute the source videos' derived data before sharing this dataset.
"""


# ----------------------------------------------------------------------------- orchestration
def _rotate(final_dir: Path, data_dir: Path, videos: list[str], include_media: bool, mode: str, index: int) -> tuple[int, int]:
    """Move (archive) or delete the uploaded shard so final/ starts empty. Returns (paths moved/removed, bytes)."""
    archive = final_dir / UPLOADED_DIR / shard_name(index)
    moved = 0
    size = 0
    targets: list[tuple[Path, Path]] = []
    for name in DATASET_ROOT_FILES:
        f = final_dir / name
        if f.exists():
            targets.append((f, archive / name))
    for name in DATASET_ROOT_DIRS:
        d = final_dir / name
        if d.is_dir():
            targets.append((d, archive / name))
    for vid in videos:
        d = final_dir / "per_video" / vid
        if d.exists():
            targets.append((d, archive / "per_video" / vid))
    if mode == "delete" and include_media:
        for d, rel in media_dirs(data_dir, videos):
            targets.append((d, archive / rel))
    for src, dst in targets:
        size += _dir_size(src) if src.is_dir() else src.stat().st_size
        if mode == "archive":
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        elif src.is_dir():
            shutil.rmtree(src)
        else:
            src.unlink()
        moved += 1
    return moved, size


def check_and_upload(
    config: PipelineConfig,
    *,
    force: bool = False,
    dry_run: bool = False,
    uploader: Uploader | None = None,
) -> dict[str, Any] | None:
    """Upload the current shard when it has reached the threshold (or `force`). Returns a summary
    dict, or None when nothing was (or would be) uploaded. Never raises for a disabled/unconfigured
    uploader; raises UploadError for a misconfiguration when uploading was actually attempted."""
    cfg = config.upload
    provider = (cfg.provider or "none").lower()
    if provider == "none":
        return None
    if provider != "huggingface":
        raise UploadError(f"unknown upload.provider '{cfg.provider}' (none | huggingface)")
    mode = (cfg.after_upload or "archive").lower()
    if mode not in AFTER_UPLOAD_MODES:
        raise UploadError(f"upload.after_upload must be one of {AFTER_UPLOAD_MODES}")
    if not cfg.repo_id or "/" not in cfg.repo_id:
        raise UploadError("upload.repo_id must be '<user-or-org>/<dataset-name>'")

    final_dir = config.export_dir
    data_dir = config.data_dir
    for stale in final_dir.glob("vd-upload-*") if final_dir.exists() else []:
        if stale.is_dir():  # staging left behind by a process killed mid-upload (hard links: no data lost)
            shutil.rmtree(stale, ignore_errors=True)
            log.info("removed stale upload staging dir %s", stale.name)
    size, files, videos = measure_shard(final_dir, data_dir, bool(cfg.include_media))
    threshold = float(cfg.threshold_mb) * 1024 * 1024
    if files == 0 or not videos:
        log.debug("upload: nothing exported yet")
        return None
    if size < threshold and not force:
        log.info("upload: current shard is %.1f MB of %.0f MB, %d video(s) - waiting", size / 1e6, threshold / 1e6, len(videos))
        return None

    state = UploadState.load(final_dir)
    index = state.next_shard
    prefix = cfg.path_in_repo.strip("/") if cfg.path_in_repo else ""
    path_in_repo = f"{prefix}/{shard_name(index)}" if prefix else shard_name(index)
    summary: dict[str, Any] = {
        "shard": shard_name(index), "repo_id": cfg.repo_id, "path_in_repo": path_in_repo, "bytes": size, "files": files,
        "videos": videos, "include_media": bool(cfg.include_media), "after_upload": mode, "dry_run": dry_run,
    }
    if dry_run:
        return summary

    if uploader is None:
        token = os.environ.get(cfg.token_env or "HF_TOKEN")
        if not token:
            raise UploadError(f"no Hugging Face token: set {cfg.token_env or 'HF_TOKEN'} (a *write* token) in the environment or .env")
        uploader = HuggingFaceUploader(cfg, token)

    staging = Path(tempfile.mkdtemp(prefix=f"vd-upload-{shard_name(index)}-", dir=str(final_dir)))
    t0 = time.time()
    try:
        staged = stage_shard(final_dir, data_dir, videos, bool(cfg.include_media), staging)
        log.info("uploading %s: %d file(s), %.1f MB, %d video(s) -> %s/%s", shard_name(index), staged, size / 1e6, len(videos), cfg.repo_id, path_in_repo)
        repo_url = retry_call(uploader.ensure_repo, retries=2, base_delay=2.0, label="create repo")
        card_added = retry_call(lambda: uploader.ensure_card(dataset_card(cfg, config)), retries=2, base_delay=2.0, label="dataset card")
        message = f"Add {shard_name(index)}: {len(videos)} video(s), {size / 1e6:.1f} MB"
        commit = retry_call(lambda: uploader.upload_dir(staging, path_in_repo, message), retries=int(cfg.retries), base_delay=5.0, max_delay=120.0, label="upload shard")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    elapsed = time.time() - t0

    moved, rotated_bytes = _rotate(final_dir, data_dir, videos, bool(cfg.include_media), mode, index)
    state.shards.append({
        "index": index, "name": shard_name(index), "path_in_repo": path_in_repo, "repo_id": cfg.repo_id, "videos": videos,
        "bytes": size, "files": staged, "commit": commit, "uploaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seconds": round(elapsed, 1), "after_upload": mode,
    })
    state.next_shard = index + 1
    state.save(final_dir)
    summary.update({"repo_url": repo_url, "commit": commit, "card_added": card_added, "seconds": round(elapsed, 1), "rotated_paths": moved, "rotated_bytes": rotated_bytes})
    log.info("uploaded %s in %.0fs (%s); local files %s", shard_name(index), elapsed, commit, "archived under final/uploaded" if mode == "archive" else "deleted")
    return summary
