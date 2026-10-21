"""Hugging Face dataset source: metadata inspection without downloading records, and a bounded
ingest of the record-type Parquet files plus the per-shard manifests (never media, never
YouTube). Everything downloaded lands under ``<data_root>/raw/<repo>`` and is treated as
immutable from then on."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from dataset_factory.config import FactoryConfig
from dataset_factory.source.local_source import iter_parquet_rows, load_video_meta


def _token(cfg: FactoryConfig) -> str | None:
    return os.environ.get(cfg.source.token_env or "HF_TOKEN") or None


def list_shards(cfg: FactoryConfig) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi(token=_token(cfg))
    prefix = cfg.source.shard_glob.rstrip("*")
    out = []
    for item in api.list_repo_tree(cfg.source.repo_id, repo_type="dataset", revision=cfg.source.revision, recursive=False):
        name = item.path.split("/")[-1]
        if getattr(item, "path", "") and name.startswith(prefix) and not hasattr(item, "size"):
            out.append(item.path)
    return sorted(out)


def inspect_source(cfg: FactoryConfig) -> dict[str, Any]:
    """Measured facts about the live dataset: files, shards, viewer row counts, per-type Parquet
    schema and row counts (from footers), media presence, rights evidence. No record download."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

    token = _token(cfg)
    api = HfApi(token=token)
    fs = HfFileSystem(token=token)
    repo = cfg.source.repo_id
    files = api.list_repo_files(repo, repo_type="dataset", revision=cfg.source.revision)
    shards = sorted({f.split("/")[0] for f in files if f.split("/")[0].startswith(cfg.source.shard_glob.rstrip("*")) and "/" in f})
    media = [f for f in files if any(p in ("frames", "clips") for p in f.split("/")[1:3]) and f.rsplit(".", 1)[-1].lower() in ("mp4", "jpg", "jpeg", "png", "webm", "mkv")]
    outside = sorted(f for f in files if not f.split("/")[0].startswith(cfg.source.shard_glob.rstrip("*")) and f not in ("README.md", ".gitattributes", ".gitignore"))
    viewer: dict[str, int] = {}
    try:
        import requests

        r = requests.get("https://datasets-server.huggingface.co/size", params={"dataset": repo}, headers={"Authorization": f"Bearer {token}"} if token else {}, timeout=60)
        if r.status_code == 200:
            viewer = {str(c["config"]): int(c.get("num_rows") or 0) for c in (r.json().get("size") or {}).get("configs", [])}
    except Exception as exc:
        viewer = {"error": str(exc)[:120]}  # type: ignore[dict-item]
    per_type: dict[str, dict[str, Any]] = {}
    for rtype in cfg.source.record_types:
        rows = 0
        columns: list[str] = []
        per_shard: dict[str, int] = {}
        for sh in shards:
            p = f"{sh}/{rtype}.parquet"
            if p not in files:
                per_shard[sh] = -1
                continue
            with fs.open(f"datasets/{repo}/{p}", "rb") as fh:
                md = pq.read_metadata(fh)
                per_shard[sh] = md.num_rows
                rows += md.num_rows
                if not columns:
                    fh.seek(0)
                    columns = pq.read_schema(fh).names
        per_type[rtype] = {"rows_from_parquet_footers": rows, "per_shard": per_shard, "columns": columns, "viewer_rows": viewer.get(rtype)}
    rights: dict[str, int] = {}
    licenses: dict[str, int] = {}
    videos = 0
    sample_video: dict[str, Any] = {}
    for sh in shards:
        p = f"{sh}/{cfg.source.manifest_file}"
        if p not in files:
            continue
        try:
            m = json.loads(Path(hf_hub_download(repo, p, repo_type="dataset", token=token, revision=cfg.source.revision)).read_text())
        except Exception:
            continue
        for v in m.get("videos") or []:
            videos += 1
            rights[str(v.get("rights_status") or "absent")] = rights.get(str(v.get("rights_status") or "absent"), 0) + 1
            licenses[str(v.get("license_status") or "absent")] = licenses.get(str(v.get("license_status") or "absent"), 0) + 1
            if not sample_video:
                sample_video = {k: v.get(k) for k in ("video_id", "url", "license", "license_status", "rights_status", "channel", "duration")}
    return {
        "repo_id": repo, "revision": cfg.source.revision, "files": len(files), "shards": shards, "media_files_in_repo": len(media),
        "files_outside_shards": len(outside), "files_outside_shards_examples": outside[:10],
        "viewer_rows": viewer, "record_types": per_type, "videos_in_manifests": videos, "rights_status_in_manifests": rights, "license_status_in_manifests": licenses,
        "sample_video_meta": sample_video,
        "media_availability": "none: the repository holds annotation records only; clip_path values are relative to the producer's data directory (source.media_root)" if not media else f"{len(media)} media files under the shards",
    }


def ingest_shards(cfg: FactoryConfig, dest: Path, limit_shards: int | None = None) -> list[Path]:
    """Download ``<shard>/<type>.parquet`` + manifest/provenance for every shard into ``dest``
    (content-addressed, re-runs are no-ops). Returns the local shard directories."""
    from huggingface_hub import snapshot_download

    patterns = []
    for rtype in cfg.source.record_types:
        patterns.append(f"{cfg.source.shard_glob}/{rtype}.parquet")
    patterns += [f"{cfg.source.shard_glob}/{cfg.source.manifest_file}", f"{cfg.source.shard_glob}/{cfg.source.provenance_file}"]
    snapshot_download(repo_id=cfg.source.repo_id, repo_type="dataset", revision=cfg.source.revision, token=_token(cfg), local_dir=str(dest), allow_patterns=patterns)
    shards = sorted(p for p in dest.glob(cfg.source.shard_glob) if p.is_dir())
    return shards[:limit_shards] if limit_shards else shards


def iter_source_rows(cfg: FactoryConfig, shard_dirs: list[Path]) -> Iterator[tuple[str, dict[str, Any], dict[str, Any]]]:
    """(record type, row, per-video meta) streamed shard by shard, batch by batch."""
    for sh in shard_dirs:
        meta = load_video_meta(sh, cfg)
        for rtype in cfg.source.record_types:
            p = sh / f"{rtype}.parquet"
            if not p.exists():
                continue
            for row in iter_parquet_rows(p, cfg.batch_size):
                yield rtype, row, meta.get(str(row.get("video_id")), {})
