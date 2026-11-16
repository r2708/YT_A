"""Read the existing annotation dataset without mutating it (Change 1).

Source: a local export directory (``data/final``, or a folder of ``shard_*`` directories) or the
Hugging Face repository, from which only the small metadata files are downloaded. Produces

    legacy_inventory.json        every JSONL / Parquet file: size, sha256, rows, schema versions
    legacy_schema_report.json    columns per file
    legacy_source_index.parquet  one row per source video (+ .jsonl copy) linking its records
    legacy_dataset_revision.txt  the exact revision that was read
"""

from __future__ import annotations

import datetime as dt
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from video_dataset.utils.io import ensure_dir, file_sha256, read_jsonl, write_json_atomic, write_jsonl
from video_dataset.utils.logging import get_logger

log = get_logger("v3.legacy")

RECORD_FILES = ("clips", "scenes", "frames", "events", "sequences", "temporal_qa", "long_video_qa", "video_descriptions")
_DERIVED_DIRS = {"strict", "cinematic", "exports", "benchmark", "per_video"}  # subsets / copies of the root files
_HF_PATTERNS = ["provenance.jsonl", "clips.jsonl", "manifest.json", "inventory.json", "splits.json", "schemas.json"]
_SAMPLE_ROWS = 200


def legacy_dir(config: Any) -> Path:
    return ensure_dir(config.v3_dir / "legacy")


def resolve_local_root(config: Any) -> Path:
    leg = config.v3.legacy
    if leg.local_dir:
        p = Path(leg.local_dir)
        return p if p.is_absolute() else (Path.cwd() / p)
    return config.export_dir


def fetch_metadata(config: Any) -> tuple[Path, str, list[dict[str, Any]] | None]:
    """(root holding the metadata files, revision text, remote file listing or None)."""
    leg = config.v3.legacy
    if (leg.source or "local").lower() != "huggingface":
        root = resolve_local_root(config)
        if not root.exists():
            raise FileNotFoundError(f"legacy dataset directory not found: {root} (v3.legacy.local_dir)")
        revision = f"local:{root}"
        manifest = root / "manifest.json"
        if manifest.exists():
            m = json.loads(manifest.read_text(encoding="utf-8"))
            revision += f"\nrelease_id: {m.get('release_id')}\nsource_revision: {m.get('source_revision')}\ngenerated_at: {m.get('generated_at')}"
        return root, revision, None
    import os

    from huggingface_hub import HfApi, snapshot_download

    repo_id = leg.repo_id or config.upload.repo_id
    if not repo_id:
        raise ValueError("v3.legacy.repo_id (or upload.repo_id) is required for v3.legacy.source=huggingface")
    token = os.environ.get(config.v3.repos.token_env) or None
    api = HfApi(token=token)
    info = api.dataset_info(repo_id, revision=leg.revision)
    commit = str(info.sha)
    patterns = [p for name in _HF_PATTERNS for p in (name, f"**/{name}")]
    root = Path(snapshot_download(repo_id, repo_type="dataset", revision=commit, allow_patterns=patterns, local_dir=str(legacy_dir(config) / "snapshot"), token=token))
    remote: list[dict[str, Any]] = []
    for item in api.list_repo_tree(repo_id, repo_type="dataset", revision=commit, recursive=True, expand=True):
        if item.__class__.__name__ != "RepoFile":
            continue
        lfs = getattr(item, "lfs", None)
        sha = getattr(lfs, "sha256", None) if lfs is not None and not isinstance(lfs, dict) else (lfs or {}).get("sha256") if isinstance(lfs, dict) else None
        remote.append({"path": item.path, "size_bytes": getattr(item, "size", None), "sha256": sha})
    return root, f"huggingface:{repo_id}\ncommit: {commit}", remote


def _roots(root: Path) -> list[Path]:
    """Directories that each hold one set of record files: the root itself and / or its shards."""
    out = [root] if (root / "clips.jsonl").exists() or (root / "provenance.jsonl").exists() else []
    for pattern in ("shard_*", "uploaded/shard_*"):
        out.extend(sorted(p for p in root.glob(pattern) if p.is_dir()))
    return out


def _file_entry(path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    rel = str(path.relative_to(root))
    entry: dict[str, Any] = {"path": rel, "size_bytes": path.stat().st_size, "sha256": file_sha256(path), "format": path.suffix.lstrip("."), "rows": None, "schema_versions": {}}
    schema: dict[str, Any] = {"path": rel, "columns": []}
    if path.suffix == ".jsonl":
        rows = 0
        versions: Counter[str] = Counter()
        columns: dict[str, str] = {}
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rows += 1
                if rows <= _SAMPLE_ROWS:
                    rec = json.loads(line)
                    versions[str(rec.get("schema_version"))] += 1
                    for k, v in rec.items():
                        columns.setdefault(k, type(v).__name__)
        entry.update(rows=rows, schema_versions=dict(versions))
        schema.update(columns=[{"name": k, "type": t} for k, t in columns.items()], sampled_rows=min(rows, _SAMPLE_ROWS))
    elif path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(str(path))
            entry["rows"] = pf.metadata.num_rows
            schema["columns"] = [{"name": f.name, "type": str(f.type)} for f in pf.schema_arrow]
        except Exception as exc:
            entry["error"] = f"unreadable parquet: {exc}"
    return entry, schema


def load_records(root: Path, name: str) -> list[dict[str, Any]]:
    """Every record of one type across the root / shards (a later shard replaces an earlier copy)."""
    seen: dict[str, dict[str, Any]] = {}
    for base in _roots(root):
        f = base / f"{name}.jsonl"
        if not f.exists():
            continue
        for rec in read_jsonl(f):
            key = str(rec.get("record_id") or rec.get("video_id") or len(seen))
            seen[key] = rec
    return list(seen.values())


def build_source_index(root: Path) -> list[dict[str, Any]]:
    """One row per source video: URL, licence as declared, legacy split and the ids of its records."""
    rows: dict[str, dict[str, Any]] = {}
    for prov in load_records(root, "provenance"):
        vid = prov.get("video_id")
        if not vid:
            continue
        rows[vid] = {
            "source_video_id": vid, "source_url": prov.get("url"), "youtube_id": prov.get("youtube_id"), "channel": prov.get("channel"),
            "title": prov.get("title"), "duration": prov.get("duration"), "license_status": prov.get("license_status"),
            "legacy_rights_status": prov.get("rights_status"), "source_sha256": prov.get("source_sha256"),
            "source_group_id": prov.get("source_group_id") or vid, "split": None, "clip_ids": [], "counts": {},
        }
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for base in _roots(root):
        manifest = base / "manifest.json"
        if manifest.exists():
            for v in json.loads(manifest.read_text(encoding="utf-8")).get("videos") or []:
                if v.get("video_id") in rows:
                    for k, n in (v.get("counts") or {}).items():
                        counts[v["video_id"]][k] = int(n or 0)
                    rows[v["video_id"]]["split"] = v.get("split") or rows[v["video_id"]]["split"]
    for clip in load_records(root, "clips"):
        vid = clip.get("video_id")
        if vid not in rows:  # a clip without provenance: keep it visible, it can never be approved without a URL
            rows[vid] = {"source_video_id": vid, "source_url": None, "youtube_id": None, "channel": None, "title": None, "duration": None, "license_status": None,
                         "legacy_rights_status": (clip.get("rights") or {}).get("status"), "source_sha256": None, "source_group_id": clip.get("source_group_id") or vid,
                         "split": None, "clip_ids": [], "counts": {}}
        row = rows[vid]
        row["clip_ids"].append(clip.get("record_id"))
        row["split"] = row["split"] or clip.get("split")
        row["source_group_id"] = clip.get("source_group_id") or row["source_group_id"]
    for vid, row in rows.items():
        c = dict(counts.get(vid) or {})
        c["clips"] = len(row["clip_ids"]) or c.get("clips", 0)
        row["counts"] = c
    return [rows[k] for k in sorted(rows)]


def build_inventory(config: Any) -> dict[str, Any]:
    """Freeze + inventory. Writes the four legacy_* files into <work_dir>/legacy and returns a summary."""
    out_dir = legacy_dir(config)
    root, revision, remote = fetch_metadata(config)
    files: list[dict[str, Any]] = []
    schemas: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in (".jsonl", ".parquet")):
        if ".cache" in path.parts:
            continue
        entry, schema = _file_entry(path, root)
        entry["derived_subset"] = bool(_DERIVED_DIRS & set(path.relative_to(root).parts[:-1]))
        files.append(entry)
        schemas.append(schema)
    local_paths = {f["path"] for f in files}
    remote_only = [r for r in (remote or []) if r["path"] not in local_paths and r["path"].endswith((".jsonl", ".parquet"))]
    index = build_source_index(root)
    versions: Counter[str] = Counter()
    for f in files:
        for v, n in (f.get("schema_versions") or {}).items():
            versions[v] += n
    inventory = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "source": (config.v3.legacy.source or "local").lower(),
        "revision": revision.splitlines(),
        "root": str(root),
        "files": files,
        "remote_files_not_downloaded": remote_only,  # listed with the Hub's size / LFS sha256, rows unknown
        "totals": {
            "files": len(files), "bytes": sum(f["size_bytes"] for f in files), "rows": sum(f["rows"] or 0 for f in files if not f["derived_subset"]),
            "source_videos": len(index), "clips": sum(len(r["clip_ids"]) for r in index), "schema_versions_sampled": dict(versions),
        },
    }
    write_json_atomic(out_dir / "legacy_inventory.json", inventory)
    write_json_atomic(out_dir / "legacy_schema_report.json", {"generated_at": inventory["generated_at"], "files": schemas})
    (out_dir / "legacy_dataset_revision.txt").write_text(revision + "\n", encoding="utf-8")
    write_jsonl(out_dir / "legacy_source_index.jsonl", index)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist([{**r, "counts": json.dumps(r["counts"], sort_keys=True)} for r in index]) if index else pa.table({"source_video_id": pa.array([], pa.string())})
        pq.write_table(table, str(out_dir / "legacy_source_index.parquet"))
    except Exception as exc:  # pragma: no cover - pyarrow is a core dependency
        log.warning("legacy_source_index.parquet not written: %s", exc)
    log.info("legacy inventory: %d files, %d source videos, %d clips -> %s", len(files), len(index), inventory["totals"]["clips"], out_dir)
    return {"out_dir": str(out_dir), "root": str(root), **inventory["totals"]}


def load_source_index(config: Any) -> list[dict[str, Any]]:
    path = legacy_dir(config) / "legacy_source_index.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `video-dataset v3 inventory` first")
    return list(read_jsonl(path))


def legacy_root(config: Any) -> Path:
    """Where the inventoried metadata files live (the snapshot for a Hub source)."""
    inv = legacy_dir(config) / "legacy_inventory.json"
    if inv.exists():
        return Path(json.loads(inv.read_text(encoding="utf-8"))["root"])
    return resolve_local_root(config)


def legacy_revision(config: Any) -> str:
    path = legacy_dir(config) / "legacy_dataset_revision.txt"
    return " | ".join(path.read_text(encoding="utf-8").split("\n")).strip(" |") if path.exists() else "unknown"
