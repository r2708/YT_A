"""Re-annotate an already exported shard without re-running the pipeline.

A shard is a directory in the layout the uploader produces: `<type>.jsonl` at the root (or only
`per_video/<video_id>/<type>.jsonl`), `manifest.json`, `statistics.json`. For every shard:

1. records are upgraded to the current schema (renamed fields, record_id / record_type, defaults)
2. hard negatives are rebuilt from the exported records themselves (events give QA negatives,
   scene records give caption negatives) - nothing needs the frames or the working files
3. split / tier / subsets are assigned (`annotate_records`)
4. the shard is rewritten: per-type JSONL + Parquet, dataset.jsonl / dataset.parquet, cinematic/,
   splits.json, schemas.json, statistics.json, manifest.json and the per_video/ copies

Fields that need pixels (aesthetic score, subject motion, camera speed) stay empty for old shards.
Shards can be pulled from the Hub (`download_shard`) and pushed back in place (`push_shard`).
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

from video_dataset.config import PipelineConfig
from video_dataset.dataset.aggregate import _write_dataset_files
from video_dataset.dataset.curation import annotate_records, split_manifest, subset_records
from video_dataset.dataset.export import (
    CINEMATIC_DIR,
    DATASET_FILES,
    SCHEMA_FILE,
    SPLITS_FILE,
    upgrade_records,
    write_jsonl_files,
    write_schema_file,
)
from video_dataset.dataset.negatives import caption_hard_negatives, qa_hard_negatives
from video_dataset.dataset.stats import compute_statistics
from video_dataset.schemas.events import Event
from video_dataset.schemas.qa import QARecord
from video_dataset.schemas.vision import ObjectAnnotation, SceneAnalysis
from video_dataset.utils.io import read_json, read_jsonl, write_json_atomic
from video_dataset.utils.logging import get_logger

log = get_logger("dataset.reannotate")


# ----------------------------------------------------------------------------- loading
def load_shard(shard_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """(records per type, per-video manifests). Root files win; per_video/ is the fallback."""
    records: dict[str, list[dict[str, Any]]] = {name: [] for name in DATASET_FILES}
    root_found = False
    for name in DATASET_FILES:
        f = shard_dir / f"{name}.jsonl"
        if f.exists():
            root_found = True
            records[name].extend(read_jsonl(f))
    manifests: list[dict[str, Any]] = []
    per_video = shard_dir / "per_video"
    if per_video.exists():
        for vdir in sorted(p for p in per_video.iterdir() if p.is_dir()):
            mp = vdir / "manifest.json"
            if mp.exists():
                manifests.append(read_json(mp))
            if not root_found:
                for name in DATASET_FILES:
                    f = vdir / f"{name}.jsonl"
                    if f.exists():
                        records[name].extend(read_jsonl(f))
    if not manifests:
        mp = shard_dir / "manifest.json"
        if mp.exists():
            try:
                manifests = list(read_json(mp).get("videos", []))
            except (ValueError, AttributeError):
                manifests = []
    return records, manifests


# ----------------------------------------------------------------------------- negatives from exports
def _event_from_record(rec: dict[str, Any]) -> Event | None:
    try:
        return Event(
            event_id=rec["event_id"], video_id=rec["video_id"], event_type=rec["event_type"],
            start_time=rec["start_time"], end_time=rec["end_time"], event=rec["event"],
            entities=rec.get("entities") or [], action=rec.get("action"), scene_ids=rec.get("scene_ids") or [],
            frame_ids=rec.get("frame_ids") or [], clip_ids=rec.get("clip_ids") or [], source=rec["source"],
            confidence=rec.get("confidence"), confidence_source=rec.get("confidence_source") or "unavailable",
        )
    except Exception as exc:  # pragma: no cover - malformed legacy row
        log.debug("skipping event %s: %s", rec.get("event_id"), exc)
        return None


def _analysis_from_scene(rec: dict[str, Any]) -> SceneAnalysis | None:
    try:
        objects = []
        for o in rec.get("object_details") or []:
            try:
                objects.append(ObjectAnnotation.model_validate(o))
            except Exception:
                if o.get("name"):
                    objects.append(ObjectAnnotation(name=str(o["name"])))
        if not objects:
            objects = [ObjectAnnotation(name=str(n)) for n in (rec.get("objects") or [])]
        return SceneAnalysis(
            scene_id=rec["scene_id"], video_id=rec["video_id"], start_time=rec["start_time"], end_time=rec["end_time"],
            summary=rec.get("summary"), environment=rec.get("environment") or {}, objects=objects, actions=rec.get("actions") or [],
            camera=rec.get("camera") or {}, visual_style=rec.get("visual_style") or {}, measurements=rec.get("measurements"),
            provider=rec.get("provider"), confidence=rec.get("confidence"), confidence_source=rec.get("confidence_source") or "unavailable",
        )
    except Exception as exc:  # pragma: no cover - malformed legacy row
        log.debug("skipping scene %s: %s", rec.get("scene_id"), exc)
        return None


def rebuild_hard_negatives(records: dict[str, list[dict[str, Any]]], limit: int = 3, seed: int = 1234, only_missing: bool = True) -> dict[str, int]:
    """Attach hard negatives to QA, scene and clip dicts using only what the export already holds.
    Returns how many records of each type received negatives."""
    counts = {"temporal_qa": 0, "long_video_qa": 0, "scenes": 0, "clips": 0}
    if limit <= 0:
        return counts
    events_by_video: dict[str, list[Event]] = {}
    for rec in records.get("events", []):
        e = _event_from_record(rec)
        if e is not None:
            events_by_video.setdefault(e.video_id, []).append(e)
    duration_by_video: dict[str, float] = {}
    for rows in records.values():
        for rec in rows:
            end = rec.get("end_time") or (rec.get("evidence") or {}).get("end_time") or rec.get("timestamp") or 0.0
            vid = str(rec.get("video_id"))
            duration_by_video[vid] = max(duration_by_video.get(vid, 0.0), float(end))
    analyses_by_video: dict[str, dict[str, SceneAnalysis]] = {}
    for rec in records.get("scenes", []):
        a = _analysis_from_scene(rec)
        if a is not None:
            analyses_by_video.setdefault(a.video_id, {})[a.scene_id] = a

    for kind in ("temporal_qa", "long_video_qa"):
        for rec in records.get(kind, []):
            if only_missing and rec.get("hard_negatives"):
                continue
            try:
                q = QARecord.model_validate({k: v for k, v in rec.items() if k != "record_type"})
            except Exception:
                continue
            rng = random.Random(f"{seed}:{q.video_id}:qa:{q.question_id}")
            negs = qa_hard_negatives(q, events_by_video.get(q.video_id, []), duration_by_video.get(q.video_id, q.evidence.end_time), rng, limit)
            rec["hard_negatives"] = [n.model_dump(mode="json") for n in negs]
            counts[kind] += int(bool(negs))
    for kind in ("scenes", "clips"):
        for rec in records.get(kind, []):
            if only_missing and rec.get("hard_negatives"):
                continue
            vid = str(rec.get("video_id"))
            a = analyses_by_video.get(vid, {}).get(str(rec.get("scene_id")))
            if a is None:
                continue
            rng = random.Random(f"{seed}:{vid}:caption:{rec.get('scene_id')}")  # scene and clip of one shot share negatives
            negs = caption_hard_negatives(a, list(analyses_by_video[vid].values()), rng, limit)
            rec["hard_negatives"] = [n.model_dump(mode="json") for n in negs]
            counts[kind] += int(bool(negs))
    return counts


# ----------------------------------------------------------------------------- rewrite
def reannotate_shard(shard_dir: Path, config: PipelineConfig, out_dir: Path | None = None, rebuild_negatives: bool = True) -> dict[str, Any]:
    """Upgrade, re-annotate and rewrite one shard. Writes into ``out_dir`` (default: in place)."""
    shard_dir = Path(shard_dir)
    out = Path(out_dir) if out_dir else shard_dir
    records, manifests = load_shard(shard_dir)
    n_in = sum(len(r) for r in records.values())
    if n_in == 0:
        raise FileNotFoundError(f"no dataset records found in {shard_dir}")
    records = upgrade_records(records)
    neg_counts = rebuild_hard_negatives(records, int(config.export.hard_negatives_per_record), int(config.qa.seed), only_missing=False) if rebuild_negatives and config.export.hard_negatives else {}
    curation = annotate_records(records, config.export)
    splits = split_manifest(records)

    counts, combined_rows, typed_counts, parquet_rows = _write_dataset_files(records, out, config)
    write_schema_file(out / SCHEMA_FILE)
    write_json_atomic(out / SPLITS_FILE, {"salt": config.export.split.salt, "fractions": {"validation": config.export.split.validation_fraction, "test": config.export.split.test_fraction}, "videos": splits, "counts": {k: len(v) for k, v in splits.items()}})
    cinematic_counts: dict[str, int] = {}
    if config.export.cinematic.enabled and config.export.cinematic.write_files:
        cinematic_counts, _c, _t, _p = _write_dataset_files(subset_records(records, "cinematic"), out / CINEMATIC_DIR, config)

    # per-video copies: same records grouped by source video, so the shard stays internally consistent
    by_video: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for name, rows in records.items():
        for rec in rows:
            by_video.setdefault(str(rec.get("video_id")), {n: [] for n in DATASET_FILES})[name].append(rec)
    manifest_by_video = {m.get("video_id"): m for m in manifests if m.get("video_id")}
    new_manifests: list[dict[str, Any]] = []
    for vid, rows in sorted(by_video.items()):
        vdir = out / "per_video" / vid
        if (shard_dir / "per_video" / vid).exists() or out != shard_dir:
            vcounts = write_jsonl_files(rows, vdir)
        else:
            vcounts = {n: len(r) for n, r in rows.items()}
        m = dict(manifest_by_video.get(vid) or {"video_id": vid})
        m.update({"counts": vcounts, "split": next((r.get("split") for rs in rows.values() for r in rs if r.get("split")), None), "reannotated": True})
        if (out / "per_video" / vid).exists():
            write_json_atomic(vdir / "manifest.json", m)
        new_manifests.append(m)

    videos = [{"video_id": m.get("video_id"), "duration": m.get("duration"), "status": "done"} for m in new_manifests]
    stats = compute_statistics(videos, records, [m.get("validation_summary", {}) for m in new_manifests], extra={
        "reannotated": True, "typed_parquet_counts": typed_counts, "parquet_rows": parquet_rows, "combined_jsonl_rows": combined_rows,
        "cinematic": {**curation, "files": cinematic_counts}, "hard_negatives_rebuilt": neg_counts, "jsonl_counts": counts,
    })
    write_json_atomic(out / "statistics.json", stats)
    write_json_atomic(out / "manifest.json", {"videos": new_manifests, "counts": counts, "parquet_rows": parquet_rows, "combined_jsonl_rows": combined_rows, "splits": {k: len(v) for k, v in splits.items()}, "cinematic": cinematic_counts, "reannotated": True})
    summary = {
        "shard": str(shard_dir), "output": str(out), "videos": len(by_video), "records_in": n_in, "records_out": sum(counts.values()),
        "splits": {k: len(v) for k, v in splits.items()}, "tiers": stats.get("records_by_tier", {}).get("scenes", {}),
        "qa_tiers": stats.get("records_by_tier", {}).get("temporal_qa", {}), "hard_negatives": neg_counts, "cinematic": curation,
        "typed_parquet": typed_counts,
    }
    log.info("re-annotated %s: %d video(s), %d records, splits %s, cinematic scenes %d", shard_dir.name, len(by_video), n_in, summary["splits"], curation.get("cinematic_scenes", 0))
    return summary


# ----------------------------------------------------------------------------- Hub
def hub_token(config: PipelineConfig) -> str | None:
    return os.environ.get(config.upload.token_env or "HF_TOKEN") or None


def list_hub_shards(config: PipelineConfig, token: str | None) -> list[str]:
    """Shard folders (`.../shard_NNNN`) present in the dataset repo."""
    from huggingface_hub import HfApi

    cfg = config.upload
    api = HfApi(token=token)
    prefix = cfg.path_in_repo.strip("/") if cfg.path_in_repo else ""
    out: list[str] = []
    for item in api.list_repo_tree(cfg.repo_id, path_in_repo=prefix or None, repo_type=cfg.repo_type, recursive=False):
        name = Path(item.path).name
        if name.startswith("shard_") and item.__class__.__name__ == "RepoFolder":
            out.append(item.path)
    return sorted(out)


def download_shard(config: PipelineConfig, path_in_repo: str, dest: Path, token: str | None) -> Path:
    """Download one shard folder (dataset files + per_video, no media) into ``dest/<shard>``."""
    from huggingface_hub import snapshot_download

    cfg = config.upload
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=cfg.repo_id, repo_type=cfg.repo_type, token=token, local_dir=str(dest),
        allow_patterns=[f"{path_in_repo}/*.jsonl", f"{path_in_repo}/*.parquet", f"{path_in_repo}/*.json", f"{path_in_repo}/per_video/*/*"],
        ignore_patterns=[f"{path_in_repo}/frames/*", f"{path_in_repo}/clips/*"],
    )
    local = dest / path_in_repo
    if not local.exists():
        raise FileNotFoundError(f"{path_in_repo} was not found in {cfg.repo_id}")
    return local


def shard_files(local_dir: Path, ignore_dirs: tuple[str, ...] = ("frames", "clips", ".cache")) -> list[Path]:
    """Dataset files of a shard, root files first, then cinematic/, then per_video/ (media excluded)."""
    files = [p for p in local_dir.rglob("*") if p.is_file() and not any(part in ignore_dirs for part in p.relative_to(local_dir).parts)]

    def order(p: Path) -> tuple[int, str]:
        rel = p.relative_to(local_dir)
        return (0 if len(rel.parts) == 1 else (1 if rel.parts[0] == CINEMATIC_DIR else 2), str(rel))

    return sorted(files, key=order)


def push_shard(
    config: PipelineConfig,
    local_dir: Path,
    path_in_repo: str,
    token: str,
    message: str | None = None,
    files_per_commit: int = 60,
    retries: int = 4,
    api: Any | None = None,
) -> str:
    """Upload the rewritten shard back to the same folder in several small commits (a *write* token is required).

    One commit per ``files_per_commit`` files: a 1 GB shard with 300+ files in a single commit makes
    the Hub's commit endpoint time out (HTTP 408). Each commit is retried with back-off; when a run
    dies half-way, re-running simply re-uploads the same files (uploads are content-addressed, so
    unchanged files cost nothing). Returns the url of the last commit.
    """
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
    from huggingface_hub import CommitOperationAdd

    from video_dataset.utils.retry import retry_call

    cfg = config.upload
    files = shard_files(Path(local_dir))
    if not files:
        raise FileNotFoundError(f"no dataset files to push in {local_dir}")
    base = message or f"Re-annotate {Path(path_in_repo).name}: schema upgrade, split, tiers, hard negatives, Parquet"
    chunks = [files[i : i + max(1, int(files_per_commit))] for i in range(0, len(files), max(1, int(files_per_commit)))]
    last = ""
    for n, chunk in enumerate(chunks, start=1):
        ops = [CommitOperationAdd(path_in_repo=f"{path_in_repo}/{p.relative_to(local_dir).as_posix()}", path_or_fileobj=str(p)) for p in chunk]
        msg = f"{base} ({n}/{len(chunks)})" if len(chunks) > 1 else base
        log.info("pushing %s: commit %d/%d, %d file(s)", path_in_repo, n, len(chunks), len(ops))
        info = retry_call(
            lambda ops=ops, msg=msg: api.create_commit(repo_id=cfg.repo_id, repo_type=cfg.repo_type, operations=ops, commit_message=msg),
            retries=int(retries), base_delay=5.0, max_delay=120.0, label=f"commit {n}/{len(chunks)} of {path_in_repo}",
        )
        last = str(getattr(info, "commit_url", None) or info)
    return last


def update_dataset_card(config: PipelineConfig, token: str) -> str:
    """Refresh the loader `configs:` block of README.md on the Hub (or create the card when there is
    none). The rest of an existing card is kept. Returns the commit URL, or "" when it was current."""
    from huggingface_hub import HfApi

    from video_dataset.dataset.upload import dataset_card, merge_card_configs

    cfg = config.upload
    api = HfApi(token=token)
    if api.file_exists(cfg.repo_id, "README.md", repo_type=cfg.repo_type):
        current = Path(api.hf_hub_download(cfg.repo_id, "README.md", repo_type=cfg.repo_type)).read_text(encoding="utf-8")
        body = merge_card_configs(current, cfg)
        if body == current:
            return ""
    else:
        body = dataset_card(cfg, config)
    info = api.upload_file(
        path_or_fileobj=body.encode("utf-8"), path_in_repo="README.md", repo_id=cfg.repo_id,
        repo_type=cfg.repo_type, commit_message="Update dataset card: loader configs point at the per-type Parquet files",
    )
    return str(getattr(info, "commit_url", None) or info)
