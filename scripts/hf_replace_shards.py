#!/usr/bin/env python
"""Replace every shard on the Hugging Face Hub with its re-annotated version (current schema).

    .venv/bin/python scripts/hf_replace_shards.py --dry-run        # list shards + stale root files, change nothing
    .venv/bin/python scripts/hf_replace_shards.py --no-push        # download + re-annotate + gate only
    .venv/bin/python scripts/hf_replace_shards.py                  # the real thing
    .venv/bin/python scripts/hf_replace_shards.py --only shard_0002,shard_0003
    .venv/bin/python scripts/hf_replace_shards.py --clean-root     # also delete stale files at the repo root

What happens, in order:

1. every shard folder (``shard_NNNN``) is downloaded (dataset files + per_video, no media);
2. each one is upgraded and re-annotated into ``<work-dir>/rewritten/<shard>`` (current schema,
   split / tier / subsets / source groups / strict subset / hard negatives / typed Parquet);
3. the release gates run on every rewritten shard (schema + timestamps, cross-record references,
   JSONL == Parquet ids, reconciled counts, non-empty validation / test splits, source pool,
   portable paths, accepted == verified, strict subset) and the Parquet schema of every record
   type is compared across shards - **nothing is pushed unless every shard passes**, so the Hub
   never ends up with mixed schemas. Re-annotation applies the acceptance policy at export
   (``export.enforce_policy``): legacy ``accepted`` records without check states or without a
   passed clip boundary check become ``review`` (bronze) with the missing checks named - the
   only way back to ``accepted`` is reprocessing the video;
4. each shard is pushed back to its folder in small commits; files the old shard had that the new
   one no longer produces are deleted in a final commit (media under ``frames/`` / ``clips/`` is
   kept unless ``--delete-media``), so the folder really is replaced, not merged;
5. with ``--clean-root`` the stale non-shard files at the repository root (old ``*.jsonl`` /
   ``*.parquet`` / ``per_video`` / ``cinematic`` / ``manifest.json`` ...) are removed; ``README.md``
   and ``.gitattributes`` are never touched;
6. the dataset card's loader ``configs:`` block is refreshed.

Afterwards run ``scripts/hf_verify_counts.py``: it reconciles every shard's Parquet row counts
with its manifest / statistics and with what the viewer reports, and lists leftover root files.

HF_TOKEN (a *write* token) is read from the environment or ``.env``. The script refuses to run
beside a live ``video-dataset run`` (``data/run.lock``): re-annotation loads whole shards into
memory. Re-running after a failure is safe: downloads and uploads are content-addressed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

KEEP_ROOT = {"README.md", ".gitattributes", ".gitignore"}


def _overrides(items: list[str]) -> dict[str, str]:
    return dict(x.split("=", 1) for x in items)


def _live_run_pid(data_dir: Path) -> int | None:
    lock = data_dir / "run.lock"
    if not lock.exists():
        return None
    try:
        pid = int(lock.read_text().strip().split()[0])
    except (ValueError, IndexError):
        return None
    from video_dataset.utils.runlock import pid_alive

    return pid if pid != os.getpid() and pid_alive(pid) else None


def _parquet_schemas(shard_dir: Path) -> dict[str, str]:
    import pyarrow.parquet as pq

    from video_dataset.dataset.export import DATASET_FILES

    out: dict[str, str] = {}
    for name in DATASET_FILES:
        p = shard_dir / f"{name}.parquet"
        if p.exists():
            out[name] = pq.read_schema(p).to_string()
    return out


def _push_replace(api, cfg, local_dir: Path, path_in_repo: str, *, delete_media: bool, files_per_commit: int = 60, retries: int = 4) -> tuple[str, int]:  # type: ignore[no-untyped-def]
    """Upload the rewritten shard and delete every old file of that folder the new shard no longer
    has. Returns (last commit url, deleted files)."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete

    from video_dataset.dataset.reannotate import shard_files
    from video_dataset.utils.retry import retry_call

    files = shard_files(local_dir)
    if not files:
        raise FileNotFoundError(f"no dataset files to push in {local_dir}")
    new_paths = {f"{path_in_repo}/{p.relative_to(local_dir).as_posix()}" for p in files}
    existing = [f for f in api.list_repo_files(cfg.repo_id, repo_type=cfg.repo_type) if f.startswith(path_in_repo + "/")]
    stale = [f for f in existing if f not in new_paths and (delete_media or not any(part in ("frames", "clips") for part in f[len(path_in_repo) + 1 :].split("/")))]
    chunks = [files[i : i + files_per_commit] for i in range(0, len(files), files_per_commit)]
    from video_dataset.schemas.common import SCHEMA_VERSION

    base = f"Replace {Path(path_in_repo).name}: schema {SCHEMA_VERSION} re-annotation (policy enforced at export, source groups, strict subset, typed Parquet)"
    last = ""
    for n, chunk in enumerate(chunks, start=1):
        ops = [CommitOperationAdd(path_in_repo=f"{path_in_repo}/{p.relative_to(local_dir).as_posix()}", path_or_fileobj=str(p)) for p in chunk]
        msg = f"{base} ({n}/{len(chunks) + (1 if stale else 0)})"
        print(f"  pushing {path_in_repo}: commit {n}/{len(chunks) + (1 if stale else 0)}, {len(ops)} file(s)", flush=True)
        info = retry_call(lambda ops=ops, msg=msg: api.create_commit(repo_id=cfg.repo_id, repo_type=cfg.repo_type, operations=ops, commit_message=msg), retries=retries, base_delay=5.0, max_delay=120.0, label=f"commit {n} of {path_in_repo}")
        last = str(getattr(info, "commit_url", None) or info)
    if stale:
        print(f"  deleting {len(stale)} stale file(s) of {path_in_repo} the new shard does not produce", flush=True)
        ops = [CommitOperationDelete(path_in_repo=f) for f in stale]
        info = retry_call(lambda: api.create_commit(repo_id=cfg.repo_id, repo_type=cfg.repo_type, operations=ops, commit_message=f"{base}: remove {len(stale)} superseded file(s)"), retries=retries, base_delay=5.0, max_delay=120.0, label=f"delete stale files of {path_in_repo}")
        last = str(getattr(info, "commit_url", None) or info)
    return last, len(stale)


def _stale_root_files(api, cfg) -> list[str]:  # type: ignore[no-untyped-def]
    """Files outside every shard_*/ folder (and outside upload.path_in_repo when set) that are not the card."""
    prefix = cfg.path_in_repo.strip("/") + "/" if cfg.path_in_repo else ""
    out = []
    for f in api.list_repo_files(cfg.repo_id, repo_type=cfg.repo_type):
        rel = f[len(prefix) :] if prefix and f.startswith(prefix) else f
        if rel.split("/")[0].startswith("shard_") or rel in KEEP_ROOT or f in KEEP_ROOT:
            continue
        out.append(f)
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma-separated shard names, e.g. shard_0002,shard_0004 (default: every shard on the Hub)")
    ap.add_argument("--work-dir", help="where downloaded and rewritten shards go (default data/reannotate)")
    ap.add_argument("--no-push", action="store_true", help="download + re-annotate + gate only; do not upload, do not touch the card")
    ap.add_argument("--no-card", action="store_true", help="push the shards but leave README.md alone")
    ap.add_argument("--no-negatives", action="store_true", help="skip rebuilding hard negatives (faster)")
    ap.add_argument("--clean-root", action="store_true", help="also delete stale non-shard files at the repository root")
    ap.add_argument("--delete-media", action="store_true", help="also delete frames/ and clips/ files of a shard that the rewritten shard does not list")
    ap.add_argument("--ignore-gates", action="store_true", help="push even when a release gate fails (not recommended)")
    ap.add_argument("--clean", action="store_true", help="delete the work dir after a successful push")
    ap.add_argument("--force", action="store_true", help="run even while a pipeline batch is live (not recommended on 8 GB)")
    ap.add_argument("--dry-run", action="store_true", help="show the shards and the stale root files, change nothing")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="config override (repeatable)")
    args = ap.parse_args()

    from video_dataset.cli import load_env_file
    from video_dataset.config import load_config
    from video_dataset.dataset.reannotate import (
        download_shard,
        hub_token,
        list_hub_shards,
        reannotate_shard,
        update_dataset_card,
    )
    from video_dataset.dataset.release import format_report, run_release_check

    load_env_file()
    cfg = load_config(None, _overrides(args.overrides))
    token = hub_token(cfg)
    if not token:
        print(f"no Hugging Face token: set {cfg.upload.token_env or 'HF_TOKEN'} (a write token) in the environment or .env")
        return 1
    pid = _live_run_pid(cfg.data_dir)
    if pid and not args.force and not args.dry_run:
        print(f"a pipeline run is live (pid {pid}); wait for it or pass --force")
        return 3
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    shards = list_hub_shards(cfg, token)
    wanted = {x.strip() for x in args.only.split(",") if x.strip()} if args.only else None
    shards = [s for s in shards if not wanted or Path(s).name in wanted]
    stale_root = _stale_root_files(api, cfg.upload)
    print(f"repo: {cfg.upload.repo_id}\nshards: {', '.join(Path(s).name for s in shards) or '(none)'}")
    if stale_root:
        print(f"stale files outside the shards ({len(stale_root)}){' - would be deleted with --clean-root' if not args.clean_root else ' - will be deleted'}:")
        for f in stale_root[:40]:
            print(f"  {f}")
        if len(stale_root) > 40:
            print(f"  ... and {len(stale_root) - 40} more")
    if args.dry_run:
        return 0
    if not shards:
        print("nothing to do")
        return 1

    work = Path(args.work_dir) if args.work_dir else cfg.data_dir / "reannotate"
    downloads, rewritten = work / "hub", work / "rewritten"
    t0 = time.time()

    # ---- phase 1: download, re-annotate, gate every shard -------------------------------------
    results: dict[str, Path] = {}
    schemas: dict[str, dict[str, str]] = {}
    failed: list[str] = []
    for path_in_repo in shards:
        name = Path(path_in_repo).name
        print(f"\n== {name}: downloading", flush=True)
        local = download_shard(cfg, path_in_repo, downloads, token)
        out_dir = rewritten / name
        if out_dir.exists():
            shutil.rmtree(out_dir)
        print(f"== {name}: re-annotating -> {out_dir}", flush=True)
        summary = reannotate_shard(local, cfg, out_dir, rebuild_negatives=not args.no_negatives)
        print(f"   {summary['videos']} videos, {summary['records_in']:,} records, splits {summary['splits']}, scene tiers {summary['tiers']}, strict {summary.get('cinematic', {}).get('strict', '?')}")
        report = run_release_check(cfg, out_dir, require_media=False)
        print("   " + format_report(report).replace("\n", "\n   "))
        if not report["ok"]:
            failed.append(name)
        results[path_in_repo] = out_dir
        schemas[name] = _parquet_schemas(out_dir)
    # Parquet schema must be identical for every record type across shards (one viewer table per type)
    names = list(schemas)
    for other in names[1:]:
        for rtype, schema in schemas[names[0]].items():
            if rtype in schemas[other] and schemas[other][rtype] != schema:
                print(f"schema mismatch for {rtype}.parquet between {names[0]} and {other}")
                failed.append(other)
    if failed and not args.ignore_gates:
        print(f"\nrelease gates failed for {sorted(set(failed))}; nothing pushed. See <shard>/release_report.json under {rewritten}, fix, re-run (or --ignore-gates).")
        return 2
    if args.no_push:
        print(f"\nre-annotated {len(results)} shard(s) into {rewritten} in {(time.time() - t0) / 60:.1f} min; nothing pushed (--no-push)")
        return 0

    # ---- phase 2: replace every shard on the Hub ----------------------------------------------
    for path_in_repo, out_dir in results.items():
        print(f"\n== {Path(path_in_repo).name}: replacing on the Hub", flush=True)
        url, deleted = _push_replace(api, cfg.upload, out_dir, path_in_repo, delete_media=args.delete_media)
        print(f"   done ({deleted} superseded file(s) removed): {url}")
    if args.clean_root and stale_root:
        from huggingface_hub import CommitOperationDelete

        print(f"\n== removing {len(stale_root)} stale root file(s)", flush=True)
        api.create_commit(repo_id=cfg.upload.repo_id, repo_type=cfg.upload.repo_type, operations=[CommitOperationDelete(path_in_repo=f) for f in stale_root], commit_message=f"Remove {len(stale_root)} stale pre-shard file(s) from the repository root")
    if not args.no_card:
        print("\n== dataset card:", update_dataset_card(cfg, token) or "already current")
    if args.clean and work.exists():
        shutil.rmtree(work, ignore_errors=True)
        print(f"removed {work}")
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")
    print(f"dataset: https://huggingface.co/datasets/{cfg.upload.repo_id}\nviewer:  https://huggingface.co/datasets/{cfg.upload.repo_id}/viewer (rebuilds a few minutes after the last commit)")
    print("verify:  .venv/bin/python scripts/hf_verify_counts.py   (once the viewer has rebuilt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
