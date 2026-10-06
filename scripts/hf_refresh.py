#!/usr/bin/env python
"""Pull every shard from the Hugging Face dataset repo, re-annotate it to the current schema
(split / tier / subsets / hard negatives / typed Parquet) and push it back, then refresh the
dataset card's loader configs so the Hub viewer reads the per-type Parquet files.

    .venv/bin/python scripts/hf_refresh.py                 # all shards, push + card update
    .venv/bin/python scripts/hf_refresh.py --only shard_0002,shard_0004
    .venv/bin/python scripts/hf_refresh.py --dry-run       # list the Hub shards, change nothing
    .venv/bin/python scripts/hf_refresh.py --no-push       # rewrite locally (data/reannotate), no upload
    .venv/bin/python scripts/hf_refresh.py --wait          # if a batch is running, wait for it to finish first
    .venv/bin/python scripts/hf_refresh.py --clean         # delete data/reannotate after a successful push

Works on macOS and Windows (same interpreter as the pipeline). HF_TOKEN (write) is read from the
environment or .env by the CLI. Refuses to run beside a live `video-dataset run` (data/run.lock)
unless --wait or --force is given: re-annotation loads whole shards into memory.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-c", "from video_dataset.cli import app; app()"]


def _data_dir(overrides: list[str]) -> Path:
    for item in overrides:
        if item.startswith("project.data_dir="):
            return Path(item.split("=", 1)[1]).expanduser()
    return ROOT / "data"


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


def _repo_id(overrides: list[str]) -> str:
    from video_dataset.config import load_config

    cfg = load_config(None, dict(x.split("=", 1) for x in overrides))
    return cfg.upload.repo_id or ""


def _viewer_status(repo_id: str) -> str:
    """Best effort: ask the datasets-server whether the viewer can build the dataset (may lag a few minutes)."""
    try:
        import requests

        r = requests.get("https://datasets-server.huggingface.co/is-valid", params={"dataset": repo_id},
                         headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}, timeout=20)
        return f"{r.status_code} {r.text[:300]}"
    except Exception as exc:  # informational only
        return f"unavailable ({exc.__class__.__name__})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma-separated shard names, e.g. shard_0002,shard_0004 (default: every shard on the Hub)")
    ap.add_argument("--no-push", action="store_true", help="re-annotate into data/reannotate only; do not upload, do not touch the card")
    ap.add_argument("--no-card", action="store_true", help="push the shards but leave README.md alone")
    ap.add_argument("--no-negatives", action="store_true", help="skip rebuilding hard negatives (faster)")
    ap.add_argument("--work-dir", help="where the downloaded shards go (default data/reannotate)")
    ap.add_argument("--clean", action="store_true", help="delete the work dir after a successful push")
    ap.add_argument("--wait", action="store_true", help="wait for a running `video-dataset run` to finish instead of aborting")
    ap.add_argument("--force", action="store_true", help="run even while a batch is live (not recommended on 8 GB)")
    ap.add_argument("--dry-run", action="store_true", help="show the Hub shards and the command, change nothing")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="config override passed to the CLI (repeatable)")
    args = ap.parse_args()

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "src"))
    data_dir = _data_dir(args.overrides)

    pid = _live_run_pid(data_dir)
    if pid and not args.force and not args.dry_run:
        if not args.wait:
            print(f"a pipeline run is live (pid {pid}, {data_dir / 'run.lock'}). Re-run with --wait to queue behind it, or --force.")
            return 2
        print(f"waiting for the live run (pid {pid}) to finish ...", flush=True)
        while pid:
            time.sleep(30)
            pid = _live_run_pid(data_dir)
        print("live run finished; starting")

    cmd = [*CLI, *(x for o in args.overrides for x in ("--set", o)), "reannotate", "--from-hub"]
    if args.only:
        cmd += ["--only", args.only]
    if args.work_dir:
        cmd += ["--work-dir", args.work_dir]
    if args.no_negatives:
        cmd.append("--no-negatives")
    if not args.no_push:
        cmd.append("--push")
        if not args.no_card:
            cmd.append("--update-card")

    repo_id = _repo_id(args.overrides)
    if args.dry_run:
        from video_dataset.cli import load_env_file
        from video_dataset.config import load_config
        from video_dataset.dataset.reannotate import hub_token, list_hub_shards

        load_env_file()  # exports HF_TOKEN from .env the same way the CLI does
        cfg = load_config(None, dict(x.split("=", 1) for x in args.overrides))
        shards = list_hub_shards(cfg, hub_token(cfg))
        print(f"repo: {repo_id}\nshards on the Hub: {', '.join(Path(s).name for s in shards) or '(none)'}")
        if pid:
            print(f"note: a pipeline run is live (pid {pid}); the real run needs --wait or --force")
        print("would run:", " ".join(cmd[3:]))
        return 0

    t0 = time.time()
    print("running:", "video-dataset", " ".join(cmd[3:]), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"reannotate failed (exit {rc}); nothing was cleaned up, re-run to resume (uploads are content-addressed)")
        return rc
    work = Path(args.work_dir) if args.work_dir else data_dir / "reannotate"
    if args.clean and not args.no_push and work.exists():
        shutil.rmtree(work, ignore_errors=True)
        print(f"removed {work}")
    print(f"done in {(time.time() - t0) / 60:.1f} min")
    if not args.no_push:
        from video_dataset.cli import load_env_file

        load_env_file()
        print(f"dataset: https://huggingface.co/datasets/{repo_id}")
        print(f"viewer:  https://huggingface.co/datasets/{repo_id}/viewer")
        print("viewer status (rebuilds a few minutes after the last commit):", _viewer_status(repo_id))
    return 0


if __name__ == "__main__":
    sys.exit(main())
