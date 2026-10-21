#!/usr/bin/env python
"""Download every shard of the Hub dataset, re-annotate it to the current schema, upload it back
and delete the local copies - one shard at a time, so an 8 GB machine never holds more than one.

    .venv/bin/python scripts/hf_reannotate_all.py --dry-run          # list shards + stale root files
    .venv/bin/python scripts/hf_reannotate_all.py                    # the real thing
    .venv/bin/python scripts/hf_reannotate_all.py --only shard_0003  # one shard
    .venv/bin/python scripts/hf_reannotate_all.py --clean-root       # also remove stale root files
    .venv/bin/python scripts/hf_reannotate_all.py --redo             # ignore the done markers

Per shard, in order:

1. a **child interpreter** downloads the shard (dataset files + per_video, no media), re-annotates
   it into ``<work-dir>/rewritten/<shard>`` (current schema, export-time acceptance policy,
   splits with minimums, tiers, strict subset, hard negatives, typed Parquet, licence / tier
   policy blocks), runs the release gates and compares the Parquet schema of every record type
   with the one the current code produces. The child exits, so its memory goes back to the OS
   before the next shard starts (one long-lived process was SIGKILLed on 8 GB).
2. the parent pushes the rewritten shard in small commits and deletes every old file of that
   folder the new shard no longer produces (media under ``frames/`` / ``clips/`` is kept unless
   ``--delete-media``) - a true replacement;
3. the downloaded and rewritten copies are deleted (``--keep-local`` keeps them) and a done
   marker is written to ``<work-dir>/done/<shard>.json``; a re-run skips shards with a marker.

A shard whose gates fail is **not pushed** and its rewritten copy is kept for inspection; the run
stops there (``--continue-on-failure`` moves on to the next shard instead). After the last shard
``--clean-root`` removes the stale non-shard files at the repository root and the dataset card's
loader ``configs:`` block is refreshed. Then run ``scripts/hf_verify_counts.py`` once the viewer
has rebuilt.

``--source-dir DIR`` re-annotates ``DIR/<shard>`` folders already on disk (an earlier download)
instead of downloading them again.

HF_TOKEN (a *write* token) comes from the environment or ``.env``. The script refuses to run beside
a live ``video-dataset run`` (``data/run.lock``) unless ``--force``. Nothing is deleted locally
before the shard's push has succeeded; nothing on the Hub is deleted before its replacement is up.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

HIDDEN_WORKER = "--worker"


def _overrides(items: list[str]) -> dict[str, str]:
    return dict(x.split("=", 1) for x in items)


def _normalised(schema) -> list[tuple[str, str]]:  # type: ignore[no-untyped-def]
    """(name, type) pairs without metadata; Parquet renames list children item -> element, which is cosmetic."""
    return [(f.name, str(f.type).replace("<element:", "<item:")) for f in schema]


def _expected_schemas() -> dict[str, list[tuple[str, str]]]:
    from video_dataset.dataset.arrow import arrow_schema
    from video_dataset.dataset.export import DATASET_FILES, RECORD_MODELS

    return {name: _normalised(arrow_schema(RECORD_MODELS[name])[0]) for name in DATASET_FILES}


def _parquet_schemas(shard_dir: Path) -> dict[str, list[tuple[str, str]]]:
    import pyarrow.parquet as pq

    from video_dataset.dataset.export import DATASET_FILES

    out: dict[str, list[tuple[str, str]]] = {}
    for name in DATASET_FILES:
        p = shard_dir / f"{name}.parquet"
        if p.exists():
            out[name] = _normalised(pq.read_schema(p))
    return out


# ----------------------------------------------------------------------------- child: one shard
def worker(path_in_repo: str, work: Path, overrides: dict[str, str], rebuild_negatives: bool, source_dir: Path | None = None) -> int:
    """Download (or take from ``source_dir``) + re-annotate + gate one shard; write
    <work>/status/<shard>.json; exit 0 when pushable."""
    from video_dataset.cli import load_env_file
    from video_dataset.config import load_config
    from video_dataset.dataset.reannotate import download_shard, hub_token, reannotate_shard
    from video_dataset.dataset.release import format_report, run_release_check
    from video_dataset.schemas.common import SCHEMA_VERSION

    load_env_file()
    cfg = load_config(None, overrides)
    token = hub_token(cfg)
    name = Path(path_in_repo).name
    status_file = work / "status" / f"{name}.json"
    status_file.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if source_dir is not None:
        local = source_dir / name
        if not local.is_dir():
            print(f"== {name}: {local} does not exist")
            return 2
        print(f"== {name}: using {local}", flush=True)
    else:
        print(f"== {name}: downloading", flush=True)
        local = download_shard(cfg, path_in_repo, work / "hub", token)
    out_dir = work / "rewritten" / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    print(f"== {name}: re-annotating -> {out_dir}", flush=True)
    summary = reannotate_shard(local, cfg, out_dir, rebuild_negatives=rebuild_negatives)
    print(f"   {summary['videos']} videos, {summary['records_in']:,} records, splits {summary['splits']}, scene tiers {summary['tiers']}, strict {summary.get('cinematic', {}).get('strict', '?')}, policy downgraded {summary.get('cinematic', {}).get('policy_downgraded', '?')}", flush=True)
    report = run_release_check(cfg, out_dir, require_media=False)
    print("   " + format_report(report).replace("\n", "\n   "), flush=True)
    expected = _expected_schemas()
    schema_mismatch = {k: v for k, v in _parquet_schemas(out_dir).items() if expected.get(k) not in (None, v)}
    if schema_mismatch:
        print(f"   Parquet schema differs from the current models for: {', '.join(sorted(schema_mismatch))}", flush=True)
    ok = bool(report["ok"]) and not schema_mismatch
    status = {
        "shard": path_in_repo, "ok": ok, "failed_gates": report["failed_gates"], "schema_mismatch": sorted(schema_mismatch), "schema_version": SCHEMA_VERSION,
        "videos": summary["videos"], "records": summary["records_out"], "splits": summary["splits"], "seconds": round(time.time() - t0, 1),
        "downloaded": str(local), "rewritten": str(out_dir),
    }
    status_file.write_text(json.dumps(status, indent=2))
    return 0 if ok else 2


# ----------------------------------------------------------------------------- parent
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma-separated shard names, e.g. shard_0002,shard_0004 (default: every shard on the Hub)")
    ap.add_argument("--work-dir", help="downloads, rewritten shards, status and done markers (default data/reannotate)")
    ap.add_argument("--source-dir", help="re-annotate DIR/<shard> folders already on disk instead of downloading (they are deleted after the push like a download)")
    ap.add_argument("--no-push", action="store_true", help="download + re-annotate + gate only; keep the local copies, touch nothing on the Hub")
    ap.add_argument("--no-card", action="store_true", help="leave README.md alone")
    ap.add_argument("--no-negatives", action="store_true", help="skip rebuilding hard negatives (faster)")
    ap.add_argument("--clean-root", action="store_true", help="after the shards: delete stale non-shard files at the repository root")
    ap.add_argument("--delete-media", action="store_true", help="also delete frames/ and clips/ files of a shard that the rewritten shard does not list")
    ap.add_argument("--keep-local", action="store_true", help="do not delete the downloaded / rewritten shard after a successful push")
    ap.add_argument("--redo", action="store_true", help="re-process shards that already have a done marker")
    ap.add_argument("--continue-on-failure", action="store_true", help="a shard whose gates fail is skipped (kept locally) instead of stopping the run")
    ap.add_argument("--ignore-gates", action="store_true", help="push even when a release gate fails (not recommended)")
    ap.add_argument("--force", action="store_true", help="run even while a pipeline batch is live (not recommended on 8 GB)")
    ap.add_argument("--dry-run", action="store_true", help="show the shards, the done markers and the stale root files; change nothing")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="config override (repeatable)")
    ap.add_argument(HIDDEN_WORKER, help=argparse.SUPPRESS)
    args = ap.parse_args()

    from video_dataset.cli import load_env_file
    from video_dataset.config import load_config

    load_env_file()
    cfg = load_config(None, _overrides(args.overrides))
    work = Path(args.work_dir) if args.work_dir else cfg.data_dir / "reannotate"
    if args.worker:
        return worker(args.worker, work, _overrides(args.overrides), rebuild_negatives=not args.no_negatives, source_dir=Path(args.source_dir) if args.source_dir else None)

    import hf_replace_shards as base

    from video_dataset.dataset.reannotate import hub_token, list_hub_shards, update_dataset_card
    from video_dataset.schemas.common import SCHEMA_VERSION

    token = hub_token(cfg)
    if not token:
        print(f"no Hugging Face token: set {cfg.upload.token_env or 'HF_TOKEN'} (a write token) in the environment or .env")
        return 1
    pid = base._live_run_pid(cfg.data_dir)
    if pid and not args.force and not args.dry_run:
        print(f"a pipeline run is live (pid {pid}); wait for it or pass --force")
        return 3
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    shards = list_hub_shards(cfg, token)
    wanted = {x.strip() for x in args.only.split(",") if x.strip()} if args.only else None
    shards = [s for s in shards if not wanted or Path(s).name in wanted]
    done_dir = work / "done"
    done = {p.stem for p in done_dir.glob("*.json")} if done_dir.exists() else set()
    stale_root = base._stale_root_files(api, cfg.upload)
    print(f"repo: {cfg.upload.repo_id}  target schema: {SCHEMA_VERSION}  work dir: {work}")
    print("shards: " + (", ".join(f"{Path(s).name}{' (done)' if Path(s).name in done and not args.redo else ''}" for s in shards) or "(none)"))
    if stale_root:
        print(f"stale files outside the shards ({len(stale_root)}){' - will be deleted after the shards' if args.clean_root else ' - pass --clean-root to delete them'}:")
        for f in stale_root[:20]:
            print(f"  {f}")
        if len(stale_root) > 20:
            print(f"  ... and {len(stale_root) - 20} more")
    if args.dry_run:
        return 0
    if not shards:
        print("nothing to do")
        return 1

    t0 = time.time()
    pushed: list[str] = []
    failed: list[str] = []
    for path_in_repo in shards:
        name = Path(path_in_repo).name
        if name in done and not args.redo:
            print(f"\n== {name}: already done ({done_dir / (name + '.json')}); --redo to repeat")
            continue
        cmd = [sys.executable, str(Path(__file__).resolve()), HIDDEN_WORKER, path_in_repo, "--work-dir", str(work)]
        if args.no_negatives:
            cmd.append("--no-negatives")
        if args.source_dir:
            cmd += ["--source-dir", args.source_dir]
        for ov in args.overrides:
            cmd += ["--set", ov]
        print(f"\n== {name}: child process {' '.join(cmd[2:4])}", flush=True)
        rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
        status_file = work / "status" / f"{name}.json"
        status = json.loads(status_file.read_text()) if status_file.exists() else {}
        if rc != 0 or not status:
            why = f"gates {status.get('failed_gates')}" + (f", schema mismatch {status.get('schema_mismatch')}" if status.get("schema_mismatch") else "") if status else f"child exit code {rc} (killed? check `sysctl vm.swapusage` and close other apps)"
            print(f"   {name}: NOT pushable - {why}; rewritten copy kept under {work / 'rewritten' / name}")
            failed.append(name)
            if status and args.ignore_gates:
                print("   --ignore-gates: pushing anyway")
            elif args.continue_on_failure:
                continue
            else:
                break
        if args.no_push:
            print(f"   {name}: re-annotated, not pushed (--no-push)")
            continue
        out_dir = Path(status["rewritten"])
        print(f"== {name}: replacing on the Hub", flush=True)
        url, deleted = base._push_replace(api, cfg.upload, out_dir, path_in_repo, delete_media=args.delete_media)
        print(f"   done ({deleted} superseded file(s) removed): {url}")
        done_dir.mkdir(parents=True, exist_ok=True)
        (done_dir / f"{name}.json").write_text(json.dumps({**status, "pushed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "commit": url, "superseded_files_removed": deleted}, indent=2))
        pushed.append(name)
        if not args.keep_local:
            for p in (Path(status["downloaded"]), out_dir):
                shutil.rmtree(p, ignore_errors=True)
            print(f"   local copies removed ({status['downloaded']}, {out_dir})")

    if args.no_push:
        print(f"\nre-annotated {len(shards) - len(failed)} shard(s) into {work / 'rewritten'} in {(time.time() - t0) / 60:.1f} min; nothing pushed")
        return 2 if failed else 0
    if failed and not args.ignore_gates:
        print(f"\nstopped: {', '.join(failed)} failed the gates; pushed before that: {', '.join(pushed) or 'none'}. Fix, then re-run (done shards are skipped).")
        return 2
    if args.clean_root and stale_root:
        from huggingface_hub import CommitOperationDelete

        print(f"\n== removing {len(stale_root)} stale root file(s)", flush=True)
        api.create_commit(repo_id=cfg.upload.repo_id, repo_type=cfg.upload.repo_type, operations=[CommitOperationDelete(path_in_repo=f) for f in stale_root], commit_message=f"Remove {len(stale_root)} stale pre-shard file(s) from the repository root")
    if not args.no_card:
        print("\n== dataset card:", update_dataset_card(cfg, token) or "already current")
    hub_dir = work / "hub"
    if not args.keep_local and hub_dir.exists() and not any(p for p in hub_dir.rglob("*") if p.is_file() and ".cache" not in p.parts):
        shutil.rmtree(hub_dir, ignore_errors=True)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min: pushed {len(pushed)} shard(s) ({', '.join(pushed) or 'none'})")
    print(f"dataset: https://huggingface.co/datasets/{cfg.upload.repo_id}\nviewer:  https://huggingface.co/datasets/{cfg.upload.repo_id}/viewer (rebuilds a few minutes after the last commit)")
    print("verify:  .venv/bin/python scripts/hf_verify_counts.py   (once the viewer has rebuilt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
