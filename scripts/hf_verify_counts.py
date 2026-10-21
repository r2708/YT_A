#!/usr/bin/env python
"""Verify that every published count of the Hub dataset reconciles, without downloading records.

    .venv/bin/python scripts/hf_verify_counts.py            # exit 1 on any disagreement
    .venv/bin/python scripts/hf_verify_counts.py --json out.json

For every ``shard_NNNN`` folder: ``manifest.json["counts"]`` and ``statistics.json`` per-type counts
versus the row counts in the Parquet footers of ``<type>.parquet`` (read over HfFileSystem, a few
KB per file). Across shards: the per-type sums versus what the datasets-server reports for each
loader config (what the viewer shows). Files outside the shards that no loader config matches are
listed as stale: they are what makes a root ``manifest.json`` disagree with the viewer. Run it after
every push; the dataset is only consistent when it prints ``OK``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

KEEP_ROOT = {"README.md", ".gitattributes", ".gitignore"}


def _sizes_from_server(repo_id: str, token: str | None) -> dict[str, int]:
    """{config_name: num_rows} as the datasets-server (the viewer) sees the dataset."""
    import requests

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.get("https://datasets-server.huggingface.co/size", params={"dataset": repo_id}, headers=headers, timeout=60)
    if r.status_code != 200:
        return {}
    out: dict[str, int] = {}
    for c in (r.json().get("size") or {}).get("configs", []):
        out[str(c.get("config"))] = int(c.get("num_rows") or 0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", help="write the full comparison to this file")
    ap.add_argument("--no-server", action="store_true", help="skip the datasets-server (viewer) comparison")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

    from video_dataset.cli import load_env_file
    from video_dataset.config import load_config
    from video_dataset.dataset.export import DATASET_FILES
    from video_dataset.dataset.reannotate import hub_token, list_hub_shards

    load_env_file()
    cfg = load_config(None, dict(x.split("=", 1) for x in args.overrides))
    token = hub_token(cfg) or os.environ.get("HF_TOKEN")
    up = cfg.upload
    api = HfApi(token=token)
    fs = HfFileSystem(token=token)
    import pyarrow.parquet as pq

    shards = list_hub_shards(cfg, token)
    all_files = api.list_repo_files(up.repo_id, repo_type=up.repo_type)
    prefix = up.path_in_repo.strip("/") + "/" if up.path_in_repo else ""
    stale = sorted(f for f in all_files if not (f[len(prefix):] if prefix and f.startswith(prefix) else f).split("/")[0].startswith("shard_") and f not in KEEP_ROOT)

    problems: list[str] = []
    per_shard: dict[str, dict[str, dict[str, int | None]]] = {}
    totals: dict[str, int] = {name: 0 for name in DATASET_FILES}
    for path_in_repo in shards:
        name = Path(path_in_repo).name
        row: dict[str, dict[str, int | None]] = {}
        manifest = stats = {}
        try:
            manifest = json.loads(Path(hf_hub_download(up.repo_id, f"{path_in_repo}/manifest.json", repo_type=up.repo_type, token=token)).read_text())
        except Exception as exc:
            problems.append(f"{name}: manifest.json unreadable ({str(exc)[:80]})")
        try:
            stats = json.loads(Path(hf_hub_download(up.repo_id, f"{path_in_repo}/statistics.json", repo_type=up.repo_type, token=token)).read_text())
        except Exception as exc:
            problems.append(f"{name}: statistics.json unreadable ({str(exc)[:80]})")
        for rtype in DATASET_FILES:
            parquet_rows: int | None = None
            p = f"datasets/{up.repo_id}/{path_in_repo}/{rtype}.parquet"
            if f"{path_in_repo}/{rtype}.parquet" in all_files:
                try:
                    with fs.open(p, "rb") as fh:
                        parquet_rows = int(pq.read_metadata(fh).num_rows)
                except Exception as exc:
                    problems.append(f"{name}/{rtype}.parquet: footer unreadable ({str(exc)[:80]})")
            else:
                problems.append(f"{name}/{rtype}.parquet missing (loader config '{rtype}' skips this shard)")
            m = (manifest.get("counts") or {}).get(rtype)
            s = stats.get(rtype) if isinstance(stats.get(rtype), int) else (stats.get("jsonl_counts") or {}).get(rtype)
            row[rtype] = {"parquet": parquet_rows, "manifest": m, "statistics": s}
            vals = {v for v in (parquet_rows, m, s) if v is not None}
            if len(vals) > 1:
                problems.append(f"{name}/{rtype}: parquet={parquet_rows} manifest={m} statistics={s}")
            totals[rtype] += int(parquet_rows or 0)
        per_shard[name] = row
        listed = len(manifest.get("videos") or [])
        splits = manifest.get("splits") or {}
        if splits and sum(int(v) for v in splits.values()) != listed:
            problems.append(f"{name}: manifest lists {listed} videos but splits sum to {sum(int(v) for v in splits.values())}")
        for sp in ("validation", "test"):
            if splits and not int(splits.get(sp, 0)):
                problems.append(f"{name}: {sp} split is empty")
        print(f"{name}: " + ", ".join(f"{t}={row[t]['parquet']}" for t in DATASET_FILES) + f"; videos={listed}, splits={splits}")

    server = {} if args.no_server else _sizes_from_server(up.repo_id, token)
    for rtype, n in totals.items():
        seen = server.get(rtype)
        flag = "" if seen is None or seen == n else "   <-- viewer disagrees"
        if flag:
            problems.append(f"viewer config '{rtype}' shows {seen} rows, shards hold {n}")
        print(f"total {rtype:<20} shards={n:<8} viewer={seen if seen is not None else 'n/a'}{flag}")
    if stale:
        print(f"\n{len(stale)} file(s) outside every shard (no loader config reads them; a root manifest.json there describes a different dataset than the viewer):")
        for f in stale[:30]:
            print(f"  {f}")
        if len(stale) > 30:
            print(f"  ... and {len(stale) - 30} more")
        print("  remove them with: scripts/hf_replace_shards.py --clean-root")
        problems.append(f"{len(stale)} stale file(s) outside the shards")
    if args.json:
        Path(args.json).write_text(json.dumps({"repo": up.repo_id, "shards": per_shard, "totals": totals, "viewer": server, "stale_root_files": stale, "problems": problems}, indent=2))
    if problems:
        print("\nFAILED:\n  " + "\n  ".join(problems))
        return 1
    print("\nOK: every published count reconciles")
    return 0


if __name__ == "__main__":
    sys.exit(main())
