"""`video-dataset v3 ...` commands. Registered on the main app by cli.py."""

from __future__ import annotations

import json
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

console = Console()
ConfigOpt = Annotated[str | None, typer.Option("--config", "-c", help="Path to a YAML config overriding config/default.yaml")]
SetOpt = Annotated[list[str] | None, typer.Option("--set", "-s", help="Override a config value, e.g. --set resolution.target=1080")]


def register(app: typer.Typer, config_loader: Any, apply_local_options: Any) -> None:
    v3 = typer.Typer(help="V3: visual T2V / V2V training datasets on top of the existing annotations.", no_args_is_help=True)
    app.add_typer(v3, name="v3")

    def _cfg(config: str | None, set_: list[str] | None) -> Any:
        apply_local_options(config, set_)
        return config_loader()

    @v3.command()
    def inventory(config: ConfigOpt = None, set_: SetOpt = None) -> None:
        """Freeze + inventory the existing annotation dataset (read-only) and build the source index."""
        from video_dataset.v3.legacy import build_inventory

        out = build_inventory(_cfg(config, set_))
        console.print(f"[green]{out['files']} files, {out['source_videos']} source videos, {out['clips']} clips[/green] -> {out['out_dir']}")

    @v3.command("rights-init")
    def rights_init(config: ConfigOpt = None, set_: SetOpt = None) -> None:
        """Create rights manifest entries (all `metadata_only`) for sources that have none. Existing entries are never changed."""
        from video_dataset.v3 import legacy, rights

        cfg = _cfg(config, set_)
        path = rights.default_rights_path(cfg)
        added, total = rights.init_rights(legacy.load_source_index(cfg), path)
        console.print(f"{added} entr{'y' if added == 1 else 'ies'} added, {total} total -> {path}")
        console.print("Edit an entry to `approved` only with the permission / licence evidence filled in (see docs/v3_pipeline.md).")

    @v3.command("rights-check")
    def rights_check(config: ConfigOpt = None, set_: SetOpt = None) -> None:
        """Rights report: counts per status, which sources are eligible, which `approved` entries are incomplete."""
        from video_dataset.utils.io import write_json_atomic
        from video_dataset.v3 import legacy, rights

        cfg = _cfg(config, set_)
        path = rights.default_rights_path(cfg)
        report = rights.rights_report(legacy.load_source_index(cfg), rights.load_rights(path), path.parent)
        write_json_atomic(cfg.v3_dir / "rights" / "rights_report.json", report)
        t = Table(title=f"rights manifest: {path}")
        t.add_column("status")
        t.add_column("sources", justify="right")
        for status, n in report["by_status"].items():
            t.add_row(status, str(n))
        t.add_row("[green]eligible for visual processing[/green]", str(len(report["eligible"])))
        console.print(t)
        for item in report["approved_but_incomplete"]:
            console.print(f"[yellow]{item['source_video_id']} is marked approved but blocked:[/yellow] {'; '.join(item['problems'])}")

    @v3.command()
    def run(
        video_id: Annotated[list[str] | None, typer.Argument(help="Source video ids (default: every approved source)")] = None,
        limit: Annotated[int | None, typer.Option("--limit", help="Process at most this many approved sources (pilot: 10-25)")] = None,
        force: Annotated[bool, typer.Option("--force", help="Restart the selected jobs from the beginning")] = False,
        config: ConfigOpt = None, set_: SetOpt = None,
    ) -> None:
        """Rights gate -> acquire -> probe -> decode check -> match annotations -> clips -> T2V / V2V records. Resumable."""
        from video_dataset.v3.pipeline import run as run_v3

        out = run_v3(_cfg(config, set_), video_ids=video_id, limit=limit, force=force)
        console.print(f"{out['sources']} source(s), {out['approved']} approved")
        for state, n in sorted(out["states"].items()):
            console.print(f"  {state}: {n}")
        if not out["approved"]:
            console.print("[yellow]No approved source: nothing was downloaded. See `video-dataset v3 rights-check`.[/yellow]")

    @v3.command()
    def status(config: ConfigOpt = None, set_: SetOpt = None) -> None:
        """Job states, failure reasons and the upload / cleanup flags."""
        from video_dataset.v3.pipeline import open_store

        store = open_store(_cfg(config, set_))
        try:
            jobs = store.all()
        finally:
            store.close()
        t = Table(title="v3 jobs")
        for col in ("source", "state", "attempts", "metadata verified", "visual verified", "cleanup allowed", "note"):
            t.add_column(col)
        for j in jobs:
            f = j["flags"]
            state = f"[red]{j['failure']}[/red] (at {j['state']})" if j["failure"] else j["state"]
            t.add_row(j["source_video_id"], state, str(j["attempts"]), str(bool(f.get("metadata_upload_verified"))), str(bool(f.get("visual_asset_upload_verified"))), str(bool(f.get("cleanup_allowed"))), (j["reason"] or "")[:70])
        console.print(t)

    @v3.command()
    def publish(
        dry_run: Annotated[bool, typer.Option("--dry-run", help="Stage both repositories locally, upload nothing")] = False,
        shell: Annotated[bool, typer.Option("--shell", help="Create the repositories with card + schemas only (no records, no media)")] = False,
        config: ConfigOpt = None, set_: SetOpt = None,
    ) -> None:
        """Stage, upload and verify the T2V and V2V repositories. Cleanup is allowed only after verification."""
        from video_dataset.v3.publish import PublishError
        from video_dataset.v3.publish import publish as publish_v3

        try:
            out = publish_v3(_cfg(config, set_), dry_run=dry_run, shell=shell)
        except PublishError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from None
        console.print(json.dumps(out, indent=2, default=str))
        if not dry_run and not out.get("verified"):
            console.print("[red]remote verification failed: local media is kept, nothing was marked verified[/red]")
            raise typer.Exit(2)

    @v3.command()
    def cleanup(dry_run: Annotated[bool, typer.Option("--dry-run")] = False, config: ConfigOpt = None, set_: SetOpt = None) -> None:
        """Delete local media of sources whose remote copy is verified (never anything else)."""
        from video_dataset.v3.cleanup import run_cleanup

        out = run_cleanup(_cfg(config, set_), dry_run=dry_run)
        console.print(f"{'would free' if dry_run else 'freed'} {out['bytes'] / 1e6:.1f} MB for {len(out['cleaned'])} source(s)")
        for vid, why in out["kept"].items():
            console.print(f"  kept {vid}: {why}")

    @v3.command("loader-check")
    def loader_check(
        split: Annotated[str, typer.Option("--split")] = "train",
        steps: Annotated[int, typer.Option("--steps", help="Batches to read")] = 8,
        local: Annotated[bool, typer.Option("--local", help="Read the staged repositories under <work_dir>/staging instead of the Hub")] = False,
        config: ConfigOpt = None, set_: SetOpt = None,
    ) -> None:
        """Read a few batches through the unified T2V / V2V loader and report cache usage."""
        from video_dataset.v3.loader import UnifiedLoader

        cfg = _cfg(config, set_)
        local_dirs = {cfg.v3.repos.t2v: cfg.v3_dir / "staging" / "t2v", cfg.v3.repos.v2v: cfg.v3_dir / "staging" / "v2v"} if local else None
        loader = UnifiedLoader.from_config(cfg, split=split, local_dirs=local_dirs)
        n = 0
        for batch in loader.batches(steps=steps):
            n += 1
            console.print(f"{batch['task']}: " + ", ".join(s["record"]["sample_id"] for s in batch["samples"]))
            loader.release(batch)
        console.print(json.dumps({**loader.stats, "cache_bytes": loader.cache.bytes, "cache_limit_bytes": loader.cache.max_bytes, "batches_read": n}, indent=2, default=str))
