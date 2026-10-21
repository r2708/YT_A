"""``python -m dataset_factory`` / ``dataset-factory``: inspect, ingest, validate, extract-frames,
deduplicate, captions review, curate, export, report. Every command is resumable; ``--limit``
bounds work; ``--offline`` forbids any provider but ``none``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from dataset_factory import PIPELINE_VERSION
from dataset_factory.config import ConfigError, FactoryConfig, load_factory_config

app = typer.Typer(add_completion=False, no_args_is_help=True, help=f"dataset_factory {PIPELINE_VERSION}: T2V / I2V curation from the youtube-video-dataset exports. Offline by default.")
captions_app = typer.Typer(no_args_is_help=True, help="Caption reuse, deterministic checks and optional provider review.")
app.add_typer(captions_app, name="captions")
console = Console()

ConfigOpt = Annotated[str | None, typer.Option("--config", "-c", help="Factory YAML (default config/factory/dataset.yaml)")]
SetOpt = Annotated[list[str] | None, typer.Option("--set", help="Override a.b.c=value (repeatable)")]
LimitOpt = Annotated[int | None, typer.Option("--limit", help="Process at most this many records")]
OfflineOpt = Annotated[bool, typer.Option("--offline/--online", help="Forbid any provider but 'none' (default: config.offline)")]


def _load_env() -> None:
    """HF_TOKEN & co from a local .env (never printed, never logged)."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def _cfg(config: str | None, set_: list[str] | None, **extra: Any) -> FactoryConfig:
    _load_env()
    overrides = dict(x.split("=", 1) for x in (set_ or []))
    overrides.update({k: v for k, v in extra.items() if v is not None})
    try:
        return load_factory_config(config, overrides)
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from None


def _log(msg: str) -> None:
    console.print(f"[dim]{msg}[/dim]")


def _show(result: dict[str, Any]) -> None:
    console.print(json.dumps(result, indent=2, default=str))


@app.command()
def inspect(source: Annotated[str | None, typer.Option("--source", help="Hub repo id (default: source.repo_id)")] = None, out: Annotated[str, typer.Option("--out", help="Directory for source_audit.json / .md")] = "reports", config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """Inspect the source dataset's files, schema, row counts, media presence and rights evidence without downloading records."""
    from dataset_factory.source.huggingface_source import inspect_source

    cfg = _cfg(config, set_, **({"source.repo_id": source} if source else {}))
    audit = inspect_source(cfg)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "source_audit.json").write_text(json.dumps(audit, indent=2))
    (out_dir / "source_audit.md").write_text(_audit_markdown(audit))
    _show({k: v for k, v in audit.items() if k not in ("record_types",)})
    for rtype, info in audit["record_types"].items():
        console.print(f"{rtype}: {info['rows_from_parquet_footers']} rows in Parquet footers (viewer: {info['viewer_rows']}), {len(info['columns'])} columns")
    console.print(f"written: {out_dir / 'source_audit.md'}")


def _audit_markdown(a: dict[str, Any]) -> str:
    lines = [f"# Source audit: {a['repo_id']}", "", "Measured from repository metadata and Parquet footers (no record download).", ""]
    lines += [f"- files: {a['files']}; shards: {len(a['shards'])} ({', '.join(a['shards'])})", f"- media files in the repository: {a['media_files_in_repo']} -> {a['media_availability']}", f"- files outside the shards: {a['files_outside_shards']} (examples: {', '.join(a['files_outside_shards_examples'][:5]) or 'none'})", f"- viewer rows per config: {a['viewer_rows']}", f"- videos in shard manifests: {a['videos_in_manifests']}; rights_status: {a['rights_status_in_manifests']}; license_status: {a['license_status_in_manifests']}", f"- sample video metadata: {a['sample_video_meta']}", ""]
    for rtype, info in a["record_types"].items():
        lines += [f"## {rtype}", "", f"- rows (Parquet footers): {info['rows_from_parquet_footers']}; viewer: {info['viewer_rows']}", f"- per shard: {info['per_shard']}", f"- columns ({len(info['columns'])}): {', '.join(info['columns'])}", ""]
    lines += ["## Interpretation", "", "- Only `clips` rows are video-caption candidates; scenes / frames / events / QA are annotations, not training media.", "- `clip_path` is relative to the producer's data directory; set `source.media_root` to where those files live. Without it every record is `media_validation_status: not_checked` and nothing is eligible.", "- The repository licence covers annotations only; `license_status: standard` videos are `restricted` and never enter approved exports unless rights evidence is supplied."]
    return "\n".join(lines) + "\n"


@app.command()
def ingest(limit: LimitOpt = None, source_dir: Annotated[str | None, typer.Option("--source-dir", help="Ingest a local final/ or shard directory instead of the Hub")] = None, shards: Annotated[int | None, typer.Option("--shards", help="Only the first N shards")] = None, config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """Fetch the record Parquet files (never media) and map rows into the canonical manifest."""
    from dataset_factory.pipeline import ingest as run

    cfg = _cfg(config, set_)
    _show(run(cfg, limit=limit, source_root=Path(source_dir) if source_dir else None, limit_shards=shards, log=_log))


@app.command()
def validate(limit: LimitOpt = None, force: Annotated[bool, typer.Option("--force", help="Re-check records whose inputs did not change")] = False, offline: OfflineOpt = True, config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """Rule-based media integrity checks (probe, decode, black / frozen frames). No remote calls."""
    from dataset_factory.media.probe import MACOS_INSTALL_HINT, tools_available
    from dataset_factory.pipeline import validate as run

    cfg = _cfg(config, set_)
    tools = tools_available()
    if not tools["ffprobe"]:
        console.print(f"[yellow]{MACOS_INSTALL_HINT}[/yellow]")
    if not cfg.source.media_root:
        console.print("[yellow]source.media_root is unset: assets cannot be resolved, every record will be not_checked[/yellow]")
    _show(run(cfg, limit=limit, force=force, log=_log))


@app.command("extract-frames")
def extract_frames(limit: LimitOpt = None, force: Annotated[bool, typer.Option("--force")] = False, config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """Extract and checksum the I2V starting frame of every media-valid record."""
    from dataset_factory.pipeline import extract_frames as run

    _show(run(_cfg(config, set_), limit=limit, force=force, log=_log))


@app.command()
def deduplicate(config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """Group byte-identical clips; flag perceptual near-duplicate candidates for review."""
    from dataset_factory.pipeline import deduplicate as run

    _show(run(_cfg(config, set_), log=_log))


@captions_app.command("review")
def captions_review(limit: LimitOpt = None, max_calls: Annotated[int | None, typer.Option("--max-calls", help="Provider call budget for this run (default provider.max_calls)")] = None, dry_run: Annotated[bool, typer.Option("--dry-run", help="Count what would be called; change nothing")] = False, all_records: Annotated[bool, typer.Option("--all", help="Send every record to the provider, not only weak ones")] = False, offline: Annotated[bool | None, typer.Option("--offline/--online", help="Forbid any provider but 'none' (default config.offline)")] = None, config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """Choose the best source caption, run the deterministic checks and, only when a provider is configured, review weak captions within a call budget."""
    from dataset_factory.pipeline import review_captions as run

    try:
        _show(run(_cfg(config, set_), limit=limit, max_calls=max_calls, dry_run=dry_run, select_all=all_records, offline=offline, log=_log))
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from None


@app.command()
def curate(config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """Resolve rights, verify I2V pairs, assign leakage-safe splits and compute T2V / I2V eligibility."""
    from dataset_factory.pipeline import curate as run

    _show(run(_cfg(config, set_), log=_log))


@app.command()
def export(target: Annotated[str, typer.Option("--target", help="canonical | wan-t2v | wan-i2v (see exports.targets)")], overwrite: Annotated[bool, typer.Option("--overwrite", help="Replace an existing export of the same version")] = False, config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """Write a versioned export into a staging directory, verify every path, then promote it atomically."""
    from dataset_factory.exports.release import ExportConflict
    from dataset_factory.pipeline import export as run

    try:
        _show(run(_cfg(config, set_), target, overwrite=overwrite, log=_log))
    except (ExportConflict, KeyError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None


@app.command()
def report(config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """Write reports/summary.json + summary.md with measured counts per status and eligibility."""
    from dataset_factory.pipeline import report as run

    res = run(_cfg(config, set_))
    _show(res["summary"])
    console.print(f"written: {res['markdown']}")


@app.command("show-config")
def show_config(config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """Print the resolved configuration and its hash."""
    cfg = _cfg(config, set_)
    console.print(json.dumps({"config_hash": cfg.config_hash, **cfg.model_dump(mode="json")}, indent=2))


if __name__ == "__main__":
    app()
