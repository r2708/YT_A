"""video-dataset command line interface."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from video_dataset import __version__
from video_dataset.config import PipelineConfig, dump_config, load_config
from video_dataset.stages import STAGE_ORDER, Stage, parse_stage
from video_dataset.utils.logging import get_logger, setup_logging

app = typer.Typer(help="YouTube video -> temporally grounded multimodal dataset pipeline.", no_args_is_help=True, add_completion=False)
console = Console()
log = get_logger("cli")


class State:
    config: PipelineConfig | None = None
    config_path: str | None = None
    overrides: dict[str, Any] = {}
    log_level: str | None = None


state = State()


def _config() -> PipelineConfig:
    if state.config is None:
        cfg = load_config(state.config_path, state.overrides)
        if state.log_level:
            cfg.project.log_level = state.log_level
        setup_logging(cfg.project.log_level, cfg.data_dir / "logs")
        state.config = cfg
    return state.config


def _parse_overrides(values: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for v in values or []:
        if "=" not in v:
            raise typer.BadParameter(f"--set expects key.path=value, got '{v}'")
        k, val = v.split("=", 1)
        out[k.strip()] = val.strip()
    return out


@app.callback()
def main(
    config: Annotated[str | None, typer.Option("--config", "-c", help="Path to a YAML config overriding config/default.yaml")] = None,
    set_: Annotated[list[str] | None, typer.Option("--set", "-s", help="Override a config value, e.g. --set vision.provider=hf")] = None,
    data_dir: Annotated[str | None, typer.Option("--data-dir", help="Data directory (default: config project.data_dir)")] = None,
    log_level: Annotated[str | None, typer.Option("--log-level", help="DEBUG, INFO, WARNING, ERROR")] = None,
) -> None:
    state.config_path = config
    state.overrides = _parse_overrides(set_)
    if data_dir:
        state.overrides["project.data_dir"] = data_dir
    state.log_level = log_level


ConfigOpt = Annotated[str | None, typer.Option("--config", "-c", help="Path to a YAML config overriding config/default.yaml")]
SetOpt = Annotated[list[str] | None, typer.Option("--set", "-s", help="Override a config value, e.g. --set vision.provider=hf")]


def _apply_local_options(config: str | None, set_: list[str] | None) -> None:
    """Commands accept --config/--set too, so `video-dataset run urls.txt --config x.yaml` works."""
    if config:
        state.config_path = config
    if set_:
        state.overrides.update(_parse_overrides(set_))
    if config or set_:
        state.config = None  # rebuild on next access


def _runner():  # type: ignore[no-untyped-def]
    from video_dataset.pipeline.runner import PipelineRunner

    return PipelineRunner(_config())


def _print_results(results) -> None:  # type: ignore[no-untyped-def]
    table = Table(title="Run summary")
    table.add_column("video_id")
    table.add_column("result")
    table.add_column("stages run", justify="right")
    table.add_column("cached", justify="right")
    table.add_column("error")
    for r in results:
        table.add_row(r.video_id, "[green]ok[/green]" if r.completed else f"[red]failed at {r.failed_stage}[/red]", str(len(r.stages_run)), str(len(r.stages_cached)), (r.error or "")[:80])
    console.print(table)


# ---------------------------------------------------------------------------- commands
@app.command()
def download(
    urls: Annotated[list[str], typer.Argument(help="YouTube URLs, ids or local video files")],
    process: Annotated[bool, typer.Option("--process/--no-process", help="Run the full pipeline after downloading")] = False,
    config: ConfigOpt = None,
    set_: SetOpt = None,
) -> None:
    """Download one or more videos (metadata + media), optionally processing them fully."""
    from video_dataset.utils.urls import classify_input

    _apply_local_options(config, set_)

    items = [it for it in (classify_input(u) for u in urls) if it]
    if not items:
        console.print("[red]No valid inputs[/red]")
        raise typer.Exit(1)
    runner = _runner()
    try:
        registered = runner.register_inputs(items)
        results = runner.run_batch(registered, until=None if process else Stage.PREPROCESS)  # type: ignore[arg-type]
        _print_results(results)
        raise typer.Exit(0 if all(r.completed for r in results) else 2)
    finally:
        runner.close()


@app.command()
def run(
    source: Annotated[str, typer.Argument(help="URL, video id, local video, a .txt of URLs, or a directory of URL files")],
    workers: Annotated[int | None, typer.Option("--workers", "-w", help="Parallel videos for CPU stages")] = None,
    force_from: Annotated[str | None, typer.Option("--force-from", help="Re-run this stage and everything after it")] = None,
    until: Annotated[str | None, typer.Option("--until", help="Stop after this stage")] = None,
    no_aggregate: Annotated[bool, typer.Option("--no-aggregate", help="Skip merging per-video exports into final/")] = False,
    config: ConfigOpt = None,
    set_: SetOpt = None,
) -> None:
    """Run the whole pipeline for every input, resuming completed stages."""
    from video_dataset.dataset.aggregate import aggregate_exports
    from video_dataset.utils.urls import collect_inputs

    _apply_local_options(config, set_)

    items = collect_inputs(source)
    if not items:
        console.print("[red]No inputs found[/red]")
        raise typer.Exit(1)
    console.print(f"Found {len(items)} input(s)")
    runner = _runner()
    try:
        registered = runner.register_inputs(items)
        results = runner.run_batch(registered, force_from=parse_stage(force_from) if force_from else None, until=parse_stage(until) if until else None, workers=workers)  # type: ignore[arg-type]
        _print_results(results)
        if not no_aggregate and (until is None or parse_stage(until) == Stage.EXPORT):
            stats = aggregate_exports(runner.config, runner.db)
            console.print(f"Final dataset written to {runner.config.export_dir} ({stats['temporal_qa']} temporal QA, {stats['long_video_qa']} long-video QA)")
        raise typer.Exit(0 if all(r.completed for r in results) else 2)
    finally:
        runner.close()


@app.command()
def process(
    video_id: Annotated[str, typer.Argument(help="Registered video id (see `status`)")],
    force_from: Annotated[str | None, typer.Option("--force-from", help="Re-run this stage and everything after it")] = None,
    until: Annotated[str | None, typer.Option("--until")] = None,
    config: ConfigOpt = None,
    set_: SetOpt = None,
) -> None:
    """Process (or resume) a single registered video."""
    _apply_local_options(config, set_)
    runner = _runner()
    try:
        if runner.db.get_video(video_id) is None:
            console.print(f"[red]Unknown video id {video_id}[/red]")
            raise typer.Exit(1)
        r = runner.run_video(video_id, force_from=parse_stage(force_from) if force_from else None, until=parse_stage(until) if until else None)
        _print_results([r])
        raise typer.Exit(0 if r.completed else 2)
    finally:
        runner.close()


@app.command()
def status(
    video_id: Annotated[str | None, typer.Argument(help="Show one video in detail")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine readable output")] = False,
    config: ConfigOpt = None,
    set_: SetOpt = None,
) -> None:
    """Show per-video stage status (the checkpoint table)."""
    _apply_local_options(config, set_)
    runner = _runner()
    try:
        rows = runner.status_rows(video_id)
        if as_json:
            console.print_json(json.dumps(rows, default=str))
            return
        if not rows:
            console.print("No videos registered yet. Use `video-dataset run <urls.txt>`.")
            return
        if video_id:
            row = rows[0]
            console.print(f"[bold]{row['video_id']}[/bold]  {row.get('title') or ''}  status={row['status']}")
            t = Table(show_header=True)
            t.add_column("stage")
            t.add_column("status")
            t.add_column("metrics / error")
            for st in STAGE_ORDER:
                s = row["stages"][st.value]
                color = {"DONE": "green", "FAILED": "red", "RUNNING": "yellow", "SKIPPED": "cyan"}.get(s, "white")
                detail = row["errors"].get(st.value) or json.dumps(row["metrics"].get(st.value, {}), default=str)[:100]
                t.add_row(st.value, f"[{color}]{s}[/{color}]", detail)
            console.print(t)
            return
        t = Table(title=f"{len(rows)} video(s)")
        t.add_column("video_id")
        t.add_column("title")
        t.add_column("status")
        for st in STAGE_ORDER:
            t.add_column(st.value.split("_")[0][:6], justify="center")
        marks = {"DONE": "[green]✓[/green]", "FAILED": "[red]✗[/red]", "RUNNING": "[yellow]…[/yellow]", "SKIPPED": "[cyan]-[/cyan]", "PENDING": "·"}
        for row in rows:
            t.add_row(row["video_id"], (row.get("title") or "")[:32], row["status"], *[marks.get(row["stages"][st.value], "·") for st in STAGE_ORDER])
        console.print(t)
        summary = runner.db.summary()
        console.print(f"videos by status: {summary['videos_by_status']}")
    finally:
        runner.close()


@app.command("retry-failed")
def retry_failed(until: Annotated[str | None, typer.Option("--until")] = None, config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """Re-run every video from its first failed stage."""
    _apply_local_options(config, set_)
    runner = _runner()
    try:
        results = runner.retry_failed(until=parse_stage(until) if until else None)
        if not results:
            console.print("Nothing to retry.")
            return
        _print_results(results)
        raise typer.Exit(0 if all(r.completed for r in results) else 2)
    finally:
        runner.close()


@app.command()
def validate(video_id: Annotated[str | None, typer.Argument(help="Validate one video, or all if omitted")] = None, config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """(Re-)run validation for one or all videos that have QA generated."""
    _apply_local_options(config, set_)
    runner = _runner()
    try:
        vids = [video_id] if video_id else [v["video_id"] for v in runner.db.list_videos() if runner.db.is_done(v["video_id"], Stage.QA_GENERATION)]
        if not vids:
            console.print("No videos ready for validation.")
            return
        results = [runner.run_video(v, stages=[Stage.VALIDATION], force_from=Stage.VALIDATION) for v in vids]
        _print_results(results)
    finally:
        runner.close()


@app.command()
def export(
    output: Annotated[str | None, typer.Option("--output", "-o", help="Output directory (default data/final)")] = None,
    rerun: Annotated[bool, typer.Option("--rerun", help="Re-run the per-video EXPORT stage first")] = False,
    config: ConfigOpt = None,
    set_: SetOpt = None,
) -> None:
    """Merge per-video exports into final JSONL + Parquet + statistics."""
    from video_dataset.dataset.aggregate import aggregate_exports
    from video_dataset.dataset.stats import format_statistics

    _apply_local_options(config, set_)
    runner = _runner()
    try:
        if rerun:
            for v in runner.db.list_videos():
                if runner.db.is_done(v["video_id"], Stage.VALIDATION):
                    runner.run_video(v["video_id"], stages=[Stage.EXPORT], force_from=Stage.EXPORT)
        stats = aggregate_exports(runner.config, runner.db, Path(output) if output else None)
        console.print(format_statistics(stats))
        console.print(f"\nfiles: {(Path(output) if output else runner.config.export_dir)}")
    finally:
        runner.close()


@app.command()
def package(
    output: Annotated[str, typer.Option("--output", "-o", help="Zip file to write")] = "data/final/dataset_bundle.zip",
    include_media: Annotated[bool, typer.Option("--include-media/--no-media", help="Also bundle every referenced frame and clip")] = False,
    config: ConfigOpt = None,
    set_: SetOpt = None,
) -> None:
    """Bundle the exported dataset into ONE zip file for upload (optionally with frames and clips)."""
    from video_dataset.dataset.package import package_dataset

    _apply_local_options(config, set_)
    summary = package_dataset(_config(), Path(output), include_media)
    console.print(f"wrote {summary['output']} ({summary['size_bytes'] / 1e6:.1f} MB): {', '.join(summary['dataset_files'])}")
    if include_media:
        console.print(f"media files: {summary['media_files']} ({summary['media_bytes'] / 1e6:.1f} MB), missing: {summary['media_missing']}")


@app.command()
def stats(config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """Print dataset statistics from the last export."""
    from video_dataset.dataset.aggregate import load_statistics
    from video_dataset.dataset.stats import format_statistics

    _apply_local_options(config, set_)
    cfg = _config()
    s = load_statistics(cfg)
    if s is None:
        console.print("No statistics yet. Run `video-dataset export` first.")
        raise typer.Exit(1)
    console.print(format_statistics(s))
    console.print(f"\nfull statistics: {cfg.export_dir / 'statistics.json'}")


@app.command("show-config")
def show_config(config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """Print the effective merged configuration."""
    _apply_local_options(config, set_)
    console.print(dump_config(_config()))


@app.command()
def stages() -> None:
    """List pipeline stages in order."""
    for i, s in enumerate(STAGE_ORDER, start=1):
        console.print(f"{i:2d}. {s.value}")


@app.command()
def doctor(config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """Check the environment: ffmpeg, GPU, optional model libraries."""
    _apply_local_options(config, set_)
    cfg = _config()
    from video_dataset.utils.device import resolve_device
    from video_dataset.utils.ffmpeg import find_ffmpeg, find_ffprobe

    t = Table(title=f"video-dataset {__version__} environment")
    t.add_column("component")
    t.add_column("status")

    def check(name: str, fn):  # type: ignore[no-untyped-def]
        try:
            val = fn()
            t.add_row(name, f"[green]{val}[/green]")
        except Exception as exc:
            t.add_row(name, f"[red]missing[/red] ({str(exc)[:60]})")

    t.add_row("python", sys.version.split()[0])
    check("ffmpeg", lambda: find_ffmpeg(cfg.project.ffmpeg_path))
    check("ffprobe", lambda: find_ffprobe(cfg.project.ffprobe_path) or "not found (OpenCV fallback)")
    check("yt-dlp", lambda: __import__("yt_dlp").version.__version__)
    check("scenedetect", lambda: __import__("scenedetect").__version__)
    check("opencv", lambda: __import__("cv2").__version__)
    check("torch / device", lambda: f"{__import__('torch').__version__} -> {resolve_device(cfg.project.device)}")
    check("faster-whisper", lambda: __import__("faster_whisper").__version__)
    check("transformers", lambda: __import__("transformers").__version__)
    check("rapidocr", lambda: "installed" if __import__("rapidocr_onnxruntime") else "")
    check("anthropic sdk", lambda: __import__("anthropic").__version__)
    check("openai sdk", lambda: __import__("openai").__version__)
    check("datasketch", lambda: __import__("datasketch").__version__)
    t.add_row("data dir", str(cfg.data_dir))
    free = shutil.disk_usage(cfg.data_dir if cfg.data_dir.exists() else Path.cwd()).free / 1e9
    t.add_row("free disk", f"{free:.1f} GB")
    t.add_row("vision provider", f"{cfg.vision.provider} ({cfg.vision.model or 'n/a'})")
    t.add_row("asr provider", f"{cfg.transcription.provider} ({cfg.transcription.model})")
    t.add_row("ocr provider", cfg.ocr.provider)
    console.print(t)


@app.command()
def clean(video_id: Annotated[str, typer.Argument(help="Video id to forget (artifacts are NOT deleted)")]) -> None:
    """Remove a video's checkpoint state so it is fully re-processed next run."""
    runner = _runner()
    try:
        runner.db.delete_video(video_id)
        console.print(f"removed state for {video_id} (files left in place)")
    finally:
        runner.close()


if __name__ == "__main__":  # pragma: no cover
    app()
