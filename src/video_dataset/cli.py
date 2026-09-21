"""video-dataset command line interface."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
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


def load_env_file(path: str | Path | None = None) -> list[str]:
    """Export the non-empty entries of `.env` (API keys, HF_TOKEN, HF_HOME ...) that are not already set.

    Blank lines such as `HF_HOME=` from a copied .env.example are skipped on purpose: exporting an empty
    HF_HOME makes huggingface_hub cache models in ./hub inside the project. Returns the names set."""
    try:
        from dotenv import dotenv_values, find_dotenv
    except Exception as exc:  # pragma: no cover
        log.debug("python-dotenv unavailable: %s", exc)
        return []
    env_path = str(path) if path else find_dotenv(usecwd=True)
    if not env_path or not Path(env_path).exists():
        return []
    applied: list[str] = []
    for key, value in dotenv_values(env_path).items():
        if not key or value is None or not str(value).strip():
            continue
        if os.environ.get(key):
            continue  # the real environment wins
        os.environ[key] = str(value)
        applied.append(key)
    return applied


def _config() -> PipelineConfig:
    if state.config is None:
        load_env_file()
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


def _fmt_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    s = int(max(0, seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _print_results(results, elapsed: float | None = None) -> None:  # type: ignore[no-untyped-def]
    table = Table(title="Run summary" + (f" ({_fmt_seconds(elapsed)})" if elapsed is not None else ""))
    table.add_column("video_id")
    table.add_column("result")
    table.add_column("stages run", justify="right")
    table.add_column("cached", justify="right")
    table.add_column("time", justify="right")
    table.add_column("error")
    for r in results:
        table.add_row(
            r.video_id,
            "[green]ok[/green]" if r.completed else f"[red]failed at {r.failed_stage}[/red]",
            str(len(r.stages_run)),
            str(len(r.stages_cached)),
            _fmt_seconds(r.seconds),
            (r.error or "")[:80],
        )
    console.print(table)
    if results:
        failed = [r for r in results if not r.completed]
        console.print(f"{len(results) - len(failed)}/{len(results)} videos completed" + (f", [red]{len(failed)} failed[/red]" if failed else ""))


def _progress_printer():  # type: ignore[no-untyped-def]
    """Callback for PipelineRunner.run_batch: one line per finished video with counts, elapsed and ETA."""

    def on_progress(ev) -> None:  # type: ignore[no-untyped-def]
        mark = "[green]ok[/green]" if ev.result.completed else f"[red]failed ({ev.result.failed_stage})[/red]"
        eta = ev.eta_seconds
        console.print(
            f"[bold][{ev.phase} {ev.done}/{ev.total}][/bold] {ev.video_id} {mark}  "
            f"failed={ev.failed}  elapsed={_fmt_seconds(ev.elapsed)}  eta={_fmt_seconds(eta)}",
            highlight=False,
        )

    return on_progress


def _maybe_upload(cfg: PipelineConfig, force: bool = False, dry_run: bool = False) -> bool:
    """Run the threshold-triggered Hugging Face upload; prints the outcome, never raises inside `run`."""
    from video_dataset.dataset.upload import UploadError, check_and_upload

    if (cfg.upload.provider or "none").lower() == "none" and not force:
        return False
    try:
        summary = check_and_upload(cfg, force=force, dry_run=dry_run)
    except UploadError as exc:
        console.print(f"[red]upload failed: {exc}[/red]")
        return False
    except Exception as exc:
        console.print(f"[red]upload failed: {type(exc).__name__}: {exc}[/red]")
        log.debug("upload traceback", exc_info=True)
        return False
    if summary is None:
        return False
    if summary.get("dry_run"):
        console.print(f"[cyan]dry run:[/cyan] would upload {summary['shard']} ({summary['bytes'] / 1e6:.1f} MB, {summary['files']} files, {len(summary['videos'])} videos) to {summary['repo_id']}/{summary['path_in_repo']}")
        return False
    console.print(
        f"[green]uploaded {summary['shard']}[/green] ({summary['bytes'] / 1e6:.1f} MB, {len(summary['videos'])} videos) "
        f"to {summary['repo_id']}/{summary['path_in_repo']} in {summary['seconds']:.0f}s -> {summary['commit']}"
    )
    console.print(f"local copy {'archived under ' + str(cfg.export_dir / 'uploaded' / summary['shard']) if summary['after_upload'] == 'archive' else 'deleted'}; final/ starts a new shard")
    return True


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
    t0 = time.time()
    try:
        registered = runner.register_inputs(items)
        results = runner.run_batch(
            registered,  # type: ignore[arg-type]
            force_from=parse_stage(force_from) if force_from else None,
            until=parse_stage(until) if until else None,
            workers=workers,
            progress=_progress_printer() if len(registered) > 1 else None,
        )
        _print_results(results, elapsed=time.time() - t0)
        if not no_aggregate and (until is None or parse_stage(until) == Stage.EXPORT):
            stats = aggregate_exports(runner.config, runner.db)
            console.print(f"Final dataset written to {runner.config.export_dir} ({stats['temporal_qa']} temporal QA, {stats['long_video_qa']} long-video QA)")
            _maybe_upload(runner.config)
        raise typer.Exit(0 if all(r.completed for r in results) else 2)
    finally:
        runner.close()


@app.command()
def resume(
    workers: Annotated[int | None, typer.Option("--workers", "-w", help="Parallel videos for CPU stages")] = None,
    until: Annotated[str | None, typer.Option("--until", help="Stop after this stage")] = None,
    include_rejected: Annotated[bool, typer.Option("--include-rejected", help="Also retry videos rejected by URL/duration/size limits")] = False,
    no_aggregate: Annotated[bool, typer.Option("--no-aggregate", help="Skip merging per-video exports into final/")] = False,
    config: ConfigOpt = None,
    set_: SetOpt = None,
) -> None:
    """Continue every unfinished video after a crash or shutdown - no URL file needed (inputs come from state.db)."""
    from video_dataset.dataset.aggregate import aggregate_exports

    _apply_local_options(config, set_)
    runner = _runner()
    t0 = time.time()
    try:
        pending = runner.unfinished_videos(include_rejected)
        if not pending:
            console.print("Nothing to resume: every registered video is done.")
            return
        missing = [vid for vid, item in pending if item is None and not runner.paths.source_candidates(vid)]
        for vid in missing:
            console.print(f"[yellow]{vid}: no stored input and no download on disk; it will fail at DOWNLOAD[/yellow]")
        console.print(f"Resuming {len(pending)} unfinished video(s)")
        results = runner.run_batch(
            pending, until=parse_stage(until) if until else None, workers=workers,
            progress=_progress_printer() if len(pending) > 1 else None,
        )
        _print_results(results, elapsed=time.time() - t0)
        if not no_aggregate and (until is None or parse_stage(until) == Stage.EXPORT):
            stats = aggregate_exports(runner.config, runner.db)
            console.print(f"Final dataset written to {runner.config.export_dir} ({stats['temporal_qa']} temporal QA, {stats['long_video_qa']} long-video QA)")
            _maybe_upload(runner.config)
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


def _status_renderable(runner, video_id: str | None):  # type: ignore[no-untyped-def]
    """Build the rich renderable for `status` (shared by the one-shot and --watch modes)."""
    from rich.console import Group

    rows = runner.status_rows(video_id)
    if not rows:
        return "No videos registered yet. Use `video-dataset run <urls.txt>`."
    if video_id:
        row = rows[0]
        t = Table(show_header=True, title=f"{row['video_id']}  {row.get('title') or ''}  status={row['status']}")
        t.add_column("stage")
        t.add_column("status")
        t.add_column("metrics / error")
        for st in STAGE_ORDER:
            s = row["stages"][st.value]
            color = {"DONE": "green", "FAILED": "red", "RUNNING": "yellow", "SKIPPED": "cyan"}.get(s, "white")
            detail = row["errors"].get(st.value) or json.dumps(row["metrics"].get(st.value, {}), default=str)[:100]
            t.add_row(st.value, f"[{color}]{s}[/{color}]", detail)
        return t
    t = Table(title=f"{len(rows)} video(s)")
    t.add_column("video_id")
    t.add_column("title")
    t.add_column("status")
    for st in STAGE_ORDER:
        t.add_column(st.value.split("_")[0][:6], justify="center")
    marks = {"DONE": "[green]✓[/green]", "FAILED": "[red]✗[/red]", "RUNNING": "[yellow]…[/yellow]", "SKIPPED": "[cyan]-[/cyan]", "PENDING": "·"}
    for row in rows:
        t.add_row(row["video_id"], (row.get("title") or "")[:32], row["status"], *[marks.get(row["stages"][st.value], "·") for st in STAGE_ORDER])
    summary = runner.db.summary()
    running = sum(n for per in summary["stages"].values() for s, n in per.items() if s == "RUNNING")
    done = summary["videos_by_status"].get("done", 0)
    line = f"videos by status: {summary['videos_by_status']}  |  {done}/{summary['videos_total']} done, {running} stage(s) running"
    return Group(t, line)


@app.command()
def status(
    video_id: Annotated[str | None, typer.Argument(help="Show one video in detail")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine readable output")] = False,
    watch: Annotated[float | None, typer.Option("--watch", "-w", help="Refresh every N seconds (monitor a running batch from another terminal)")] = None,
    config: ConfigOpt = None,
    set_: SetOpt = None,
) -> None:
    """Show per-video stage status (the checkpoint table); --watch keeps it refreshing."""
    _apply_local_options(config, set_)
    runner = _runner()
    try:
        if as_json:
            console.print_json(json.dumps(runner.status_rows(video_id), default=str))
            return
        if watch:
            from rich.live import Live

            interval = max(0.5, float(watch))
            with Live(_status_renderable(runner, video_id), console=console, refresh_per_second=4, screen=False) as live:
                try:
                    while True:
                        time.sleep(interval)
                        live.update(_status_renderable(runner, video_id))
                except KeyboardInterrupt:
                    pass
            return
        console.print(_status_renderable(runner, video_id))
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
        if not output:
            _maybe_upload(runner.config)
    finally:
        runner.close()


@app.command()
def upload(
    force: Annotated[bool, typer.Option("--force", help="Upload the current shard even if it is below upload.threshold_mb")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", "-n", help="Show what would be uploaded")] = False,
    status: Annotated[bool, typer.Option("--status", help="Show uploaded shards and the current shard size")] = False,
    config: ConfigOpt = None,
    set_: SetOpt = None,
) -> None:
    """Upload the exported dataset to the Hugging Face Hub (automatic once final/ reaches upload.threshold_mb)."""
    from video_dataset.dataset.upload import UploadState, measure_shard

    _apply_local_options(config, set_)
    cfg = _config()
    if status or dry_run:
        size, files, videos = measure_shard(cfg.export_dir, cfg.data_dir, bool(cfg.upload.include_media))
        st = UploadState.load(cfg.export_dir)
        console.print(f"provider={cfg.upload.provider} repo={cfg.upload.repo_id or '-'} threshold={cfg.upload.threshold_mb:.0f} MB include_media={cfg.upload.include_media} after_upload={cfg.upload.after_upload}")
        console.print(f"current shard: {size / 1e6:.1f} MB, {files} files, {len(videos)} video(s)  ->  next {'upload' if size >= cfg.upload.threshold_mb * 1024 * 1024 else 'threshold not reached'}")
        if st.shards:
            t = Table(title=f"{len(st.shards)} uploaded shard(s)")
            for col in ("shard", "videos", "MB", "uploaded_at", "path_in_repo", "local"):
                t.add_column(col)
            for s in st.shards:
                t.add_row(s["name"], str(len(s.get("videos", []))), f"{s.get('bytes', 0) / 1e6:.1f}", s.get("uploaded_at", ""), s.get("path_in_repo", ""), s.get("after_upload", ""))
            console.print(t)
        if status:
            return
    if (cfg.upload.provider or "none").lower() == "none":
        console.print("[red]upload.provider is 'none'. Set --set upload.provider=huggingface --set upload.repo_id=user/name (and HF_TOKEN).[/red]")
        raise typer.Exit(1)
    ok = _maybe_upload(cfg, force=force, dry_run=dry_run)
    if not ok and not dry_run:
        console.print("nothing uploaded (below threshold, nothing exported, or an error above); use --force to upload now")
        raise typer.Exit(1)


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
def clean(
    video_ids: Annotated[list[str] | None, typer.Argument(help="Video id(s) to forget (state only unless --files)")] = None,
    files: Annotated[bool, typer.Option("--files", help="With VIDEO_ID: also delete every artifact of that video")] = False,
    intermediate: Annotated[bool, typer.Option("--intermediate", help="Delete original downloads and audio.wav for videos whose EXPORT is done")] = False,
    orphans: Annotated[bool, typer.Option("--orphans", help="Delete artifacts of video ids no longer in the state DB")] = False,
    failed: Annotated[bool, typer.Option("--failed", help="Delete artifacts of videos whose DOWNLOAD/PREPROCESS failed")] = False,
    logs: Annotated[bool, typer.Option("--logs", help="Delete data/logs/*.log")] = False,
    report: Annotated[bool, typer.Option("--report", help="Only print disk usage per data sub-directory")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", "-n", help="Show what would be deleted, delete nothing")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation")] = False,
    config: ConfigOpt = None,
    set_: SetOpt = None,
) -> None:
    """Free disk space or reset videos. Plans first, deletes only after confirmation (see README > Cleanup)."""
    from video_dataset.storage import cleanup as cl

    _apply_local_options(config, set_)
    runner = _runner()
    try:
        paths = runner.paths
        if report:
            t = Table(title=f"disk usage under {paths.root}")
            t.add_column("directory")
            t.add_column("size", justify="right")
            total = 0
            for sub, size in sorted(cl.data_dir_usage(paths).items(), key=lambda kv: -kv[1]):
                t.add_row(sub, f"{size / 1e9:.2f} GB")
                total += size
            t.add_row("[bold]total[/bold]", f"[bold]{total / 1e9:.2f} GB[/bold]")
            console.print(t)
            console.print(f"free: {shutil.disk_usage(paths.root).free / 1e9:.1f} GB")
            return

        plan = cl.CleanupPlan()
        forget: list[str] = []
        for vid in video_ids or []:
            if runner.db.get_video(vid) is None and not files:
                console.print(f"[yellow]{vid} is not registered[/yellow]")
                continue
            forget.append(vid)
            if files:
                plan.extend(cl.plan_video(paths, vid))
        if intermediate:
            plan.extend(cl.plan_intermediate(paths, runner.db, video_ids or None))
        if orphans:
            plan.extend(cl.plan_orphans(paths, runner.db))
        if failed:
            plan.extend(cl.plan_failed(paths, runner.db))
        if logs:
            plan.extend(cl.plan_logs(paths))
        if not forget and not plan.paths:
            console.print("Nothing to do. Pass VIDEO_ID, --intermediate, --orphans, --failed, --logs or --report.")
            for note in plan.notes[:10]:
                console.print(f"  {note}")
            return

        for note in plan.notes[:10]:
            console.print(f"[dim]{note}[/dim]")
        if forget:
            console.print(f"forget state of {len(forget)} video(s): {', '.join(forget)}")
        if plan.paths:
            console.print(f"delete {len(plan.paths)} path(s), {plan.bytes / 1e9:.2f} GB:")
            for p in plan.paths[:40]:
                console.print(f"  {p}")
            if len(plan.paths) > 40:
                console.print(f"  ... and {len(plan.paths) - 40} more")
        if dry_run:
            console.print("[cyan]dry run: nothing deleted[/cyan]")
            return
        if not yes and not typer.confirm("Proceed?", default=False):
            console.print("aborted")
            raise typer.Exit(1)
        for vid in forget:
            runner.db.delete_video(vid)
        removed, freed = cl.apply_plan(plan)
        console.print(f"removed {removed} path(s), freed {freed / 1e9:.2f} GB" + (f"; forgot {len(forget)} video(s)" if forget else ""))
    finally:
        runner.close()


if __name__ == "__main__":  # pragma: no cover
    app()
