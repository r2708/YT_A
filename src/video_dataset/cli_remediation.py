"""Release-remediation commands (issue doc sections 3, 5, 6, 7 and P1-07 / P1-09): inventory and
checksums, release comparison, repair queues, lineage, stratified human review with agreement,
camera confusion matrix, task exports and the benchmark. Registered on the main ``video-dataset``
app by ``cli.py`` so every command shares ``--config`` / ``--set``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

console = Console()
ConfigOpt = Annotated[str | None, typer.Option("--config", "-c", help="Path to a YAML config overriding config/default.yaml")]
SetOpt = Annotated[list[str] | None, typer.Option("--set", "-s", help="Override a config value")]
FinalOpt = Annotated[str | None, typer.Option("--final-dir", help="Dataset directory (default: data/final)")]


def register(app: typer.Typer, config_loader: Any, apply_local_options: Any) -> None:
    """``config_loader()`` returns the PipelineConfig, ``apply_local_options(config, set_)`` applies the CLI overrides."""

    def _final(final_dir: str | None, config: str | None, set_: list[str] | None) -> tuple[Any, Path]:
        apply_local_options(config, set_)
        cfg = config_loader()
        return cfg, (Path(final_dir) if final_dir else cfg.export_dir)

    def _records(final: Path) -> dict[str, list[dict[str, Any]]]:
        from video_dataset.dataset.release import load_final

        records = load_final(final)
        if not any(records.values()):
            console.print(f"[red]no exported dataset under {final}; run `video-dataset export` first[/red]")
            raise typer.Exit(1)
        return records

    # ------------------------------------------------------------------ inventory / checksums
    @app.command()
    def inventory(
        final_dir: FinalOpt = None,
        include_media: Annotated[bool, typer.Option("--include-media/--no-media", help="Also hash frames/ and clips/ under the directory")] = False,
        workers: Annotated[int, typer.Option("--workers", help="Hashing threads")] = 2,
        config: ConfigOpt = None,
        set_: SetOpt = None,
    ) -> None:
        """Write inventory.json (path, size, sha256, format, rows, schema versions, record type per
        file) and checksums.sha256 (sha256sum format) for a release / shard directory. Streaming;
        never loads a file into memory."""
        from video_dataset.dataset.inventory import write_inventory

        _cfg, final = _final(final_dir, config, set_)
        inv = write_inventory(final, include_media=include_media, workers=workers)
        t = inv["totals"]
        console.print(f"{t['files']} files, {t['bytes'] / 1e6:.1f} MB, rows by type {json.dumps(t.get('rows_by_type'))}, schema versions {json.dumps(t.get('schema_versions'))}")
        console.print(f"written: {final / 'inventory.json'}, {final / 'checksums.sha256'}")

    @app.command("verify-checksums")
    def verify_checksums_cmd(final_dir: FinalOpt = None, config: ConfigOpt = None, set_: SetOpt = None) -> None:
        """Re-hash every file listed in checksums.sha256 and report mismatched / missing / unlisted files (exit 1 on any)."""
        from video_dataset.dataset.inventory import verify_checksums

        _cfg, final = _final(final_dir, config, set_)
        res = verify_checksums(final)
        console.print(f"checked {res['checked']}: mismatched {len(res['mismatched'])}, missing {len(res['missing'])}, unlisted {len(res['unlisted'])}")
        for key in ("mismatched", "missing", "unlisted"):
            for p in res[key][:20]:
                console.print(f"  {key}: {p}")
        if not res["ok"]:
            raise typer.Exit(1)

    # ------------------------------------------------------------------ compare / repair / lineage
    @app.command("compare-releases")
    def compare_releases_cmd(
        old_dir: Annotated[str, typer.Argument(help="Previous release / shard directory (the frozen baseline)")],
        new_dir: Annotated[str, typer.Argument(help="Candidate release directory")],
        allow_split_changes: Annotated[bool, typer.Option("--allow-split-changes", help="Do not fail when a video moved between splits")] = False,
        tolerance: Annotated[float, typer.Option("--tolerance", help="Allowed fraction of unexplained lost records per type")] = 0.0,
        as_json: Annotated[bool, typer.Option("--json")] = False,
    ) -> None:
        """Compare two releases (counts, videos, record ids, statuses, lost ids not explained by
        removed_sources.json, duplicates, temporal failures, split moves, missing media, checksums).
        Exit 1 when the candidate loses records unexplained or regresses."""
        from video_dataset.dataset.compare import compare_releases, format_comparison

        rep = compare_releases(Path(old_dir), Path(new_dir), allow_split_changes=allow_split_changes, tolerance=tolerance)
        console.print(json.dumps(rep, indent=2) if as_json else format_comparison(rep))
        if not rep["ok"]:
            raise typer.Exit(1)

    @app.command("repair-queues")
    def repair_queues_cmd(
        final_dir: FinalOpt = None,
        shard: Annotated[str | None, typer.Option("--shard", help="Shard name recorded on every row")] = None,
        config: ConfigOpt = None,
        set_: SetOpt = None,
    ) -> None:
        """Write final/repairs/<queue>.jsonl: missing media / frames, invalid timestamps, broken
        evidence, schema failures, duplicates, low-confidence or ambiguous camera, unverified
        descriptions, rights-unverified sources; each row names a proposed repair action."""
        from video_dataset.dataset.repair import build_repair_queues

        cfg, final = _final(final_dir, config, set_)
        counts = build_repair_queues(final, cfg.data_dir, shard=shard, release_rights=tuple(cfg.licensing.release_rights or ()))
        t = Table(title=f"repair queues of {final.name}")
        t.add_column("queue")
        t.add_column("records", justify="right")
        for k, v in counts.items():
            t.add_row(k, str(v))
        console.print(t)
        console.print(f"written: {final / 'repairs'}")

    @app.command()
    def lineage(final_dir: FinalOpt = None, config: ConfigOpt = None, set_: SetOpt = None) -> None:
        """Summarise final/lineage.jsonl (every status change with reason, actor, policy version)."""
        from video_dataset.dataset.repair import lineage_summary

        _cfg, final = _final(final_dir, config, set_)
        console.print(json.dumps(lineage_summary(final), indent=2))

    # ------------------------------------------------------------------ human review
    @app.command("review-sample")
    def review_sample_cmd(
        final_dir: FinalOpt = None,
        n: Annotated[int | None, typer.Option("--n", help="Sample size (default: Cochran 95% / 5% margin over the exported records)")] = None,
        double_fraction: Annotated[float, typer.Option("--double-fraction", help="Share of the sample labelled by two reviewers")] = 0.2,
        reviewers: Annotated[str, typer.Option("--reviewers", help="Comma-separated reviewer ids")] = "reviewer_a,reviewer_b",
        per_video_cap: Annotated[int | None, typer.Option("--per-video-cap")] = None,
        split: Annotated[str | None, typer.Option("--split", help="Restrict to one split (e.g. test for the benchmark)")] = None,
        config: ConfigOpt = None,
        set_: SetOpt = None,
    ) -> None:
        """Draw a stratified human verification sample (record type x tier x failure category) with a
        double-annotated share -> final/reviews/sample.jsonl. Fill label / reviewer / camera_movement_label /
        boundary_ok per row, then run `review-agreement`."""
        from video_dataset.dataset.human_review import sample_size, stratified_sample, write_sample

        cfg, final = _final(final_dir, config, set_)
        records = _records(final)
        population = sum(len(v) for v in records.values())
        size = int(n) if n else sample_size(population)
        rows = stratified_sample(records, size, double_fraction=double_fraction, reviewers=tuple(r.strip() for r in reviewers.split(",") if r.strip()), per_video_cap=per_video_cap, seed=int(cfg.qa.seed), split=split)
        out = write_sample(rows, final)
        strata: dict[str, int] = {}
        for r in rows:
            k = r["stratum"].get("failure_category", "?")
            strata[k] = strata.get(k, 0) + 1
        console.print(f"{len(rows)} records sampled from {population} (requested {size}); double-annotated {sum(1 for r in rows if r.get('double_annotated'))}; by failure category {json.dumps(strata)}")
        console.print(f"written: {out}")

    @app.command("review-agreement")
    def review_agreement_cmd(
        final_dir: FinalOpt = None,
        adjudicate_file: Annotated[str | None, typer.Option("--adjudications", help="JSON {record_id: label} resolving disagreements")] = None,
        write_labels: Annotated[bool, typer.Option("--write-labels", help="Merge the adjudicated labels into reviews/labels.jsonl (applied by the next export)")] = False,
        config: ConfigOpt = None,
        set_: SetOpt = None,
    ) -> None:
        """Measure inter-annotator agreement (percent, Cohen's kappa), camera-label agreement and the
        boundary audit over the labelled sample; list unresolved disagreements; optionally adjudicate."""
        from video_dataset.dataset.human_review import (
            LABELS_FILE,
            SAMPLE_FILE,
            adjudicate,
            agreement,
            read_labelled,
            read_sample,
            verification_report,
            write_adjudicated,
        )

        _cfg, final = _final(final_dir, config, set_)
        sample_path = final / SAMPLE_FILE
        if not sample_path.exists():
            console.print("no reviews/sample.jsonl; run `video-dataset review-sample` first")
            raise typer.Exit(1)
        sample = read_sample(sample_path)
        labelled = read_labelled(sample_path) + (read_labelled(final / LABELS_FILE) if (final / LABELS_FILE).exists() else [])
        rep = verification_report(sample, labelled)
        ag = rep.get("agreement") or agreement(labelled)
        console.print(f"sample {rep.get('sample_size')}, labelled {rep.get('labelled')}, multi-reviewed {ag.get('records_multi_reviewed')}, agreement {ag.get('percent_agreement')}, kappa binary {ag.get('kappa_binary')}, kappa 3-class {ag.get('kappa_multiclass')}, disagreement rate {ag.get('disagreement_rate')}")
        console.print(f"camera agreement: {json.dumps(ag.get('camera_agreement'))}; boundary audit: {json.dumps(ag.get('boundary_audit'))}")
        if ag.get("disagreements"):
            console.print("unresolved: " + ", ".join(ag["disagreements"][:20]))
        adj = json.loads(Path(adjudicate_file).read_text()) if adjudicate_file else None
        final_labels = adjudicate(labelled, adj)
        if write_labels:
            out = write_adjudicated(final, final_labels)
            console.print(f"labels merged into {out}; run `video-dataset export` to apply them")
        (final / "reviews").mkdir(parents=True, exist_ok=True)
        (final / "reviews" / "agreement.json").write_text(json.dumps(rep, indent=2))
        console.print(f"written: {final / 'reviews' / 'agreement.json'}")

    # ------------------------------------------------------------------ exports / benchmark
    @app.command("export-tasks")
    def export_tasks_cmd(
        final_dir: FinalOpt = None,
        tasks: Annotated[str | None, typer.Option("--tasks", help="Comma-separated subset of understanding,temporal_reasoning,cinematic,generation_ready")] = None,
        config: ConfigOpt = None,
        set_: SetOpt = None,
    ) -> None:
        """Write final/exports/<task>/ (records + filters.json + README) for each task-specific
        training export. generation_ready needs decodable, rights-cleared media and is otherwise
        empty with the reasons recorded."""
        from video_dataset.dataset.task_exports import write_task_exports
        from video_dataset.utils.io import read_jsonl

        cfg, final = _final(final_dir, config, set_)
        records = _records(final)
        available = {str(r["path"]) for r in read_jsonl(final / "media_manifest.jsonl") if r.get("exists")} if (final / "media_manifest.jsonl").exists() else set()
        rights = {str(r["video_id"]): str(r.get("rights_status") or "unknown") for r in read_jsonl(final / "provenance.jsonl")} if (final / "provenance.jsonl").exists() else {}
        summary = write_task_exports(records, final, cfg, media_available=lambda rec: str(rec.get("clip_path") or rec.get("frame_path") or "") in available, rights=rights, release_rights=list(cfg.licensing.release_rights or []), tasks=[t.strip() for t in tasks.split(",")] if tasks else None)
        t = Table(title="task exports")
        t.add_column("task")
        t.add_column("eligible", justify="right")
        for k, v in summary.items():
            t.add_row(k, str(v.get("eligible_total", 0)))
        console.print(t)
        console.print(f"written: {final / 'exports'}")

    @app.command()
    def benchmark(
        final_dir: FinalOpt = None,
        split: Annotated[str, typer.Option("--split")] = "test",
        allow_unverified: Annotated[bool, typer.Option("--allow-unverified", help="Include accepted records without a human label (not a real benchmark)")] = False,
        config: ConfigOpt = None,
        set_: SetOpt = None,
    ) -> None:
        """Build final/benchmark/: fixed, held-out, human-verified evaluation items per task with
        versioned metric definitions and guidelines. Empty (with a note) until human labels exist."""
        from video_dataset.dataset.benchmark import build_benchmark

        _cfg, final = _final(final_dir, config, set_)
        man = build_benchmark(_records(final), final, split=split, require_human=not allow_unverified)
        console.print(json.dumps({k: man.get(k) for k in ("version", "split", "require_human", "counts", "frozen_hash", "note")}, indent=2))
        console.print(f"written: {final / 'benchmark'}")
