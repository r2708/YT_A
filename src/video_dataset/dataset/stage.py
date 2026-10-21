"""EXPORT stage (per video): build dataset records and write them under final/per_video/<video_id>/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from video_dataset.dataset.builders import DatasetBuilder
from video_dataset.dataset.coverage import stage_coverage
from video_dataset.dataset.curation import annotate_records, rights_status_for
from video_dataset.dataset.export import upgrade_records, write_jsonl_files
from video_dataset.pipeline.context import StageOutput, VideoContext
from video_dataset.schemas.ocr import OCRResult
from video_dataset.schemas.scene import FrameSamplingResult, SceneDetectionResult
from video_dataset.schemas.transcript import Transcript
from video_dataset.schemas.validated import ValidatedVideo
from video_dataset.stages import Stage
from video_dataset.utils.environment import environment_manifest
from video_dataset.utils.io import read_json, write_json_atomic


def per_video_export_dir(ctx: VideoContext) -> Path:
    return ctx.config.export_dir / "per_video" / ctx.video_id


def _load(path: Path, model):  # type: ignore[no-untyped-def]
    if not path.exists():
        return None
    data = read_json(path)
    return model.model_validate(data) if model is not None else data


def _retrieved_at(ctx: VideoContext) -> str | None:
    """When the source file was fetched (mtime of the download metadata, ISO-8601 UTC)."""
    from datetime import UTC, datetime

    p = ctx.paths.metadata_file(ctx.video_id)
    try:
        return datetime.fromtimestamp(p.stat().st_mtime, tz=UTC).isoformat(timespec="seconds")
    except OSError:
        return None


def export_stage(ctx: VideoContext) -> StageOutput:
    log = ctx.logger(Stage.EXPORT)
    cfg = ctx.config
    validated_path = ctx.paths.validated_file(ctx.video_id)
    if not validated_path.exists():
        raise FileNotFoundError("validated annotations missing; run VALIDATION first")
    validated = ValidatedVideo.model_validate(read_json(validated_path))
    scenes = SceneDetectionResult.model_validate(read_json(ctx.paths.scenes_file(ctx.video_id))).scenes
    sampling = FrameSamplingResult.model_validate(read_json(ctx.paths.frames_file(ctx.video_id)))
    transcript = _load(ctx.paths.transcript_file(ctx.video_id), Transcript)
    ocr = _load(ctx.paths.ocr_file(ctx.video_id), OCRResult)

    builder = DatasetBuilder(
        cfg.export.include_review, cfg.export.include_rejected, ctx.config.data_dir if cfg.export.relative_paths else None,
        hard_negatives=bool(cfg.export.hard_negatives), negatives_per_record=int(cfg.export.hard_negatives_per_record), seed=int(cfg.qa.seed),
    )
    temporal_qa, long_qa = builder.qa(validated)
    records: dict[str, list[Any]] = {
        "frames": builder.frames(validated, sampling.frames, ocr),
        "clips": builder.clips(validated, sampling.clips, transcript),
        "scenes": builder.scenes(validated, sampling.clips, transcript, ocr, scenes),
        "sequences": [],
        "events": builder.events(validated),
        "temporal_qa": temporal_qa,
        "long_video_qa": long_qa,
        "video_descriptions": builder.video_descriptions(validated, scenes, ctx.metadata.title),
    }
    rows = upgrade_records(records)  # plain dicts with record_type; split / tier / subsets filled below
    try:
        media = ctx.media_info
        media_summary: dict[str, Any] = {"width": media.width, "height": media.height, "fps": media.fps, "duration": media.duration, "bitrate": media.bitrate, "codec": media.video_codec}
        video_flags = list(media.quality_flags)
    except FileNotFoundError:
        media_summary, video_flags = {}, []
    coverage = stage_coverage(ctx, transcript, ocr, validated)
    rights = rights_status_for(ctx.metadata.license_status, str(ctx.metadata.source_type))
    video_info = {ctx.video_id: {"canonical_video_id": sampling.near_duplicate_of, "license_status": ctx.metadata.license_status, "rights_status": rights, "quality_flags": video_flags, "coverage": coverage}}
    curation = annotate_records(rows, cfg.export, video_info, cfg.licensing, seed=int(cfg.qa.seed))
    out_dir = per_video_export_dir(ctx)
    counts = write_jsonl_files(rows, out_dir)
    download = _load(ctx.paths.download_record_file(ctx.video_id), None) if ctx.paths.download_record_file(ctx.video_id).exists() else None
    manifest = {
        "video_id": ctx.video_id,
        "title": ctx.metadata.title,
        "url": ctx.metadata.url,
        "youtube_id": ctx.metadata.youtube_id,
        "channel": ctx.metadata.channel,
        "upload_date": ctx.metadata.upload_date,
        "duration": ctx.duration,
        "counts": counts,
        "split": next((r.get("split") for rs in rows.values() for r in rs if r.get("split")), None),
        "curation": curation,
        "validation_summary": validated.summary,
        "transcript_segments": len(transcript.segments) if transcript else 0,
        "transcript_status": getattr(transcript, "status", None) if transcript else "not_run",
        "ocr_tracks": len(ocr.tracks) if ocr else 0,
        "coverage": coverage,
        "files": {k: f"per_video/{ctx.video_id}/{k}.jsonl" for k in counts},  # relative to the export root: portable
        # video-level facts the aggregation needs without the working files
        "license": ctx.metadata.license,
        "license_status": ctx.metadata.license_status,
        "rights_status": rights,
        "source_sha256": (download or {}).get("sha256") if isinstance(download, dict) else None,
        "retrieved_at": _retrieved_at(ctx),
        "media": media_summary,
        "quality_flags": video_flags,
        "fingerprint": list(sampling.fingerprint),
        "near_duplicate_of": sampling.near_duplicate_of,
        "near_duplicate_similarity": sampling.near_duplicate_similarity,
        "duplicate_frames_removed": sampling.duplicate_frames_removed,
        "sampling": {"analysis_fps": cfg.frame_sampling.analysis_fps, "max_frame_gap_seconds": cfg.frame_sampling.max_frame_gap_seconds, "min_frames_per_scene": cfg.frame_sampling.min_frames_per_scene, "max_frames_per_scene": cfg.frame_sampling.max_frames_per_scene, "motion_aware": cfg.frame_sampling.motion_aware, "clip_codec": cfg.frame_sampling.clip_codec, "verify_clip_boundaries": cfg.frame_sampling.verify_clip_boundaries},
        "environment": environment_manifest(cfg),
    }
    write_json_atomic(out_dir / "manifest.json", manifest)
    log.info("exported %s", counts)
    return StageOutput(artifact_path=str(out_dir), metrics=counts, message=", ".join(f"{k}={v}" for k, v in counts.items()))
