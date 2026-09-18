"""EXPORT stage (per video): build dataset records and write them under final/per_video/<video_id>/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from video_dataset.dataset.builders import DatasetBuilder
from video_dataset.dataset.export import write_jsonl_files
from video_dataset.pipeline.context import StageOutput, VideoContext
from video_dataset.schemas.ocr import OCRResult
from video_dataset.schemas.scene import FrameSamplingResult, SceneDetectionResult
from video_dataset.schemas.transcript import Transcript
from video_dataset.schemas.validated import ValidatedVideo
from video_dataset.stages import Stage
from video_dataset.utils.io import read_json, write_json_atomic


def per_video_export_dir(ctx: VideoContext) -> Path:
    return ctx.config.export_dir / "per_video" / ctx.video_id


def _load(path: Path, model):  # type: ignore[no-untyped-def]
    return model.model_validate(read_json(path)) if path.exists() else None


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

    builder = DatasetBuilder(cfg.export.include_review, cfg.export.include_rejected, ctx.config.data_dir if cfg.export.relative_paths else None)
    temporal_qa, long_qa = builder.qa(validated)
    records: dict[str, list[Any]] = {
        "frames": builder.frames(validated, sampling.frames, ocr),
        "clips": builder.clips(validated, sampling.clips, transcript),
        "scenes": builder.scenes(validated, sampling.clips, transcript, ocr),
        "events": builder.events(validated),
        "temporal_qa": temporal_qa,
        "long_video_qa": long_qa,
        "video_descriptions": builder.video_descriptions(validated, scenes, ctx.metadata.title),
    }
    out_dir = per_video_export_dir(ctx)
    counts = write_jsonl_files(records, out_dir)
    manifest = {
        "video_id": ctx.video_id,
        "title": ctx.metadata.title,
        "url": ctx.metadata.url,
        "duration": ctx.duration,
        "counts": counts,
        "validation_summary": validated.summary,
        "transcript_segments": len(transcript.segments) if transcript else 0,
        "ocr_tracks": len(ocr.tracks) if ocr else 0,
        "files": {k: str(out_dir / f"{k}.jsonl") for k in counts},
    }
    write_json_atomic(out_dir / "manifest.json", manifest)
    log.info("exported %s", counts)
    return StageOutput(artifact_path=str(out_dir), metrics=counts, message=", ".join(f"{k}={v}" for k, v in counts.items()))
