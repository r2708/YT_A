"""TEMPORAL_ANALYSIS stage: events from every modality -> relations -> timeline.json."""

from __future__ import annotations

from pathlib import Path

from video_dataset.pipeline.context import StageOutput, VideoContext
from video_dataset.schemas.ocr import OCRResult
from video_dataset.schemas.scene import FrameSamplingResult, SceneDetectionResult
from video_dataset.schemas.transcript import AudioAnalysis, Transcript
from video_dataset.schemas.vision import VisionResult
from video_dataset.stages import Stage
from video_dataset.temporal.causal import LLMCausalInferencer, NullCausalInferencer
from video_dataset.temporal.events import EventExtractor
from video_dataset.temporal.relations import RelationBuilder
from video_dataset.temporal.timeline import build_timeline, timeline_stats
from video_dataset.utils.io import read_json, write_json_atomic


def _load(path: Path, model):  # type: ignore[no-untyped-def]
    return model.model_validate(read_json(path)) if path.exists() else None


def temporal_stage(ctx: VideoContext) -> StageOutput:
    log = ctx.logger(Stage.TEMPORAL_ANALYSIS)
    cfg = ctx.config
    if not cfg.temporal.enabled:
        return StageOutput(message="temporal analysis disabled", skipped=True)

    detection = SceneDetectionResult.model_validate(read_json(ctx.paths.scenes_file(ctx.video_id)))
    sampling = FrameSamplingResult.model_validate(read_json(ctx.paths.frames_file(ctx.video_id)))
    vision = _load(ctx.paths.vision_file(ctx.video_id), VisionResult)
    transcript = _load(ctx.paths.transcript_file(ctx.video_id), Transcript)
    ocr = _load(ctx.paths.ocr_file(ctx.video_id), OCRResult)
    audio = _load(ctx.paths.audio_events_file(ctx.video_id), AudioAnalysis)
    duration = ctx.duration

    extractor = EventExtractor(cfg.temporal, scene_threshold=detection.threshold)
    events = extractor.extract(ctx.video_id, duration, detection.scenes, sampling.frames, vision, transcript, ocr, audio, sampling.scans)
    relations = RelationBuilder(cfg.temporal).build(ctx.video_id, events)

    causal = NullCausalInferencer()
    if cfg.temporal.causal_inference == "llm" and cfg.llm.provider not in ("none", None):
        from video_dataset.llm.base import create_llm_client

        client = ctx.get_model(f"llm:{cfg.llm.provider}:{cfg.llm.model}", lambda: create_llm_client(cfg.llm.provider, cfg.llm.model, cfg.llm.base_url, cfg.llm.api_key_env, float(cfg.llm.timeout_seconds)))
        causal = LLMCausalInferencer(client)  # type: ignore[assignment]
    relations += causal.infer(ctx.video_id, events, relations)

    timeline = build_timeline(ctx.video_id, duration, events, relations)
    out = ctx.paths.timeline_file(ctx.video_id)
    write_json_atomic(out, timeline)
    stats = timeline_stats(timeline)
    log.info("%d events %s, %d relations %s", stats["events"], stats["events_by_type"], stats["relations"], stats["relations_by_type"])
    return StageOutput(artifact_path=str(out), metrics=stats, message=f"{stats['events']} events, {stats['relations']} relations")
