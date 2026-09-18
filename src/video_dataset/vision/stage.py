"""VISION_ANALYSIS stage: per-scene multimodal analysis with measured signals attached and
optional model verification. Results are checkpointed after every scene, so an interrupted
run resumes at the next un-analysed scene instead of starting over."""

from __future__ import annotations

from pathlib import Path

from video_dataset.pipeline.context import StageOutput, VideoContext
from video_dataset.schemas.common import ConfidenceSource
from video_dataset.schemas.ocr import OCRResult
from video_dataset.schemas.scene import FrameSamplingResult, SceneDetectionResult
from video_dataset.schemas.transcript import Transcript
from video_dataset.schemas.vision import (
    CameraMovement,
    SceneAnalysis,
    Setting,
    VerificationVerdict,
    VisionResult,
)
from video_dataset.stages import Stage
from video_dataset.utils.device import empty_cache, is_oom_error, resolve_device
from video_dataset.utils.io import read_json, write_json_atomic
from video_dataset.vision.base import AnalysisContext, create_vision_analyzer, select_evenly
from video_dataset.vision.enrichers.base import create_enrichers
from video_dataset.vision.heuristic import HeuristicVisionAnalyzer
from video_dataset.vision.measurements import measure_scene


def _load_optional(path: Path, model):  # type: ignore[no-untyped-def]
    return model.model_validate(read_json(path)) if path.exists() else None


def _apply_enrichments(analysis: SceneAnalysis, enrichment: dict, min_score: float = 0.75) -> None:
    attrs = (enrichment or {}).get("attributes") or {}
    setting = attrs.get("setting")
    if setting and analysis.environment.setting == Setting.UNKNOWN and setting["score"] >= min_score:
        analysis.environment.setting = Setting(setting["label"])
    tod = attrs.get("time_of_day")
    if tod and not analysis.environment.time_of_day and tod["score"] >= min_score:
        analysis.environment.time_of_day = tod["label"]


def vision_stage(ctx: VideoContext) -> StageOutput:
    log = ctx.logger(Stage.VISION_ANALYSIS)
    cfg = ctx.config
    vcfg = cfg.vision
    device = resolve_device(vcfg.device or cfg.project.device)

    scenes = SceneDetectionResult.model_validate(read_json(ctx.paths.scenes_file(ctx.video_id))).scenes
    sampling = FrameSamplingResult.model_validate(read_json(ctx.paths.frames_file(ctx.video_id)))
    transcript = _load_optional(ctx.paths.transcript_file(ctx.video_id), Transcript)
    ocr = _load_optional(ctx.paths.ocr_file(ctx.video_id), OCRResult)
    frames_by_scene: dict[str, list] = {}
    for f in sampling.frames:
        frames_by_scene.setdefault(f.scene_id, []).append(f)
    scans = {s.scene_id: s for s in sampling.scans}
    clips = {c.scene_id: c for c in sampling.clips}

    analyzer = ctx.get_model(f"vision:{vcfg.provider}:{vcfg.model}", lambda: create_vision_analyzer(vcfg, device))
    heuristic = HeuristicVisionAnalyzer(cfg.temporal.min_camera_motion, cfg.temporal.min_camera_consistency)
    enrichers = ctx.get_model(f"enrichers:{','.join(vcfg.enrichers)}", lambda: create_enrichers(vcfg, device)) if vcfg.enrichers else []

    out_path = ctx.paths.vision_file(ctx.video_id)
    done: dict[str, SceneAnalysis] = {}
    if out_path.exists():  # resume inside the stage
        try:
            prev = VisionResult.model_validate(read_json(out_path))
            if prev.provider == analyzer.name and prev.model == analyzer.model:
                done = {s.scene_id: s for s in prev.scenes}
                if done:
                    log.info("resuming vision analysis: %d/%d scenes already done", len(done), len(scenes))
        except Exception:
            done = {}

    results: list[SceneAnalysis] = []
    previous_summary: str | None = None
    failures = 0
    verified = 0
    for scene in scenes:
        frames = sorted(frames_by_scene.get(scene.scene_id, []), key=lambda f: f.timestamp)
        if scene.scene_id in done:
            results.append(done[scene.scene_id])
            previous_summary = done[scene.scene_id].summary
            continue
        scan = scans.get(scene.scene_id)
        measurements = measure_scene([Path(f.frame_path) for f in frames], scan)
        context = AnalysisContext(
            video_id=ctx.video_id,
            transcript_text=transcript.text_between(scene.start_time, scene.end_time) if transcript and transcript.segments else None,
            ocr_texts=[t.text for t in ocr.tracks if scene.scene_id in t.scene_ids] if ocr else [],
            measurements=measurements,
            scan=scan,
            previous_summary=previous_summary,
            video_title=ctx.metadata.title,
            extra={"clip_path": clips[scene.scene_id].clip_path} if scene.scene_id in clips else {},
        )
        measured = heuristic.analyze_scene(scene, frames, context)
        if not frames:
            analysis = measured
            analysis.errors.append("no_frames")
        elif analyzer.name == heuristic.name:
            analysis = measured
        else:
            try:
                analysis = analyzer.analyze_scene(scene, frames, context)
            except Exception as exc:
                failures += 1
                if is_oom_error(exc):
                    empty_cache(device)
                log.warning("vision analysis failed for %s (%s); keeping measured annotation", scene.scene_id, str(exc)[:200])
                analysis = measured
                analysis.errors.append(f"analyzer_error: {type(exc).__name__}: {str(exc)[:200]}")

        # Always attach measured signals; fill camera movement from optical flow when the model did not say.
        analysis.measurements = measured.measurements
        analysis.enrichments["measured_camera"] = measured.camera.model_dump(mode="json")
        if analysis.camera.movement == CameraMovement.UNKNOWN and measured.camera.movement != CameraMovement.UNKNOWN:
            analysis.camera.movement = measured.camera.movement
            analysis.camera.confidence = measured.camera.confidence
            analysis.camera.confidence_source = measured.camera.confidence_source
            if measured.camera.stability and not analysis.camera.stability:
                analysis.camera.stability = measured.camera.stability
        elif analysis.camera.confidence is None and analysis.camera.movement == measured.camera.movement and measured.camera.confidence is not None:
            analysis.camera.confidence = measured.camera.confidence
            analysis.camera.confidence_source = measured.camera.confidence_source
        if not analysis.environment.lighting and measured.environment.lighting:
            analysis.environment.lighting = measured.environment.lighting
        if not analysis.actions and measured.actions:
            analysis.actions = list(measured.actions)

        if vcfg.frame_captions and frames and analyzer.is_generative:
            for fr in select_evenly(frames, int(vcfg.frame_captions_per_scene)):
                try:
                    analysis.frame_analyses.append(analyzer.analyze_frame(fr, context))
                except Exception as exc:
                    log.warning("frame caption failed for %s: %s", fr.frame_id, str(exc)[:120])

        for enr in enrichers:
            try:
                data = enr.enrich(scene, frames)
                analysis.enrichments[enr.name] = data
                if enr.name == "clip":
                    _apply_enrichments(analysis, data)
            except Exception as exc:
                log.warning("enricher %s failed on %s: %s", enr.name, scene.scene_id, str(exc)[:120])

        if analyzer.is_generative and vcfg.verify_with_model and analyzer.supports_verification and analysis.summary and frames and not analysis.errors:
            verification = analyzer.verify(analysis.summary, frames, context)
            analysis.verification = verification
            if verification.verdict != VerificationVerdict.UNKNOWN and verification.score is not None:
                analysis.confidence = verification.score
                analysis.confidence_source = ConfidenceSource.VERIFIER
                verified += 1

        results.append(analysis)
        previous_summary = analysis.summary
        write_json_atomic(out_path, VisionResult(video_id=ctx.video_id, provider=analyzer.name, model=analyzer.model, scenes=results))

    result = VisionResult(video_id=ctx.video_id, provider=analyzer.name, model=analyzer.model, scenes=results)
    write_json_atomic(out_path, result)
    n_objects = sum(len(s.objects) for s in results)
    n_actions = sum(len(s.actions) for s in results)
    log.info("%d scenes analysed by %s (%d failures, %d verified, %d objects, %d actions)", len(results), analyzer.name, failures, verified, n_objects, n_actions)
    return StageOutput(
        artifact_path=str(out_path),
        metrics={"scenes": len(results), "provider": analyzer.name, "model": analyzer.model, "failures": failures, "verified": verified, "objects": n_objects, "actions": n_actions},
        message=f"{len(results)} scenes analyzed ({analyzer.name})",
    )
