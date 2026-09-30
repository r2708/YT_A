"""VISION_ANALYSIS stage: per-scene multimodal analysis with measured signals attached and
optional model verification. Results are checkpointed after every scene, so an interrupted
run resumes at the next un-analysed scene instead of starting over."""

from __future__ import annotations

from pathlib import Path

from video_dataset.pipeline.context import StageOutput, VideoContext
from video_dataset.schemas.common import ConfidenceSource
from video_dataset.schemas.ocr import OCRResult
from video_dataset.schemas.scene import FrameSamplingResult, Scene, SceneDetectionResult
from video_dataset.schemas.transcript import Transcript
from video_dataset.schemas.vision import (
    CameraMovement,
    Measurements,
    SceneAnalysis,
    VerificationVerdict,
    VisionResult,
)
from video_dataset.stages import Stage
from video_dataset.utils.device import empty_cache, is_oom_error, resolve_device
from video_dataset.utils.io import read_json, write_json_atomic
from video_dataset.utils.parallel import chunk_evenly, parallel_map, stage_workers
from video_dataset.utils.progress import ProgressLog
from video_dataset.vision.base import AnalysisContext, create_vision_analyzer, select_evenly
from video_dataset.vision.enrichers.base import create_enrichers
from video_dataset.vision.enrichers.clip import apply_clip_attributes
from video_dataset.vision.heuristic import HeuristicVisionAnalyzer
from video_dataset.vision.measurements import measure_scene


def _load_optional(path: Path, model):  # type: ignore[no-untyped-def]
    return model.model_validate(read_json(path)) if path.exists() else None


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

    def _load_analyzer():  # type: ignore[no-untyped-def]
        log.info("loading vision analyzer '%s'%s on %s (first use in this process)", vcfg.provider, f" model={vcfg.model}" if vcfg.model else "", device)
        return create_vision_analyzer(vcfg, device)

    analyzer = ctx.get_model(f"vision:{vcfg.provider}:{vcfg.model}", _load_analyzer)
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

    workers = stage_workers(ctx, vcfg, len(scenes))
    analyzer_parallel = workers > 1 and bool(getattr(analyzer, "parallel_safe", False))
    failures = 0
    verified = 0
    log.info(
        "analyzing %d scenes with %s%s on %s (%d already done%s; %d worker%s, model calls %s)", len(scenes), analyzer.name,
        f" ({analyzer.model})" if getattr(analyzer, "model", None) else "", device, len(done),
        f", enrichers: {', '.join(e.name for e in enrichers)}" if enrichers else "", workers, "s" if workers != 1 else "",
        "parallel" if analyzer_parallel else "sequential",
    )
    progress = ProgressLog(log, "vision", len(scenes), unit="scenes", every_seconds=15)

    def _measure(scene: Scene) -> Measurements:
        frames = sorted(frames_by_scene.get(scene.scene_id, []), key=lambda f: f.timestamp)
        return measure_scene([Path(f.frame_path) for f in frames], scans.get(scene.scene_id))

    def _context(scene: Scene, measurements: Measurements, previous_summary: str | None) -> AnalysisContext:
        return AnalysisContext(
            video_id=ctx.video_id,
            transcript_text=transcript.text_between(scene.start_time, scene.end_time) if transcript and transcript.segments else None,
            ocr_texts=[t.text for t in ocr.tracks if scene.scene_id in t.scene_ids] if ocr else [],
            measurements=measurements,
            scan=scans.get(scene.scene_id),
            previous_summary=previous_summary,
            video_title=ctx.metadata.title,
            extra={"clip_path": clips[scene.scene_id].clip_path} if scene.scene_id in clips else {},
        )

    def _analyze(scene: Scene, context: AnalysisContext) -> tuple[SceneAnalysis, bool, bool]:
        """Measured annotation + model annotation + merge + captions + verification for one scene.
        Returns (analysis, analyzer_failed, verified). Enrichers run afterwards on the main thread."""
        frames = sorted(frames_by_scene.get(scene.scene_id, []), key=lambda f: f.timestamp)
        failed = False
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
                failed = True
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

        is_verified = False
        if analyzer.is_generative and vcfg.verify_with_model and analyzer.supports_verification and analysis.summary and frames and not analysis.errors:
            verification = analyzer.verify(analysis.summary, frames, context)
            analysis.verification = verification
            if verification.verdict != VerificationVerdict.UNKNOWN and verification.score is not None:
                analysis.confidence = verification.score
                analysis.confidence_source = ConfidenceSource.VERIFIER
                is_verified = True
        return analysis, failed, is_verified

    def _enrich(scene: Scene, analysis: SceneAnalysis) -> None:
        frames = sorted(frames_by_scene.get(scene.scene_id, []), key=lambda f: f.timestamp)
        for enr in enrichers:
            if enr.name in analysis.enrichments:
                continue  # the analyzer already ran this model (cv_models runs CLIP itself)
            try:
                data = enr.enrich(scene, frames)
                analysis.enrichments[enr.name] = data
                if enr.name == "clip":
                    apply_clip_attributes(analysis, data)
            except Exception as exc:
                log.warning("enricher %s failed on %s: %s", enr.name, scene.scene_id, str(exc)[:120])

    def _ordered() -> list[SceneAnalysis]:
        return [done[s.scene_id] for s in scenes if s.scene_id in done]

    def _checkpoint() -> None:
        write_json_atomic(out_path, VisionResult(video_id=ctx.video_id, provider=analyzer.name, model=analyzer.model, scenes=_ordered()))

    # Scenes are processed in order, in chunks of `workers`. Measurements always run in parallel; the
    # model runs in parallel only when it is thread-safe, in which case every scene of a chunk sees the
    # summary of the scene before the chunk as `previous_summary` (exact predecessor otherwise).
    pending = [(i, sc) for i, sc in enumerate(scenes) if sc.scene_id not in done]
    n_done = len(done)
    for chunk in chunk_evenly(pending, max(1, -(-len(pending) // workers))) if pending else []:
        progress.update(n_done, extra=f"{failures} failed" if failures else "")
        measurements = parallel_map(lambda item: _measure(item[1]), chunk, workers)
        outcomes: list[tuple[SceneAnalysis, bool, bool]]
        if analyzer_parallel:
            i0 = chunk[0][0]
            prev_summary = done[scenes[i0 - 1].scene_id].summary if i0 > 0 and scenes[i0 - 1].scene_id in done else None

            def _run(pair: tuple[tuple[int, Scene], Measurements], ps: str | None = prev_summary) -> tuple[SceneAnalysis, bool, bool]:
                return _analyze(pair[0][1], _context(pair[0][1], pair[1], ps))

            outcomes = parallel_map(_run, list(zip(chunk, measurements)), workers)
        else:
            outcomes = []
            for (i, sc), m in zip(chunk, measurements):
                prev_summary = done[scenes[i - 1].scene_id].summary if i > 0 and scenes[i - 1].scene_id in done else None
                res = _analyze(sc, _context(sc, m, prev_summary))
                done[sc.scene_id] = res[0]
                outcomes.append(res)
        for (_i, sc), (analysis, failed, is_verified) in zip(chunk, outcomes):
            _enrich(sc, analysis)
            done[sc.scene_id] = analysis
            failures += int(failed)
            verified += int(is_verified)
            n_done += 1
        _checkpoint()
    results = _ordered()
    result = VisionResult(video_id=ctx.video_id, provider=analyzer.name, model=analyzer.model, scenes=results)
    progress.finish(f"{failures} failed" if failures else "")
    write_json_atomic(out_path, result)
    n_objects = sum(len(s.objects) for s in results)
    n_actions = sum(len(s.actions) for s in results)
    log.info("%d scenes analysed by %s (%d failures, %d verified, %d objects, %d actions)", len(results), analyzer.name, failures, verified, n_objects, n_actions)
    return StageOutput(
        artifact_path=str(out_path),
        metrics={"scenes": len(results), "provider": analyzer.name, "model": analyzer.model, "failures": failures, "verified": verified, "objects": n_objects, "actions": n_actions, "workers": workers, "model_parallel": analyzer_parallel},
        message=f"{len(results)} scenes analyzed ({analyzer.name})",
    )
