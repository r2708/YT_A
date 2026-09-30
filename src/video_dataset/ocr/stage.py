"""OCR stage: run the engine on a subset of sampled frames, merge repeated text into tracks."""

from __future__ import annotations

import os
import queue
from pathlib import Path

from video_dataset.ocr.base import create_ocr_engine
from video_dataset.ocr.merge import classify_tracks, merge_detections
from video_dataset.pipeline.context import StageOutput, VideoContext
from video_dataset.schemas.ocr import OCRDetection, OCRResult
from video_dataset.schemas.scene import Frame, FrameSamplingResult
from video_dataset.stages import Stage
from video_dataset.utils.device import resolve_device
from video_dataset.utils.io import read_json, write_json_atomic
from video_dataset.utils.parallel import parallel_map, stage_workers
from video_dataset.utils.progress import ProgressLog


def pick_frames_for_ocr(frames: list[Frame], max_per_scene: int) -> list[Frame]:
    by_scene: dict[str, list[Frame]] = {}
    for f in frames:
        by_scene.setdefault(f.scene_id, []).append(f)
    chosen: list[Frame] = []
    for scene_frames in by_scene.values():
        scene_frames.sort(key=lambda f: f.timestamp)
        if len(scene_frames) <= max_per_scene:
            chosen.extend(scene_frames)
            continue
        step = (len(scene_frames) - 1) / max(1, max_per_scene - 1)
        idxs = sorted({int(round(i * step)) for i in range(max_per_scene)})
        chosen.extend(scene_frames[i] for i in idxs)
    chosen.sort(key=lambda f: f.timestamp)
    return chosen


def ocr_stage(ctx: VideoContext) -> StageOutput:
    log = ctx.logger(Stage.OCR)
    cfg = ctx.config.ocr
    out = ctx.paths.ocr_file(ctx.video_id)
    frames = FrameSamplingResult.model_validate(read_json(ctx.paths.frames_file(ctx.video_id))).frames
    def _load_engine():  # type: ignore[no-untyped-def]
        log.info("loading OCR engine '%s' (first use in this process)", cfg.provider)
        return create_ocr_engine(cfg, resolve_device(ctx.config.project.device))

    engine = ctx.get_model(f"ocr:{cfg.provider}", _load_engine)
    if engine is None:
        result = OCRResult(video_id=ctx.video_id, engine="none", frames_processed=0)
        write_json_atomic(out, result)
        return StageOutput(artifact_path=str(out), metrics={"detections": 0, "tracks": 0}, message="skipped (no OCR engine)", skipped=True)

    targets = pick_frames_for_ocr(frames, cfg.max_frames_per_scene)
    workers = stage_workers(ctx, cfg, len(targets))
    engines: queue.Queue = queue.Queue()
    if workers > 1:
        # One engine instance per thread, each capped to its share of the cores: ONNX runtime already
        # multi-threads a single instance, so N uncapped instances only oversubscribe the CPU.
        threads = max(1, (os.cpu_count() or 1) // workers)

        def _load_pool() -> list:
            log.info("loading %d OCR engine instances (%d thread%s each) for parallel OCR", workers, threads, "s" if threads != 1 else "")
            pool = [create_ocr_engine(cfg, resolve_device(ctx.config.project.device), threads=threads) for _ in range(workers)]
            return [e for e in pool if e is not None]

        for eng in ctx.get_model(f"ocr:{cfg.provider}:pool:{workers}", _load_pool) or [engine]:
            engines.put(eng)
    else:
        engines.put(engine)
    workers = engines.qsize()

    log.info(
        "running %s on %d of %d frames (max %d per scene, %d worker%s)",
        getattr(engine, "name", cfg.provider), len(targets), len(frames), cfg.max_frames_per_scene, workers, "s" if workers != 1 else "",
    )
    progress = ProgressLog(log, "OCR", len(targets), unit="frames", every_seconds=15)
    counts = {"detections": 0}

    def _detect(frame: Frame) -> list | None:
        """Raw detections for one frame, None on failure / missing file. Runs on a worker thread."""
        path = Path(frame.frame_path)
        if not path.exists():
            return []
        eng = engines.get()
        try:
            raw = eng.detect(path)
        except Exception as exc:
            log.warning("OCR failed on %s: %s", path.name, exc)
            return None
        finally:
            engines.put(eng)
        return [d for d in raw if not (d.confidence is not None and d.confidence < cfg.min_confidence) and len(d.text.strip()) >= cfg.min_text_length]

    def _tick(i: int, raw: list | None) -> None:
        counts["detections"] += len(raw or [])
        progress.update(i + 1, extra=f"{counts['detections']} detections")

    per_frame = parallel_map(_detect, targets, workers, on_done=_tick)
    detections: list[OCRDetection] = []
    failures = 0
    n = 0
    for frame, raw in zip(targets, per_frame):  # in frame order -> deterministic detection ids
        if raw is None:
            failures += 1
            continue
        for det in raw:
            n += 1
            detections.append(
                OCRDetection(
                    detection_id=f"det_{n:05d}",
                    frame_id=frame.frame_id,
                    scene_id=frame.scene_id,
                    timestamp=frame.timestamp,
                    text=det.text.strip(),
                    confidence=det.confidence,
                    bbox=det.bbox,
                )
            )
    tracks = merge_detections(detections, cfg.merge_similarity, cfg.merge_max_gap_seconds)
    progress.finish(f"{len(detections)} detections")
    total_scenes = len({f.scene_id for f in frames})
    tracks, overlays = classify_tracks(tracks, total_scenes, cfg.static_overlay_min_scene_fraction, cfg.static_overlay_min_scenes, cfg.merge_max_gap_seconds)
    n_overlay = sum(1 for t in tracks if t.is_static_overlay)
    n_fragment = sum(1 for t in tracks if t.is_fragment)
    result = OCRResult(video_id=ctx.video_id, engine=engine.name, frames_processed=len(targets), detections=detections, tracks=tracks, static_overlays=overlays)
    write_json_atomic(out, result)
    log.info("%d detections -> %d text tracks over %d frames (%d failures); %d overlay tracks %s, %d fragments", len(detections), len(tracks), len(targets), failures, n_overlay, overlays, n_fragment)
    return StageOutput(
        artifact_path=str(out),
        metrics={"frames": len(targets), "workers": workers, "detections": len(detections), "tracks": len(tracks), "static_overlay_tracks": n_overlay, "fragment_tracks": n_fragment, "static_overlays": overlays, "failures": failures, "engine": engine.name},
        message=f"{len(detections)} text detections ({len(tracks) - n_overlay - n_fragment} usable, {n_overlay} overlay, {n_fragment} fragments)",
    )
