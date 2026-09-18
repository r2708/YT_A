"""OCR stage: run the engine on a subset of sampled frames, merge repeated text into tracks."""

from __future__ import annotations

from pathlib import Path

from video_dataset.ocr.base import create_ocr_engine
from video_dataset.ocr.merge import merge_detections
from video_dataset.pipeline.context import StageOutput, VideoContext
from video_dataset.schemas.ocr import OCRDetection, OCRResult
from video_dataset.schemas.scene import Frame, FrameSamplingResult
from video_dataset.stages import Stage
from video_dataset.utils.device import resolve_device
from video_dataset.utils.io import read_json, write_json_atomic


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
    engine = ctx.get_model(f"ocr:{cfg.provider}", lambda: create_ocr_engine(cfg, resolve_device(ctx.config.project.device)))
    if engine is None:
        result = OCRResult(video_id=ctx.video_id, engine="none", frames_processed=0)
        write_json_atomic(out, result)
        return StageOutput(artifact_path=str(out), metrics={"detections": 0, "tracks": 0}, message="skipped (no OCR engine)", skipped=True)

    targets = pick_frames_for_ocr(frames, cfg.max_frames_per_scene)
    detections: list[OCRDetection] = []
    failures = 0
    n = 0
    for frame in targets:
        path = Path(frame.frame_path)
        if not path.exists():
            continue
        try:
            raw = engine.detect(path)
        except Exception as exc:
            failures += 1
            log.warning("OCR failed on %s: %s", path.name, exc)
            continue
        for det in raw:
            if det.confidence is not None and det.confidence < cfg.min_confidence:
                continue
            if len(det.text.strip()) < cfg.min_text_length:
                continue
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
    result = OCRResult(video_id=ctx.video_id, engine=engine.name, frames_processed=len(targets), detections=detections, tracks=tracks)
    write_json_atomic(out, result)
    log.info("%d detections -> %d text tracks over %d frames (%d failures)", len(detections), len(tracks), len(targets), failures)
    return StageOutput(
        artifact_path=str(out),
        metrics={"frames": len(targets), "detections": len(detections), "tracks": len(tracks), "failures": failures, "engine": engine.name},
        message=f"{len(detections)} text detections ({len(tracks)} unique)",
    )
