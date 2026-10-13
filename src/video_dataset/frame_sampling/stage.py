"""FRAME_EXTRACTION stage: scan -> select -> extract frames -> drop duplicate frames -> cut clips
-> video fingerprint -> frames.json."""

from __future__ import annotations

from pathlib import Path

from video_dataset.deduplication.frames import duplicate_frames
from video_dataset.deduplication.videos import fingerprint_similarity, video_fingerprint
from video_dataset.frame_sampling.clips import extract_clips
from video_dataset.frame_sampling.extractor import FrameExtractor
from video_dataset.frame_sampling.sampler import select_frames
from video_dataset.frame_sampling.scan import scan_video
from video_dataset.pipeline.context import StageOutput, VideoContext
from video_dataset.schemas.scene import Frame, FrameSamplingResult, SceneDetectionResult
from video_dataset.stages import Stage
from video_dataset.utils.io import read_json, write_json_atomic
from video_dataset.utils.parallel import stage_workers


def remove_duplicate_frames(frames: list[Frame], max_distance: int, min_per_scene: int = 1) -> tuple[list[Frame], int]:
    """Drop frames whose perceptual hash repeats an earlier frame of the same scene (their files are
    deleted). A scene always keeps at least ``min_per_scene`` frames. Returns (kept frames, dropped)."""
    dups = duplicate_frames(frames, max_distance, within_scene=True)
    if not dups:
        return frames, 0
    per_scene: dict[str, int] = {}
    for f in frames:
        per_scene[f.scene_id] = per_scene.get(f.scene_id, 0) + 1
    kept: list[Frame] = []
    dropped = 0
    for f in frames:
        if f.frame_id in dups and per_scene.get(f.scene_id, 0) > min_per_scene:
            per_scene[f.scene_id] -= 1
            dropped += 1
            try:
                Path(f.frame_path).unlink(missing_ok=True)
            except OSError:
                pass
            continue
        if f.frame_id in dups:
            f.duplicate_of = dups[f.frame_id]  # kept to honour the minimum, but marked
        kept.append(f)
    return kept, dropped


def frame_extraction_stage(ctx: VideoContext) -> StageOutput:
    log = ctx.logger(Stage.FRAME_EXTRACTION)
    cfg = ctx.config.frame_sampling
    scenes = SceneDetectionResult.model_validate(read_json(ctx.paths.scenes_file(ctx.video_id))).scenes
    info = ctx.media_info

    scans = []
    if cfg.motion_aware or cfg.compute_optical_flow:
        scans = scan_video(ctx.video_path, info, scenes, cfg, ctx.config.project.ffmpeg_path)
    scan_by_scene = {s.scene_id: s for s in scans}

    plan = [(scene, select_frames(scene, scan_by_scene.get(scene.scene_id), cfg, info.fps)) for scene in scenes]
    extractor = FrameExtractor(ctx.video_path, info.fps, cfg.max_frame_side, cfg.jpeg_quality, ctx.config.project.ffmpeg_path)
    frames_dir = ctx.paths.frames_dir(ctx.video_id)
    workers = stage_workers(ctx, cfg)
    frames = extractor.extract(ctx.video_id, plan, frames_dir, workers=workers)
    if not frames:
        raise RuntimeError("no frames could be extracted")

    dropped = 0
    if cfg.drop_duplicate_frames:
        frames, dropped = remove_duplicate_frames(frames, int(cfg.duplicate_frame_hamming))
        if dropped:
            log.info("dropped %d duplicate frame(s) (dHash distance <= %d within a scene)", dropped, int(cfg.duplicate_frame_hamming))

    clips: list = []
    clip_metrics: dict = {}
    if cfg.extract_clips:
        clips, clip_stats = extract_clips(
            ctx.video_path, ctx.video_id, scenes, frames, ctx.paths.clips_dir(ctx.video_id), cfg, info.has_audio,
            ctx.config.project.ffmpeg_path, ctx.config.project.ffprobe_path, workers=workers,
        )
        clip_metrics = clip_stats.as_metrics()
        if clip_stats.reencoded:
            log.info("%d of %d clips re-encoded for exact scene boundaries", clip_stats.reencoded, clip_stats.total)

    # perceptual fingerprint of the whole video -> near-duplicate videos already in the state DB
    fingerprint = video_fingerprint(s.dhash for sc in scans for s in sc.samples) if scans else video_fingerprint(f.dhash for f in frames)
    near_of: str | None = None
    near_sim: float | None = None
    dcfg = ctx.config.deduplication
    if dcfg.video_fingerprint and fingerprint:
        best = 0.0
        for other, hashes in ctx.db.fingerprints(exclude=ctx.video_id).items():
            sim = fingerprint_similarity(fingerprint, hashes)
            if sim > best:
                best, near_of = sim, other
        if near_of is not None and best >= float(dcfg.video_similarity_threshold):
            near_sim = round(best, 3)
            log.warning("video content matches %s (fingerprint similarity %.2f): near-duplicate video", near_of, best)
        else:
            near_of = None
        ctx.db.register_fingerprint(ctx.video_id, fingerprint)

    result = FrameSamplingResult(
        video_id=ctx.video_id, frames=frames, scans=scans, clips=clips, fingerprint=fingerprint,
        duplicate_frames_removed=dropped, near_duplicate_of=near_of, near_duplicate_similarity=near_sim,
    )
    out = ctx.paths.frames_file(ctx.video_id)
    write_json_atomic(out, result)
    reasons: dict[str, int] = {}
    for f in frames:
        reasons[str(f.sampling_reason)] = reasons.get(str(f.sampling_reason), 0) + 1
    log.info("%d frames from %d scenes (%s), %d clips", len(frames), len(scenes), reasons, len(clips))
    return StageOutput(
        artifact_path=str(out),
        metrics={
            "frames": len(frames), "clips": len(clips), **clip_metrics, "workers": workers, "scan_samples": sum(len(s.samples) for s in scans),
            "reasons": reasons, "duplicate_frames_removed": dropped, "fingerprint_hashes": len(fingerprint), "near_duplicate_of": near_of,
        },
        message=f"{len(frames)} frames extracted, {len(clips)} clips" + (f", {dropped} duplicate frames dropped" if dropped else "") + (f", near-duplicate of {near_of}" if near_of else ""),
    )
