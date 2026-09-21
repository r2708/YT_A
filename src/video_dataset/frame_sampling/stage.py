"""FRAME_EXTRACTION stage: scan -> select -> extract frames -> cut clips -> frames.json."""

from __future__ import annotations

from video_dataset.frame_sampling.clips import extract_clips
from video_dataset.frame_sampling.extractor import FrameExtractor
from video_dataset.frame_sampling.sampler import select_frames
from video_dataset.frame_sampling.scan import scan_video
from video_dataset.pipeline.context import StageOutput, VideoContext
from video_dataset.schemas.scene import FrameSamplingResult, SceneDetectionResult
from video_dataset.stages import Stage
from video_dataset.utils.io import read_json, write_json_atomic


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
    frames = extractor.extract(ctx.video_id, plan, frames_dir)
    if not frames:
        raise RuntimeError("no frames could be extracted")

    clips: list = []
    clip_metrics: dict = {}
    if cfg.extract_clips:
        clips, clip_stats = extract_clips(
            ctx.video_path, ctx.video_id, scenes, frames, ctx.paths.clips_dir(ctx.video_id), cfg, info.has_audio,
            ctx.config.project.ffmpeg_path, ctx.config.project.ffprobe_path,
        )
        clip_metrics = clip_stats.as_metrics()
        if clip_stats.reencoded:
            log.info("%d of %d clips re-encoded for exact scene boundaries", clip_stats.reencoded, clip_stats.total)

    result = FrameSamplingResult(video_id=ctx.video_id, frames=frames, scans=scans, clips=clips)
    out = ctx.paths.frames_file(ctx.video_id)
    write_json_atomic(out, result)
    reasons: dict[str, int] = {}
    for f in frames:
        reasons[str(f.sampling_reason)] = reasons.get(str(f.sampling_reason), 0) + 1
    log.info("%d frames from %d scenes (%s), %d clips", len(frames), len(scenes), reasons, len(clips))
    return StageOutput(
        artifact_path=str(out),
        metrics={"frames": len(frames), "clips": len(clips), **clip_metrics, "scan_samples": sum(len(s.samples) for s in scans), "reasons": reasons},
        message=f"{len(frames)} frames extracted, {len(clips)} clips",
    )
