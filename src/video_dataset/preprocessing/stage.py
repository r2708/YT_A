"""PREPROCESS stage: probe -> canonical video.mp4 -> media_info.json -> audio.wav."""

from __future__ import annotations

from video_dataset.errors import VideoRejected
from video_dataset.pipeline.context import StageOutput, VideoContext
from video_dataset.preprocessing.normalize import ensure_canonical_video, extract_audio
from video_dataset.preprocessing.probe import probe_media
from video_dataset.stages import Stage
from video_dataset.utils.io import write_json_atomic


def preprocess_stage(ctx: VideoContext) -> StageOutput:
    log = ctx.logger(Stage.PREPROCESS)
    cfg = ctx.config
    sources = ctx.paths.source_candidates(ctx.video_id)
    canonical = ctx.paths.video_file(ctx.video_id)
    if not sources and not canonical.exists():
        raise FileNotFoundError(f"no downloaded source for {ctx.video_id}")
    source = sources[0] if sources else canonical

    src_info = probe_media(source, cfg.project.ffprobe_path)
    max_dur = cfg.limits.max_duration_seconds
    if max_dur and src_info.duration > float(max_dur):
        raise VideoRejected(
            f"video is {src_info.duration / 60:.1f} min long, over limits.max_duration_seconds ({float(max_dur) / 60:.0f} min); "
            "raise the limit or trim the video"
        )
    video_path, transcoded = ensure_canonical_video(source, canonical, src_info, cfg.preprocess, cfg.project.ffmpeg_path)
    info = probe_media(video_path, cfg.project.ffprobe_path) if transcoded or video_path != source else src_info
    info = info.model_copy(update={"path": str(video_path), "has_audio": src_info.has_audio, "audio": src_info.audio})
    if info.duration <= 0:
        raise ValueError("video duration is zero or unknown; file is probably corrupt")

    has_audio = False
    wav = ctx.paths.audio_file(ctx.video_id)
    if src_info.has_audio or not info.video_codec:  # unknown -> try
        has_audio = extract_audio(video_path, wav, cfg.preprocess.audio_sample_rate, cfg.preprocess.audio_channels, cfg.project.ffmpeg_path)
    if not has_audio:
        log.info("no audio stream; transcription will be skipped")
        wav.unlink(missing_ok=True)
    info = info.model_copy(update={"has_audio": has_audio})

    write_json_atomic(ctx.paths.media_info_file(ctx.video_id), info)
    ctx.db.update_video_meta(ctx.video_id, duration=info.duration)
    ctx.invalidate()
    return StageOutput(
        artifact_path=str(ctx.paths.media_info_file(ctx.video_id)),
        metrics={
            "duration": round(info.duration, 3),
            "fps": round(info.fps, 3),
            "width": info.width,
            "height": info.height,
            "frame_count": info.frame_count,
            "codec": info.video_codec,
            "has_audio": has_audio,
            "transcoded": transcoded,
        },
        message=f"{info.width}x{info.height} @ {info.fps:.2f} fps, {info.duration:.1f}s, audio={'yes' if has_audio else 'no'}",
    )
