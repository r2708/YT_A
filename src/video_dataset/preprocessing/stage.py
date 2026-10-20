"""PREPROCESS stage: probe -> canonical video.mp4 -> media_info.json -> audio.wav."""

from __future__ import annotations

from video_dataset.errors import VideoRejected
from video_dataset.pipeline.context import StageOutput, VideoContext
from video_dataset.preprocessing.normalize import ensure_canonical_video, extract_audio
from video_dataset.preprocessing.probe import probe_media
from video_dataset.stages import Stage
from video_dataset.utils.io import write_json_atomic
from video_dataset.utils.progress import fmt_seconds


def preprocess_stage(ctx: VideoContext) -> StageOutput:
    log = ctx.logger(Stage.PREPROCESS)
    cfg = ctx.config
    sources = ctx.paths.source_candidates(ctx.video_id)
    canonical = ctx.paths.video_file(ctx.video_id)
    if not sources and not canonical.exists():
        raise FileNotFoundError(f"no downloaded source for {ctx.video_id}")
    source = sources[0] if sources else canonical

    log.info("probing %s", source.name)
    src_info = probe_media(source, cfg.project.ffprobe_path)
    log.info("source: %dx%d @ %.2f fps, %s, codec=%s, audio=%s", src_info.width, src_info.height, src_info.fps, fmt_seconds(src_info.duration), src_info.video_codec, "yes" if src_info.has_audio else "no")
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

    # --- duration vs the source's own claim, resolution / bitrate of the source: measured flags ---
    qcfg = cfg.quality
    flags: list[str] = []
    meta_duration: float | None = None
    mismatch: float | None = None
    try:
        meta_duration = ctx.metadata.duration
    except FileNotFoundError:
        meta_duration = None
    if meta_duration:
        mismatch = round(abs(float(info.duration) - float(meta_duration)), 3)
        if mismatch > float(qcfg.duration_tolerance_seconds):
            flags.append("duration_mismatch")
            log.warning("probed duration %.1fs differs from the reported %.1fs by %.1fs", info.duration, meta_duration, mismatch)
    if int(qcfg.min_height) > 0 and src_info.height < int(qcfg.min_height):
        flags.append("low_resolution")
    if float(qcfg.min_bitrate_kbps) > 0 and src_info.bitrate and src_info.bitrate / 1000.0 < float(qcfg.min_bitrate_kbps):
        flags.append("low_bitrate")
    if float(getattr(qcfg, "min_fps", 0) or 0) > 0 and src_info.fps and src_info.fps < float(qcfg.min_fps):
        flags.append("low_fps")
    max_ar = float(getattr(qcfg, "max_aspect_ratio", 0) or 0)
    if max_ar > 0 and src_info.width and src_info.height:
        ar = src_info.width / float(src_info.height)
        if ar > max_ar or ar < 1.0 / max_ar:
            flags.append("extreme_aspect_ratio")
    if info.frame_count and info.duration and info.fps and abs(info.frame_count - info.duration * info.fps) > max(2.0, 0.02 * info.frame_count) + info.fps:
        flags.append("frame_count_mismatch")  # container says N frames, duration * fps says otherwise: truncated / VFR stream
    info = info.model_copy(update={"metadata_duration": meta_duration, "duration_mismatch_seconds": mismatch, "quality_flags": flags})
    if flags:
        log.info("source quality flags: %s", ", ".join(flags))

    has_audio = False
    wav = ctx.paths.audio_file(ctx.video_id)
    if src_info.has_audio or not info.video_codec:  # unknown -> try
        log.info("extracting audio -> %s (%d Hz, %d ch)", wav.name, cfg.preprocess.audio_sample_rate, cfg.preprocess.audio_channels)
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
            "quality_flags": flags,
            "duration_mismatch_seconds": mismatch,
        },
        message=f"{info.width}x{info.height} @ {info.fps:.2f} fps, {info.duration:.1f}s, audio={'yes' if has_audio else 'no'}",
    )
