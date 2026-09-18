"""AUDIO stage: make sure audio.wav exists, measure it, and detect non-speech sound segments."""

from __future__ import annotations

from video_dataset.audio.events import create_audio_event_detector
from video_dataset.audio.wav import wav_info
from video_dataset.pipeline.context import StageOutput, VideoContext
from video_dataset.preprocessing.normalize import extract_audio
from video_dataset.schemas.transcript import AudioAnalysis
from video_dataset.stages import Stage
from video_dataset.utils.device import resolve_device
from video_dataset.utils.io import write_json_atomic


def audio_stage(ctx: VideoContext) -> StageOutput:
    log = ctx.logger(Stage.AUDIO)
    cfg = ctx.config
    wav = ctx.paths.audio_file(ctx.video_id)
    out = ctx.paths.audio_events_file(ctx.video_id)
    has_audio = ctx.media_info.has_audio
    if has_audio and not wav.exists():
        has_audio = extract_audio(ctx.video_path, wav, cfg.preprocess.audio_sample_rate, cfg.preprocess.audio_channels, cfg.project.ffmpeg_path)
    if not has_audio:
        analysis = AudioAnalysis(video_id=ctx.video_id, provider="none", has_audio=False)
        write_json_atomic(out, analysis)
        return StageOutput(artifact_path=str(out), metrics={"has_audio": False, "events": 0}, message="no audio stream", skipped=False)

    _sr, _ch, duration = wav_info(wav)
    detector = ctx.get_model(
        f"audio_events:{cfg.audio_events.provider}:{cfg.audio_events.model}",
        lambda: create_audio_event_detector(cfg.audio_events, resolve_device(cfg.project.device)),
    )
    events = detector.detect(wav) if cfg.audio_events.enabled else []
    analysis = AudioAnalysis(
        video_id=ctx.video_id,
        provider=detector.name,
        model=getattr(detector, "model_name", None),
        has_audio=True,
        duration=round(duration, 3),
        events=events,
    )
    write_json_atomic(out, analysis)
    cats: dict[str, int] = {}
    for e in events:
        cats[str(e.category)] = cats.get(str(e.category), 0) + 1
    log.info("audio %.1fs, %d sound segments %s", duration, len(events), cats)
    return StageOutput(artifact_path=str(out), metrics={"has_audio": True, "events": len(events), "categories": cats, "provider": detector.name}, message=f"{len(events)} audio segments ({detector.name})")
