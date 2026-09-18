"""TRANSCRIPTION stage: ASR over audio.wav -> transcripts/<video_id>.json with scene ids attached."""

from __future__ import annotations

from video_dataset.pipeline.context import StageOutput, VideoContext
from video_dataset.schemas.scene import SceneDetectionResult
from video_dataset.schemas.transcript import Transcript
from video_dataset.stages import Stage
from video_dataset.transcription.base import create_transcriber
from video_dataset.utils.io import read_json, write_json_atomic


def attach_scene_ids(transcript: Transcript, scenes: SceneDetectionResult | None) -> Transcript:
    if scenes is None:
        return transcript
    for seg in transcript.segments:
        seg.scene_ids = [s.scene_id for s in scenes.scenes if s.start_time < seg.end and s.end_time > seg.start]
    return transcript


def transcription_stage(ctx: VideoContext) -> StageOutput:
    log = ctx.logger(Stage.TRANSCRIPTION)
    cfg = ctx.config.transcription
    out = ctx.paths.transcript_file(ctx.video_id)
    wav = ctx.paths.audio_file(ctx.video_id)
    scenes_path = ctx.paths.scenes_file(ctx.video_id)
    scenes = SceneDetectionResult.model_validate(read_json(scenes_path)) if scenes_path.exists() else None

    if not cfg.enabled or cfg.provider == "none" or not ctx.media_info.has_audio or not wav.exists():
        reason = "disabled" if (not cfg.enabled or cfg.provider == "none") else "no audio"
        transcript = Transcript(video_id=ctx.video_id, provider="none", has_speech=False)
        write_json_atomic(out, transcript)
        return StageOutput(artifact_path=str(out), metrics={"segments": 0, "reason": reason}, message=f"skipped ({reason})", skipped=(reason == "disabled"))

    transcriber = ctx.get_model(f"asr:{cfg.provider}:{cfg.model}", lambda: create_transcriber(cfg, ctx.config.project.device))
    transcript = transcriber.transcribe(wav, ctx.video_id)
    transcript = attach_scene_ids(transcript, scenes)
    write_json_atomic(out, transcript)
    words = sum(len(s.text.split()) for s in transcript.segments)
    log.info("%d segments, %d words, language=%s", len(transcript.segments), words, transcript.language)
    return StageOutput(
        artifact_path=str(out),
        metrics={"segments": len(transcript.segments), "words": words, "language": transcript.language, "provider": transcriber.name, "model": transcriber.model_name},
        message=f"{len(transcript.segments)} transcript segments",
    )
