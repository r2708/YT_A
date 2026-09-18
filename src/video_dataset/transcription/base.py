from __future__ import annotations

from pathlib import Path
from typing import Protocol

from video_dataset.config import TranscriptionConfig
from video_dataset.schemas.transcript import Transcript
from video_dataset.utils.logging import get_logger

log = get_logger("transcription")


class Transcriber(Protocol):
    name: str
    model_name: str | None

    def transcribe(self, wav_path: Path, video_id: str, language: str | None = None) -> Transcript: ...


def create_transcriber(cfg: TranscriptionConfig, device_pref: str = "auto") -> Transcriber:
    provider = (cfg.provider or "none").lower()
    if provider == "mock":
        from video_dataset.transcription.mock import MockTranscriber

        return MockTranscriber()
    if provider == "none":
        from video_dataset.transcription.mock import MockTranscriber

        return MockTranscriber(name="none")
    if provider == "transformers":
        from video_dataset.transcription.transformers_adapter import TransformersWhisperTranscriber

        return TransformersWhisperTranscriber(cfg, cfg.device or device_pref)
    if provider == "faster_whisper":
        try:
            from video_dataset.transcription.faster_whisper_adapter import FasterWhisperTranscriber

            return FasterWhisperTranscriber(cfg, cfg.device or device_pref)
        except ImportError as exc:
            log.warning("faster-whisper not installed (%s); trying transformers Whisper", exc)
            from video_dataset.transcription.transformers_adapter import TransformersWhisperTranscriber

            return TransformersWhisperTranscriber(cfg, cfg.device or device_pref)
    raise ValueError(f"Unknown transcription provider '{cfg.provider}'")
