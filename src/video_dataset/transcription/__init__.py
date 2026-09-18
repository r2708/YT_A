"""Speech recognition adapters (faster-whisper, transformers Whisper, mock)."""

from video_dataset.transcription.base import Transcriber, create_transcriber

__all__ = ["Transcriber", "create_transcriber"]
