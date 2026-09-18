"""Hugging Face transformers Whisper adapter (works on MPS / CUDA / CPU). No per-segment confidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from video_dataset.config import TranscriptionConfig
from video_dataset.schemas.common import ConfidenceSource
from video_dataset.schemas.transcript import Transcript, TranscriptSegment
from video_dataset.utils.device import resolve_device, torch_dtype
from video_dataset.utils.ids import segment_id
from video_dataset.utils.logging import get_logger

log = get_logger("transcription.transformers")


class TransformersWhisperTranscriber:
    name = "transformers"

    def __init__(self, cfg: TranscriptionConfig, device_pref: str = "auto"):
        from transformers import pipeline

        self.cfg = cfg
        device = resolve_device(device_pref)
        model = cfg.model if "/" in cfg.model else f"openai/whisper-{cfg.model}"
        log.info("loading transformers ASR '%s' on %s", model, device)
        dev: Any = 0 if device == "cuda" else (device if device == "mps" else -1)
        self.pipe = pipeline("automatic-speech-recognition", model=model, device=dev, torch_dtype=torch_dtype(device))
        self.model_name: str | None = model

    def transcribe(self, wav_path: Path, video_id: str, language: str | None = None) -> Transcript:
        lang = None if (language or self.cfg.language) in (None, "", "auto") else (language or self.cfg.language)
        gen: dict[str, Any] = {"task": "transcribe"}
        if lang:
            gen["language"] = lang
        result: Any = self.pipe(str(wav_path), return_timestamps=True, chunk_length_s=30, stride_length_s=5, generate_kwargs=gen)
        if isinstance(result, list):
            result = result[0] if result else {}
        segments: list[TranscriptSegment] = []
        n = 0
        last_end = 0.0
        for chunk in result.get("chunks", []):
            text = (chunk.get("text") or "").strip()
            ts = chunk.get("timestamp") or (None, None)
            start = float(ts[0]) if ts[0] is not None else last_end
            end = float(ts[1]) if ts[1] is not None else start
            if not text:
                continue
            n += 1
            last_end = max(end, start)
            segments.append(
                TranscriptSegment(
                    segment_id=segment_id(n),
                    start=round(start, 3),
                    end=round(max(start, end), 3),
                    text=text,
                    confidence=None,
                    confidence_source=ConfidenceSource.UNAVAILABLE,
                )
            )
        return Transcript(video_id=video_id, provider=self.name, model=self.model_name, language=lang, has_speech=bool(segments), segments=segments)
