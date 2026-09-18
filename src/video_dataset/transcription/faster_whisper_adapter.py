"""faster-whisper (CTranslate2) adapter. CUDA float16 when available, otherwise CPU int8."""

from __future__ import annotations

import math
from pathlib import Path

from video_dataset.config import TranscriptionConfig
from video_dataset.schemas.common import ConfidenceSource
from video_dataset.schemas.transcript import Transcript, TranscriptSegment, Word
from video_dataset.utils.device import resolve_device
from video_dataset.utils.ids import segment_id
from video_dataset.utils.logging import get_logger

log = get_logger("transcription.faster_whisper")


def segment_confidence(avg_logprob: float | None, no_speech_prob: float | None) -> float | None:
    """exp(avg token logprob) * (1 - no_speech_prob): both are real decoder signals, not guesses."""
    if avg_logprob is None:
        return None
    p = math.exp(min(0.0, float(avg_logprob)))
    if no_speech_prob is not None:
        p *= 1.0 - max(0.0, min(1.0, float(no_speech_prob)))
    return round(max(0.0, min(1.0, p)), 4)


class FasterWhisperTranscriber:
    name = "faster_whisper"

    def __init__(self, cfg: TranscriptionConfig, device_pref: str = "auto"):
        from faster_whisper import WhisperModel

        self.cfg = cfg
        device = resolve_device(device_pref)
        ct2_device = "cuda" if device == "cuda" else "cpu"  # CTranslate2 has no MPS backend
        compute = cfg.compute_type if cfg.compute_type != "auto" else ("float16" if ct2_device == "cuda" else "int8")
        log.info("loading faster-whisper '%s' on %s (%s)", cfg.model, ct2_device, compute)
        self.model = WhisperModel(cfg.model, device=ct2_device, compute_type=compute)
        self.model_name: str | None = cfg.model

    def transcribe(self, wav_path: Path, video_id: str, language: str | None = None) -> Transcript:
        lang = None if (language or self.cfg.language) in (None, "", "auto") else (language or self.cfg.language)
        segments_iter, info = self.model.transcribe(
            str(wav_path),
            language=lang,
            beam_size=int(self.cfg.beam_size),
            vad_filter=bool(self.cfg.vad_filter),
            word_timestamps=bool(self.cfg.word_timestamps),
            condition_on_previous_text=False,  # reduces repetition hallucinations on long audio
            initial_prompt=self.cfg.initial_prompt or None,
        )
        segments: list[TranscriptSegment] = []
        n = 0
        for seg in segments_iter:
            text = (seg.text or "").strip()
            if not text:
                continue
            conf = segment_confidence(seg.avg_logprob, seg.no_speech_prob)
            if conf is not None and conf < self.cfg.min_segment_confidence:
                continue
            n += 1
            words = [
                Word(start=round(w.start, 3), end=round(w.end, 3), word=w.word.strip(), probability=round(w.probability, 4) if w.probability is not None else None)
                for w in (seg.words or [])
            ]
            segments.append(
                TranscriptSegment(
                    segment_id=segment_id(n),
                    start=round(max(0.0, seg.start), 3),
                    end=round(max(seg.start, seg.end), 3),
                    text=text,
                    confidence=conf,
                    confidence_source=ConfidenceSource.ASR_LOGPROB if conf is not None else ConfidenceSource.UNAVAILABLE,
                    avg_logprob=round(seg.avg_logprob, 4) if seg.avg_logprob is not None else None,
                    no_speech_prob=round(seg.no_speech_prob, 4) if seg.no_speech_prob is not None else None,
                    words=words,
                )
            )
        return Transcript(
            video_id=video_id,
            provider=self.name,
            model=self.model_name,
            language=getattr(info, "language", None),
            language_probability=round(float(info.language_probability), 4) if getattr(info, "language_probability", None) is not None else None,
            has_speech=bool(segments),
            segments=segments,
        )
