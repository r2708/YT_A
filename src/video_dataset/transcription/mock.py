"""Deterministic mock transcriber for tests / pipelines without ASR."""

from __future__ import annotations

from pathlib import Path

from video_dataset.schemas.common import ConfidenceSource
from video_dataset.schemas.transcript import Transcript, TranscriptSegment
from video_dataset.utils.ids import segment_id


class MockTranscriber:
    def __init__(self, segments: list[tuple[float, float, str]] | None = None, name: str = "mock"):
        self.name = name
        self.model_name: str | None = None
        self._segments = segments or []

    def transcribe(self, wav_path: Path, video_id: str, language: str | None = None) -> Transcript:
        segs = [
            TranscriptSegment(
                segment_id=segment_id(i + 1),
                start=s,
                end=e,
                text=t,
                confidence=None,
                confidence_source=ConfidenceSource.UNAVAILABLE,
            )
            for i, (s, e, t) in enumerate(self._segments)
        ]
        return Transcript(video_id=video_id, provider=self.name, model=None, language="en" if segs else None, has_speech=bool(segs), segments=segs)
