"""Speech transcription and non-speech audio event schemas."""

from __future__ import annotations

from pydantic import Field

from video_dataset.schemas.common import BaseSchema, ConfidenceSource, StrEnum


class Word(BaseSchema):
    start: float
    end: float
    word: str
    probability: float | None = None


class TranscriptSegment(BaseSchema):
    segment_id: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_source: ConfidenceSource = ConfidenceSource.UNAVAILABLE
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    words: list[Word] = Field(default_factory=list)
    scene_ids: list[str] = Field(default_factory=list)


class Transcript(BaseSchema):
    video_id: str
    provider: str
    model: str | None = None
    language: str | None = None
    language_probability: float | None = None
    has_speech: bool = True
    segments: list[TranscriptSegment] = Field(default_factory=list)

    def text_between(self, start: float, end: float) -> str:
        parts = [s.text.strip() for s in self.segments if s.end > start and s.start < end]
        return " ".join(p for p in parts if p)


class AudioEventCategory(StrEnum):
    MUSIC = "music"
    SPEECH = "speech"
    APPLAUSE = "applause"
    TRAFFIC = "traffic"
    EXPLOSION = "explosion"
    FOOTSTEPS = "footsteps"
    ENVIRONMENTAL = "environmental"
    CROWD = "crowd"
    ANIMAL = "animal"
    SILENCE = "silence"
    NON_SPEECH_SOUND = "non_speech_sound"  # energy present but type not identified (never guessed)
    OTHER = "other"


class AudioEvent(BaseSchema):
    event_id: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    label: str  # raw model label (e.g. AudioSet class name) or heuristic label
    category: AudioEventCategory
    score: float | None = Field(default=None, ge=0, le=1)
    confidence_source: ConfidenceSource = ConfidenceSource.UNAVAILABLE


class AudioAnalysis(BaseSchema):
    video_id: str
    provider: str
    model: str | None = None
    has_audio: bool
    duration: float | None = None
    speech_ratio: float | None = None  # fraction of duration covered by transcript segments
    events: list[AudioEvent] = Field(default_factory=list)
