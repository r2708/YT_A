"""Non-speech audio event detection.

Two implementations:
* EnergyAudioEventDetector - model free. Measures where sound is present / absent. It never claims a
  sound *type* (no guessing), only `non_speech_sound` / `silence` segments with a coverage score.
* TransformersAudioEventDetector - an AudioSet classifier (default: MIT AST). Reports the classifier's
  own class probabilities; labels below `min_score` are dropped.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from video_dataset.audio.wav import iter_windows, rms_db, wav_info
from video_dataset.config import AudioEventsConfig
from video_dataset.schemas.common import ConfidenceSource
from video_dataset.schemas.transcript import AudioEvent, AudioEventCategory
from video_dataset.utils.ids import audio_event_id
from video_dataset.utils.logging import get_logger

log = get_logger("audio.events")

# AudioSet label keyword -> our coarse category. Unknown labels stay OTHER with their raw label preserved.
_CATEGORY_KEYWORDS: list[tuple[AudioEventCategory, tuple[str, ...]]] = [
    (AudioEventCategory.SPEECH, ("speech", "conversation", "narration", "monologue", "talk", "whispering", "shout", "yell")),
    (AudioEventCategory.MUSIC, ("music", "song", "singing", "guitar", "piano", "drum", "orchestra", "synthesizer", "melody", "choir", "violin", "bass", "strings", "hip hop", "jazz", "rock", "electronic")),
    (AudioEventCategory.APPLAUSE, ("applause", "clapping", "cheering")),
    (AudioEventCategory.CROWD, ("crowd", "chatter", "hubbub", "babble", "children playing", "stadium")),
    (AudioEventCategory.TRAFFIC, ("vehicle", "car", "traffic", "engine", "motor", "truck", "bus", "motorcycle", "horn", "siren", "train", "aircraft", "helicopter", "airplane", "tire", "skidding", "race car")),
    (AudioEventCategory.EXPLOSION, ("explosion", "gunshot", "fireworks", "bang", "boom", "burst", "eruption", "artillery", "blast")),
    (AudioEventCategory.FOOTSTEPS, ("footsteps", "walk", "running", "shuffle", "stomp")),
    (AudioEventCategory.ANIMAL, ("dog", "cat", "bird", "animal", "bark", "meow", "chirp", "horse", "cow", "insect", "roar", "howl", "livestock", "frog", "crow", "duck", "goat", "sheep", "pig")),
    (AudioEventCategory.ENVIRONMENTAL, ("wind", "rain", "thunder", "water", "waves", "stream", "ocean", "fire", "crackle", "rustling", "leaves", "environmental", "outside", "ambient", "noise", "hum", "rumble", "splash", "gurgling")),
    (AudioEventCategory.SILENCE, ("silence",)),
]


def categorize_label(label: str) -> AudioEventCategory:
    low = label.lower()
    for cat, kws in _CATEGORY_KEYWORDS:
        if any(k in low for k in kws):
            return cat
    return AudioEventCategory.OTHER


class AudioEventDetector(Protocol):
    name: str

    def detect(self, wav_path: Path) -> list[AudioEvent]: ...


def _merge_windows(windows: Iterable[tuple[float, float, str, AudioEventCategory, float | None]], max_gap: float) -> list[tuple[float, float, str, AudioEventCategory, list[float]]]:
    """Merge consecutive windows with the same label into continuous segments."""
    merged: list[tuple[float, float, str, AudioEventCategory, list[float]]] = []
    for start, end, label, cat, score in sorted(windows, key=lambda w: (w[0], w[2])):
        for i, (ms, me, ml, mc, scores) in enumerate(merged):
            if ml == label and start <= me + max_gap and start >= ms - 1e-6:
                merged[i] = (ms, max(me, end), ml, mc, scores + ([score] if score is not None else []))
                break
        else:
            merged.append((start, end, label, cat, [score] if score is not None else []))
    return merged


class EnergyAudioEventDetector:
    name = "energy"

    def __init__(self, cfg: AudioEventsConfig):
        self.cfg = cfg

    def detect(self, wav_path: Path) -> list[AudioEvent]:
        window, hop = 1.0, 0.5
        raw: list[tuple[float, float, str, AudioEventCategory, float | None]] = []
        for start, samples, _sr in iter_windows(wav_path, window, hop):
            db = rms_db(samples)
            loud = db > self.cfg.energy_threshold_db
            label = "sound_present" if loud else "silence"
            cat = AudioEventCategory.NON_SPEECH_SOUND if loud else AudioEventCategory.SILENCE
            raw.append((start, start + window, label, cat, 1.0 if loud else 0.0))
        merged = _merge_windows(raw, max_gap=hop + 1e-3)
        events: list[AudioEvent] = []
        n = 0
        _sr, _ch, duration = wav_info(wav_path)
        for start, end, label, cat, scores in merged:
            end = min(end, duration) if duration else end
            if end - start < 1.0:  # ignore blips
                continue
            n += 1
            events.append(
                AudioEvent(
                    event_id=audio_event_id(n),
                    start=round(start, 3),
                    end=round(end, 3),
                    label=label,
                    category=cat,
                    # coverage = fraction of windows consistent with the label; a measured quantity, not a type probability
                    score=round(float(np.mean(scores)) if cat == AudioEventCategory.NON_SPEECH_SOUND else 1.0 - float(np.mean(scores)), 3) if scores else None,
                    confidence_source=ConfidenceSource.MEASUREMENT,
                )
            )
        return events


class TransformersAudioEventDetector:
    name = "transformers"

    def __init__(self, cfg: AudioEventsConfig, device: str = "cpu"):
        from transformers import pipeline

        self.cfg = cfg
        dev: Any = 0 if device == "cuda" else (device if device == "mps" else -1)
        self.pipe = pipeline("audio-classification", model=cfg.model, device=dev, top_k=int(cfg.top_k))
        self.model_name = cfg.model

    def detect(self, wav_path: Path) -> list[AudioEvent]:
        raw: list[tuple[float, float, str, AudioEventCategory, float | None]] = []
        for start, samples, sr in iter_windows(wav_path, self.cfg.window_seconds, self.cfg.hop_seconds):
            if len(samples) < sr * 0.25:
                continue
            preds = self.pipe({"raw": samples, "sampling_rate": sr})
            for p in preds:
                score = float(p["score"])
                if score < self.cfg.min_score:
                    continue
                label = str(p["label"])
                cat = categorize_label(label)
                if cat == AudioEventCategory.SPEECH:
                    continue  # speech is covered by ASR
                raw.append((start, start + self.cfg.window_seconds, label, cat, score))
        merged = _merge_windows(raw, max_gap=self.cfg.hop_seconds + 1e-3)
        events: list[AudioEvent] = []
        for n, (start, end, label, cat, scores) in enumerate(merged, start=1):
            events.append(
                AudioEvent(
                    event_id=audio_event_id(n),
                    start=round(start, 3),
                    end=round(end, 3),
                    label=label,
                    category=cat,
                    score=round(float(np.mean(scores)), 3) if scores else None,
                    confidence_source=ConfidenceSource.DETECTOR_SCORE,
                )
            )
        return events


class NullAudioEventDetector:
    name = "none"

    def detect(self, wav_path: Path) -> list[AudioEvent]:
        return []


def create_audio_event_detector(cfg: AudioEventsConfig, device: str = "cpu") -> AudioEventDetector:
    if not cfg.enabled or cfg.provider == "none":
        return NullAudioEventDetector()
    if cfg.provider == "transformers":
        try:
            return TransformersAudioEventDetector(cfg, device)
        except Exception as exc:
            log.warning("audio classifier unavailable (%s); falling back to energy detector", exc)
            return EnergyAudioEventDetector(cfg)
    return EnergyAudioEventDetector(cfg)
