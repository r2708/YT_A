"""Audio analysis: level/silence measurement and optional non-speech sound event classification."""

from video_dataset.audio.events import (
    EnergyAudioEventDetector,
    TransformersAudioEventDetector,
    create_audio_event_detector,
)

__all__ = ["EnergyAudioEventDetector", "TransformersAudioEventDetector", "create_audio_event_detector"]
