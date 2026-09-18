from pathlib import Path

from video_dataset.audio.events import EnergyAudioEventDetector, categorize_label
from video_dataset.config import load_config
from video_dataset.ocr.merge import merge_detections
from video_dataset.preprocessing.normalize import extract_audio
from video_dataset.schemas.ocr import OCRDetection
from video_dataset.schemas.scene import Scene, SceneDetectionResult, TransitionType
from video_dataset.schemas.transcript import AudioEventCategory
from video_dataset.transcription.faster_whisper_adapter import segment_confidence
from video_dataset.transcription.mock import MockTranscriber
from video_dataset.transcription.stage import attach_scene_ids


def test_segment_confidence_derivation():
    assert segment_confidence(None, None) is None
    assert segment_confidence(0.0, 0.0) == 1.0
    c = segment_confidence(-0.5, 0.2)
    assert 0.0 < c < 1.0 and abs(c - (2.718281828 ** -0.5) * 0.8) < 1e-3


def test_mock_transcriber_and_scene_attachment(tmp_path: Path):
    tr = MockTranscriber([(0.5, 2.0, "hello there"), (4.5, 6.0, "second shot")]).transcribe(tmp_path / "x.wav", "vid")
    assert tr.has_speech and len(tr.segments) == 2 and tr.segments[0].segment_id == "seg_0001"
    scenes = SceneDetectionResult(video_id="vid", detector="t", duration=8.0, scenes=[
        Scene(scene_id="scene_001", video_id="vid", index=0, start_time=0, end_time=4, start_frame=0, end_frame=96, transition_in=TransitionType.START),
        Scene(scene_id="scene_002", video_id="vid", index=1, start_time=4, end_time=8, start_frame=96, end_frame=192),
    ])
    tr = attach_scene_ids(tr, scenes)
    assert tr.segments[0].scene_ids == ["scene_001"] and tr.segments[1].scene_ids == ["scene_002"]
    assert tr.text_between(0, 3) == "hello there"


def test_ocr_merge_groups_persisting_text():
    dets = [
        OCRDetection(detection_id="d1", frame_id="f1", scene_id="scene_001", timestamp=1.0, text="NEW YORK", confidence=0.9, bbox=[10, 10, 100, 30]),
        OCRDetection(detection_id="d2", frame_id="f2", scene_id="scene_001", timestamp=2.0, text="NEW Y0RK", confidence=0.7, bbox=[12, 10, 102, 30]),
        OCRDetection(detection_id="d3", frame_id="f3", scene_id="scene_001", timestamp=3.0, text="New York", confidence=0.95, bbox=[10, 11, 100, 31]),
        OCRDetection(detection_id="d4", frame_id="f9", scene_id="scene_003", timestamp=30.0, text="NEW YORK", confidence=0.9, bbox=[10, 10, 100, 30]),
        OCRDetection(detection_id="d5", frame_id="f2", scene_id="scene_001", timestamp=2.0, text="EXIT", confidence=0.8, bbox=[300, 300, 340, 320]),
    ]
    tracks = merge_detections(dets, similarity=0.85, max_gap=3.0)
    texts = sorted(t.normalized_text for t in tracks)
    assert texts == ["exit", "new york", "new york"]  # far-apart repeat is a separate track
    ny = next(t for t in tracks if t.first_seen == 1.0)
    assert ny.num_detections == 3 and ny.last_seen == 3.0 and ny.text == "New York"  # highest confidence spelling wins
    assert ny.bbox == [10, 10, 102, 31] and abs(ny.mean_confidence - 0.85) < 1e-6


def test_categorize_audio_labels():
    assert categorize_label("Music") == AudioEventCategory.MUSIC
    assert categorize_label("Vehicle horn, car horn, honking") == AudioEventCategory.TRAFFIC
    assert categorize_label("Dog") == AudioEventCategory.ANIMAL
    assert categorize_label("Zipper (clothing)") == AudioEventCategory.OTHER


def test_energy_detector_finds_tone_and_silence(synthetic_video: Path, tmp_path: Path):
    wav = tmp_path / "a.wav"
    assert extract_audio(synthetic_video, wav, 16000, 1)
    cfg = load_config(None, {}).audio_events
    events = EnergyAudioEventDetector(cfg).detect(wav)
    sound = [e for e in events if e.category == AudioEventCategory.NON_SPEECH_SOUND]
    silence = [e for e in events if e.category == AudioEventCategory.SILENCE]
    assert sound and silence
    assert any(e.start < 0.6 and 4.0 <= e.end <= 6.0 for e in sound), sound
    assert any(4.5 <= e.start <= 6.0 and 8.5 <= e.end <= 10.0 for e in silence), silence
    assert all(e.score is not None and 0 <= e.score <= 1 for e in events)
