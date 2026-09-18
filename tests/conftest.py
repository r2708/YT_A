from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_test_video import SCENES, make_test_video  # noqa: E402

from video_dataset.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("video") / "synthetic_test.mp4"
    return make_test_video(out)


@pytest.fixture(scope="session")
def expected_scenes() -> list[tuple[float, float]]:
    return SCENES


@pytest.fixture()
def test_config(tmp_path: Path):
    """Fast, network-free configuration: mock ASR, heuristic vision, energy audio, real OCR if available."""
    overrides = {
        "project.data_dir": str(tmp_path / "data"),
        "project.log_level": "WARNING",
        "transcription.provider": "mock",
        "audio_events.provider": "energy",
        "vision.provider": "heuristic",
        "ocr.provider": "rapidocr",
        "frame_sampling.extract_clips": "true",
        "frame_sampling.clip_codec": "libx264",
        "scene_detection.max_scene_duration": "60",
        "qa.min_questions": "8",
        "pipeline.stage_retries": "0",
    }
    return load_config(None, overrides, config_dir=ROOT / "config")
