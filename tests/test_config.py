from pathlib import Path

import pytest

from video_dataset.config import DEFAULT_CONFIG_DIR, apply_dotted_overrides, deep_merge, load_config


def test_load_defaults_and_overrides(tmp_path: Path):
    cfg = load_config(None, {"vision.provider": "mock", "qa.questions_per_minute": "7", "project.data_dir": str(tmp_path)})
    assert cfg.vision.provider == "mock"
    assert cfg.qa.questions_per_minute == 7.0
    assert cfg.data_dir == tmp_path
    assert cfg.db_path == tmp_path / "state.db"
    assert [s for s in cfg.pipeline.stages][:2] == ["DOWNLOAD", "PREPROCESS"]


def test_presets_resolve():
    cfg = load_config(None, {"vision.preset": "smolvlm2_256m"})
    assert cfg.vision.provider == "hf"
    assert cfg.vision.model == "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
    with pytest.raises(ValueError):
        load_config(None, {"vision.preset": "does_not_exist"})


def test_user_config_file(tmp_path: Path):
    p = tmp_path / "my.yaml"
    p.write_text("scene_detection:\n  threshold: 15\ntranscription:\n  model: tiny\n")
    cfg = load_config(p)
    assert cfg.scene_detection.threshold == 15
    assert cfg.transcription.model == "tiny"
    assert (DEFAULT_CONFIG_DIR / "default.yaml").exists()


def test_deep_merge_and_dotted():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    out = deep_merge(base, {"a": {"c": 5}})
    assert out == {"a": {"b": 1, "c": 5}, "d": 3} and base["a"]["c"] == 2
    out2 = apply_dotted_overrides(base, {"a.e.f": "true", "d": "[1,2]"})
    assert out2["a"]["e"]["f"] is True and out2["d"] == [1, 2]
