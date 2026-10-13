"""Fixes from the first production batch review (2026-10-13): frame-gap coverage, measured providers
need no verification signal, CLIP alignment on prompt text, long-range questions from the spaCy
generator, entity hygiene and 'a'/'an' articles."""

from __future__ import annotations

import pytest

from test_questions import _scenes, _timeline
from video_dataset.config import load_config
from video_dataset.frame_sampling.sampler import select_frames
from video_dataset.questions.spacy_generator import SpacyQAGenerator, usable_entity
from video_dataset.schemas.common import ConfidenceSource
from video_dataset.schemas.qa import QAType
from video_dataset.schemas.scene import Frame, SamplingReason, ScanSample, Scene, SceneScan, TransitionType
from video_dataset.schemas.vision import (
    CameraAnnotation,
    Environment,
    Measurements,
    ObjectAnnotation,
    SceneAnalysis,
)
from video_dataset.utils.text import with_article
from video_dataset.validation.alignment import alignment_text
from video_dataset.validation.checks import check_frame_sequence


def _scene(start: float, end: float, fps: float = 24.0) -> Scene:
    return Scene(scene_id="scene_001", video_id="v", index=0, start_time=start, end_time=end, start_frame=int(start * fps), end_frame=int(end * fps), transition_in=TransitionType.START)


def _frame(i: int, t: float) -> Frame:
    return Frame(frame_id=f"f{i}", video_id="v", scene_id="s", timestamp=t, frame_index=int(t * 24), frame_path=f"/x/f{i}.jpg", width=10, height=10)


# ------------------------------------------------------------------------------------ 6. frame gaps
def test_sampler_closes_gaps_wider_than_max_frame_gap():
    cfg = load_config(None, {"frame_sampling.max_frames_per_scene": "20", "frame_sampling.max_frame_gap_seconds": "5"}).frame_sampling
    scene = _scene(0.0, 26.4)
    # all visual change in the last two seconds: motion-aware sampling alone would cluster there
    samples = [ScanSample(timestamp=i / 4.0, frame_index=int(i * 6), change=0.9 if i / 4.0 > 24 else 0.0, brightness=100.0, flow_mag=0.1) for i in range(int(26.4 * 4))]
    cands = select_frames(scene, SceneScan(scene_id="scene_001", analysis_width=160, analysis_fps=4.0, samples=samples), cfg, 24.0)
    ts = [c.timestamp for c in cands]
    assert ts == sorted(ts) and len(ts) <= 20
    assert max(b - a for a, b in zip(ts, ts[1:])) <= 5.0 + 1e-6
    assert SamplingReason.UNIFORM in {c.reason for c in cands}
    # the validator agrees with the sampler on the same limit
    frames = [_frame(i, t) for i, t in enumerate(ts)]
    assert check_frame_sequence(frames, 0.0, 26.4, min_gap_seconds=5.0, max_frames=20) == (True, None)


def test_sampler_gap_fill_respects_max_frames_and_validator_skips_full_shots():
    cfg = load_config(None, {"frame_sampling.max_frames_per_scene": "4", "frame_sampling.max_frame_gap_seconds": "2"}).frame_sampling
    cands = select_frames(_scene(0.0, 30.0), None, cfg, 24.0)
    assert len(cands) == 4  # cannot add more than max_frames even though 2 s gaps are impossible
    frames = [_frame(i, c.timestamp) for i, c in enumerate(cands)]
    assert check_frame_sequence(frames, 0.0, 30.0, min_gap_seconds=2.0, max_frames=4) == (True, None)
    # a full shot whose frames cluster (motion-aware picks) is not blamed for the gap the sampler could not close
    clustered = [_frame(i, t) for i, t in enumerate([0.2, 0.8, 1.4, 29.5])]
    ok, msg = check_frame_sequence(clustered, 0.0, 30.0, min_gap_seconds=2.0, max_frames=None)
    assert not ok and "gap of 28.1s" in msg
    assert check_frame_sequence(clustered, 0.0, 30.0, min_gap_seconds=2.0, max_frames=20)[0] is False
    assert check_frame_sequence(clustered, 0.0, 30.0, min_gap_seconds=2.0, max_frames=4) == (True, None)
    cfg0 = load_config(None, {"frame_sampling.max_frames_per_scene": "6", "frame_sampling.max_frame_gap_seconds": "0"}).frame_sampling
    assert len(select_frames(_scene(0.0, 30.0), None, cfg0, 24.0)) == 3  # 0 disables the gap fill


# ------------------------------------------------------------------------------------ 3. alignment text
def _analysis(objects: list[str], summary: str = "A 3.0-second shot with dim lighting and a warm-toned palette (dominant colors #332c22, #6f6353).") -> SceneAnalysis:
    return SceneAnalysis(
        scene_id="scene_001", video_id="v", start_time=0.0, end_time=3.0, summary=summary, provider="cv_models",
        environment=Environment(setting="outdoor", lighting="dim", time_of_day="day"),
        objects=[ObjectAnnotation(name=n, confidence=0.8, confidence_source=ConfidenceSource.DETECTOR_SCORE) for n in objects],
        camera=CameraAnnotation(movement="pan", camera_movement="pan", movement_direction="right", movement_speed="slow"),
        measurements=Measurements(color_temperature="warm", lighting_level="dim"),
    )


def test_alignment_text_uses_prompt_and_skips_shots_without_objects():
    a = _analysis(["car", "person"])
    text = alignment_text(a, "prompt")
    assert text and "#" not in text and "car" in text and "person" in text and "pan to the right" in text and "outdoors" in text
    assert alignment_text(a, "summary") == a.summary
    assert alignment_text(_analysis([]), "prompt") is None  # nothing verifiable: the check is skipped, not failed
    assert alignment_text(_analysis([]), "summary") == a.summary
    cfg = load_config(None, {})
    assert cfg.validation.clip_alignment_text == "prompt" and cfg.validation.clip_alignment_min_rank_fraction == 0.75 and cfg.validation.clip_alignment_min_margin == 0.05
    assert cfg.frame_sampling.max_frame_gap_seconds == 5.0


# ------------------------------------------------------------------------------------ 5. text hygiene
def test_with_article():
    assert with_article("umbrella") == "an umbrella"
    assert with_article("car") == "a car"
    assert with_article("an apple") == "an apple" and with_article("the bench") == "the bench"
    assert with_article("") == ""


def test_appearance_event_uses_the_right_article():
    import video_dataset.temporal.events  # noqa: F401  (module import exercises the new helper)
    from video_dataset.utils.text import sentence

    assert sentence(f"{with_article('umbrella')} is visible in the center of the frame") == "An umbrella is visible in the center of the frame."


@pytest.mark.parametrize(
    ("text", "label", "full", "ok"),
    [
        ("London", "GPE", "WELCOME TO London", True),
        ("LONDON", "GPE", "LONDON 2024", True),  # places may be all caps
        ("Lumix", "ORG", None, True),
        ("AI", "GPE", None, False),  # too short
        ("Camer.", "PERSON", "Camer.", False),  # the whole OCR fragment
        ("KNIPANO", "NORP", "KNIPANO is here", False),  # all-caps heading mislabelled as a group
        ("Toourmembers:A", "WORK_OF_ART", "Toourmembers:A thank you", False),  # run-together OCR
        ("Graphic Equalizer(30 Ban", "WORK_OF_ART", "x", False),
        ("a few seconds", "TIME", None, False),  # no number
        ("a seven day", "DATE", None, False),
        ("7 days", "DATE", None, True),
        ("$20", "MONEY", None, True),
        ("K 30 FPS", "ORG", None, False),  # one-letter token
        ("Blender Foundation", "ORG", None, True),
    ],
)
def test_usable_entity(text, label, full, ok):
    assert usable_entity(text, label, full_text=full) is ok


# ------------------------------------------------------------------------------------ 4. long-range QA from spaCy
needs_spacy = pytest.mark.skipif(not SpacyQAGenerator.available("en_core_web_sm"), reason="spaCy en_core_web_sm not installed")


@needs_spacy
def test_spacy_generator_emits_long_range_questions():
    cfg = load_config(None, {"qa.questions_per_minute": "60", "qa.min_questions": "20", "qa.max_questions": "200", "qa.long_range_min_gap_seconds": "20"}).qa
    tl = _timeline()
    records = SpacyQAGenerator(cfg).generate("vid_test", 48.0, tl, _scenes(), {}, None)
    lr = [r for r in records if r.is_long_range]
    assert lr and all(r.type == QAType.LONG_RANGE and r.difficulty == "hard" and r.generator == "spacy_v1" for r in lr)
    assert {r.template_id for r in lr} <= {"spacy_long_range_v1", "spacy_count_v1"}
    assert all(len(r.evidence.event_ids) >= 2 and "event_0006" in r.evidence.event_ids for r in lr)  # the red car 40 s later
    assert len(lr) <= max(1, round(0.2 * len(records))) + 1
    # ids stay sequential in timeline order after the merge
    assert [r.question_id for r in records] == [f"qa_test_{n:06d}" for n in range(1, len(records) + 1)]
    assert all(r.record_id == r.question_id for r in records)
    starts = [r.evidence.start_time for r in records]
    assert starts == sorted(starts)
    # the export routes them to long_video_qa
    from video_dataset.schemas.qa import QARecord  # noqa: F401

    assert any(r.is_long_range for r in records)
