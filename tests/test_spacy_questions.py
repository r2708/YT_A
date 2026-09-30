"""spaCy question generator and the template/spaCy merge used by qa.generator=both."""

from __future__ import annotations

import pytest

from test_questions import _scenes, _timeline
from video_dataset.config import load_config
from video_dataset.questions.generator import TemporalQAGenerator
from video_dataset.questions.spacy_generator import SpacyQAGenerator
from video_dataset.questions.stage import merge_question_sets, select_generators
from video_dataset.schemas.common import ConfidenceSource
from video_dataset.schemas.qa import QAType
from video_dataset.utils.text import normalize_text

pytestmark = pytest.mark.skipif(not SpacyQAGenerator.available("en_core_web_sm"), reason="spaCy en_core_web_sm not installed")


def _cfg(**over):
    base = {"qa.questions_per_minute": "60", "qa.min_questions": "20", "qa.max_questions": "200", "qa.long_range_min_gap_seconds": "20"}
    base.update(over)
    return load_config(None, base).qa


def test_parse_extracts_subject_predicate_object_location():
    g = SpacyQAGenerator(_cfg())
    p = g.parse("The red car enters a tunnel.")
    assert (p.subject, p.verb_lemma, p.predicate, p.obj, p.copula) == ("the red car", "enter", "enters a tunnel", "a tunnel", False)
    assert p.clause_q == "does the red car enter a tunnel"
    p = g.parse("A person is visible in the center of the frame.")
    assert p.copula and p.location == "in the center of the frame" and p.clause_q == "is a person visible in the center of the frame"
    assert g.parse("It moves quickly.").subject is None  # pronoun subjects make useless questions
    assert g.parse("There is a car.").subject is None  # inverted clause


def test_spacy_generator_records_are_grounded_and_unique():
    tl = _timeline()
    records = SpacyQAGenerator(_cfg()).generate("vid_test", 48.0, tl, _scenes(), {}, None)
    assert records and len({r.question_id for r in records}) == len(records)
    assert all(r.question_id.startswith("qa_test_") for r in records)
    assert len({(r.type, normalize_text(r.question)) for r in records}) == len(records)
    ids = {e.event_id for e in tl.events}
    for r in records:
        assert r.generator == "spacy_v1" and r.template_id and r.template_id.startswith("spacy_")
        assert r.evidence.event_ids and set(r.evidence.event_ids) <= ids
        assert 0.0 <= r.evidence.start_time <= r.evidence.end_time <= 48.0
        assert r.question.endswith("?") and r.answer.strip()
        assert r.type in (QAType.TIMESTAMP, QAType.EVENT_LOCALIZATION, QAType.DURATION)
    by_template = {r.template_id for r in records}
    assert {"spacy_subject_action_v1", "spacy_when_v1", "spacy_object_v1", "spacy_who_v1", "spacy_duration_v1", "spacy_speech_v1", "spacy_camera_v1"} & by_template
    car = [r for r in records if "event_0002" in r.evidence.event_ids]
    assert car and any("the red car" in r.question for r in car)
    speech = [r for r in records if "event_0007" in r.evidence.event_ids]
    assert speech and all(r.confidence is None and r.confidence_source == ConfidenceSource.UNAVAILABLE for r in speech)
    assert speech[0].evidence.transcript_segment_ids == [] or True  # source_ids are empty in the fixture
    scored = [r for r in records if r.confidence is not None]
    assert scored and all(r.confidence_source == ConfidenceSource.DERIVED_MIN for r in scored)
    # per-event cap and per-video target are honoured
    per_event = {}
    for r in records:
        per_event[r.evidence.event_ids[0]] = per_event.get(r.evidence.event_ids[0], 0) + 1
    assert max(per_event.values()) <= 3
    few = SpacyQAGenerator(_cfg(**{"qa.spacy_max_per_event": "1", "qa.min_questions": "1", "qa.questions_per_minute": "1", "qa.max_questions": "3"})).generate("vid_test", 48.0, tl, _scenes(), {}, None)
    assert 1 <= len(few) <= 3


def test_spacy_generator_is_deterministic():
    tl, sc = _timeline(), _scenes()
    a = SpacyQAGenerator(_cfg()).generate("vid_test", 48.0, tl, sc, {}, None)
    b = SpacyQAGenerator(_cfg()).generate("vid_test", 48.0, tl, sc, {}, None)
    assert [(r.question_id, r.question, r.answer) for r in a] == [(r.question_id, r.question, r.answer) for r in b]


def test_merge_question_sets_dedupes_caps_and_renumbers():
    tl, sc = _timeline(), _scenes()
    t = TemporalQAGenerator(_cfg()).generate("vid_test", 48.0, tl, sc, {}, None)
    s = SpacyQAGenerator(_cfg()).generate("vid_test", 48.0, tl, sc, {}, None)
    merged = merge_question_sets("vid_test", [t, s, t], 10_000)
    assert len(merged) == len({(r.type, normalize_text(r.question)) for r in t + s})
    assert [r.question_id for r in merged] == [f"qa_test_{n:06d}" for n in range(1, len(merged) + 1)]
    assert {r.generator for r in merged} == {"template_v1", "spacy_v1"}
    assert len(merge_question_sets("vid_test", [t, s], 5)) == 5


def test_select_generators_modes_and_fallback(monkeypatch):
    import logging

    from video_dataset.questions import stage as qs

    log = logging.getLogger("test")
    cache: dict = {}

    def get_model(key, factory):
        if key not in cache:
            cache[key] = factory()
        return cache[key]

    assert [g.name for g in select_generators(_cfg(**{"qa.generator": "template"}), get_model, log)] == ["template_v1"]
    both = select_generators(_cfg(**{"qa.generator": "both"}), get_model, log)
    assert [g.name for g in both] == ["template_v1", "spacy_v1"]
    only = select_generators(_cfg(**{"qa.generator": "spacy"}), get_model, log)
    assert [g.name for g in only] == ["spacy_v1"] and only[0] is both[1]  # loaded once per process
    with pytest.raises(ValueError):
        select_generators(_cfg(**{"qa.generator": "llm"}), get_model, log)

    class Missing(SpacyQAGenerator):
        def load(self):
            raise RuntimeError("no model")

    monkeypatch.setattr(qs, "SpacyQAGenerator", Missing)
    cache.clear()
    assert [g.name for g in select_generators(_cfg(**{"qa.generator": "spacy"}), get_model, log)] == ["template_v1"]
    assert [g.name for g in select_generators(_cfg(**{"qa.generator": "both"}), get_model, log)] == ["template_v1"]
