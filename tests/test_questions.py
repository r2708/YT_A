from video_dataset.config import load_config
from video_dataset.questions.generator import TemporalQAGenerator
from video_dataset.schemas.common import ConfidenceSource
from video_dataset.schemas.events import Event, EventSource, EventType, Timeline
from video_dataset.schemas.qa import QAType
from video_dataset.schemas.scene import Scene, TransitionType
from video_dataset.temporal.relations import RelationBuilder


def _ev(i, start, end, text, etype=EventType.ACTION, conf=0.9, entities=None, action=None, scenes=None, precision="scene", attrs=None):
    return Event(event_id=f"event_{i:04d}", video_id="v", event_type=etype, start_time=start, end_time=end, event=text, entities=entities or [], action=action, scene_ids=scenes or ["scene_001"], frame_ids=[f"frame_{i:03d}_000"], source=EventSource.VISION, confidence=conf, confidence_source=ConfidenceSource.VERIFIER if conf is not None else ConfidenceSource.UNAVAILABLE, boundary_precision=precision, attributes=attrs or {})


def _timeline():
    events = [
        _ev(1, 0.0, 4.0, "A red car drives along a forest road.", entities=["red car"], action="drives along a forest road", scenes=["scene_001"]),
        _ev(2, 4.0, 8.0, "The red car enters a tunnel.", entities=["red car", "tunnel"], action="enters a tunnel", scenes=["scene_002"]),
        _ev(3, 4.5, 7.5, "The camera pans left.", EventType.CAMERA, conf=0.8, entities=["camera"], action="pan_left", scenes=["scene_002"], precision="measured"),
        _ev(4, 8.0, 12.0, "A person walks out of the tunnel.", entities=["person", "tunnel"], action="walks out of the tunnel", scenes=["scene_003"]),
        _ev(5, 7.0, 9.0, "The lighting changes from bright to dark.", EventType.STATE_CHANGE, conf=0.7, entities=["lighting"], action="changes", scenes=["scene_002", "scene_003"], precision="frame", attrs={"attribute": "lighting", "boundary_time": 8.0}),
        _ev(6, 40.0, 44.0, "The red car is visible.", EventType.APPEARANCE, entities=["red car"], action="appears", scenes=["scene_010"]),
        _ev(7, 1.0, 3.0, 'A speaker says: "we are entering the forest"', EventType.SPEECH, conf=None, entities=["speaker"], action="speaking", scenes=["scene_001"], precision="asr", attrs={"text": "we are entering the forest"}),
    ]
    rels = RelationBuilder(load_config(None, {"temporal.long_range_min_gap_seconds": "20"}).temporal).build("v", events)
    return Timeline(video_id="v", duration=48.0, events=events, relations=rels)


def _scenes():
    bounds = [(0, 4), (4, 8), (8, 12), (12, 40), (40, 48)]
    return [Scene(scene_id=f"scene_{i + 1:03d}" if i < 4 else "scene_010", video_id="v", index=i, start_time=s, end_time=e, start_frame=int(s * 24), end_frame=int(e * 24), transition_in=TransitionType.START if i == 0 else TransitionType.CUT) for i, (s, e) in enumerate(bounds)]


def test_generator_covers_types_with_evidence():
    cfg = load_config(None, {"qa.questions_per_minute": "60", "qa.min_questions": "20", "qa.max_questions": "200", "qa.long_range_min_gap_seconds": "20"}).qa
    tl = _timeline()
    records = TemporalQAGenerator(cfg).generate("vid_test", 48.0, tl, _scenes(), {}, None)
    assert len(records) >= 15
    types = {r.type for r in records}
    for t in (QAType.TIMESTAMP, QAType.BEFORE_AFTER, QAType.TEMPORAL_ORDERING, QAType.DURATION, QAType.EVENT_LOCALIZATION, QAType.STATE_CHANGE, QAType.LONG_RANGE):
        assert t in types, f"missing {t}: {types}"
    ids = {e.event_id for e in tl.events}
    for r in records:
        assert r.evidence.event_ids and set(r.evidence.event_ids) <= ids
        assert r.evidence.start_time <= r.evidence.end_time <= 48.0
        assert r.question.strip() and r.answer.strip()
        assert r.question_id.startswith("qa_test_")
    assert len({r.question_id for r in records}) == len(records)
    assert len({(r.type, r.question) for r in records}) == len(records)
    # confidence is min over evidence or None when evidence is unscored
    speech_q = [r for r in records if "event_0007" in r.evidence.event_ids]
    assert all(r.confidence is None and r.confidence_source == ConfidenceSource.UNAVAILABLE for r in speech_q)
    scored = [r for r in records if r.confidence is not None]
    assert scored and all(r.confidence_source == ConfidenceSource.DERIVED_MIN for r in scored)
    lr = [r for r in records if r.type == QAType.LONG_RANGE]
    assert lr and all(r.is_long_range and r.difficulty == "hard" for r in lr)
    sc = next(r for r in records if r.type == QAType.STATE_CHANGE)
    assert "lighting" in sc.question and "bright to dark" in sc.answer


def test_generator_is_deterministic():
    cfg = load_config(None, {"qa.questions_per_minute": "30"}).qa
    tl = _timeline()
    a = TemporalQAGenerator(cfg).generate("vid_test", 48.0, tl, _scenes(), {}, None)
    b = TemporalQAGenerator(cfg).generate("vid_test", 48.0, tl, _scenes(), {}, None)
    assert [(r.question, r.answer) for r in a] == [(r.question, r.answer) for r in b]
