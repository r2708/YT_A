from video_dataset.config import load_config
from video_dataset.schemas.common import ConfidenceSource
from video_dataset.schemas.events import Event, EventSource, EventType, RelationType
from video_dataset.temporal.events import transition_confidence
from video_dataset.temporal.relations import RelationBuilder, classify_pair, derived_confidence


def _ev(i: int, start: float, end: float, text: str, etype: EventType = EventType.ACTION, conf: float | None = 0.9, entities=None, action=None, scenes=None) -> Event:
    return Event(event_id=f"event_{i:04d}", video_id="v", event_type=etype, start_time=start, end_time=end, event=text, entities=entities or [], action=action, scene_ids=scenes or ["scene_001"], source=EventSource.VISION, confidence=conf, confidence_source=ConfidenceSource.VERIFIER if conf is not None else ConfidenceSource.UNAVAILABLE)


def test_classify_pairs():
    a, b = _ev(1, 0, 2, "A"), _ev(2, 3, 5, "B")
    assert classify_pair(a, b)[0] == RelationType.BEFORE
    inner = _ev(3, 3.5, 4.0, "C")
    rel, s, o = classify_pair(b, inner)
    assert rel == RelationType.DURING and s.event_id == inner.event_id and o.event_id == b.event_id
    assert classify_pair(_ev(4, 3, 4, "D"), b)[0] == RelationType.STARTS
    assert classify_pair(b, _ev(5, 4, 5, "E"))[0] == RelationType.ENDS
    assert classify_pair(_ev(6, 2, 4, "F"), b)[0] == RelationType.OVERLAPS
    assert classify_pair(a, _ev(7, 0, 2, "same span")) is None


def test_relation_builder_emits_expected_relations():
    cfg = load_config(None, {"temporal.long_range_min_gap_seconds": "10"}).temporal
    events = [
        _ev(1, 0.0, 4.0, "A red car drives along the road.", entities=["red car"], action="drives along the road", scenes=["scene_001"]),
        _ev(2, 4.0, 8.0, "A red car drives along the road.", entities=["red car"], action="drives along the road", scenes=["scene_002"]),
        _ev(3, 8.2, 9.0, "The red car stops.", entities=["red car"], action="stops", scenes=["scene_003"]),
        _ev(4, 5.0, 5.5, "A loud bang is audible.", EventType.SOUND, conf=None, entities=["bang"], scenes=["scene_002"]),
        _ev(5, 25.0, 27.0, "A red car is visible.", EventType.APPEARANCE, entities=["red car"], action="appears", scenes=["scene_009"]),
    ]
    rels = RelationBuilder(cfg).build("v", events)
    kinds = {(r.event_a, r.event_b, r.relation) for r in rels}
    assert ("event_0001", "event_0002", RelationType.BEFORE) in kinds
    assert ("event_0002", "event_0001", RelationType.AFTER) in kinds
    assert ("event_0001", "event_0002", RelationType.CONTINUES) in kinds
    assert ("event_0002", "event_0003", RelationType.CHANGES_TO) in kinds
    assert ("event_0004", "event_0002", RelationType.DURING) in kinds
    assert ("event_0004", "event_0002", RelationType.INTERRUPTS) in kinds
    long_range = [r for r in rels if r.derivation == "long_range_shared_entity"]
    assert long_range and all(r.event_b == "event_0005" for r in long_range)
    assert not any(r.relation == RelationType.CAUSES for r in rels)
    # confidence derived only when both sides are scored
    r12 = next(r for r in rels if (r.event_a, r.event_b, r.relation) == ("event_0001", "event_0002", RelationType.BEFORE))
    assert r12.confidence == 0.9 and r12.confidence_source == ConfidenceSource.DERIVED_MIN and r12.gap_seconds == 0.0
    r42 = next(r for r in rels if (r.event_a, r.event_b, r.relation) == ("event_0004", "event_0002", RelationType.DURING))
    assert r42.confidence is None and r42.confidence_source == ConfidenceSource.UNAVAILABLE
    assert len({r.relation_id for r in rels}) == len(rels)


def test_confidence_helpers():
    assert derived_confidence(_ev(1, 0, 1, "a", conf=0.5), _ev(2, 0, 1, "b", conf=0.8)) == (0.5, ConfidenceSource.DERIVED_MIN)
    assert transition_confidence(None, 27.0) is None
    assert transition_confidence(27.0, 27.0) == 0.5
    assert transition_confidence(54.0, 27.0) == 1.0
    assert transition_confidence(500.0, 27.0) == 1.0
