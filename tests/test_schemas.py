import pytest
from pydantic import ValidationError

from video_dataset.schemas import (
    ConfidenceSource,
    Event,
    EventSource,
    EventType,
    Evidence,
    QARecord,
    QAType,
    QualityScore,
    RelationType,
    TemporalRelation,
)


def test_event_schema_rejects_bad_times():
    with pytest.raises(ValidationError):
        Event(event_id="e1", video_id="v", event_type=EventType.ACTION, start_time=5.0, end_time=4.0, event="x", source=EventSource.VISION)
    e = Event(event_id="e1", video_id="v", event_type=EventType.ACTION, start_time=4.0, end_time=5.5, event="The car moves.", source=EventSource.VISION)
    assert e.duration == 1.5 and e.confidence is None and e.confidence_source == ConfidenceSource.UNAVAILABLE
    with pytest.raises(ValidationError):
        Event(event_id="e1", video_id="v", event_type=EventType.ACTION, start_time=0, end_time=1, event="x", source=EventSource.VISION, confidence=1.5)


def test_event_schema_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        Event(event_id="e1", video_id="v", event_type="action", start_time=0, end_time=1, event="x", source="vision", bogus=1)


def test_evidence_requires_reference():
    with pytest.raises(ValidationError):
        Evidence(start_time=0, end_time=1)
    ev = Evidence(start_time=1.0, end_time=2.0, event_ids=["event_0001"])
    assert ev.timestamps == []
    with pytest.raises(ValidationError):
        Evidence(start_time=3.0, end_time=2.0, event_ids=["e"])


def test_qa_record_roundtrip():
    q = QARecord(question_id="qa_x_000001", video_id="v", type=QAType.TEMPORAL_ORDERING, question="Which happens first?", answer="A.", evidence=Evidence(start_time=0, end_time=5, event_ids=["e1", "e2"], scene_ids=["scene_001"]), confidence=0.8, confidence_source=ConfidenceSource.DERIVED_MIN)
    d = q.to_json_dict()
    assert d["type"] == "temporal_ordering" and d["difficulty"] == "medium"
    assert QARecord.model_validate(d) == q


def test_quality_score_only_uses_available_components():
    q = QualityScore.from_components(grounding=0.9, temporal_accuracy=None, description_quality=0.5)
    assert q.overall == 0.7 and q.components_used == ["grounding", "description_quality"]
    empty = QualityScore.from_components()
    assert empty.overall is None and empty.components_used == []


def test_relation_schema():
    r = TemporalRelation(relation_id="rel_1", video_id="v", event_a="a", event_b="b", relation=RelationType.BEFORE, event_a_text="A", event_b_text="B", gap_seconds=1.2)
    assert r.confidence is None and r.derivation == "interval_algebra"
