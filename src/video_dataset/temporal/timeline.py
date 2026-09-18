"""Timeline assembly and query helpers shared by QA generation and validation."""

from __future__ import annotations

from video_dataset.schemas.events import Event, EventType, RelationType, TemporalRelation, Timeline
from video_dataset.utils.ids import relation_id as make_relation_id


def build_timeline(video_id: str, duration: float, events: list[Event], relations: list[TemporalRelation]) -> Timeline:
    for k, r in enumerate(relations, start=1):
        r.relation_id = make_relation_id(k)
    return Timeline(video_id=video_id, duration=duration, events=events, relations=relations)


def relations_of(timeline: Timeline, relation: RelationType) -> list[TemporalRelation]:
    return [r for r in timeline.relations if r.relation == relation]


def events_of_type(timeline: Timeline, *types: EventType) -> list[Event]:
    wanted = set(types)
    return [e for e in timeline.events if e.event_type in wanted]


def describable(e: Event) -> bool:
    """Events whose description reads naturally inside a question."""
    return e.event_type in (EventType.ACTION, EventType.APPEARANCE, EventType.CAMERA, EventType.SPEECH, EventType.TEXT_ON_SCREEN, EventType.SOUND, EventType.STATE_CHANGE)


def timeline_stats(timeline: Timeline) -> dict:
    by_type: dict[str, int] = {}
    for e in timeline.events:
        by_type[str(e.event_type)] = by_type.get(str(e.event_type), 0) + 1
    by_rel: dict[str, int] = {}
    for r in timeline.relations:
        by_rel[str(r.relation)] = by_rel.get(str(r.relation), 0) + 1
    scored = [e for e in timeline.events if e.confidence is not None]
    return {
        "events": len(timeline.events),
        "events_by_type": by_type,
        "events_with_confidence": len(scored),
        "relations": len(timeline.relations),
        "relations_by_type": by_rel,
    }
