"""Interval-algebra relations between events (BEFORE/AFTER/DURING/OVERLAPS/STARTS/ENDS) plus
rule-based CONTINUES / INTERRUPTS / CHANGES_TO. CAUSES is only produced by an explicit inferencer."""

from __future__ import annotations

from video_dataset.config import TemporalConfig
from video_dataset.schemas.common import ConfidenceSource
from video_dataset.schemas.events import Event, EventType, RelationType, TemporalRelation
from video_dataset.utils.ids import relation_id as make_relation_id
from video_dataset.utils.text import normalize_text

_GENERIC_ENTITIES = {"camera", "speaker", "sound", "lighting", "setting", "location", "time of day", "weather"}


def derived_confidence(a: Event, b: Event) -> tuple[float | None, ConfidenceSource]:
    confs = [c for c in (a.confidence, b.confidence) if c is not None]
    if len(confs) == 2:
        return round(min(confs), 3), ConfidenceSource.DERIVED_MIN
    return None, ConfidenceSource.UNAVAILABLE


def shared_entities(a: Event, b: Event) -> set[str]:
    ea = {normalize_text(e) for e in a.entities} - _GENERIC_ENTITIES
    eb = {normalize_text(e) for e in b.entities} - _GENERIC_ENTITIES
    return {e for e in ea & eb if e}


def classify_pair(a: Event, b: Event, tol: float = 0.25) -> tuple[RelationType, Event, Event] | None:
    """Return (relation, subject, object) for two events sorted by start (a.start <= b.start), or None."""
    if a.end_time <= b.start_time + tol and not (abs(a.start_time - b.start_time) <= tol and abs(a.end_time - b.end_time) <= tol):
        return RelationType.BEFORE, a, b
    same_start = abs(a.start_time - b.start_time) <= tol
    same_end = abs(a.end_time - b.end_time) <= tol
    if same_start and same_end:
        return None  # co-extensive; DURING would be arbitrary
    a_in_b = a.start_time >= b.start_time - tol and a.end_time <= b.end_time + tol
    b_in_a = b.start_time >= a.start_time - tol and b.end_time <= a.end_time + tol
    if same_start:
        shorter, longer = (a, b) if a.duration < b.duration else (b, a)
        return RelationType.STARTS, shorter, longer
    if same_end:
        shorter, longer = (a, b) if a.duration < b.duration else (b, a)
        return RelationType.ENDS, shorter, longer
    if b_in_a:
        return RelationType.DURING, b, a
    if a_in_b:
        return RelationType.DURING, a, b
    if a.start_time < b.start_time < a.end_time < b.end_time:
        return RelationType.OVERLAPS, a, b
    return None


class RelationBuilder:
    def __init__(self, cfg: TemporalConfig, tol: float = 0.25):
        self.cfg = cfg
        self.tol = tol

    def build(self, video_id: str, events: list[Event]) -> list[TemporalRelation]:
        evs = sorted(events, key=lambda e: (e.start_time, e.end_time))
        rels: list[TemporalRelation] = []
        seen: set[tuple[str, str, str]] = set()
        by_key: dict[tuple[str, str, str], TemporalRelation] = {}

        def add(rel: RelationType, a: Event, b: Event, derivation: str = "interval_algebra", rationale: str | None = None, gap: float | None = None) -> None:
            key = (a.event_id, b.event_id, rel.value)
            if a.event_id == b.event_id:
                return
            if key in seen:
                # already related by the neighbour pass: annotate instead of dropping the extra derivation
                existing = by_key[key]
                if rationale and (existing.rationale or "") != rationale:
                    existing.rationale = f"{existing.rationale}; {rationale}" if existing.rationale else rationale
                if derivation != "interval_algebra" and existing.derivation == "interval_algebra":
                    existing.derivation = derivation
                return
            seen.add(key)
            conf, src = derived_confidence(a, b)
            by_key[key] = TemporalRelation(
                    relation_id="tmp",
                    video_id=video_id,
                    event_a=a.event_id,
                    event_b=b.event_id,
                    relation=rel,
                    event_a_text=a.event,
                    event_b_text=b.event,
                    gap_seconds=round(gap, 3) if gap is not None else None,
                    confidence=conf,
                    confidence_source=src,
                    derivation=derivation,
                    rationale=rationale,
                )
            rels.append(by_key[key])

        n = len(evs)
        for i, a in enumerate(evs):
            for j in range(i + 1, min(n, i + 1 + int(self.cfg.max_relation_neighbors))):
                b = evs[j]
                res = classify_pair(a, b, self.tol)
                if res is None:
                    continue
                rel, s, o = res
                gap = (o.start_time - s.end_time) if rel == RelationType.BEFORE else None
                add(rel, s, o, gap=gap)
                if rel == RelationType.BEFORE and self.cfg.emit_inverse_relations:
                    add(RelationType.AFTER, o, s, gap=gap)
                # CONTINUES: same kind of action/entity in adjacent shots, effectively back-to-back
                if rel == RelationType.BEFORE and gap is not None and gap <= 1.0 and a.event_type == b.event_type and a.event_type in (EventType.ACTION, EventType.CAMERA, EventType.APPEARANCE):
                    if (a.action and b.action and normalize_text(a.action) == normalize_text(b.action)) or (a.event_type == EventType.APPEARANCE and shared_entities(a, b)):
                        if set(a.scene_ids) != set(b.scene_ids):
                            add(RelationType.CONTINUES, a, b, derivation="same_action_adjacent_shots", rationale="same action/entity continues across a shot boundary", gap=gap)
                # CHANGES_TO: same entities, different action, close in time (state transition of the same subject)
                if rel == RelationType.BEFORE and gap is not None and gap <= 2.0 and a.event_type == b.event_type == EventType.ACTION and a.action and b.action and normalize_text(a.action) != normalize_text(b.action) and shared_entities(a, b):
                    add(RelationType.CHANGES_TO, a, b, derivation="same_entity_new_action", rationale=f"shared entities: {', '.join(sorted(shared_entities(a, b)))}", gap=gap)
                if rel == RelationType.BEFORE and gap is not None and gap <= 1.0 and a.event_type == b.event_type == EventType.CAMERA and a.action != b.action and set(a.scene_ids) & set(b.scene_ids):
                    add(RelationType.CHANGES_TO, a, b, derivation="camera_motion_change", rationale="camera movement changes within the shot", gap=gap)
                # INTERRUPTS: a short disruptive event starts strictly inside a longer continuous one
                if rel == RelationType.DURING and s.event_type in (EventType.SOUND, EventType.TRANSITION) and o.event_type in (EventType.SPEECH, EventType.ACTION, EventType.CAMERA) and s.start_time > o.start_time + self.tol and s.duration < o.duration / 2:
                    add(RelationType.INTERRUPTS, s, o, derivation="short_disruptive_event_inside_longer_event", rationale=f"{s.event_type} occurs in the middle of {o.event_type}")

        # Long-range links: same (non-generic) entity far apart in time
        min_gap = float(self.cfg.long_range_min_gap_seconds)
        for i, a in enumerate(evs):
            if a.event_type not in (EventType.ACTION, EventType.APPEARANCE, EventType.TEXT_ON_SCREEN, EventType.SOUND):
                continue
            for b in evs[i + 1 :]:
                gap = b.start_time - a.end_time
                if gap < min_gap:
                    continue
                shared = shared_entities(a, b)
                if shared and not (set(a.scene_ids) & set(b.scene_ids)):
                    add(RelationType.BEFORE, a, b, derivation="long_range_shared_entity", rationale=f"long_range; shared entities: {', '.join(sorted(shared))}", gap=gap)

        for k, r in enumerate(rels, start=1):
            r.relation_id = make_relation_id(k)
        return rels
