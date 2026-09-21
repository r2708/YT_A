"""Template-based temporal QA generation from the event timeline.

Every question is built from concrete events with timestamps, so every answer is grounded by
construction and carries the evidence (event ids, scene ids, timestamps, frames) it was made from.
Confidence is the minimum confidence of the supporting events (or None when any is unscored).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from video_dataset.config import QAConfig
from video_dataset.schemas.common import ConfidenceSource
from video_dataset.schemas.events import Event, EventType, RelationType, Timeline
from video_dataset.schemas.qa import Difficulty, Evidence, QARecord, QAType
from video_dataset.schemas.scene import Scene
from video_dataset.schemas.transcript import Transcript
from video_dataset.schemas.vision import SceneAnalysis
from video_dataset.temporal.timeline import describable
from video_dataset.utils.ids import qa_id as make_qa_id
from video_dataset.utils.text import (
    approx_duration_phrase,
    approx_timestamp_phrase,
    lower_first,
    normalize_text,
    sentence,
)


@dataclass
class Candidate:
    type: QAType
    question: str
    answer: str
    events: list[Event]
    template_id: str
    timestamps: list[float] = field(default_factory=list)
    options: list[str] = field(default_factory=list)
    scene_ids: list[str] = field(default_factory=list)
    long_range: bool = False
    extra_frames: list[str] = field(default_factory=list)
    restrict_scenes: bool = False  # evidence scene_ids come only from `scene_ids`, not from the events


def _phrase(e: Event) -> str:
    """Event description as a clause: 'the car enters the tunnel'."""
    text = e.event.strip().rstrip(".")
    if e.event_type == EventType.SPEECH:
        quoted = e.attributes.get("text") or ""
        short = quoted if len(quoted) <= 80 else quoted[:77].rstrip() + "…"
        return f'the speaker says "{short}"'
    return lower_first(text)


def _conf(events: list[Event]) -> tuple[float | None, ConfidenceSource]:
    confs = [e.confidence for e in events]
    if not confs or any(c is None for c in confs):
        return None, ConfidenceSource.UNAVAILABLE
    return round(min(c for c in confs if c is not None), 3), ConfidenceSource.DERIVED_MIN


def _difficulty(c: Candidate) -> Difficulty:
    if c.long_range or len(c.events) >= 3 or c.type in (QAType.LONG_RANGE,):
        return Difficulty.HARD
    span = (max(e.end_time for e in c.events) - min(e.start_time for e in c.events)) if c.events else 0.0
    if len(c.events) >= 2 or span > 10.0 or c.type in (QAType.STATE_CHANGE, QAType.MULTI_EVENT, QAType.DURATION):
        return Difficulty.MEDIUM
    return Difficulty.EASY


class TemporalQAGenerator:
    name = "template_v1"

    def __init__(self, cfg: QAConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------ public
    def generate(
        self,
        video_id: str,
        duration: float,
        timeline: Timeline,
        scenes: list[Scene],
        analyses: dict[str, SceneAnalysis],
        transcript: Transcript | None = None,
    ) -> list[QARecord]:
        rng = random.Random(f"{self.cfg.seed}:{video_id}")
        events = [e for e in timeline.events if self.cfg.allow_unscored_evidence or e.confidence is not None]
        by_id = {e.event_id: e for e in timeline.events}
        target = int(round(duration / 60.0 * float(self.cfg.questions_per_minute)))
        target = max(int(self.cfg.min_questions), min(int(self.cfg.max_questions), target))

        pools: dict[QAType, list[Candidate]] = {
            QAType.TIMESTAMP: self._timestamp(events, scenes, analyses, rng),
            QAType.BEFORE_AFTER: self._before_after(timeline, by_id, rng),
            QAType.TEMPORAL_ORDERING: self._ordering(events, rng),
            QAType.DURATION: self._duration(events, scenes, analyses),
            QAType.EVENT_LOCALIZATION: self._localization(events, scenes, analyses),
            QAType.STATE_CHANGE: self._state_change(events, scenes),
            QAType.MULTI_EVENT: self._multi_event(timeline, by_id),
            QAType.LONG_RANGE: self._long_range(timeline, by_id, duration),
        }
        for pool in pools.values():
            rng.shuffle(pool)

        chosen = self._select(pools, target, rng)
        records: list[QARecord] = []
        for n, c in enumerate(chosen, start=1):
            conf, src = _conf(c.events)
            start = min([e.start_time for e in c.events] + (c.timestamps or [duration]))
            end = max([e.end_time for e in c.events] + (c.timestamps or [0.0]))
            frame_ids: list[str] = []
            for e in c.events:
                for fid in e.frame_ids:
                    if fid not in frame_ids:
                        frame_ids.append(fid)
            frame_ids += [f for f in c.extra_frames if f not in frame_ids]
            scene_ids: list[str] = []
            if not c.restrict_scenes:
                for e in c.events:
                    for sid in e.scene_ids:
                        if sid not in scene_ids:
                            scene_ids.append(sid)
            for sid in c.scene_ids:
                if sid not in scene_ids:
                    scene_ids.append(sid)
            evidence = Evidence(
                start_time=round(max(0.0, start), 3),
                end_time=round(min(duration, max(end, start)), 3),
                timestamps=[round(t, 3) for t in (c.timestamps or sorted({e.start_time for e in c.events}))],
                scene_ids=scene_ids,
                event_ids=[e.event_id for e in c.events],
                frame_ids=frame_ids[:16],
                relation_ids=[r.relation_id for r in timeline.relations if {r.event_a, r.event_b} <= {e.event_id for e in c.events} and len(c.events) >= 2][:6],
                transcript_segment_ids=[sid for e in c.events if e.event_type == EventType.SPEECH for sid in e.source_ids],
                ocr_track_ids=[sid for e in c.events if e.event_type == EventType.TEXT_ON_SCREEN for sid in e.source_ids],
            )
            records.append(
                QARecord(
                    question_id=make_qa_id(video_id, n),
                    video_id=video_id,
                    type=c.type,
                    question=c.question,
                    answer=c.answer,
                    evidence=evidence,
                    difficulty=_difficulty(c),
                    confidence=conf,
                    confidence_source=src,
                    generator=self.name,
                    template_id=c.template_id,
                    options=c.options,
                    is_long_range=c.long_range,
                )
            )
        return records

    # ------------------------------------------------------------------ selection
    def _select(self, pools: dict[QAType, list[Candidate]], target: int, rng: random.Random) -> list[Candidate]:
        weights = {QAType(k): float(v) for k, v in self.cfg.type_weights.items() if v and v > 0}
        chosen: list[Candidate] = []
        seen_q: set[str] = set()
        usage: dict[str, int] = {}
        # spread questions over many events, but relax the cap when the timeline is sparse
        n_events = len({e.event_id for pool in pools.values() for c in pool for e in c.events}) or 1
        max_use = max(3, -(-2 * target // n_events))
        active = {t: list(p) for t, p in pools.items() if p and weights.get(t, 0) > 0}

        def take(t: QAType) -> Candidate | None:
            pool = active.get(t)
            if pool is None:
                return None
            cand = None
            while pool:
                c = pool.pop()
                key = normalize_text(c.question) + "|" + str(c.type)
                if key in seen_q or any(usage.get(e.event_id, 0) >= max_use for e in c.events):
                    continue
                cand = c
                break
            if not pool:
                del active[t]
            if cand is None:
                return None
            seen_q.add(normalize_text(cand.question) + "|" + str(cand.type))
            for e in cand.events:
                usage[e.event_id] = usage.get(e.event_id, 0) + 1
            chosen.append(cand)
            return cand

        # coverage first: one question per available type (heaviest weight first) ...
        for t in sorted(list(active.keys()), key=lambda x: -weights.get(x, 1.0)):
            if len(chosen) >= target:
                break
            take(t)
        # ... then weighted sampling until the target is reached or the pools run dry
        while len(chosen) < target and active:
            types = list(active.keys())
            t = rng.choices(types, weights=[weights.get(x, 1.0) for x in types], k=1)[0]
            take(t)
        chosen.sort(key=lambda c: (min(e.start_time for e in c.events) if c.events else 0.0, str(c.type)))
        return chosen

    # ------------------------------------------------------------------ generators
    def _timestamp(self, events: list[Event], scenes: list[Scene], analyses: dict[str, SceneAnalysis], rng: random.Random) -> list[Candidate]:
        out: list[Candidate] = []
        anchors = [e for e in events if describable(e) and e.event_type not in (EventType.STATE_CHANGE, EventType.SOUND) and e.duration >= 1.0]
        for anchor in anchors:
            t = round(anchor.start_time + anchor.duration / 2.0, 1)
            active = [
                e for e in events
                if e.contains_time(t) and describable(e) and e.event_type != EventType.STATE_CHANGE
                and not (e.event_type == EventType.SOUND and e.attributes.get("label") == "sound_present")  # unnamed background sound is not an answer
            ]
            if not active:
                continue
            active.sort(key=lambda e: (e.event_type != EventType.ACTION, e.start_time))
            scene = next((s for s in scenes if s.start_time <= t <= s.end_time), None)
            parts: list[str] = []
            summary = analyses[scene.scene_id].summary if scene and scene.scene_id in analyses else None
            if scene and summary and analyses[scene.scene_id].provider not in ("heuristic",):
                parts.append(summary.strip())
            seen: set[str] = set()
            for e in active[:4]:
                p = sentence(_phrase(e))
                if normalize_text(p) not in seen:
                    seen.add(normalize_text(p))
                    parts.append(p)
            answer = " ".join(parts)
            q = rng.choice([
                f"What is happening at {t:.1f} seconds?",
                f"Describe what is taking place at {t:.1f} seconds into the video.",
                f"At {t:.1f}s, what can be observed in the video?",
            ])
            out.append(Candidate(QAType.TIMESTAMP, q, answer, active[:4], "timestamp_v1", timestamps=[t], scene_ids=[scene.scene_id] if scene else [], restrict_scenes=bool(scene)))
        return out

    def _before_after(self, timeline: Timeline, by_id: dict[str, Event], rng: random.Random) -> list[Candidate]:
        out: list[Candidate] = []
        for r in timeline.relations:
            if r.relation != RelationType.BEFORE or r.gap_seconds is None or r.gap_seconds > 5.0 or "long_range" in (r.rationale or ""):
                continue
            a, b = by_id.get(r.event_a), by_id.get(r.event_b)
            if not a or not b or not describable(a) or not describable(b):
                continue
            if a.event_type == b.event_type == EventType.SPEECH:
                continue
            if normalize_text(_phrase(a)) == normalize_text(_phrase(b)):
                continue
            if rng.random() < 0.5:
                out.append(Candidate(QAType.BEFORE_AFTER, f"What happens immediately before {_phrase(b)}?", sentence(_phrase(a)), [a, b], "before_v1"))
            else:
                out.append(Candidate(QAType.BEFORE_AFTER, f"What happens right after {_phrase(a)}?", sentence(_phrase(b)), [a, b], "after_v1"))
        return out

    def _ordering(self, events: list[Event], rng: random.Random) -> list[Candidate]:
        out: list[Candidate] = []
        evs = [e for e in events if describable(e)]
        for i, a in enumerate(evs):
            for b in evs[i + 1 : i + 12]:
                if b.start_time - a.end_time < self.cfg.ordering_min_gap_seconds:
                    continue
                if normalize_text(_phrase(a)) == normalize_text(_phrase(b)):
                    continue
                first, second = a, b
                opts = [_phrase(first), _phrase(second)]
                if rng.random() < 0.5:
                    opts.reverse()
                if rng.random() < 0.5:
                    q = f"Which happens first: {opts[0]}, or {opts[1]}?"
                    ans = f"{sentence(_phrase(first))[:-1]} happens first."
                else:
                    q = f"Does {opts[0]} happen before or after {opts[1]}?"
                    ans = "Before." if opts[0] == _phrase(first) else "After."
                    ans += f" {sentence(_phrase(first))[:-1]} occurs around {first.start_time:.1f}s and {_phrase(second)} around {second.start_time:.1f}s."
                out.append(Candidate(QAType.TEMPORAL_ORDERING, q, ans, [a, b], "ordering_v1", options=opts))
        return out

    def _duration(self, events: list[Event], scenes: list[Scene], analyses: dict[str, SceneAnalysis]) -> list[Candidate]:
        out: list[Candidate] = []
        for e in events:
            if not describable(e) or e.event_type == EventType.STATE_CHANGE or e.duration < 1.0:
                continue
            if e.boundary_precision == "scene":
                q = f"Approximately how long does the shot last in which {_phrase(e)}?"
            else:
                q = f"Approximately how long does it last while {_phrase(e)}?" if e.event_type in (EventType.CAMERA, EventType.SOUND) else f"Approximately how long does the following last: {_phrase(e)}?"
            ans = f"{approx_duration_phrase(e.duration).capitalize()} (from {e.start_time:.1f}s to {e.end_time:.1f}s)."
            out.append(Candidate(QAType.DURATION, q, ans, [e], "duration_v1"))
        return out

    def _localization(self, events: list[Event], scenes: list[Scene], analyses: dict[str, SceneAnalysis]) -> list[Candidate]:
        out: list[Candidate] = []
        for e in events:
            if not describable(e) or e.event_type == EventType.STATE_CHANGE:
                continue
            if e.boundary_precision == "scene":
                q = f"At approximately what time does the shot begin in which {_phrase(e)}?"
            else:
                q = f"At approximately what timestamp does it start that {_phrase(e)}?"
            ans = f"{approx_timestamp_phrase(e.start_time).capitalize()} (starts at {e.start_time:.1f}s)."
            out.append(Candidate(QAType.EVENT_LOCALIZATION, q, ans, [e], "localization_v1"))
        return out

    def _state_change(self, events: list[Event], scenes: list[Scene]) -> list[Candidate]:
        out: list[Candidate] = []
        for e in events:
            if e.event_type != EventType.STATE_CHANGE:
                continue
            t = float(e.attributes.get("boundary_time", e.start_time))
            prev_scene = next((s for s in scenes if s.scene_id == (e.scene_ids[0] if e.scene_ids else "")), None)
            next_scene = next((s for s in scenes if s.scene_id == (e.scene_ids[-1] if e.scene_ids else "")), None)
            t1 = round(prev_scene.start_time + prev_scene.duration / 2, 1) if prev_scene else round(max(0.0, t - 3.0), 1)
            t2 = round(next_scene.start_time + next_scene.duration / 2, 1) if next_scene else round(t + 3.0, 1)
            attr = e.attributes.get("attribute", "environment")
            q = f"What changes in the {attr} between {t1:.1f} and {t2:.1f} seconds?"
            ans = f"{e.event.rstrip('.')} (the change happens at the shot boundary around {t:.1f}s)."
            out.append(Candidate(QAType.STATE_CHANGE, q, ans, [e], "state_change_v1", timestamps=[t1, t, t2]))
        return out

    def _multi_event(self, timeline: Timeline, by_id: dict[str, Event]) -> list[Candidate]:
        out: list[Candidate] = []
        before: dict[str, list[str]] = {}
        for r in timeline.relations:
            if r.relation == RelationType.BEFORE and r.gap_seconds is not None and r.gap_seconds <= 8.0 and "long_range" not in (r.rationale or ""):
                before.setdefault(r.event_a, []).append(r.event_b)
        for a_id, bs in before.items():
            a = by_id.get(a_id)
            if not a or not describable(a):
                continue
            for b_id in bs:
                b = by_id.get(b_id)
                if not b or not describable(b) or (b.event_type == EventType.SPEECH and a.event_type == EventType.SPEECH):
                    continue
                for c_id in before.get(b_id, []):
                    c = by_id.get(c_id)
                    if not c or not describable(c) or c.event_id == a.event_id:
                        continue
                    phrases = {normalize_text(_phrase(x)) for x in (a, b, c)}
                    if len(phrases) < 3:
                        continue
                    out.append(Candidate(QAType.MULTI_EVENT, f"After {_phrase(a)}, what happens next, and what follows after that?", f"First {_phrase(b)}, then {_phrase(c)}.", [a, b, c], "chain_v1"))
                    out.append(Candidate(QAType.MULTI_EVENT, f"What happens between {_phrase(a)} and {_phrase(c)}?", sentence(_phrase(b)), [a, b, c], "between_v1"))
                    break
        return out

    _STOPWORD_ENTITIES = frozenset({"the", "and", "for", "you", "are", "with", "this", "that", "from", "com", "www", "http", "https"})

    def _long_range(self, timeline: Timeline, by_id: dict[str, Event], duration: float) -> list[Candidate]:
        out: list[Candidate] = []
        for r in timeline.relations:
            if r.relation != RelationType.BEFORE or "long_range" not in (r.rationale or ""):
                continue
            a, b = by_id.get(r.event_a), by_id.get(r.event_b)
            if not a or not b or (r.gap_seconds or 0) < self.cfg.long_range_min_gap_seconds:
                continue
            if a.event_type == EventType.TEXT_ON_SCREEN and b.event_type == EventType.TEXT_ON_SCREEN and normalize_text(a.event) == normalize_text(b.event):
                continue  # "text X is visible ... later text X is visible again" teaches nothing
            entities = (r.rationale or "").split("shared entities:")[-1].strip() if r.rationale else ""
            ent = entities.split(",")[0].strip() if entities else "the same subject"
            if len(ent) < 3 or ent.lower() in self._STOPWORD_ENTITIES:
                continue
            if b.start_time >= 0.75 * duration:
                q = f"Earlier in the video, {_phrase(a)} (around {a.start_time:.0f}s). What happens near the end of the video that involves {ent}?"
            else:
                q = f"Earlier in the video, {_phrase(a)} (around {a.start_time:.0f}s). What happens later that involves {ent}?"
            ans = f"{sentence(_phrase(b))[:-1]}, around {b.start_time:.1f}s."
            out.append(Candidate(QAType.LONG_RANGE, q, ans, [a, b], "long_range_v1", long_range=True))
        # counting: how many separate shots show entity X (needs >= 2 non-adjacent appearances)
        appearances: dict[str, list[Event]] = {}
        for e in timeline.events:
            if e.event_type == EventType.APPEARANCE and e.entities:
                appearances.setdefault(normalize_text(e.entities[0]), []).append(e)
        for ent, evs in appearances.items():
            evs = sorted(evs, key=lambda e: e.start_time)
            if len(evs) >= 2 and (evs[-1].start_time - evs[0].end_time) >= self.cfg.long_range_min_gap_seconds:
                times = ", ".join(f"{e.start_time:.0f}s" for e in evs)
                out.append(Candidate(QAType.LONG_RANGE, f"In how many separate shots does {ent} newly come into view over the whole video?", f"{len(evs)} shots (starting around {times}).", evs, "count_v1", long_range=True))
        return out
