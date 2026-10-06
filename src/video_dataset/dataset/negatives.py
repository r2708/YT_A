"""Rule-built hard negatives: texts that are wrong for a record for a reason we can state.

QA negatives reuse the structure of the right answer (order swapped, time shifted, duration
changed, the answer of another event). Caption negatives swap a *measured* attribute (camera
movement, lighting level) or borrow the caption of another shot of the same video whose measured
attributes differ. Nothing here invents content; every negative is derived from existing records.
"""

from __future__ import annotations

import random
import re

from video_dataset.questions.generator import _phrase
from video_dataset.schemas.curation import HardNegative
from video_dataset.schemas.events import Event, EventType
from video_dataset.schemas.qa import QARecord, QAType
from video_dataset.schemas.vision import CameraMovement, SceneAnalysis
from video_dataset.temporal.timeline import describable
from video_dataset.utils.text import approx_duration_phrase, approx_timestamp_phrase, normalize_text, sentence
from video_dataset.vision.motion import MOVEMENT_PHRASES

OPPOSITE_MOVEMENT: dict[CameraMovement, CameraMovement] = {
    CameraMovement.PAN_LEFT: CameraMovement.PAN_RIGHT,
    CameraMovement.PAN_RIGHT: CameraMovement.PAN_LEFT,
    CameraMovement.TILT_UP: CameraMovement.TILT_DOWN,
    CameraMovement.TILT_DOWN: CameraMovement.TILT_UP,
    CameraMovement.ZOOM_IN: CameraMovement.ZOOM_OUT,
    CameraMovement.ZOOM_OUT: CameraMovement.ZOOM_IN,
    CameraMovement.STATIC: CameraMovement.PAN_RIGHT,
    CameraMovement.HANDHELD: CameraMovement.STATIC,
    CameraMovement.TRACKING: CameraMovement.STATIC,
    CameraMovement.TRACKING_LEFT: CameraMovement.TRACKING_RIGHT,
    CameraMovement.TRACKING_RIGHT: CameraMovement.TRACKING_LEFT,
    CameraMovement.TRACKING_FORWARD: CameraMovement.TRACKING_BACKWARD,
    CameraMovement.TRACKING_BACKWARD: CameraMovement.TRACKING_FORWARD,
    CameraMovement.DOLLY_IN: CameraMovement.DOLLY_OUT,
    CameraMovement.DOLLY_OUT: CameraMovement.DOLLY_IN,
    CameraMovement.CRANE_UP: CameraMovement.CRANE_DOWN,
    CameraMovement.CRANE_DOWN: CameraMovement.CRANE_UP,
    CameraMovement.CRANE: CameraMovement.STATIC,
    CameraMovement.ORBIT_CLOCKWISE: CameraMovement.ORBIT_COUNTERCLOCKWISE,
    CameraMovement.ORBIT_COUNTERCLOCKWISE: CameraMovement.ORBIT_CLOCKWISE,
    CameraMovement.ORBIT: CameraMovement.STATIC,
    CameraMovement.FPV: CameraMovement.STATIC,
    CameraMovement.DRONE: CameraMovement.STATIC,
}
OPPOSITE_LIGHTING = {"dark": "bright", "dim": "bright", "normal": "dark", "bright": "dark"}

_FIRST_THEN = re.compile(r"^First (?P<a>.+?), then (?P<b>.+?)\.$", re.DOTALL)
_DURATION_ANS = re.compile(r"^(?P<phrase>.+?) \(from (?P<a>[\d.]+)s to (?P<b>[\d.]+)s\)\.$")
_LOCALIZATION_ANS = re.compile(r"^(?P<phrase>.+?) \(starts at (?P<t>[\d.]+)s\)\.$")


def _distinct(text: str, answer: str) -> bool:
    a, b = normalize_text(text), normalize_text(answer)
    return bool(a) and a != b and a not in b and b not in a


def _window(q: QARecord) -> tuple[float, float]:
    return q.evidence.start_time, q.evidence.end_time


def _other_event(q: QARecord, events: list[Event], rng: random.Random, min_gap: float = 3.0) -> HardNegative | None:
    """The description of a real event that lies at least ``min_gap`` seconds outside the evidence window."""
    start, end = _window(q)
    used = set(q.evidence.event_ids)
    pool = [
        e for e in events
        if e.event_id not in used and describable(e) and e.event_type not in (EventType.STATE_CHANGE, EventType.TRANSITION)
        and (e.start_time >= end + min_gap or e.end_time <= start - min_gap)
    ]
    rng.shuffle(pool)
    for e in pool:
        text = sentence(_phrase(e))
        if _distinct(text, q.answer):
            return HardNegative(text=text, kind="other_event", source_ids=[e.event_id], note=f"happens at {e.start_time:.1f}s, outside the evidence window {start:.1f}-{end:.1f}s")
    return None


def _swapped_order(q: QARecord) -> HardNegative | None:
    m = _FIRST_THEN.match(q.answer.strip())
    if m:
        text = f"First {m.group('b')}, then {m.group('a')}."
        return HardNegative(text=text, kind="swapped_order", source_ids=list(q.evidence.event_ids), note="order of the two events reversed")
    ans = q.answer.strip()
    for a, b in (("Before.", "After."), ("After.", "Before.")):
        if ans.startswith(a):
            return HardNegative(text=b + ans[len(a):], kind="swapped_order", source_ids=list(q.evidence.event_ids), note="before/after verdict flipped")
    return None


def _wrong_duration(q: QARecord) -> HardNegative | None:
    m = _DURATION_ANS.match(q.answer.strip())
    if not m:
        return None
    a, b = float(m.group("a")), float(m.group("b"))
    d = max(0.1, b - a)
    factor = 2.5 if d < 8.0 else 0.4
    d2 = round(d * factor, 1)
    if abs(d2 - d) < 2.0:
        d2 = round(d + 4.0, 1)
    text = f"{approx_duration_phrase(d2).capitalize()} (from {a:.1f}s to {a + d2:.1f}s)."
    if not _distinct(text, q.answer):
        return None
    return HardNegative(text=text, kind="wrong_duration", source_ids=list(q.evidence.event_ids), note=f"measured duration {d:.1f}s, negative claims {d2:.1f}s")


def _shifted_time(q: QARecord, duration: float) -> HardNegative | None:
    m = _LOCALIZATION_ANS.match(q.answer.strip())
    if not m:
        return None
    t = float(m.group("t"))
    delta = max(5.0, 0.2 * duration)
    t2 = t + delta if t + delta <= duration else max(0.0, t - delta)
    if abs(t2 - t) < 3.0:
        return None
    text = f"{approx_timestamp_phrase(t2).capitalize()} (starts at {t2:.1f}s)."
    return HardNegative(text=text, kind="shifted_time", source_ids=list(q.evidence.event_ids), note=f"measured start {t:.1f}s, negative claims {t2:.1f}s")


def qa_hard_negatives(q: QARecord, events: list[Event], duration: float, rng: random.Random, limit: int = 3) -> list[HardNegative]:
    out: list[HardNegative] = []
    structural: HardNegative | None = None
    if q.type in (QAType.TEMPORAL_ORDERING, QAType.MULTI_EVENT, QAType.BEFORE_AFTER):
        structural = _swapped_order(q)
    elif q.type == QAType.DURATION:
        structural = _wrong_duration(q)
    elif q.type == QAType.EVENT_LOCALIZATION:
        structural = _shifted_time(q, duration)
    if structural is not None and _distinct(structural.text, q.answer):
        out.append(structural)
    seen = {normalize_text(n.text) for n in out}
    tries = 0
    while len(out) < limit and tries < limit * 3:
        tries += 1
        neg = _other_event(q, events, rng)
        if neg is None:
            break
        key = normalize_text(neg.text)
        if key in seen:
            continue
        seen.add(key)
        out.append(neg)
    return out[:limit]


# ------------------------------------------------------------------------------------ captions
def _attribute_swaps(a: SceneAnalysis) -> list[HardNegative]:
    out: list[HardNegative] = []
    text = a.summary or ""
    if not text:
        return out
    mov = a.camera.movement
    known_motion = mov != CameraMovement.UNKNOWN and (a.camera.confidence or 0.0) >= 0.6
    if known_motion and mov in OPPOSITE_MOVEMENT:
        phrase, other = MOVEMENT_PHRASES[mov], MOVEMENT_PHRASES[OPPOSITE_MOVEMENT[mov]]
        if phrase in text:
            out.append(HardNegative(text=text.replace(phrase, other, 1), kind="attribute_swap", source_ids=[a.scene_id], note=f"camera measured {mov}, negative claims {OPPOSITE_MOVEMENT[mov]}"))
    level = a.measurements.lighting_level if a.measurements else None
    if level in OPPOSITE_LIGHTING and f"{level} lighting" in text:
        other_level = OPPOSITE_LIGHTING[level]
        out.append(HardNegative(text=text.replace(f"{level} lighting", f"{other_level} lighting", 1), kind="attribute_swap", source_ids=[a.scene_id], note=f"lighting measured {level}, negative claims {other_level}"))
    return out


def _differences(a: SceneAnalysis, b: SceneAnalysis) -> list[str]:
    diffs: list[str] = []
    if a.camera.movement != CameraMovement.UNKNOWN and b.camera.movement != CameraMovement.UNKNOWN and a.camera.movement != b.camera.movement:
        diffs.append(f"camera {a.camera.movement} vs {b.camera.movement}")
    la = a.measurements.lighting_level if a.measurements else None
    lb = b.measurements.lighting_level if b.measurements else None
    if la and lb and la != lb:
        diffs.append(f"lighting {la} vs {lb}")
    oa = {o.name.lower() for o in a.objects}
    ob = {o.name.lower() for o in b.objects}
    if (oa or ob) and oa != ob:
        diffs.append(f"objects {sorted(oa) or 'none'} vs {sorted(ob) or 'none'}")
    sa, sb = str(a.environment.setting), str(b.environment.setting)
    if sa != "unknown" and sb != "unknown" and sa != sb:
        diffs.append(f"setting {sa} vs {sb}")
    return diffs


def caption_hard_negatives(a: SceneAnalysis, others: list[SceneAnalysis], rng: random.Random, limit: int = 3) -> list[HardNegative]:
    """Negatives for a scene / clip caption: measured-attribute swaps first, then the caption of the
    most different other shot of the same video."""
    out = _attribute_swaps(a)
    scored: list[tuple[int, SceneAnalysis, list[str]]] = []
    for b in others:
        if b.scene_id == a.scene_id or not b.summary or not _distinct(b.summary, a.summary or ""):
            continue
        diffs = _differences(a, b)
        if diffs:
            scored.append((len(diffs), b, diffs))
    rng.shuffle(scored)
    scored.sort(key=lambda x: -x[0])
    for _n, b, diffs in scored:
        if len(out) >= limit:
            break
        out.append(HardNegative(text=b.summary or "", kind="other_shot", source_ids=[b.scene_id], note="; ".join(diffs)))
    return out[:limit]
