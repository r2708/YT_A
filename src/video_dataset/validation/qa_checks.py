"""Independent re-derivation of QA answers from their evidence.

The generators build answers from event timestamps; a second, independent implementation
recomputes what the answer *must* say from the evidence events alone (without reading the stored
answer) and compares. Numeric and ordering answers can be checked exactly; descriptive answers
cannot be recomputed without a model and are reported as ``not_applicable``.
"""

from __future__ import annotations

import re

from video_dataset.schemas.events import Event
from video_dataset.schemas.qa import QARecord, QAType
from video_dataset.schemas.quality import CheckState
from video_dataset.utils.text import normalize_text

_NUM = re.compile(r"(\d+(?:\.\d+)?)")
_FROM_TO = re.compile(r"from (\d+(?:\.\d+)?)s to (\d+(?:\.\d+)?)s")
_STARTS_AT = re.compile(r"starts at (\d+(?:\.\d+)?)s")
_FIRST_THEN = re.compile(r"^first (?P<a>.+?), then (?P<b>.+?)\.?$", re.I | re.S)
_COUNT = re.compile(r"^(\d+) shots?")
_BEFORE_AFTER_Q = re.compile(r"does the moment when (?P<x>.+?) come before or after the moment when (?P<y>.+?)\?$", re.I | re.S)


def recompute_answer(q: QARecord, events: dict[str, Event], tolerance: float = 0.15) -> tuple[str, str | None]:
    """(check state, detail). ``passed`` when the stored answer agrees with what the evidence events
    imply; ``failed`` when they disagree; ``not_applicable`` for answer types that cannot be
    recomputed without a model; ``not_run`` when the evidence events are missing."""
    evs = [events[e] for e in q.evidence.event_ids if e in events]
    if not evs:
        return CheckState.NOT_RUN.value, "evidence events unavailable"
    ans = q.answer.strip()
    if q.type == QAType.DURATION:
        m = _FROM_TO.search(ans)
        e = evs[0]
        if not m:
            return CheckState.FAILED.value, "duration answer carries no interval"
        a, b = float(m.group(1)), float(m.group(2))
        if abs(a - e.start_time) > tolerance or abs(b - e.end_time) > tolerance:
            return CheckState.FAILED.value, f"answer interval {a:.1f}-{b:.1f}s vs event {e.start_time:.1f}-{e.end_time:.1f}s"
        return CheckState.PASSED.value, None
    if q.type == QAType.EVENT_LOCALIZATION:
        m = _STARTS_AT.search(ans)
        e = evs[0]
        if not m:
            return CheckState.FAILED.value, "localization answer carries no start time"
        t = float(m.group(1))
        if abs(t - e.start_time) > tolerance:
            return CheckState.FAILED.value, f"answer start {t:.1f}s vs event start {e.start_time:.1f}s"
        return CheckState.PASSED.value, None
    if q.type in (QAType.TEMPORAL_ORDERING, QAType.MULTI_EVENT) and len(evs) >= 2:
        ordered = sorted(evs, key=lambda e: (e.start_time, e.end_time))
        m = _FIRST_THEN.match(ans)
        if m:
            first_txt, second_txt = normalize_text(m.group("a")), normalize_text(m.group("b"))
            chain = ordered if q.type == QAType.TEMPORAL_ORDERING else ordered[1:]  # chain answers describe events 2 and 3
            if len(chain) < 2:
                return CheckState.NOT_APPLICABLE.value, None
            first_ev, second_ev = _best_match(first_txt, chain), _best_match(second_txt, chain)
            if first_ev is None or second_ev is None or first_ev is second_ev:
                return CheckState.NOT_APPLICABLE.value, "answer wording does not name the events unambiguously"
            if first_ev.start_time <= second_ev.start_time:
                return CheckState.PASSED.value, None
            return CheckState.FAILED.value, "answer states the reverse order of the evidence events"
        if ans.lower().startswith(("before", "after")):
            # "Does the moment when X come before or after the moment when Y?" -> X before Y iff event(X) starts first
            mq = _BEFORE_AFTER_Q.search(q.question.strip())
            if mq:
                x_txt, y_txt = normalize_text(mq.group("x")), normalize_text(mq.group("y"))
                ex, ey = _best_match(x_txt, evs), _best_match(y_txt, evs)
                if ex is not None and ey is not None and ex is not ey:
                    truly_before = ex.start_time <= ey.start_time + tolerance
                    claimed_before = ans.lower().startswith("before")
                    if truly_before != claimed_before and abs(ex.start_time - ey.start_time) > tolerance:
                        return CheckState.FAILED.value, "before/after verdict disagrees with the evidence event order"
                    return CheckState.PASSED.value, None
            return CheckState.NOT_APPLICABLE.value, "could not match the question's events"
        return CheckState.NOT_APPLICABLE.value, None
    if q.type == QAType.LONG_RANGE and (q.template_id or "").startswith("count"):
        m = _COUNT.match(ans)
        if m and int(m.group(1)) != len(evs):
            return CheckState.FAILED.value, f"answer counts {m.group(1)} shots, evidence lists {len(evs)} events"
        return CheckState.PASSED.value if m else CheckState.NOT_APPLICABLE.value, None
    return CheckState.NOT_APPLICABLE.value, None


def _overlap(text: str, e: Event) -> float:
    """Fraction of the event's content words (and its quoted speech) present in ``text``."""
    source = normalize_text(e.event) + " " + normalize_text(str(e.attributes.get("text") or ""))
    words = {w for w in source.split() if len(w) > 3}
    if not words:
        return 0.0
    have = set(text.split())
    return sum(1 for w in words if w in have) / len(words)


def _best_match(text: str, events: list[Event]) -> Event | None:
    """The event the text describes: the highest content-word overlap (>= 0.5), unambiguous."""
    scored = sorted(((_overlap(text, e), e) for e in events), key=lambda t: -t[0])
    if not scored or scored[0][0] < 0.5:
        return None
    if len(scored) > 1 and scored[1][0] == scored[0][0]:
        return None
    return scored[0][1]
