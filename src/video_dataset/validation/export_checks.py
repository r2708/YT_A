"""Export-time checks computed from the exported records themselves (no working files needed).

They fill ``validation.checks`` entries the acceptance policy requires for sequences, video
descriptions and QA evidence, and ``temporal_validation`` on every record, so shards validated
before this policy get the same checks on re-annotation and a release can prove that no accepted
record has an invalid timestamp. Everything is deterministic and pure over record dicts:

* ``temporal_validation``: ``{status, checks: {non_negative, ordered, within_duration,
  inside_parent}, boundary_error_seconds, method}`` — ``within_duration`` is ``not_run`` when the
  source duration is unknown, ``inside_parent`` compares a clip / frame with its scene span and a
  QA evidence window with the video;
* sequences: ``scene_order`` (scene ids in time order), ``contiguity`` (no gap above the
  tolerance between consecutive shots), ``scene_membership`` (every scene / clip id exists for the
  video), ``clip_overlap`` (member clips do not overlap each other), ``all_shots_accepted``;
* video descriptions: ``scenes_exist``, ``evidence_resolves`` (every evidence scene / clip / frame
  id exists), ``shots_cover_span`` (the shots cover the record's interval), ``claims_not_contradicted``;
* QA: ``evidence_covers_events`` (the evidence window contains every referenced event's interval).

Checks are recorded as ``passed`` / ``failed`` / ``not_applicable`` / ``not_run`` strings;
``policy.enforce_exported`` turns a failed or missing required check into ``review``.
"""

from __future__ import annotations

from typing import Any

from video_dataset.schemas.quality import CheckState, ValidationStatus

TEMPORAL_METHOD = "export_checks_2026.11"
PASSED, FAILED, NA, NOT_RUN = CheckState.PASSED.value, CheckState.FAILED.value, CheckState.NOT_APPLICABLE.value, CheckState.NOT_RUN.value


def _span(rec: dict[str, Any]) -> tuple[float | None, float | None]:
    ev = rec.get("evidence") if isinstance(rec.get("evidence"), dict) else None
    if ev is not None and ev.get("start_time") is not None:
        return _f(ev.get("start_time")), _f(ev.get("end_time"))
    if rec.get("timestamp") is not None and rec.get("start_time") is None:
        t = _f(rec.get("timestamp"))
        return t, t
    return _f(rec.get("start_time")), _f(rec.get("end_time"))


def _f(v: Any) -> float | None:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _state(ok: bool | None) -> str:
    return NOT_RUN if ok is None else (PASSED if ok else FAILED)


def _validation(rec: dict[str, Any]) -> dict[str, Any]:
    v = rec.get("validation")
    if not isinstance(v, dict):
        v = {"status": ValidationStatus.REVIEW.value, "issues": [], "checks": {}, "reason_codes": []}
        rec["validation"] = v
    v.setdefault("checks", {})
    v.setdefault("issues", [])
    v.setdefault("reason_codes", [])
    return v


class _Index:
    """Per-video id sets and spans gathered in one pass."""

    def __init__(self, records: dict[str, list[dict[str, Any]]]):
        self.scenes: dict[str, dict[str, tuple[float, float]]] = {}
        self.clips: dict[str, dict[str, tuple[float, float, str | None]]] = {}
        self.frames: dict[str, dict[str, float]] = {}
        self.events: dict[str, dict[str, tuple[float, float]]] = {}
        self.scene_status: dict[tuple[str, str], str | None] = {}
        for r in records.get("scenes", []):
            vid, sid = str(r.get("video_id")), str(r.get("scene_id"))
            s, e = _span(r)
            if s is not None and e is not None:
                self.scenes.setdefault(vid, {})[sid] = (s, e)
            self.scene_status[(vid, sid)] = (r.get("validation") or {}).get("status")
        for r in records.get("clips", []):
            s, e = _span(r)
            if s is not None and e is not None:
                self.clips.setdefault(str(r.get("video_id")), {})[str(r.get("clip_id"))] = (s, e, str(r.get("scene_id")) if r.get("scene_id") else None)
        for r in records.get("frames", []):
            t = _f(r.get("timestamp"))
            if t is not None:
                self.frames.setdefault(str(r.get("video_id")), {})[str(r.get("frame_id"))] = t
        for r in records.get("events", []):
            s, e = _span(r)
            if s is not None and e is not None:
                self.events.setdefault(str(r.get("video_id")), {})[str(r.get("event_id"))] = (s, e)


# ----------------------------------------------------------------------------- temporal validation
def temporal_validation(kind: str, rec: dict[str, Any], idx: _Index, duration: float | None, tol: float = 0.05) -> dict[str, Any]:
    vid = str(rec.get("video_id"))
    s, e = _span(rec)
    checks: dict[str, str] = {}
    err: float | None = None
    if s is None or e is None:
        checks["non_negative"] = checks["ordered"] = FAILED
        checks["within_duration"] = checks["inside_parent"] = NOT_RUN
        return {"status": FAILED, "checks": checks, "boundary_error_seconds": None, "method": TEMPORAL_METHOD, "detail": "missing start/end"}
    checks["non_negative"] = _state(s >= -tol and e >= -tol)
    checks["ordered"] = _state(e >= s)
    if duration is None:
        checks["within_duration"] = NOT_RUN
    else:
        checks["within_duration"] = _state(s <= duration + tol and e <= duration + tol)
        err = max(0.0, s - duration, e - duration) if checks["within_duration"] == FAILED else err
    parent: tuple[float, float] | None = None
    if (kind == "clips" and rec.get("scene_id")) or (kind == "frames" and rec.get("scene_id")):
        parent = idx.scenes.get(vid, {}).get(str(rec["scene_id"]))
    if parent is None:
        checks["inside_parent"] = NA if kind not in ("clips", "frames") else NOT_RUN
    else:
        inside = s >= parent[0] - tol and e <= parent[1] + tol
        checks["inside_parent"] = _state(inside)
        if not inside:
            err = max(err or 0.0, parent[0] - s, e - parent[1])
    if kind == "clips" and rec.get("boundary_error_seconds") is not None:
        err = max(err or 0.0, float(rec["boundary_error_seconds"]))
    status = FAILED if FAILED in checks.values() else (NOT_RUN if all(v in (NOT_RUN, NA) for v in checks.values()) else PASSED)
    return {"status": status, "checks": checks, "boundary_error_seconds": None if err is None else round(err, 3), "method": TEMPORAL_METHOD}


# ----------------------------------------------------------------------------- per-type checks
def sequence_checks(rec: dict[str, Any], idx: _Index, gap_tol: float = 0.5) -> dict[str, str]:
    vid = str(rec.get("video_id"))
    scenes = idx.scenes.get(vid, {})
    clips = idx.clips.get(vid, {})
    sids = [str(s) for s in (rec.get("scene_ids") or [])]
    cids = [str(c) for c in (rec.get("clip_ids") or [])]
    checks: dict[str, str] = {}
    known = [sid for sid in sids if sid in scenes]
    checks["scene_membership"] = _state(bool(sids) and len(known) == len(sids) and all(c in clips for c in cids))
    spans = [scenes[sid] for sid in known]
    checks["scene_order"] = _state(all(spans[i][0] <= spans[i + 1][0] for i in range(len(spans) - 1))) if len(spans) >= 2 else (NA if len(spans) == 1 else FAILED)
    checks["contiguity"] = _state(all(abs(spans[i + 1][0] - spans[i][1]) <= gap_tol for i in range(len(spans) - 1))) if len(spans) >= 2 else (NA if len(spans) == 1 else FAILED)
    cspans = sorted(clips[c][:2] for c in cids if c in clips)
    if len(cspans) >= 2:
        checks["clip_overlap"] = _state(all(cspans[i + 1][0] >= cspans[i][1] - gap_tol for i in range(len(cspans) - 1)))
    else:
        checks["clip_overlap"] = NA
    statuses = [idx.scene_status.get((vid, sid)) for sid in sids]
    checks["all_shots_accepted"] = _state(bool(statuses) and all(st == ValidationStatus.ACCEPTED.value for st in statuses))
    return checks


def description_checks(rec: dict[str, Any], idx: _Index, gap_tol: float = 0.5) -> dict[str, str]:
    vid = str(rec.get("video_id"))
    scenes = idx.scenes.get(vid, {})
    sids = [str(s) for s in (rec.get("scene_ids") or [])]
    checks: dict[str, str] = {}
    checks["scenes_exist"] = _state(bool(sids) and all(s in scenes for s in sids))
    ev = rec.get("evidence") if isinstance(rec.get("evidence"), dict) else {}
    ok = all(s in scenes for s in (ev.get("scene_ids") or [])) and all(c in idx.clips.get(vid, {}) for c in (ev.get("clip_ids") or [])) and all(f in idx.frames.get(vid, {}) for f in (ev.get("frame_ids") or []))
    checks["evidence_resolves"] = _state(ok) if ev else NOT_RUN
    s, e = _span(rec)
    shots = rec.get("shots") or []
    spans = sorted((_f(sh.get("start_time")), _f(sh.get("end_time"))) for sh in shots if isinstance(sh, dict) and sh.get("start_time") is not None and sh.get("end_time") is not None)
    if s is None or e is None or not spans:
        checks["shots_cover_span"] = FAILED
    else:
        covered = abs(spans[0][0] - s) <= gap_tol and abs(spans[-1][1] - e) <= gap_tol and all(spans[i + 1][0] - spans[i][1] <= gap_tol for i in range(len(spans) - 1))  # type: ignore[operator]
        checks["shots_cover_span"] = _state(covered)
    claims = rec.get("claims") or []
    checks["claims_not_contradicted"] = _state(not any(isinstance(c, dict) and c.get("status") == "contradicted" for c in claims))
    return checks


def qa_evidence_check(rec: dict[str, Any], idx: _Index, tol: float = 0.05) -> str:
    vid = str(rec.get("video_id"))
    ev = rec.get("evidence") if isinstance(rec.get("evidence"), dict) else {}
    ids = [str(x) for x in (ev.get("event_ids") or [])]
    if not ids:
        return NA
    s, e = _f(ev.get("start_time")), _f(ev.get("end_time"))
    if s is None or e is None:
        return FAILED
    events = idx.events.get(vid, {})
    if not all(i in events for i in ids):
        return FAILED  # the reference gate reports the missing id; coverage cannot be shown
    return _state(all(events[i][0] >= s - tol and events[i][1] <= e + tol for i in ids))


# ----------------------------------------------------------------------------- driver
def fill_export_checks(records: dict[str, list[dict[str, Any]]], durations: dict[str, float] | None = None, tol: float = 0.05) -> dict[str, int]:
    """Set ``temporal_validation`` on every record and the type-specific required checks on
    sequences, video descriptions and QA (in place). Returns counts of failed temporal validations
    per type plus ``sequence_checks_failed`` / ``description_checks_failed`` / ``qa_evidence_failed``."""
    idx = _Index(records)
    durations = durations or {}
    counts: dict[str, int] = {}
    for kind, rows in records.items():
        failed = 0
        for rec in rows:
            tv = temporal_validation(kind, rec, idx, durations.get(str(rec.get("video_id"))), tol)
            rec["temporal_validation"] = tv
            failed += tv["status"] == FAILED
            v = _validation(rec)
            if kind == "sequences":
                checks = sequence_checks(rec, idx)
                v["checks"].update(checks)
                counts["sequence_checks_failed"] = counts.get("sequence_checks_failed", 0) + (FAILED in checks.values())
            elif kind == "video_descriptions":
                checks = description_checks(rec, idx)
                v["checks"].update(checks)
                counts["description_checks_failed"] = counts.get("description_checks_failed", 0) + (FAILED in checks.values())
            elif kind in ("temporal_qa", "long_video_qa"):
                st = qa_evidence_check(rec, idx, tol)
                v["checks"]["evidence_covers_events"] = st
                counts["qa_evidence_failed"] = counts.get("qa_evidence_failed", 0) + (st == FAILED)
            if tv["status"] == FAILED:
                v["checks"]["timestamps"] = FAILED
                if "failed_check:timestamps" not in v["reason_codes"]:
                    v["reason_codes"].append("failed_check:timestamps")
        counts[f"{kind}_temporal_failed"] = failed
    return counts
