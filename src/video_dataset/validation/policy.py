"""Versioned acceptance policy: named check results in, status + machine-readable reasons out.

Individual validators never decide a record's final status. They record what they checked
(``checks``: passed / failed / not_run / not_applicable / error) and what they found (``issues``);
this module turns that into ``accepted`` / ``review`` / ``rejected`` with one rule set per record
type, and *fails closed*: a check the policy requires that did not run is a reason for review,
never a silent pass.

Reason codes (stable vocabulary, machine readable):

* ``hard:<text>``                  a hard failure (rejects)
* ``failed_check:<name>``          a required check failed (review unless the validator made it hard)
* ``missing_check:<name>``         a required check is ``not_run`` / ``error``
* ``semantic_verification:not_run`` generated text that no verifier judged
* ``confidence:<status>``          the confidence-threshold suggestion when it is not ``accepted``
* ``issue:<n>``                    number of soft issues the validator recorded
"""

from __future__ import annotations

from typing import Any

from video_dataset.schemas.quality import CheckState, ValidationStatus, check_state

POLICY_VERSION = "2026.10.19"

# Checks that must be *passed* for a record of this type to be accepted. A missing one -> review.
REQUIRED_CHECKS: dict[str, tuple[str, ...]] = {
    "scene": (
        "timestamps", "scene_exists", "frames_exist", "frame_files_exist", "summary_present",
        "frame_sequence", "clip_files_exist", "clip_duration_exact", "clip_boundaries",
        "visual_quality", "consistent", "camera_motion_agreement", "text_consistent", "single_shot",
    ),
    "event": ("timestamps", "scenes_exist", "frames_exist", "description_present"),
    "relation": ("events_exist", "interval_semantics"),
    "qa": (
        "evidence_timestamps", "timestamps_in_range", "scenes_exist", "events_exist", "frames_exist",
        "frame_files_exist", "has_evidence", "answer_grounded", "question_form", "answer_type",
    ),
}

# Checks whose *absence* is tolerated when the record carries no such thing (no clips cut, no
# transcript ...): the validator marks them NOT_APPLICABLE explicitly; anything else must be present.
# A clip record is never excused: its own ``boundary_check`` field must be ``passed`` (see
# ``enforce_exported``), because the record claims exact boundaries.
_OPTIONAL_WHEN_ABSENT = {"clip_files_exist", "clip_duration_exact", "clip_boundaries", "camera_motion_agreement"}

# Which checks feed each verification dimension published on every record (``validation.verification``).
VERIFICATION_DIMENSIONS: dict[str, tuple[str, ...]] = {
    "temporal": ("timestamps", "clip_boundaries", "clip_duration_exact", "frame_sequence", "evidence_timestamps", "timestamps_in_range"),
    "measurement": ("visual_quality", "consistent", "camera_motion_agreement", "single_shot", "answer_recomputed", "interval_semantics"),
}
HUMAN_ACCEPT = "human:accepted"
EXPORT_REEVALUATION = "export"


def normalize_checks(checks: dict[str, Any]) -> dict[str, str]:
    return {k: check_state(v) for k, v in checks.items()}


def decide(
    record_type: str,
    checks: dict[str, Any],
    hard: list[str],
    suggested: str,
    issues: list[str] | None = None,
    semantic_verification: str | None = None,
) -> tuple[ValidationStatus, list[str]]:
    """Status + reason codes for one record.

    ``hard`` are the validator's hard failures (always reject). ``suggested`` is the status the
    confidence thresholds alone would give ("accepted" / "review" / "rejected"). Required checks that
    failed or did not run cap the status at ``review``; a generative record whose semantic
    verification did not run is capped at ``review`` too.
    """
    states = normalize_checks(checks)
    reasons: list[str] = []
    for h in hard:
        reasons.append("hard:" + h[:80])
    if hard:
        return ValidationStatus.REJECTED, reasons
    status = ValidationStatus(suggested)
    if status != ValidationStatus.ACCEPTED:
        reasons.append(f"confidence:{status.value}")
    cap_review = False
    for name in REQUIRED_CHECKS.get(record_type, ()):
        st = states.get(name)
        if st is None:
            if name in _OPTIONAL_WHEN_ABSENT:
                continue
            reasons.append(f"missing_check:{name}")
            cap_review = True
        elif st == CheckState.FAILED.value:
            reasons.append(f"failed_check:{name}")
            cap_review = True
        elif st in (CheckState.NOT_RUN.value, CheckState.ERROR.value):
            reasons.append(f"missing_check:{name}")
            cap_review = True
    if semantic_verification == CheckState.NOT_RUN.value:
        reasons.append("semantic_verification:not_run")
        cap_review = True
    if issues:
        reasons.append(f"issue:{len(issues)}")
    if cap_review and status == ValidationStatus.ACCEPTED:
        status = ValidationStatus.REVIEW
    return status, reasons


def strict_ok(record: dict[str, Any]) -> tuple[bool, list[str]]:
    """Whether an exported record dict meets the strict-release policy for its type: accepted
    status, every required check passed (or explicitly not applicable), semantic verification
    present, and - for generation targets - a single-shot scope. Returns (ok, reasons)."""
    kind = _record_kind(record)
    v = record.get("validation") or {}
    why: list[str] = []
    if v.get("status") != ValidationStatus.ACCEPTED.value:
        why.append(f"status_{v.get('status') or 'missing'}")
    if not v.get("policy_version"):
        why.append("policy_version:missing")  # validated before the policy existed: its checks were never the required set
    states = normalize_checks(v.get("checks") or {})
    for name in REQUIRED_CHECKS.get(kind, ()):
        st = states.get(name)
        if st is None and name in _OPTIONAL_WHEN_ABSENT:
            continue
        if st is None:
            why.append(f"missing_check:{name}")
        elif st not in (CheckState.PASSED.value, CheckState.NOT_APPLICABLE.value):
            why.append(f"{st}:{name}")
    sem = v.get("semantic_verification")
    if kind in ("scene",) and sem in (None, CheckState.NOT_RUN.value):
        why.append("semantic_verification:" + str(sem or "missing"))
    if record.get("generation_target") == "not_for_generation" and kind in ("scene",):
        why.append("not_for_generation")
    if record.get("canonical_video_id"):
        why.append("near_duplicate_video")  # the canonical copy carries the content; strict holds it once
    return not why, why


def _states_summary(states: dict[str, str], names: tuple[str, ...]) -> str:
    """One state for a group of checks: failed > error > not_run > passed > not_applicable; a group
    none of whose checks exist is ``not_run``."""
    present = [states[n] for n in names if n in states]
    if not present:
        return CheckState.NOT_RUN.value
    for st in (CheckState.FAILED.value, CheckState.ERROR.value, CheckState.NOT_RUN.value):
        if st in present:
            return st
    return CheckState.PASSED.value if CheckState.PASSED.value in present else CheckState.NOT_APPLICABLE.value


def verification_summary(record: dict[str, Any]) -> dict[str, str]:
    """What has been verified about an exported record, by dimension, so a consumer can separate
    automated measurements, temporal checks, semantic (model / verifier) checks and human review
    without reading the raw ``checks`` dict:

    * ``temporal``: timestamps / clip boundaries / frame order checks (``passed`` means the clip's
      first and last frames were located in the source within the tolerance);
    * ``measurement``: pixel / flow / consistency checks;
    * ``semantic``: ``measured`` (no generated text), ``verifier:<name>``, ``human`` or ``not_run``;
    * ``human``: ``accepted`` / ``rejected`` when a reviewer labelled the record, else ``not_run``.
    """
    v = record.get("validation") or {}
    states = normalize_checks(v.get("checks") or {})
    codes = list(v.get("reason_codes") or [])
    kind = _record_kind(record)
    out = {dim: _states_summary(states, names) for dim, names in VERIFICATION_DIMENSIONS.items()}
    if kind == "qa":
        out["temporal"] = _states_summary(states, ("evidence_timestamps", "timestamps_in_range", "has_evidence"))
        out["measurement"] = _states_summary(states, ("answer_grounded", "answer_recomputed", "question_form", "answer_type"))
    elif kind == "event":
        out["temporal"] = _states_summary(states, ("timestamps",))
        out["measurement"] = _states_summary(states, ("scenes_exist", "frames_exist", "description_present"))
    if kind in ("scene",) and record_type_of(record) == "clip":
        bc = str(record.get("boundary_check") or CheckState.NOT_RUN.value)
        if bc != CheckState.PASSED.value or out["temporal"] in (CheckState.NOT_RUN.value, CheckState.ERROR.value, CheckState.FAILED.value):
            out["temporal"] = bc if bc in (CheckState.FAILED.value, CheckState.ERROR.value, CheckState.NOT_RUN.value) else out["temporal"]
    sem = v.get("semantic_verification")
    out["semantic"] = str(sem) if sem else CheckState.NOT_RUN.value
    if HUMAN_ACCEPT in codes:
        out["human"] = "accepted"
    elif "human:rejected" in codes:
        out["human"] = "rejected"
    else:
        out["human"] = CheckState.NOT_RUN.value
    return out


def enforce_exported(record: dict[str, Any], policy_version: str = POLICY_VERSION) -> list[str]:
    """Apply the acceptance policy to an *exported* record dict (in place): an ``accepted`` record
    stays accepted only when its validation was produced under a policy version and every required
    check for its type is ``passed`` / ``not_applicable``; a clip additionally needs
    ``boundary_check: passed``. Anything else becomes ``review`` with the missing / failed checks as
    reason codes and ``validation.reevaluated = "export"``. Human ``accept`` labels are never
    overridden. Returns the reason codes that caused a downgrade (empty = unchanged)."""
    v = record.get("validation")
    if not isinstance(v, dict) or v.get("status") != ValidationStatus.ACCEPTED.value:
        return []
    codes = list(v.get("reason_codes") or [])
    if HUMAN_ACCEPT in codes or v.get("semantic_verification") == "human":
        return []
    kind = _record_kind(record)
    why: list[str] = []
    if not v.get("policy_version"):
        why.append("policy_version:missing")
    states = normalize_checks(v.get("checks") or {})
    for name in REQUIRED_CHECKS.get(kind, ()):
        st = states.get(name)
        if st is None:
            if name not in _OPTIONAL_WHEN_ABSENT:
                why.append(f"missing_check:{name}")
        elif st == CheckState.FAILED.value:
            why.append(f"failed_check:{name}")
        elif st not in (CheckState.PASSED.value, CheckState.NOT_APPLICABLE.value):
            why.append(f"missing_check:{name}")
    if record_type_of(record) == "clip":
        bc = str(record.get("boundary_check") or CheckState.NOT_RUN.value)
        if bc == CheckState.FAILED.value:
            why.append("failed_check:clip_boundaries")
        elif bc != CheckState.PASSED.value and "missing_check:clip_boundaries" not in why:
            why.append("missing_check:clip_boundaries")
    if kind == "scene" and v.get("semantic_verification") in (None, CheckState.NOT_RUN.value):
        why.append("semantic_verification:" + str(v.get("semantic_verification") or "missing"))
    if not why:
        return []
    v["status"] = ValidationStatus.REVIEW.value
    v["reason_codes"] = codes + [c for c in why if c not in codes]
    v["policy_version"] = policy_version
    v["reevaluated"] = EXPORT_REEVALUATION
    return why


def record_type_of(record: dict[str, Any]) -> str:
    """``record_type`` of an exported record, inferred from its identifying fields when the tag is
    missing (records built by hand or by an old exporter)."""
    rt = str(record.get("record_type") or "")
    if rt:
        return rt
    if record.get("frame_id") and record.get("timestamp") is not None:
        return "frame"
    if record.get("clip_id") and record.get("clip_path"):
        return "clip"
    if record.get("sequence_id") or (record.get("scene_ids") and record.get("prompt")):
        return "sequence"
    if record.get("event_id") and record.get("event_type"):
        return "event"
    if record.get("question") is not None and record.get("evidence") is not None:
        return "long_video_qa" if record.get("is_long_range") or str(record.get("type")) == "long_range" else "temporal_qa"
    if record.get("scene_id") and ("summary" in record or "camera" in record):
        return "scene"
    return ""


def _record_kind(record: dict[str, Any]) -> str:
    rt = record_type_of(record)
    if rt in ("scene", "clip", "frame"):
        return "scene"
    if rt == "event":
        return "event"
    if rt in ("temporal_qa", "long_video_qa"):
        return "qa"
    return rt or "unknown"
