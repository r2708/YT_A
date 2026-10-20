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
_OPTIONAL_WHEN_ABSENT = {"clip_files_exist", "clip_duration_exact", "clip_boundaries", "camera_motion_agreement"}


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


def _record_kind(record: dict[str, Any]) -> str:
    rt = str(record.get("record_type") or "")
    if rt in ("scene", "clip", "frame"):
        return "scene"
    if rt == "event":
        return "event"
    if rt in ("temporal_qa", "long_video_qa"):
        return "qa"
    return rt or "unknown"
