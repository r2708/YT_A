"""Quality scoring and validation schemas."""

from __future__ import annotations

from pydantic import Field

from video_dataset.schemas.common import BaseSchema, StrEnum


class QualityScore(BaseSchema):
    """Component scores are None when they cannot be estimated from real signals.

    `overall` is the mean of the available components only, and `components_used`
    lists which ones contributed, so consumers can tell a measured 0.9 from a guess.
    """

    grounding: float | None = Field(default=None, ge=0, le=1)
    temporal_accuracy: float | None = Field(default=None, ge=0, le=1)
    description_quality: float | None = Field(default=None, ge=0, le=1)
    overall: float | None = Field(default=None, ge=0, le=1)
    components_used: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @classmethod
    def from_components(
        cls,
        grounding: float | None = None,
        temporal_accuracy: float | None = None,
        description_quality: float | None = None,
        notes: list[str] | None = None,
    ) -> QualityScore:
        comps = {
            "grounding": grounding,
            "temporal_accuracy": temporal_accuracy,
            "description_quality": description_quality,
        }
        used = [k for k, v in comps.items() if v is not None]
        values = [float(v) for v in comps.values() if v is not None]
        overall = round(sum(values) / len(values), 4) if values else None
        return cls(
            grounding=grounding,
            temporal_accuracy=temporal_accuracy,
            description_quality=description_quality,
            overall=overall,
            components_used=used,
            notes=notes or [],
        )


class ValidationStatus(StrEnum):
    ACCEPTED = "accepted"
    REVIEW = "review"
    REJECTED = "rejected"
    DUPLICATE = "duplicate"


class CheckState(StrEnum):
    """Outcome of one named validation check. ``not_run`` is never the same as ``passed``: the
    acceptance policy treats a required check that did not run as a reason for review."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_RUN = "not_run"  # the check exists but did not execute (stage skipped, model unavailable, disabled)
    NOT_APPLICABLE = "not_applicable"  # the check has no meaning for this record (e.g. no clips were cut)
    ERROR = "error"  # the check crashed; its verdict is unknown


def check_state(value: object) -> str:
    """Normalise a legacy ``checks`` value (True / False / None) or a state string to a CheckState."""
    if isinstance(value, CheckState):
        return value.value
    if value is True:
        return CheckState.PASSED.value
    if value is False:
        return CheckState.FAILED.value
    if value is None:
        return CheckState.NOT_APPLICABLE.value
    text = str(value).strip().lower()
    try:
        return CheckState(text).value
    except ValueError:
        return CheckState.ERROR.value


class ValidationInfo(BaseSchema):
    status: ValidationStatus
    issues: list[str] = Field(default_factory=list)
    # Named checks: True = passed, False = failed, None = not applicable, or an explicit CheckState
    # string (``not_run`` / ``error``). ``check_state`` maps every form to one vocabulary.
    checks: dict[str, bool | str | None] = Field(default_factory=dict)
    verifier_score: float | None = None
    duplicate_of: str | None = None
    duplicate_method: str | None = None  # file_hash | fingerprint | text_minhash | text_exact | evidence_overlap
    # Machine-readable reasons for the status (``policy.py`` vocabulary), e.g. ``missing_check:clip_duration_exact``
    reason_codes: list[str] = Field(default_factory=list)
    policy_version: str | None = None  # version of the acceptance policy that assigned ``status``
    # How the semantic content of this record was grounded: ``measured`` (detector / optical flow /
    # pixel statistics, nothing to hallucinate), ``verifier:<name>`` (a verifier pass judged it),
    # ``not_run`` (generated text that no verifier checked). A generative record whose semantic
    # verification is ``not_run`` can never be ``accepted``.
    semantic_verification: str | None = None
