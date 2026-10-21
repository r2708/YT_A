"""The canonical record: one row per video-caption candidate, traceable to its source row.

Status vocabularies (never collapsed into one score):

* ``*_status``: ``passed`` | ``failed`` | ``needs_review`` | ``not_checked``
* ``source_rights_status``: ``cleared`` | ``restricted`` | ``unknown`` | ``review_required``
* scores say what they measured in their ``*_score_basis`` sibling; a rule-based score is never
  proof that a caption is true.

Paths are relative to the factory data root. ``source_payload`` keeps every source field that has
no canonical home, so nothing is discarded.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0.0"
STATUSES = ("passed", "failed", "needs_review", "not_checked")
RIGHTS = ("cleared", "restricted", "unknown", "review_required")


class ActionSpan(BaseModel):
    """An action with optional timing in seconds inside the clip (start < end <= duration)."""

    model_config = ConfigDict(extra="allow")
    name: str
    start_time_seconds: float | None = None
    end_time_seconds: float | None = None
    source: str | None = None


class Keyframe(BaseModel):
    path: str
    timestamp_seconds: float
    sha256: str | None = None
    width: int | None = None
    height: int | None = None


class CanonicalRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    record_id: str
    source_dataset: str
    source_config: str | None = None  # record type / loader config, e.g. "clips"
    source_split: str | None = None
    source_record_id: str | None = None
    source_record_hash: str | None = None  # sha256 of the source row (sorted JSON)
    source_url: str | None = None
    canonical_video_id: str | None = None
    original_video_id: str | None = None
    video_path: str | None = None
    video_sha256: str | None = None
    video_fingerprint: list[str] = Field(default_factory=list)  # dHash of the sampled frames
    start_time_seconds: float | None = None
    end_time_seconds: float | None = None
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    frame_count: int | None = None
    codec: str | None = None
    caption_source: str | None = None  # which source field (or provider) caption_final came from
    caption_original: str | None = None
    generation_prompt_original: str | None = None
    prompt_variants: list[str] = Field(default_factory=list)
    caption_final: str | None = None
    caption_language: str | None = None
    caption_quality_status: str = "not_checked"
    caption_quality_score: float | None = None  # share of rule checks passed, 0..1 (see caption_quality_score_basis)
    caption_quality_score_basis: str | None = None
    caption_quality_reasons: list[str] = Field(default_factory=list)
    caption_review_method: str | None = None  # rules | provider:<name> | human
    caption_model: str | None = None
    caption_prompt_version: str | None = None
    start_frame_path: str | None = None
    start_frame_timestamp_seconds: float | None = None
    start_frame_sha256: str | None = None
    start_frame_width: int | None = None
    start_frame_height: int | None = None
    start_frame_clip_sha256: str | None = None  # sha256 of the clip the frame was decoded from
    start_frame_dhash: str | None = None
    start_frame_status: str = "not_checked"
    start_frame_reasons: list[str] = Field(default_factory=list)
    start_frame_extraction: dict[str, Any] = Field(default_factory=dict)  # tool + settings
    keyframe_paths: list[Keyframe] = Field(default_factory=list)
    subject_description: str | None = None
    actions: list[ActionSpan] = Field(default_factory=list)
    subject_motion: dict[str, Any] | None = None
    camera_motion: dict[str, Any] | None = None
    shot_type: str | None = None
    lighting: str | None = None
    visual_style: dict[str, Any] | None = None
    scene_description: str | None = None
    temporal_events: list[dict[str, Any]] = Field(default_factory=list)
    visual_quality_status: str = "not_checked"
    visual_quality_score: float | None = None  # 1 - share of failed frame metrics, 0..1
    visual_quality_reasons: list[str] = Field(default_factory=list)
    motion_quality_status: str = "not_checked"
    motion_quality_score: float | None = None
    motion_quality_reasons: list[str] = Field(default_factory=list)
    duplicate_group_id: str | None = None
    duplicate_kind: str | None = None  # exact | near | null
    duplicate_canonical: bool | None = None  # true for the record kept as the group's representative
    source_rights_status: str = "unknown"
    source_rights_evidence: str | None = None
    source_license: str | None = None
    media_validation_status: str = "not_checked"
    media_validation_reasons: list[str] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)
    source_validation: dict[str, Any] | None = None  # the producer's validation block, verbatim
    source_tier: str | None = None
    source_confidence: float | None = None
    validation_status: str = "not_checked"
    validation_reasons: list[str] = Field(default_factory=list)
    t2v_eligible: bool = False
    t2v_reasons: list[str] = Field(default_factory=list)
    i2v_eligible: bool = False
    i2v_reasons: list[str] = Field(default_factory=list)
    split: str | None = None
    dataset_version: str | None = None
    processed_at_utc: str | None = None
    pipeline_version: str | None = None
    processing_config_hash: str | None = None
    source_payload: dict[str, Any] = Field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
