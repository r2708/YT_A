"""Final dataset record schemas (what is exported to JSONL / Parquet).

Every record type shares the same envelope so one loader works for all of them:
``record_id``, ``record_type``, ``video_id``, ``start_time`` / ``end_time`` (frames: ``timestamp``),
``provider``, ``confidence`` (+ ``confidence_source``), ``quality``, ``validation``,
``split``, ``tier`` (+ ``tier_reasons``) and ``subsets``.
"""

from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, Field, model_validator

from video_dataset.schemas.common import AnnotationProvenance, BaseSchema, ConfidenceSource, RightsInfo
from video_dataset.schemas.curation import HardNegative, Split, Tier
from video_dataset.schemas.quality import QualityScore, ValidationInfo
from video_dataset.schemas.vision import (
    CameraAnnotation,
    Environment,
    Measurements,
    SubjectMotion,
    VisualStyle,
    camera_fields_from_style,
)


class _Curated(BaseSchema):
    """Fields filled by the export / aggregation step for every record type."""

    split: Split | None = None  # train / validation / test, assigned per source video (no leakage across scenes)
    tier: Tier | None = None  # gold / silver / bronze, derived from validation status + quality + confidence provenance
    tier_reasons: list[str] = Field(default_factory=list)  # why the record did not reach a higher tier
    subsets: list[str] = Field(default_factory=list)  # named subsets the record belongs to, e.g. ["cinematic", "licensed", "strict"]
    canonical_video_id: str | None = None  # set when the source video is a near-duplicate of another video (split follows the canonical one)
    # The split group: the canonical video id of a near-duplicate cluster, else the video id. Every
    # record of one group is in one split; `release-check` proves it.
    source_group_id: str | None = None
    schema_version: str | None = None  # SCHEMA_VERSION the record was written with
    # Which optional modalities ran for the source video: transcription / ocr / audio_events /
    # vision / tracking / verifier / clip_alignment -> ran | not_run | failed | disabled | no_audio.
    # ``no_text_detected`` and ``ocr_not_run`` are different things; filter on this, not on empty text.
    coverage: dict[str, str] = Field(default_factory=dict)
    # frame | clip | scene | sequence | video: what the text of this record describes
    annotation_scope: str | None = None
    # single_shot | multi_shot | not_for_generation: whether the record is a generation target and of what kind
    generation_target: str | None = None
    # --- schema 2.3.0: provenance, rights, temporal validation, lineage-friendly extras -----------
    provenance: AnnotationProvenance | None = None  # pipeline build / model / template / policy that produced this record
    rights: RightsInfo | None = None  # rights status of the source video (status, method, allowed use, redistribution)
    # Export-time timestamp validation of this record against the source duration and its parents:
    # {status: passed | failed | not_run, checks: {...}, boundary_error_seconds, method}
    temporal_validation: dict[str, Any] | None = None
    # Shot-to-shot relationship (scene / clip / frame records): prev / next scene, transition in and
    # out, temporal adjacency, sequence membership, subject and location continuity (None = unknown)
    continuity: dict[str, Any] | None = None
    # "uncalibrated" | "calibrated:<calibration file version>": whether ``confidence`` of this
    # confidence_source / record type was measured against human labels (``video-dataset calibrate``)
    confidence_calibration: str | None = None
    # Fields of an older schema that no migration could map; kept verbatim so nothing is lost
    extensions: dict[str, Any] = Field(default_factory=dict)


class FrameCaptionRecord(_Curated):
    record_id: str
    record_type: str = "frame"
    video_id: str
    scene_id: str
    clip_id: str | None = None  # the clip this frame belongs to (video -> clip -> frame); filled at export time
    frame_id: str
    frame_path: str
    asset_id: str | None = None  # stable id of the image in media_manifest.jsonl (sha1 of video_id + relative path)
    media_status: str | None = None  # available | missing | unreadable | not_shipped (export-time check of frame_path)
    timestamp: float
    caption: str
    caption_source: str = "scene_summary"  # frame_analysis | scene_summary | measurement
    objects: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    camera: CameraAnnotation | None = None
    environment: Environment | None = None
    measurements: Measurements | None = None
    ocr_text: list[str] = Field(default_factory=list)
    # Grounded OCR: every detection on this frame with its box (normalised 0..1 x_min, y_min, x_max,
    # y_max when the frame size is known, else pixels + ``bbox_units``), score, engine and version
    ocr_detections: list[dict[str, Any]] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)  # measured visual-quality flags of the shot
    provider: str | None = None
    confidence: float | None = None
    confidence_source: ConfidenceSource = ConfidenceSource.UNAVAILABLE
    quality: QualityScore | None = None
    validation: ValidationInfo | None = None
    annotation_scope: str | None = "frame"
    generation_target: str | None = "not_for_generation"


class ClipCaptionRecord(_Curated):
    record_id: str
    record_type: str = "clip"
    video_id: str
    scene_id: str
    clip_id: str
    clip_path: str
    asset_id: str | None = None  # stable id of the file in media_manifest.jsonl
    media_status: str | None = None  # available | missing | unreadable | not_shipped
    start_time: float
    end_time: float
    media_duration: float | None = None  # measured length of the clip file
    exact: bool = True  # the file covers exactly start_time..end_time (within frame_sampling.clip_tolerance_seconds)
    actual_start: float | None = None  # measured source time of the file's first frame (frame-hash boundary check)
    actual_end: float | None = None  # measured source time of the file's last frame
    boundary_error_seconds: float | None = None
    boundary_check: str = "not_run"  # passed | failed | not_run | error
    boundary_repair: str | None = None  # re-cut once after a failed boundary check: repaired | still_failed | None
    frame_count: int | None = None  # frames in the file (ffprobe nb_frames); compare with round(duration * fps)
    frame_count_check: str | None = None  # passed | failed | not_run
    stream_integrity: str | None = None  # passed | failed | not_run: the file has a decodable video (and audio) stream
    fps: float | None = None  # measured on the clip file
    width: int | None = None
    height: int | None = None
    description: str  # raw analysis description (what the analyzer saw / measured)
    generation_prompt: str | None = None  # the same shot as a text-to-video prompt, built from the structured fields only
    prompt_variants: list[str] = Field(default_factory=list)  # template / synonym rewrites of generation_prompt (same facts)
    quality_flags: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    camera: CameraAnnotation | None = None
    subject_motion: SubjectMotion | None = None  # measured subject motion separated from the camera motion
    transcript: str | None = None  # text of the transcript segments overlapping the clip (may extend past its edges, see transcript_scope)
    transcript_segments: list[dict[str, Any]] = Field(default_factory=list)  # segments clipped to the window: absolute + clip-relative times, partial flag
    transcript_scope: str | None = None  # word_clipped | segment_overlap | none | not_run
    frame_ids: list[str] = Field(default_factory=list)
    claims: list[dict[str, Any]] = Field(default_factory=list)  # claim-level evidence: field, value, source, status, evidence frames
    object_tracks: list[dict[str, Any]] = Field(default_factory=list)  # per-shot tracker tracks (see SceneRecord.object_tracks)
    provider: str | None = None
    confidence: float | None = None
    confidence_source: ConfidenceSource = ConfidenceSource.UNAVAILABLE
    quality: QualityScore | None = None
    validation: ValidationInfo | None = None
    hard_negatives: list[HardNegative] = Field(default_factory=list)  # captions that are wrong for this clip
    annotation_scope: str | None = "clip"
    generation_target: str | None = "single_shot"


class SceneRecord(_Curated):
    record_id: str
    record_type: str = "scene"
    video_id: str
    scene_id: str
    # `start` / `end` were the field names of earlier exports; they are still accepted on input
    start_time: float = Field(validation_alias=AliasChoices("start_time", "start"))
    end_time: float = Field(validation_alias=AliasChoices("end_time", "end"))
    duration: float
    transition_in: str | None = None  # start | cut | fade | dissolve | adaptive | split: how the shot begins
    summary: str  # raw analysis description
    generation_prompt: str | None = None  # text-to-video prompt assembled from the structured fields (no free-form claims)
    prompt_variants: list[str] = Field(default_factory=list)  # template / synonym rewrites of generation_prompt (same facts)
    quality_flags: list[str] = Field(default_factory=list)  # blurry | underexposed | overexposed | compression_artifacts | low_resolution | low_bitrate
    environment: Environment
    objects: list[str] = Field(default_factory=list)
    object_details: list[dict[str, Any]] = Field(default_factory=list)
    people: dict[str, Any] | None = None
    actions: list[str] = Field(default_factory=list)
    camera: CameraAnnotation
    subject_motion: SubjectMotion | None = None
    visual_style: VisualStyle
    temporal_progression: str | None = None
    transcript: str | None = None
    transcript_segments: list[dict[str, Any]] = Field(default_factory=list)
    transcript_scope: str | None = None
    ocr_text: list[str] = Field(default_factory=list)
    frame_ids: list[str] = Field(default_factory=list)
    clip_id: str | None = None
    measurements: Measurements | None = None
    claims: list[dict[str, Any]] = Field(default_factory=list)  # claim-level evidence (see validation/claims.py)
    # Persistent object tracks of the shot (tracker method): {track_id (stable within the video as
    # "<scene_id>:<n>"), class, first_time, last_time, n_frames, mean_confidence, uncertain
    # (few hits / long gaps -> association may be wrong)}. ``track_scope`` says ids are per shot.
    object_tracks: list[dict[str, Any]] = Field(default_factory=list)
    track_scope: str | None = None  # "shot" (ids are not re-identified across shots) | None when tracking did not run
    # Controlled lighting block: {level (measured brightness), appearance_time_of_day (label +
    # source + confidence), source, direction, softness (null unless estimated), method}
    lighting: dict[str, Any] | None = None
    # Colour metadata: container colour space / transfer / primaries / range from ffprobe when the
    # stream declares them (else null), the colour space the measured statistics were computed in
    # and the method; white balance is never guessed (null)
    color: dict[str, Any] | None = None
    provider: str | None = None
    confidence: float | None = None
    confidence_source: ConfidenceSource = ConfidenceSource.UNAVAILABLE
    quality: QualityScore | None = None
    validation: ValidationInfo | None = None
    hard_negatives: list[HardNegative] = Field(default_factory=list)
    annotation_scope: str | None = "scene"
    generation_target: str | None = "single_shot"

    @model_validator(mode="before")
    @classmethod
    def _camera_from_style(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("camera"), dict):
            data = {**data, "camera": camera_fields_from_style(data["camera"], data.get("visual_style"))}
        return data


class EventRecord(_Curated):
    record_id: str
    record_type: str = "event"
    video_id: str
    event_id: str
    event_type: str
    start_time: float
    end_time: float
    event: str
    entities: list[str] = Field(default_factory=list)
    action: str | None = None
    scene_ids: list[str] = Field(default_factory=list)
    frame_ids: list[str] = Field(default_factory=list)
    clip_ids: list[str] = Field(default_factory=list)
    source: str
    boundary_precision: str | None = None  # scene | measured | asr | frame | approximate
    start_uncertainty_ms: float | None = None
    end_uncertainty_ms: float | None = None
    confidence: float | None = None
    confidence_source: ConfidenceSource = ConfidenceSource.UNAVAILABLE
    relations: list[dict[str, Any]] = Field(default_factory=list)  # outgoing relations
    quality: QualityScore | None = None
    validation: ValidationInfo | None = None
    annotation_scope: str | None = "event"
    generation_target: str | None = "not_for_generation"


class SequenceRecord(_Curated):
    """A longer, coherent run of consecutive shots of one video (split parts of a long shot rejoined,
    or cuts whose setting and subjects agree): the unit for longer motion sequences."""

    record_id: str
    record_type: str = "sequence"
    video_id: str
    sequence_id: str
    scene_ids: list[str]
    clip_ids: list[str] = Field(default_factory=list)
    start_time: float
    end_time: float
    duration: float
    n_shots: int
    coherence: list[str] = Field(default_factory=list)  # why the shots were joined: rejoined_split | same_setting | shared_subject
    subjects: list[str] = Field(default_factory=list)  # object classes present in every joined shot
    setting: str | None = None
    time_of_day: str | None = None
    camera_movements: list[str] = Field(default_factory=list)  # per shot, standardized family (+ direction)
    summary: str  # the shot summaries in order
    generation_prompt: str | None = None
    shot_prompts: list[str] = Field(default_factory=list)  # the joined shots' own generation prompts, in order
    prompt_variants: list[str] = Field(default_factory=list)
    provider: str | None = None
    confidence: float | None = None
    confidence_source: ConfidenceSource = ConfidenceSource.UNAVAILABLE
    quality: QualityScore | None = None
    validation: ValidationInfo | None = None
    annotation_scope: str | None = "sequence"
    generation_target: str | None = "multi_shot"


class VideoDescriptionRecord(_Curated):
    """A generation-oriented description of the whole video or a multi-scene segment."""

    record_id: str
    record_type: str = "video_description"
    video_id: str
    start_time: float
    end_time: float
    scene_ids: list[str]
    subject: str | None = None
    environment: str | None = None
    action: str | None = None
    camera: str | None = None
    composition: str | None = None
    lighting: str | None = None
    motion: str | None = None
    temporal_progression: str
    transitions: str | None = None
    prompt: str  # a single flowing generation prompt assembled from the fields above
    shots: list[dict[str, Any]] = Field(default_factory=list)  # per-scene compact descriptors
    # Grounding of the description: the records it was assembled from, with timestamps
    # {scene_ids, clip_ids, frame_ids, timestamps: [[start, end], ...]}
    evidence: dict[str, Any] = Field(default_factory=dict)
    claims: list[dict[str, Any]] = Field(default_factory=list)  # claim-level evidence of the member shots (field, value, source, status)
    # Observed facts (measured / detected, with the shot they come from) separated from inferred or
    # free-text interpretation, so a reader can tell which sentences are evidence-backed
    facts: dict[str, list[str]] = Field(default_factory=lambda: {"observed": [], "inferred": []})
    grounding_assessed: bool = False  # True only when the description was compared with its frames / clips
    confidence: float | None = None
    confidence_source: ConfidenceSource = ConfidenceSource.UNAVAILABLE
    quality: QualityScore | None = None
    validation: ValidationInfo | None = None
    annotation_scope: str | None = "video"
    generation_target: str | None = "multi_shot"
