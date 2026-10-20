"""Typed pipeline configuration.

Config is layered: config/default.yaml -> config/models.yaml -> config/pipeline.yaml
-> user supplied --config file -> environment variables -> CLI --set overrides.
Every expensive or model-dependent knob lives here; nothing is hard-coded in the stages.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "config"


class _Section(BaseModel):
    model_config = ConfigDict(extra="allow")


class ProjectConfig(_Section):
    data_dir: str = "data"
    db_path: str | None = None  # default: <data_dir>/state.db
    log_level: str = "INFO"
    device: str = "auto"  # auto | cuda | mps | cpu
    seed: int = 1234
    ffmpeg_path: str | None = None
    ffprobe_path: str | None = None


class DownloadConfig(_Section):
    provider: str = "yt_dlp"
    max_resolution: int = 1080
    format: str = "mp4"
    cookies_file: str | None = None
    # Browser to read cookies from, yt-dlp style: "chrome", "chrome:Profile 1", "firefox", "edge", ...
    # (or a list [browser, profile, keyring, container])
    cookies_from_browser: str | list[str] | None = None
    # True: if cookies cannot be loaded (locked Chrome DB / DPAPI on Windows, missing file), warn and
    # download without them. False: treat that as a fatal error for every video.
    cookies_optional: bool = True
    rate_limit: str | None = None
    retries: int = 3
    concurrency: int = 2
    skip_existing: bool = True
    socket_timeout: int = 30
    extra_args: dict[str, Any] = Field(default_factory=dict)
    # JavaScript runtimes yt-dlp may use to solve YouTube's signature / n-parameter challenges.
    # yt-dlp enables only deno by default; entries are "name" or "name:/path/to/binary" and the first
    # one that is installed is used. Empty list = yt-dlp's own default.
    js_runtimes: list[str] = Field(default_factory=lambda: ["node", "deno", "bun"])
    # Remote components yt-dlp may download (and cache) on demand, e.g. the EJS challenge-solver
    # script from GitHub when the bundled yt-dlp-ejs package is missing or too old. Empty = never.
    remote_components: list[str] = Field(default_factory=lambda: ["ejs:github"])
    # Hosts a URL may point at. Empty = any http(s) host yt-dlp supports. Suffix match, so
    # "youtube.com" also allows "www.youtube.com" and "m.youtube.com".
    allowed_domains: list[str] = Field(default_factory=list)


class LimitsConfig(_Section):
    """Resource guards checked before expensive work starts."""

    max_duration_seconds: float | None = 7200.0  # videos longer than this are rejected before download
    max_file_size_gb: float | None = None  # reject a download whose reported size exceeds this
    min_free_disk_gb: float = 5.0  # refuse to download / extract frames when the data disk has less free space
    disk_check_stages: list[str] = Field(default_factory=lambda: ["DOWNLOAD", "PREPROCESS", "FRAME_EXTRACTION"])


class PreprocessConfig(_Section):
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    transcode_if_needed: bool = True
    decodable_codecs: list[str] = Field(default_factory=lambda: ["h264", "hevc", "vp9", "av1", "mpeg4", "vp8"])
    max_transcode_height: int = 1080


class SceneDetectionConfig(_Section):
    detector: str = "content"  # content | adaptive | threshold
    threshold: float = 27.0
    adaptive_threshold: float = 3.0
    fade_threshold: float = 12.0
    min_scene_len_seconds: float = 0.6
    detect_fades: bool = True
    max_scene_duration: float | None = 60.0
    frame_skip: int = 0
    downscale: int | None = None
    # A boundary whose content change is spread over several frames (instead of one spike) is a
    # gradual transition: labelled `dissolve` instead of `cut`.
    detect_dissolves: bool = True
    dissolve_min_frames: int = 3  # consecutive frames above dissolve_fraction * threshold around the boundary


class FrameSamplingConfig(_Section):
    min_frames_per_scene: int = 3
    max_frames_per_scene: int = 20
    motion_aware: bool = True
    scene_boundary_frames: bool = True
    boundary_offset_seconds: float = 0.15
    analysis_fps: float = 4.0
    analysis_width: int = 160
    change_threshold: float = 0.12
    min_frame_gap_seconds: float = 0.5
    # No two consecutive sampled frames of a shot may be further apart than this (frames are inserted
    # into the largest gaps until max_frames_per_scene is reached); the validation stage checks the
    # same limit, so a shot with fewer than max_frames_per_scene frames is never flagged for gaps.
    max_frame_gap_seconds: float = 5.0
    jpeg_quality: int = 90
    max_frame_side: int = 1280
    compute_optical_flow: bool = True
    extract_clips: bool = True
    clip_max_duration: float = 30.0
    # auto: stream-copy, then verify the file against the scene boundaries and re-encode any clip that is
    #       off by more than clip_tolerance_seconds (stream copy can only cut at keyframes)
    # libx264: always re-encode (exact, slower) | copy: keyframe-aligned only (fast, may include neighbours)
    clip_codec: str = "auto"
    clip_tolerance_seconds: float = 0.25
    clip_max_height: int = 720  # re-encoded clips are downscaled to this height
    workers: int | None = None  # threads for frame extraction + clip cutting; null -> pipeline.stage_workers
    # Duplicate frames: two sampled frames of one scene whose perceptual hashes (dHash) differ by at
    # most this many bits are the same image; the later one is dropped (its file deleted).
    drop_duplicate_frames: bool = True
    duplicate_frame_hamming: int = 4
    # Target-model media preparation: re-encode every clip to a fixed frame rate / size so a training
    # set is uniform. Off by default (clips keep the source fps and are only capped at clip_max_height).
    normalize_clips: bool = False
    target_fps: float | None = None  # e.g. 24
    target_height: int | None = None  # e.g. 480; with target_width the aspect_mode decides how to fit
    target_width: int | None = None  # e.g. 854
    aspect_mode: str = "scale"  # scale (fit inside, keep aspect) | pad (letterbox to exactly WxH) | crop (centre-crop to exactly WxH)
    # Boundary verification: decode the clip's first / last frames and match them against source
    # frames around the requested boundaries (perceptual hash) to measure where the clip really
    # starts and ends. copy: only stream-copied (keyframe-aligned) clips and clips of unknown origin;
    # all: every clip; none: skip. Results land in Clip.actual_start / actual_end / boundary_check.
    verify_clip_boundaries: str = "copy"
    boundary_search_seconds: float = 1.0  # source window searched on each side of a boundary
    boundary_max_hamming: int = 14  # dHash distance above which no source frame counts as a match (-> error)


class TranscriptionConfig(_Section):
    enabled: bool = True
    provider: str = "faster_whisper"  # faster_whisper | transformers | mock | none
    model: str = "base"
    language: str = "auto"
    compute_type: str = "auto"
    beam_size: int = 5
    vad_filter: bool = True
    word_timestamps: bool = True
    min_segment_confidence: float = 0.0
    initial_prompt: str | None = None  # bias vocabulary/script, e.g. "नमस्ते, आज हम बात करेंगे" for Devanagari Hindi
    device: str | None = None  # None -> project.device


class AudioEventsConfig(_Section):
    enabled: bool = True
    provider: str = "energy"  # energy | transformers | none
    model: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    window_seconds: float = 2.0
    hop_seconds: float = 1.0
    min_score: float = 0.3
    top_k: int = 3
    energy_threshold_db: float = -45.0


class OCRConfig(_Section):
    enabled: bool = True
    provider: str = "rapidocr"  # rapidocr | easyocr | paddleocr | mock | none
    languages: list[str] = Field(default_factory=lambda: ["en"])
    min_confidence: float = 0.5
    max_frames_per_scene: int = 6
    merge_similarity: float = 0.85
    merge_max_gap_seconds: float = 3.0
    min_text_length: int = 3  # 1-2 character reads are almost always noise
    # Text families seen in at least this fraction of scenes (and at least static_overlay_min_scenes
    # scenes) are watermarks / handles / logos: kept in the OCR file, excluded from events and QA.
    static_overlay_min_scene_fraction: float = 0.3
    static_overlay_min_scenes: int = 4
    workers: int | None = None  # OCR threads (one engine instance each); null -> pipeline.stage_workers


class VisionConfig(_Section):
    provider: str = "heuristic"  # heuristic | cv_models | hf | anthropic | openai_compatible | mock
    yolo_model: str = "yolov8n.pt"  # YOLO model for cv_models provider (n/s/m/l/x)
    yolo_confidence: float = 0.5  # Minimum confidence for YOLO detections
    yolo_max_frames: int = 8  # frames per scene sent to YOLO / CLIP (evenly spaced); bounds memory on small machines
    preset: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    max_images_per_request: int = 6
    image_max_side: int = 768
    frame_captions: bool = False
    frame_captions_per_scene: int = 3
    use_video_input: bool = False
    enrichers: list[str] = Field(default_factory=list)
    clip_model: str = "openai/clip-vit-base-patch32"
    # LAION aesthetic predictor: a linear head on the CLIP image embedding (~1 poor .. 10 excellent).
    # Runs wherever CLIP already runs (cv_models provider or the `clip` enricher); no extra model.
    aesthetic: bool = True
    aesthetic_weights: str | None = None  # path or URL of the .pth; null -> picked for clip_model and cached
    aesthetic_cache_dir: str | None = None  # null -> ~/.cache/video_dataset/aesthetic
    max_new_tokens: int = 700
    temperature: float = 0.0
    timeout_seconds: int = 120
    retries: int = 2
    requests_per_minute: float = 0.0  # API providers only; 0 = unlimited (SDK retries still handle 429s)
    device: str | None = None
    torch_dtype: str = "auto"
    verify_with_model: bool = True  # run the model as verifier when it supports verification
    # Scenes analysed concurrently. Measurements always run in parallel; the analyzer itself only when
    # it is thread-safe (heuristic, mock, API providers). Local torch models stay sequential.
    workers: int | None = None  # null -> pipeline.stage_workers
    # cv_models: follow detected subjects through the shot (YOLO on a low-rate decode of the scene +
    # IoU tracker) instead of only the handful of sampled frames. Gives per-track direction / path.
    tracking: bool = True
    track_fps: float = 3.0  # decode rate for tracking
    track_width: int = 640  # frames are downscaled to this width before detection
    track_max_seconds: float = 20.0  # longest span analysed per scene (from its start)
    track_iou_threshold: float = 0.3  # box overlap needed to continue a track
    track_max_age: int = 2  # frames a track may go undetected before it ends
    track_min_hits: int = 3  # detections a track needs before it counts


class LLMConfig(_Section):
    provider: str = "none"  # none | anthropic | openai_compatible | mock
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.2
    timeout_seconds: int = 120
    requests_per_minute: float = 0.0  # 0 = unlimited


class TemporalConfig(_Section):
    enabled: bool = True
    include_speech_events: bool = True
    include_ocr_events: bool = True
    include_static_overlay_text: bool = False  # watermark / fragment OCR tracks as events (normally noise)
    include_audio_events: bool = True
    include_camera_events: bool = True
    include_transitions: bool = True
    include_state_changes: bool = True
    min_camera_consistency: float = 0.6
    min_camera_motion: float = 0.35
    max_relation_neighbors: int = 6
    emit_inverse_relations: bool = True
    long_range_min_gap_seconds: float = 20.0
    causal_inference: str = "none"  # none | llm
    speech_min_words: int = 3


class QAConfig(_Section):
    questions_per_minute: float = 5.0
    min_questions: int = 8
    max_questions: int = 400
    seed: int = 1234
    type_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "timestamp": 1.0,
            "before_after": 1.5,
            "temporal_ordering": 1.5,
            "duration": 1.0,
            "event_localization": 1.0,
            "state_change": 1.0,
            "multi_event": 1.0,
            "long_range": 1.0,
        }
    )
    ordering_min_gap_seconds: float = 1.0
    long_range_min_gap_seconds: float = 20.0
    allow_unscored_evidence: bool = True  # events with confidence=None may be used (flagged for review)
    paraphrase: bool = False
    include_speech_in_answers: bool = True
    generator: str = "template"  # template | spacy | both - question generation method
    spacy_model: str = "en_core_web_sm"  # spaCy pipeline for generator=spacy/both (python -m spacy download en_core_web_sm)
    spacy_max_per_event: int = 3  # spaCy questions kept per event before the per-video target applies
    ocr_filter: bool = True  # skip on-screen text events whose OCR text does not look like words
    ocr_min_chars: int = 4  # on-screen text shorter than this (letters+digits) never anchors a question
    ocr_max_repeats: int = 3  # on-screen text resembling more than this many other texts in the video is a watermark/caption template
    ocr_max_question_fraction: float = 0.25  # max share of a video's questions anchored on on-screen text (>= 1 disables)


class ValidationConfig(_Section):
    minimum_confidence: float = 0.75
    review_confidence: float = 0.5
    verifier: str = "heuristic"  # none | heuristic | vlm | composite
    verify_sample_fraction: float = 1.0
    reject_on_missing_files: bool = True
    grounding_min_overlap: float = 0.3
    max_vlm_verifications_per_video: int = 200
    workers: int | None = None  # threads for verifier calls; null -> pipeline.stage_workers
    # Cross-field consistency: a scene whose annotations contradict each other (static camera but a
    # tracked subject, "dark" lighting on a bright frame, caption says pan left but flow says pan
    # right, ...) is rejected (reject_contradictions) or sent to review.
    consistency_checks: bool = True
    reject_contradictions: bool = True
    # Caption <-> frames alignment with CLIP: the scene's own caption must rank in the top
    # `clip_alignment_min_rank_fraction` of the video's captions by image-text similarity.
    # auto: only when a CLIP model is already loaded (cv_models / clip enricher); on: load one; off.
    # (YAML / --set turn "on" / "off" into booleans; both spellings are accepted.)
    clip_alignment: str | bool = "auto"
    # prompt: compare the frames with the natural-language generation prompt built from the annotation
    # (subject, setting, camera, lighting); summary: the full measurement summary (hex colours, numbers -
    # a poor CLIP text). Shots without any detected object make no verifiable claim and are skipped.
    clip_alignment_text: str = "prompt"
    clip_alignment_min_rank_fraction: float = 0.75  # bottom quarter of the video's captions ...
    clip_alignment_min_margin: float = 0.05  # ... and the best-matching caption must beat the own one by at least this cosine margin
    clip_alignment_min_candidates: int = 4  # videos with fewer captioned scenes skip the check
    clip_alignment_max_frames: int = 3
    frame_sequence_checks: bool = True  # frames of a scene must be ordered, inside the scene, without large gaps
    clip_checks: bool = True  # clip files must exist and match the scene span
    # Text-structure checks (captions: internal contradictions; QA: question form + answer type).
    # nlp_model: a spaCy pipeline for verb detection ("auto" = qa.spacy_model, null = rules only).
    text_checks: bool = True
    nlp_model: str | None = "auto"
    recompute_answers: bool = True  # re-derive numeric / ordering answers from the evidence events and compare
    relation_checks: bool = True  # every relation must satisfy its interval semantics; BEFORE graph must be acyclic
    claims: bool = True  # attach claim-level evidence (field, value, source, status, frames) to scene records
    # A shot whose scan shows a visual change at least this large (0..1) may hide a missed cut: the
    # record is marked generation_target=not_for_generation and sent to review.
    max_in_shot_change: float = 0.6


class QualityConfig(_Section):
    """Visual-quality thresholds. Flags are measured per shot (and per source video) and recorded as
    `quality_flags`; flagged shots go to review (or are rejected) and never reach gold."""

    min_height: int = 480  # source shorter than this -> low_resolution
    min_bitrate_kbps: float = 300.0  # source bitrate below this -> heavy compression (0 disables)
    min_sharpness: float = 30.0  # variance of Laplacian below this -> blurry (0 disables)
    max_dark_fraction: float = 0.6  # share of near-black pixels above this -> underexposed
    max_bright_fraction: float = 0.4  # share of near-white pixels above this -> overexposed
    max_blockiness: float = 0.35  # 8x8 block-boundary ratio above this -> compression artifacts (0 disables)
    reject_flagged: bool = False  # true: flagged shots are rejected; false: review + tier capped at bronze
    duration_tolerance_seconds: float = 1.0  # metadata duration vs probed duration mismatch
    min_fps: float = 10.0  # source frame rate below this -> low_fps (0 disables)
    max_aspect_ratio: float = 3.0  # width/height (or its inverse) above this -> extreme_aspect_ratio (0 disables)
    # A shot longer than frozen_min_duration whose scan never changes more than frozen_max_change
    # (0..1 content change) and shows no motion -> frozen (a still image / stalled stream)
    frozen_max_change: float = 0.004
    frozen_min_duration: float = 2.0
    policy_version: str = "quality-2026.10"  # recorded with the flags so thresholds are traceable


class LicensingConfig(_Section):
    """Source licensing. yt-dlp reports the YouTube license ("Creative Commons Attribution license
    (reuse allowed)" or the standard license); it is normalised to `license_status`."""

    allowed: list[str] = Field(default_factory=lambda: ["creative_commons"])  # statuses that count as licensed for reuse
    action: str = "flag"  # flag: record + subset only | reject: do not download videos outside `allowed`
    subset: bool = True  # add the "licensed" subset to every record of an allowed video
    # rights_status values that a release requiring verified rights may contain. The platform's
    # declared license is evidence, not proof: `declared_cc_by` means YouTube reported a CC-BY
    # license; `restricted` the standard license; `unknown` nothing usable; `removed` a takedown.
    release_rights: list[str] = Field(default_factory=lambda: ["declared_cc_by"])


class DeduplicationConfig(_Section):
    enabled: bool = True
    near_duplicate_threshold: float = 0.9
    question_similarity_threshold: float = 0.92
    evidence_iou_threshold: float = 0.5
    method: str = "minhash"  # minhash | tfidf
    num_perm: int = 128
    shingle_size: int = 3
    # Near-duplicate *videos* (re-uploads, re-encodes): perceptual-hash fingerprint of the scan pass;
    # two videos whose hash sets overlap at least this much are the same content.
    video_fingerprint: bool = True
    video_similarity_threshold: float = 0.8


class SplitConfig(_Section):
    """Deterministic train / validation / test assignment per *source video* (hash of the video id),
    so no scene, clip, frame or question of one video ever lands in two splits."""

    enabled: bool = True
    validation_fraction: float = 0.05
    test_fraction: float = 0.05
    salt: str = "video-dataset-v1"  # change to reshuffle; keep fixed to keep the split stable across runs


class TierConfig(_Section):
    """Gold / silver / bronze from signals already measured; nothing is guessed."""

    enabled: bool = True
    gold_min_overall: float = 0.8  # quality.overall needed for gold
    silver_min_overall: float = 0.6  # quality.overall needed for silver
    gold_requires_grounding: bool = True  # gold needs a grounding component (verifier or direct measurement)
    gold_min_confidence: float = 0.75  # and a confidence from a verifier / measurement at least this high


class CinematicSubsetConfig(_Section):
    """`subsets: ["cinematic"]` + final/cinematic/: shots that clear tier, aesthetic and camera criteria.
    QA and events join when every scene they rest on is in the subset."""

    enabled: bool = True
    min_tier: str = "silver"  # gold | silver | bronze
    min_aesthetic: float = 5.0  # LAION scale 1..10
    require_aesthetic: bool = False  # true: records without an aesthetic score are excluded
    min_scene_duration: float = 1.5
    require_known_camera: bool = True  # camera.movement must not be "unknown"
    min_sharpness: float = 50.0  # variance of Laplacian; 0 disables
    write_files: bool = True  # also write final/cinematic/<type>.jsonl (+ dataset.jsonl)


class SequenceConfig(_Section):
    """Longer coherent motion sequences: consecutive shots of one video joined into a `sequence`
    record (split parts of one long shot always rejoin; cuts join when the setting / subject agree)."""

    enabled: bool = True
    min_duration: float = 8.0  # sequences shorter than this are not emitted
    max_duration: float = 120.0
    max_shots: int = 12
    require_same_setting: bool = True  # consecutive shots must agree on indoor/outdoor and time of day when both are known
    require_shared_subject: bool = True  # ... and share at least one detected object class (when both list objects)
    allow_fades: bool = False  # a fade / dissolve ends a sequence unless true


class StrictConfig(_Section):
    """The strict training release: records whose required checks all passed. Written as the
    ``strict`` subset (and ``final/strict/`` files) so a training job never has to hand-filter."""

    enabled: bool = True
    write_files: bool = True
    generation_targets: list[str] = Field(default_factory=lambda: ["single_shot", "multi_shot"])  # what scene / clip / sequence records may be


class ExportConfig(_Section):
    formats: list[str] = Field(default_factory=lambda: ["jsonl", "parquet"])
    include_review: bool = True
    include_rejected: bool = False
    output_dir: str | None = None  # default <data_dir>/final
    relative_paths: bool = True
    combined_jsonl: bool = True  # also write one dataset.jsonl holding every record (tagged with record_type)
    per_type_jsonl: bool = True  # write frames.jsonl, scenes.jsonl, ... alongside
    per_type_parquet: bool = True  # frames.parquet, scenes.parquet, ... with an explicit schema (what loaders should read)
    # Keep records of previously exported videos when the final files are rebuilt, even if their
    # per_video/ folder is gone. New exports of the same video replace its old records (by record id).
    merge_existing: bool = True
    hard_negatives: bool = True  # attach rule-built wrong answers / captions to QA, scene and clip records
    hard_negatives_per_record: int = 3
    split: SplitConfig = Field(default_factory=SplitConfig)
    tiers: TierConfig = Field(default_factory=TierConfig)
    cinematic: CinematicSubsetConfig = Field(default_factory=CinematicSubsetConfig)
    sequences: SequenceConfig = Field(default_factory=SequenceConfig)
    prompt_variants: int = 3  # template / synonym rewrites of generation_prompt per scene, clip and sequence (0 disables)
    exclude_duplicate_videos: bool = False  # true: records of near-duplicate videos are left out of final/ (false: kept, bronze, same split as the canonical video)
    strict: StrictConfig = Field(default_factory=StrictConfig)
    # Duration bands: every scene / clip / sequence record joins the subset "duration_<band>" its length falls in
    duration_bands: dict[str, list[float]] = Field(default_factory=lambda: {"short": [0.0, 3.0], "medium": [3.0, 10.0], "long": [10.0, 1e9]})
    media_manifest: bool = True  # write final/media_manifest.jsonl (one row per referenced asset with existence + size)
    hash_media: bool = False  # also sha256 every referenced asset (slow on large sets)
    require_rights: bool = False  # true: only videos whose rights_status is in licensing.release_rights are exported


class UploadConfig(_Section):
    """Automatic upload of the exported dataset to the Hugging Face Hub in size-bounded shards."""

    provider: str = "none"  # none | huggingface
    repo_id: str | None = None  # "<user-or-org>/<dataset-name>"
    repo_type: str = "dataset"
    private: bool = True
    token_env: str = "HF_TOKEN"  # environment variable holding a *write* token (never put the token in YAML)
    threshold_mb: float = 1024.0  # upload once the current shard in final/ reaches this size
    include_media: bool = False  # also upload the frames and clips the records reference
    path_in_repo: str = ""  # optional prefix inside the repo, e.g. "v1"
    after_upload: str = "archive"  # archive (move to final/uploaded/shard_NNNN, keeps a local copy) | delete (free disk)
    retries: int = 3
    # Run the release gates (schema, record ids JSONL == Parquet, split leakage, and - with
    # include_media - media references) before every upload; a failing gate aborts the upload.
    release_gate: bool = True


class CleanupConfig(_Section):
    """What to delete automatically once a video has reached EXPORT (its records are in final/)."""

    after_export: str = "media"  # none | media | all
    # media: original download, canonical video.mp4 and audio.wav (the bulk); every JSON artifact stays
    #        so `--force-from VALIDATION` / `export --rerun` still work.
    # all:   also transcript, OCR, annotations, QA, validated and download metadata; only the
    #        per-video export and the log remain.
    frames_and_clips: bool = True  # with media/all: also delete frames/<id>/ (scene_NNN folders) and clips/<id>/
    # The exported records keep their frame_path/clip_path values but the files are gone, so keep this
    # false if you need the media locally. Ignored while upload.include_media is on: the media is
    # bundled into the shard first and, with upload.after_upload=delete, removed after the upload.


class PipelineStagesConfig(_Section):
    stages: list[str] = Field(
        default_factory=lambda: [
            "DOWNLOAD",
            "PREPROCESS",
            "SCENE_DETECTION",
            "FRAME_EXTRACTION",
            "AUDIO",
            "TRANSCRIPTION",
            "OCR",
            "VISION_ANALYSIS",
            "TEMPORAL_ANALYSIS",
            "QA_GENERATION",
            "VALIDATION",
            "EXPORT",
        ]
    )
    stage_retries: int = 1
    continue_on_error: bool = True
    workers: int = 1  # videos processed concurrently in the CPU phase (DOWNLOAD..AUDIO)
    stage_workers: int = 0  # threads inside a stage (clips, frames, OCR, vision, verification); 0 = auto (min(4, cores))
    model_stages_sequential: bool = True
    stage_timeout_seconds: int | None = None
    # A DONE stage whose config section changed since it ran is re-executed (with everything after
    # it) instead of being reused with a note in the log.
    rerun_on_config_change: bool = False


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    download: DownloadConfig = Field(default_factory=DownloadConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    scene_detection: SceneDetectionConfig = Field(default_factory=SceneDetectionConfig)
    frame_sampling: FrameSamplingConfig = Field(default_factory=FrameSamplingConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    audio_events: AudioEventsConfig = Field(default_factory=AudioEventsConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    qa: QAConfig = Field(default_factory=QAConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    licensing: LicensingConfig = Field(default_factory=LicensingConfig)
    deduplication: DeduplicationConfig = Field(default_factory=DeduplicationConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    cleanup: CleanupConfig = Field(default_factory=CleanupConfig)
    upload: UploadConfig = Field(default_factory=UploadConfig)
    pipeline: PipelineStagesConfig = Field(default_factory=PipelineStagesConfig)
    presets: dict[str, Any] = Field(default_factory=dict)

    # ----- derived paths -----
    @property
    def data_dir(self) -> Path:
        p = Path(self.project.data_dir)
        return p if p.is_absolute() else (Path.cwd() / p)

    @property
    def db_path(self) -> Path:
        if self.project.db_path:
            p = Path(self.project.db_path)
            return p if p.is_absolute() else (Path.cwd() / p)
        return self.data_dir / "state.db"

    @property
    def export_dir(self) -> Path:
        if self.export.output_dir:
            p = Path(self.export.output_dir)
            return p if p.is_absolute() else (Path.cwd() / p)
        return self.data_dir / "final"


# --------------------------------------------------------------------------------------
# Loading / merging
# --------------------------------------------------------------------------------------


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping at the top level")
    return data


def _coerce_value(raw: str) -> Any:
    """Parse CLI --set values: YAML scalars give us ints/floats/bools/null/lists for free."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def apply_dotted_overrides(data: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(data)
    for dotted, value in overrides.items():
        keys = dotted.split(".")
        cur = out
        for k in keys[:-1]:
            if not isinstance(cur.get(k), dict):
                cur[k] = {}
            cur = cur[k]
        cur[keys[-1]] = _coerce_value(value) if isinstance(value, str) else value
    return out


def _apply_env(data: dict[str, Any]) -> dict[str, Any]:
    env_map = {
        "VIDEO_DATASET_DATA_DIR": "project.data_dir",
        "VIDEO_DATASET_DEVICE": "project.device",
        "VIDEO_DATASET_LOG_LEVEL": "project.log_level",
        "VIDEO_DATASET_DB_PATH": "project.db_path",
        "YT_DLP_COOKIES": "download.cookies_file",
        "YT_DLP_COOKIES_FROM_BROWSER": "download.cookies_from_browser",
    }
    overrides = {dotted: os.environ[var] for var, dotted in env_map.items() if os.environ.get(var)}
    return apply_dotted_overrides(data, overrides) if overrides else data


def _resolve_presets(data: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    """`vision.preset: qwen2_5_vl_3b` copies presets.vision.qwen2_5_vl_3b into the vision section.

    Preset values replace keys that are still at their bundled default; keys the user set explicitly
    (user config file, env, --set) win over the preset.
    """
    presets = data.get("presets") or {}
    for section in ("vision", "transcription", "llm", "ocr"):
        sec = data.get(section) or {}
        name = sec.get("preset")
        if not name:
            continue
        preset = (presets.get(section) or {}).get(name)
        if preset is None:
            raise ValueError(f"Unknown {section} preset '{name}'. Available: {sorted((presets.get(section) or {}).keys())}")
        default_sec = defaults.get(section) or {}
        merged = dict(sec)
        for key, value in preset.items():
            user_set = key in sec and sec[key] != default_sec.get(key)
            if not user_set:
                merged[key] = copy.deepcopy(value)
        merged["preset"] = name
        data[section] = merged
    return data


def load_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    config_dir: str | Path | None = None,
) -> PipelineConfig:
    """Load the layered configuration and return a validated PipelineConfig."""
    cdir = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    data: dict[str, Any] = {}
    for name in ("default.yaml", "models.yaml", "pipeline.yaml"):
        data = deep_merge(data, _load_yaml(cdir / name))
    defaults = copy.deepcopy(data)
    if config_path:
        user_path = Path(config_path)
        if not user_path.exists():
            raise FileNotFoundError(f"Config file not found: {user_path}")
        # Do not double-apply the bundled defaults when the user points at them.
        if user_path.resolve() not in {(cdir / n).resolve() for n in ("default.yaml", "models.yaml", "pipeline.yaml")}:
            data = deep_merge(data, _load_yaml(user_path))
    data = _apply_env(data)
    if overrides:
        data = apply_dotted_overrides(data, overrides)
    data = _resolve_presets(data, defaults)
    return PipelineConfig.model_validate(data)


def dump_config(config: PipelineConfig) -> str:
    return yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
