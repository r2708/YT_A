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
    rate_limit: str | None = None
    retries: int = 3
    concurrency: int = 2
    skip_existing: bool = True
    socket_timeout: int = 30
    extra_args: dict[str, Any] = Field(default_factory=dict)
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


class VisionConfig(_Section):
    provider: str = "heuristic"  # heuristic | hf | anthropic | openai_compatible | mock
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
    max_new_tokens: int = 700
    temperature: float = 0.0
    timeout_seconds: int = 120
    retries: int = 2
    requests_per_minute: float = 0.0  # API providers only; 0 = unlimited (SDK retries still handle 429s)
    device: str | None = None
    torch_dtype: str = "auto"
    verify_with_model: bool = True  # run the model as verifier when it supports verification


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


class ValidationConfig(_Section):
    minimum_confidence: float = 0.75
    review_confidence: float = 0.5
    verifier: str = "heuristic"  # none | heuristic | vlm | composite
    verify_sample_fraction: float = 1.0
    reject_on_missing_files: bool = True
    grounding_min_overlap: float = 0.3
    max_vlm_verifications_per_video: int = 200


class DeduplicationConfig(_Section):
    enabled: bool = True
    near_duplicate_threshold: float = 0.9
    question_similarity_threshold: float = 0.92
    evidence_iou_threshold: float = 0.5
    method: str = "minhash"  # minhash | tfidf
    num_perm: int = 128
    shingle_size: int = 3


class ExportConfig(_Section):
    formats: list[str] = Field(default_factory=lambda: ["jsonl", "parquet"])
    include_review: bool = True
    include_rejected: bool = False
    output_dir: str | None = None  # default <data_dir>/final
    relative_paths: bool = True
    combined_jsonl: bool = True  # also write one dataset.jsonl holding every record (tagged with record_type)
    per_type_jsonl: bool = True  # write frames.jsonl, scenes.jsonl, ... alongside
    # Keep records of previously exported videos when the final files are rebuilt, even if their
    # per_video/ folder is gone. New exports of the same video replace its old records (by record id).
    merge_existing: bool = True


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
    workers: int = 1
    model_stages_sequential: bool = True
    stage_timeout_seconds: int | None = None


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
