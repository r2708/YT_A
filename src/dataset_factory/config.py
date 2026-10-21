"""Factory configuration: every threshold, path, mapping and policy lives here, validated by
pydantic. ``config/factory/dataset.yaml`` is the entry point; ``quality_rules.yaml`` and
``export_targets.yaml`` next to it are merged in when present. ``--set a.b=c`` overrides win.

Thresholds are starting points, not truths: calibrate them on a reviewed sample (see
``config/factory/quality_rules.yaml``) before trusting them for a release.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config" / "factory"
DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "dataset.yaml"
SIDE_FILES = {"quality_rules.yaml": ("media", "captions", "frames", "dedup"), "export_targets.yaml": ("exports",)}


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathsConfig(_Section):
    """Everything derived lives under ``data_root`` (outside the code repo if you like); the raw
    source copy under ``raw_dir`` is never modified after ingest."""

    data_root: str = "data/factory"
    raw_dir: str = "raw"
    assets_dir: str = "assets"
    frames_dir: str = "assets/frames"
    curated_dir: str = "curated"
    exports_dir: str = "exports"
    reports_dir: str = "reports"
    cache_dir: str = "cache"
    state_db: str = "curated/factory.sqlite"


class SourceConfig(_Section):
    kind: str = "huggingface"  # huggingface | local
    repo_id: str = "raj270898/youtube-video-dataset"
    revision: str | None = None
    token_env: str = "HF_TOKEN"
    record_types: list[str] = Field(default_factory=lambda: ["clips"])  # only video-caption rows are training candidates
    shard_glob: str = "shard_*"
    local_dir: str | None = None  # a final/ or shard directory with <type>.parquet / .jsonl + manifest.json
    # Where ``clip_path`` (relative to the producer's data dir) resolves; null -> assets are unavailable
    media_root: str | None = None
    manifest_file: str = "manifest.json"
    provenance_file: str = "provenance.jsonl"


DEFAULT_ALIASES: dict[str, list[str]] = {
        "source_record_id": ["record_id", "clip_id"],
        "canonical_video_id": ["source_group_id", "canonical_video_id", "video_id"],
        "original_video_id": ["video_id"],
        "video_path": ["clip_path", "video_path", "path"],
        "start_time_seconds": ["start_time", "start"],
        "end_time_seconds": ["end_time", "end"],
        "width": ["width"],
        "height": ["height"],
        "fps": ["fps"],
        "caption_original": ["description", "caption", "summary"],
        "generation_prompt_original": ["generation_prompt", "prompt"],
        "prompt_variants": ["prompt_variants"],
        "actions": ["actions"],
        "subject_motion": ["subject_motion"],
        "camera": ["camera"],
        "quality_flags": ["quality_flags"],
        "confidence": ["confidence"],
        "tier": ["tier"],
        "validation": ["validation"],
        "frame_ids": ["frame_ids"],
        "source_split": ["split"],
}


class MappingConfig(_Section):
    """Canonical field -> candidate source fields, first present wins. Entries given in the config
    are merged over ``DEFAULT_ALIASES`` (override one field without restating the rest). Unknown
    source fields are kept verbatim in ``source_payload``."""

    aliases: dict[str, list[str]] = Field(default_factory=lambda: copy.deepcopy(DEFAULT_ALIASES))
    caption_priority: list[str] = Field(default_factory=lambda: ["generation_prompt_original", "caption_original", "prompt_variants"])

    @model_validator(mode="before")
    @classmethod
    def _merge_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("aliases"), dict):
            merged = copy.deepcopy(DEFAULT_ALIASES)
            for k, v in data["aliases"].items():
                merged[k] = [v] if isinstance(v, str) else list(v)
            data = {**data, "aliases": merged}
        return data


class MediaRules(_Section):
    min_duration_seconds: float = 1.0
    max_duration_seconds: float = 30.0
    min_width: int = 256
    min_height: int = 256
    min_fps: float = 8.0
    max_fps: float = 120.0
    max_aspect_ratio: float = 3.0  # width/height or inverse above this -> implausible dimensions
    sample_frames: int = 6  # evenly spaced frames decoded for the black / duplicate / fingerprint checks
    black_mean_max: float = 16.0  # mean luminance (0..255) at or below which a frame counts as black
    black_std_max: float = 6.0
    max_black_fraction: float = 0.5  # share of sampled frames that may be black before the clip fails
    review_black_fraction: float = 0.25  # above this (and below the failure line) -> needs_review; a short fade-in stays passed
    review_duplicate_fraction: float = 0.5  # repeated sampled frames above this -> needs_review
    duplicate_frame_hamming: int = 2  # dHash distance at or below which two sampled frames are the same image
    max_duplicate_fraction: float = 0.9  # all sampled frames identical -> frozen clip
    metadata_tolerance_fraction: float = 0.1  # |record dims - probed dims| / probed above this -> mismatch
    duration_tolerance_seconds: float = 0.5  # |record span - probed duration| above this -> mismatch


class CaptionRules(_Section):
    min_chars: int = 20
    max_chars: int = 1200
    placeholder_patterns: list[str] = Field(default_factory=lambda: [r"\{[a-z_]+\}", r"\bTODO\b", r"\bN/A\b", r"\blorem ipsum\b", r"<[^>]+>"])
    template_residue_patterns: list[str] = Field(default_factory=lambda: [r"#[0-9a-fA-F]{6}", r"\bCLIP classifies\b", r"\bNo objects were detected\b", r"\bdominant colou?rs?\b"])
    motion_terms: list[str] = Field(default_factory=lambda: ["pan", "tilt", "dolly", "tracking", "zoom", "orbit", "crane", "handheld", "moves", "moving", "walks", "walking", "runs", "running", "turns", "rotates", "drifts", "flows", "flies", "drives", "camera movement", "motion", "slowly", "quickly"])
    language: str | None = None  # e.g. "en"; null = no language restriction
    max_duplicate_captions: int = 20  # a caption shared by more records than this -> template-like, needs_review


class FrameRules(_Section):
    start_timestamp_seconds: float = 0.0
    first_usable_policy: str = "first_non_black"  # first_non_black | fixed
    max_search_seconds: float = 2.0
    format: str = "png"
    consistency_hamming: int = 6  # re-decoded frame vs stored frame dHash distance that still counts as the same frame


class DedupRules(_Section):
    exact: bool = True
    perceptual: bool = True
    fingerprint_similarity: float = 0.85  # Jaccard of sampled-frame dHashes between records of *different* videos
    action: str = "review"  # review (flag candidates) | exclude (drop near-duplicate candidates from approved exports)


class RightsRules(_Section):
    """``source_rights_status`` per record. The source dataset's licence says nothing about the
    videos; the platform's declared licence is evidence, not clearance. Only ``cleared`` enters
    approved exports unless ``export_override`` says otherwise (recorded loudly)."""

    status_map: dict[str, str] = Field(default_factory=lambda: {"declared_cc_by": "review_required", "creative_commons": "review_required", "restricted": "restricted", "standard": "restricted", "removed": "restricted", "unknown": "review_required"})
    evidence_file: str | None = None  # JSONL: {"video_id" | "canonical_video_id", "status": cleared|restricted|review_required, "evidence": "..."}
    export_allowed: list[str] = Field(default_factory=lambda: ["cleared"])
    export_override: list[str] = Field(default_factory=list)  # additional statuses allowed into exports; a warning + the override are written to the release manifest


class SplitRules(_Section):
    validation_fraction: float = 0.05
    test_fraction: float = 0.05
    salt: str = "dataset-factory-v1"
    min_validation_groups: int = 1
    min_test_groups: int = 1
    persist_file: str = "splits.json"  # under curated_dir; existing group assignments are kept across runs


class ProviderConfig(_Section):
    """Caption review / improvement providers. ``none`` reuses source annotations and runs the
    deterministic checks only. Adapters register in ``captions.providers``; none is bundled that
    calls a remote API, so nothing leaves the machine unless you add one."""

    name: str = "none"
    model: str | None = None
    template_version: str = "caption-review-v1"
    max_calls: int = 0
    cache: bool = True
    timeout_seconds: float = 60.0
    retries: int = 2
    settings: dict[str, Any] = Field(default_factory=dict)


class ResolutionPolicy(_Section):
    mode: str = "any"  # any | min | exact
    width: int | None = None
    height: int | None = None
    aspect_tolerance: float = 0.05  # |w/h - target| / target allowed for exact / min modes


class ExportTarget(_Section):
    format: str = "generic-jsonl"  # generic-jsonl | canonical-parquet
    kind: str = "t2v"  # t2v | i2v | canonical
    packaging: str = "hardlink"  # hardlink | copy | reference
    video_dir: str = "videos"
    image_dir: str = "images"
    data_file: str = "data.jsonl"
    caption_field: str = "caption_final"
    require_rights: bool = True
    min_duration_seconds: float | None = None
    max_duration_seconds: float | None = None
    resolution: ResolutionPolicy = Field(default_factory=ResolutionPolicy)
    fps_min: float | None = None
    include_splits: list[str] = Field(default_factory=lambda: ["train", "validation", "test"])
    notes: str = ""


class ExportsConfig(_Section):
    version: str | None = None  # null -> <UTC date>-<config hash>
    targets: dict[str, ExportTarget] = Field(default_factory=lambda: {
        "canonical": ExportTarget(format="canonical-parquet", kind="canonical", packaging="reference", require_rights=False),
        "wan-t2v": ExportTarget(kind="t2v", notes="generic JSONL: video + caption; not a specific Wan trainer format until an adapter is verified against that trainer"),
        "wan-i2v": ExportTarget(kind="i2v", notes="generic JSONL: start image + video + caption"),
    })


class FactoryConfig(_Section):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    source: SourceConfig = Field(default_factory=SourceConfig)
    mapping: MappingConfig = Field(default_factory=MappingConfig)
    media: MediaRules = Field(default_factory=MediaRules)
    captions: CaptionRules = Field(default_factory=CaptionRules)
    frames: FrameRules = Field(default_factory=FrameRules)
    dedup: DedupRules = Field(default_factory=DedupRules)
    rights: RightsRules = Field(default_factory=RightsRules)
    splits: SplitRules = Field(default_factory=SplitRules)
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    exports: ExportsConfig = Field(default_factory=ExportsConfig)
    batch_size: int = 64  # rows per streamed Parquet batch
    offline: bool = True  # true: no provider other than ``none`` may run
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _check(self) -> FactoryConfig:
        if self.source.kind not in ("huggingface", "local"):
            raise ValueError("source.kind must be huggingface | local")
        if self.frames.first_usable_policy not in ("first_non_black", "fixed"):
            raise ValueError("frames.first_usable_policy must be first_non_black | fixed")
        if self.dedup.action not in ("review", "exclude"):
            raise ValueError("dedup.action must be review | exclude")
        for name, t in self.exports.targets.items():
            if t.kind not in ("t2v", "i2v", "canonical"):
                raise ValueError(f"exports.targets.{name}.kind must be t2v | i2v | canonical")
            if t.packaging not in ("hardlink", "copy", "reference"):
                raise ValueError(f"exports.targets.{name}.packaging must be hardlink | copy | reference")
            if t.resolution.mode not in ("any", "min", "exact"):
                raise ValueError(f"exports.targets.{name}.resolution.mode must be any | min | exact")
        if self.media.min_duration_seconds >= self.media.max_duration_seconds:
            raise ValueError("media.min_duration_seconds must be below media.max_duration_seconds")
        if self.splits.validation_fraction + self.splits.test_fraction >= 1.0:
            raise ValueError("splits: validation_fraction + test_fraction must be below 1")
        return self

    # ------------------------------------------------------------------ derived paths
    @property
    def data_root(self) -> Path:
        return Path(self.paths.data_root)

    def path(self, name: str) -> Path:
        return self.data_root / getattr(self.paths, name)

    def section_hash(self, *sections: str) -> str:
        data = {s: getattr(self, s).model_dump(mode="json") for s in sections}
        return hashlib.sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]

    @property
    def config_hash(self) -> str:
        return self.section_hash("mapping", "media", "captions", "frames", "dedup", "rights", "splits", "provider", "exports")


# ---------------------------------------------------------------------- loading
def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _coerce(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def apply_overrides(data: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    out = copy.deepcopy(data)
    for dotted, value in (overrides or {}).items():
        cur = out
        keys = dotted.split(".")
        for k in keys[:-1]:
            if not isinstance(cur.get(k), dict):
                cur[k] = {}
            cur = cur[k]
        cur[keys[-1]] = _coerce(value) if isinstance(value, str) else value
    return out


class ConfigError(ValueError):
    pass


def load_factory_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> FactoryConfig:
    """dataset.yaml (+ quality_rules.yaml / export_targets.yaml beside it) + overrides -> validated config."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_FILE
    data: dict[str, Any] = {}
    if cfg_path.exists():
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ConfigError(f"{cfg_path}: top level must be a mapping")
    elif path:
        raise ConfigError(f"config file not found: {cfg_path}")
    for side, keys in SIDE_FILES.items():
        p = cfg_path.parent / side
        if p.exists():
            extra = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            data = _deep_merge(data, {k: v for k, v in extra.items() if k in keys})
    data = apply_overrides(data, overrides)
    try:
        return FactoryConfig.model_validate(data)
    except ValidationError as exc:
        lines = [f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in exc.errors()]
        raise ConfigError("invalid factory configuration:\n  " + "\n  ".join(lines)) from None
