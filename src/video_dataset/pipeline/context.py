"""Per-video processing context passed to every stage function."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from video_dataset.config import PipelineConfig
from video_dataset.schemas.video import MediaInfo, VideoMetadata
from video_dataset.stages import Stage
from video_dataset.storage.paths import DataPaths
from video_dataset.storage.state_db import StateDB
from video_dataset.utils.io import read_json
from video_dataset.utils.logging import stage_logger
from video_dataset.utils.urls import InputItem


@dataclass
class StageOutput:
    """What a stage hands back to the runner for checkpointing."""

    artifact_path: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    skipped: bool = False  # stage disabled in config -> recorded as SKIPPED, downstream continues


@dataclass
class VideoContext:
    video_id: str
    config: PipelineConfig
    paths: DataPaths
    db: StateDB
    input_item: InputItem | None = None
    models: dict[str, Any] = field(default_factory=dict)  # shared model cache (loaded once per process)
    _metadata: VideoMetadata | None = None
    _media_info: MediaInfo | None = None

    def logger(self, stage: Stage) -> logging.LoggerAdapter:
        return stage_logger(self.video_id, stage.value)

    # ---- lazily loaded common artifacts ----
    @property
    def metadata(self) -> VideoMetadata:
        if self._metadata is None:
            p = self.paths.metadata_file(self.video_id)
            if not p.exists():
                raise FileNotFoundError(f"metadata.json missing for {self.video_id}; run DOWNLOAD first")
            self._metadata = VideoMetadata.model_validate(read_json(p))
        return self._metadata

    @property
    def media_info(self) -> MediaInfo:
        if self._media_info is None:
            p = self.paths.media_info_file(self.video_id)
            if not p.exists():
                raise FileNotFoundError(f"media_info.json missing for {self.video_id}; run PREPROCESS first")
            self._media_info = MediaInfo.model_validate(read_json(p))
        return self._media_info

    @property
    def video_path(self) -> Path:
        return self.paths.video_file(self.video_id)

    @property
    def duration(self) -> float:
        return float(self.media_info.duration)

    def invalidate(self) -> None:
        self._metadata = None
        self._media_info = None

    def config_hash(self, *sections: str) -> str:
        payload = {s: getattr(self.config, s).model_dump(mode="json") for s in sections if hasattr(self.config, s)}
        return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def get_model(self, key: str, factory: Any) -> Any:
        """Load-once cache for heavy models shared across videos in the same process."""
        if key not in self.models:
            self.models[key] = factory()
        return self.models[key]
