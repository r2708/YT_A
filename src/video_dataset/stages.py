"""Pipeline stage names and ordering (shared by storage, runner and CLI)."""

from __future__ import annotations

from enum import Enum


class Stage(str, Enum):
    DOWNLOAD = "DOWNLOAD"
    PREPROCESS = "PREPROCESS"
    SCENE_DETECTION = "SCENE_DETECTION"
    FRAME_EXTRACTION = "FRAME_EXTRACTION"
    AUDIO = "AUDIO"
    TRANSCRIPTION = "TRANSCRIPTION"
    OCR = "OCR"
    VISION_ANALYSIS = "VISION_ANALYSIS"
    TEMPORAL_ANALYSIS = "TEMPORAL_ANALYSIS"
    QA_GENERATION = "QA_GENERATION"
    VALIDATION = "VALIDATION"
    EXPORT = "EXPORT"

    def __str__(self) -> str:
        return self.value


STAGE_ORDER: list[Stage] = list(Stage)

# Stages that load heavy models; the runner executes them sequentially so models load once.
MODEL_STAGES: set[Stage] = {Stage.TRANSCRIPTION, Stage.OCR, Stage.VISION_ANALYSIS, Stage.VALIDATION}

# Short labels used in the human-readable progress lines.
STAGE_LABELS: dict[Stage, str] = {
    Stage.DOWNLOAD: "DOWNLOAD",
    Stage.PREPROCESS: "PREPROCESS",
    Stage.SCENE_DETECTION: "SCENE",
    Stage.FRAME_EXTRACTION: "FRAMES",
    Stage.AUDIO: "AUDIO",
    Stage.TRANSCRIPTION: "ASR",
    Stage.OCR: "OCR",
    Stage.VISION_ANALYSIS: "VISION",
    Stage.TEMPORAL_ANALYSIS: "TEMPORAL",
    Stage.QA_GENERATION: "QA",
    Stage.VALIDATION: "VALIDATION",
    Stage.EXPORT: "EXPORT",
}


# One-line "what is happening now" shown when a stage starts.
STAGE_DESCRIPTIONS: dict[Stage, str] = {
    Stage.DOWNLOAD: "downloading the source video with yt-dlp",
    Stage.PREPROCESS: "probing the file, writing video.mp4 and extracting audio.wav with ffmpeg",
    Stage.SCENE_DETECTION: "detecting scene cuts and fades (PySceneDetect, decodes the whole video)",
    Stage.FRAME_EXTRACTION: "scanning motion, picking key frames and cutting per-scene clips",
    Stage.AUDIO: "classifying non-speech audio events",
    Stage.TRANSCRIPTION: "transcribing speech (whisper)",
    Stage.OCR: "reading on-screen text from the sampled frames",
    Stage.VISION_ANALYSIS: "describing every scene (vision analyzer)",
    Stage.TEMPORAL_ANALYSIS: "building the event timeline and temporal relations",
    Stage.QA_GENERATION: "generating temporal question/answer pairs",
    Stage.VALIDATION: "validating and de-duplicating records",
    Stage.EXPORT: "writing the per-video JSONL/Parquet export",
}


class StageStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"

    def __str__(self) -> str:
        return self.value


def stages_from(stage: Stage) -> list[Stage]:
    idx = STAGE_ORDER.index(stage)
    return STAGE_ORDER[idx:]


def parse_stage(name: str) -> Stage:
    try:
        return Stage(name.strip().upper())
    except ValueError as exc:
        raise ValueError(f"Unknown stage '{name}'. Valid stages: {', '.join(s.value for s in STAGE_ORDER)}") from exc
