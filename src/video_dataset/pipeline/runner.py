"""Resumable per-video pipeline runner.

* Every stage is checkpointed in SQLite; DONE stages are skipped on re-run.
* A failing optional stage (AUDIO / TRANSCRIPTION / OCR) is recorded and the video continues.
* A failing required stage stops that video only; the batch continues with the next video.
* CPU/IO-bound stages can run for several videos in parallel; model stages run sequentially so
  heavy models are loaded once per process and never fight for GPU memory.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from video_dataset.audio.stage import audio_stage
from video_dataset.config import PipelineConfig
from video_dataset.dataset.stage import export_stage
from video_dataset.downloader.stage import download_stage
from video_dataset.frame_sampling.stage import frame_extraction_stage
from video_dataset.ocr.stage import ocr_stage
from video_dataset.pipeline.context import StageOutput, VideoContext
from video_dataset.preprocessing.stage import preprocess_stage
from video_dataset.questions.stage import qa_stage
from video_dataset.scene_detection.stage import scene_detection_stage
from video_dataset.stages import MODEL_STAGES, STAGE_LABELS, STAGE_ORDER, Stage, StageStatus
from video_dataset.storage.paths import DataPaths
from video_dataset.storage.state_db import StateDB
from video_dataset.temporal.stage import temporal_stage
from video_dataset.transcription.stage import transcription_stage
from video_dataset.utils.ids import make_video_id
from video_dataset.utils.logging import get_logger, stage_line, video_log_file
from video_dataset.utils.urls import InputItem
from video_dataset.validation.stage import validation_stage
from video_dataset.vision.stage import vision_stage

log = get_logger("pipeline")

StageFn = Callable[[VideoContext], StageOutput]

STAGE_FUNCTIONS: dict[Stage, StageFn] = {
    Stage.DOWNLOAD: download_stage,
    Stage.PREPROCESS: preprocess_stage,
    Stage.SCENE_DETECTION: scene_detection_stage,
    Stage.FRAME_EXTRACTION: frame_extraction_stage,
    Stage.AUDIO: audio_stage,
    Stage.TRANSCRIPTION: transcription_stage,
    Stage.OCR: ocr_stage,
    Stage.VISION_ANALYSIS: vision_stage,
    Stage.TEMPORAL_ANALYSIS: temporal_stage,
    Stage.QA_GENERATION: qa_stage,
    Stage.VALIDATION: validation_stage,
    Stage.EXPORT: export_stage,
}

# Failures here are recorded but do not block later stages (they consume the artifacts when present).
OPTIONAL_STAGES: set[Stage] = {Stage.AUDIO, Stage.TRANSCRIPTION, Stage.OCR}

# Which config sections a stage depends on (used to detect config drift on resume).
STAGE_CONFIG_SECTIONS: dict[Stage, tuple[str, ...]] = {
    Stage.DOWNLOAD: ("download",),
    Stage.PREPROCESS: ("preprocess",),
    Stage.SCENE_DETECTION: ("scene_detection",),
    Stage.FRAME_EXTRACTION: ("frame_sampling",),
    Stage.AUDIO: ("audio_events",),
    Stage.TRANSCRIPTION: ("transcription",),
    Stage.OCR: ("ocr",),
    Stage.VISION_ANALYSIS: ("vision",),
    Stage.TEMPORAL_ANALYSIS: ("temporal",),
    Stage.QA_GENERATION: ("qa",),
    Stage.VALIDATION: ("validation", "deduplication"),
    Stage.EXPORT: ("export",),
}


@dataclass
class VideoRunResult:
    video_id: str
    completed: bool
    stages_run: list[Stage] = field(default_factory=list)  # actually executed in this run
    stages_cached: list[Stage] = field(default_factory=list)  # already DONE, skipped
    stages_skipped: list[Stage] = field(default_factory=list)
    failed_stage: Stage | None = None
    error: str | None = None


class PipelineRunner:
    def __init__(self, config: PipelineConfig, db: StateDB | None = None):
        self.config = config
        self.paths = DataPaths(config.data_dir)
        self.paths.ensure_all()
        self.db = db or StateDB(config.db_path)
        self.models: dict[str, Any] = {}
        recovered = self.db.recover_interrupted()
        if recovered:
            log.warning("%d stage(s) were interrupted by a previous run and will be retried", recovered)
        self.stages = [Stage(s) for s in config.pipeline.stages]

    # ------------------------------------------------------------------ registration
    def register_inputs(self, items: list[InputItem]) -> list[tuple[str, InputItem]]:
        registered: list[tuple[str, InputItem]] = []
        seen: set[str] = set()
        for item in items:
            if item.kind == "youtube" and item.youtube_id:
                existing = self.db.find_by_youtube_id(item.youtube_id)
                video_id = existing["video_id"] if existing else make_video_id(youtube_id=item.youtube_id)
            elif item.kind == "local" and item.local_path:
                video_id = make_video_id(local_path=item.local_path)
            else:
                video_id = make_video_id(url=item.url)
            if video_id in seen:
                log.info("skipping duplicate input %s (%s)", item.raw, video_id)
                continue
            seen.add(video_id)
            self.db.upsert_video(video_id, item.url, youtube_id=item.youtube_id, source_type=item.kind if item.kind != "url" else "youtube")
            registered.append((video_id, item))
        return registered

    def context(self, video_id: str, item: InputItem | None = None) -> VideoContext:
        return VideoContext(video_id=video_id, config=self.config, paths=self.paths, db=self.db, input_item=item, models=self.models)

    # ------------------------------------------------------------------ single stage
    def run_stage(self, ctx: VideoContext, stage: Stage, force: bool = False) -> tuple[StageStatus, bool]:
        """Returns (status, executed). executed=False means the checkpoint was reused."""
        label = STAGE_LABELS[stage]
        rec = self.db.get_stage(ctx.video_id, stage)
        config_hash = ctx.config_hash(*STAGE_CONFIG_SECTIONS.get(stage, ()))
        if rec and rec.status == StageStatus.DONE and not force:
            if rec.config_hash and rec.config_hash != config_hash:
                log.info(stage_line(label, ctx.video_id, "cached (note: config changed since; use --force-from to redo)", ok=True))
            else:
                log.info(stage_line(label, ctx.video_id, "cached", ok=True))
            return StageStatus.DONE, False
        if rec and rec.status == StageStatus.SKIPPED and not force:
            log.info(stage_line(label, ctx.video_id, "skipped (disabled)", ok=None))
            return StageStatus.SKIPPED, False

        fn = STAGE_FUNCTIONS[stage]
        retries = max(0, int(self.config.pipeline.stage_retries))
        attempt = 0
        while True:
            attempt += 1
            self.db.start_stage(ctx.video_id, stage)
            t0 = time.time()
            try:
                out = fn(ctx)
                elapsed = time.time() - t0
                status = StageStatus.SKIPPED if out.skipped else StageStatus.DONE
                self.db.finish_stage(ctx.video_id, stage, out.artifact_path, out.metrics, config_hash, elapsed, status=status)
                self.db.log(ctx.video_id, stage, "INFO", out.message or status.value)
                log.info(stage_line(label, ctx.video_id, f"{out.message} ({elapsed:.1f}s)" if out.message else f"({elapsed:.1f}s)", ok=None if out.skipped else True))
                return status, True
            except Exception as exc:
                elapsed = time.time() - t0
                err = f"{type(exc).__name__}: {exc}"
                log.debug("stage %s traceback:\n%s", stage, traceback.format_exc())
                if attempt <= retries:
                    log.warning(stage_line(label, ctx.video_id, f"attempt {attempt} failed: {err[:200]} - retrying", ok=False))
                    self.db.log(ctx.video_id, stage, "WARNING", f"attempt {attempt} failed: {err}")
                    continue
                self.db.fail_stage(ctx.video_id, stage, err, elapsed)
                self.db.log(ctx.video_id, stage, "ERROR", err)
                log.error(stage_line(label, ctx.video_id, err[:300], ok=False))
                return StageStatus.FAILED, True

    # ------------------------------------------------------------------ single video
    def run_video(
        self,
        video_id: str,
        item: InputItem | None = None,
        stages: list[Stage] | None = None,
        force_from: Stage | None = None,
        until: Stage | None = None,
    ) -> VideoRunResult:
        stages = stages or self.stages
        if until is not None:
            stages = [s for s in stages if STAGE_ORDER.index(s) <= STAGE_ORDER.index(until)]
        if force_from is not None:
            self.db.reset_stages(video_id, force_from)
        result = VideoRunResult(video_id=video_id, completed=True)
        ctx = self.context(video_id, item)
        with video_log_file(video_id, self.paths.logs_dir):
            for stage in stages:
                force = force_from is not None and STAGE_ORDER.index(stage) >= STAGE_ORDER.index(force_from)
                status, executed = self.run_stage(ctx, stage, force=force)
                if status == StageStatus.DONE and not executed:
                    result.stages_cached.append(stage)
                elif status == StageStatus.DONE:
                    result.stages_run.append(stage)
                elif status == StageStatus.SKIPPED:
                    result.stages_skipped.append(stage)
                elif status == StageStatus.FAILED:
                    if stage in OPTIONAL_STAGES and self.config.pipeline.continue_on_error:
                        log.warning("optional stage %s failed for %s; continuing without it", stage, video_id)
                        result.stages_skipped.append(stage)
                        continue
                    result.completed = False
                    result.failed_stage = stage
                    rec = self.db.get_stage(video_id, stage)
                    result.error = rec.error if rec else "unknown error"
                    break
            if result.completed and stages and stages[-1] == Stage.EXPORT:
                self.db.set_video_status(video_id, "done")
            elif result.completed:
                self.db.set_video_status(video_id, "processing")
        return result

    # ------------------------------------------------------------------ batch
    def run_batch(
        self,
        registered: list[tuple[str, InputItem | None]],
        force_from: Stage | None = None,
        until: Stage | None = None,
        workers: int | None = None,
    ) -> list[VideoRunResult]:
        workers = max(1, int(workers or self.config.pipeline.workers or 1))
        stages = list(self.stages)
        if until is not None:
            stages = [s for s in stages if STAGE_ORDER.index(s) <= STAGE_ORDER.index(until)]
        cpu_stages = [s for s in stages if s not in MODEL_STAGES and STAGE_ORDER.index(s) < STAGE_ORDER.index(Stage.TRANSCRIPTION)]
        rest = [s for s in stages if s not in cpu_stages]
        results: dict[str, VideoRunResult] = {}

        if force_from is not None:
            for vid, _ in registered:
                self.db.reset_stages(vid, force_from)

        # Phase A: download .. audio, parallel across videos
        if workers > 1 and len(registered) > 1 and cpu_stages:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(self.run_video, vid, item, cpu_stages): vid for vid, item in registered}
                for fut in as_completed(futs):
                    r = fut.result()
                    results[r.video_id] = r
        else:
            for vid, item in registered:
                results[vid] = self.run_video(vid, item, cpu_stages)

        # Phase B: model stages, sequential (models load once)
        if rest and (self.config.pipeline.model_stages_sequential or workers == 1):
            for vid, item in registered:
                prev = results.get(vid)
                if prev is not None and not prev.completed:
                    continue
                r = self.run_video(vid, item, rest)
                if prev is not None:
                    r.stages_run = prev.stages_run + r.stages_run
                    r.stages_cached = prev.stages_cached + r.stages_cached
                    r.stages_skipped = prev.stages_skipped + r.stages_skipped
                results[vid] = r
        elif rest:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(self.run_video, vid, item, rest): vid for vid, item in registered if results.get(vid) is None or results[vid].completed}
                for fut in as_completed(futs):
                    r = fut.result()
                    results[r.video_id] = r

        ordered = [results[vid] for vid, _ in registered if vid in results]
        self._release_models()
        return ordered

    def _release_models(self) -> None:
        for key, model in list(self.models.items()):
            close = getattr(model, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            self.models.pop(key, None)

    # ------------------------------------------------------------------ maintenance
    def retry_failed(self, until: Stage | None = None) -> list[VideoRunResult]:
        failed = self.db.failed_stages()
        by_video: dict[str, Stage] = {}
        for rec in failed:
            cur = by_video.get(rec.video_id)
            if cur is None or STAGE_ORDER.index(rec.stage) < STAGE_ORDER.index(cur):
                by_video[rec.video_id] = rec.stage
        if not by_video:
            return []
        for vid, st in by_video.items():
            self.db.reset_stages(vid, st)
        return self.run_batch([(vid, None) for vid in by_video], until=until)

    def status_rows(self, video_id: str | None = None) -> list[dict[str, Any]]:
        videos = [self.db.get_video(video_id)] if video_id else self.db.list_videos()
        rows = []
        for v in videos:
            if not v:
                continue
            stages = self.db.get_stages(v["video_id"])
            rows.append({
                "video_id": v["video_id"],
                "title": v.get("title"),
                "duration": v.get("duration"),
                "status": v.get("status"),
                "stages": {s.value: (stages[s].status.value if s in stages else StageStatus.PENDING.value) for s in STAGE_ORDER},
                "errors": {s.value: stages[s].error for s in stages if stages[s].error},
                "metrics": {s.value: stages[s].metrics for s in stages if stages[s].metrics},
            })
        return rows

    def close(self) -> None:
        self._release_models()
        self.db.close()


def artifact_summary(paths: DataPaths, video_id: str) -> dict[str, Path]:
    return {
        "video": paths.video_file(video_id),
        "metadata": paths.metadata_file(video_id),
        "scenes": paths.scenes_file(video_id),
        "frames": paths.frames_file(video_id),
        "transcript": paths.transcript_file(video_id),
        "ocr": paths.ocr_file(video_id),
        "vision": paths.vision_file(video_id),
        "timeline": paths.timeline_file(video_id),
        "qa": paths.qa_file(video_id),
        "validated": paths.validated_file(video_id),
    }
