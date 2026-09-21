"""QA_GENERATION stage."""

from __future__ import annotations

from video_dataset.pipeline.context import StageOutput, VideoContext
from video_dataset.questions.generator import TemporalQAGenerator
from video_dataset.schemas.events import Timeline
from video_dataset.schemas.qa import QAResult
from video_dataset.schemas.scene import SceneDetectionResult
from video_dataset.schemas.transcript import Transcript
from video_dataset.schemas.vision import VisionResult
from video_dataset.stages import Stage
from video_dataset.utils.io import read_json, write_json_atomic


def qa_stage(ctx: VideoContext) -> StageOutput:
    log = ctx.logger(Stage.QA_GENERATION)
    cfg = ctx.config
    timeline_path = ctx.paths.timeline_file(ctx.video_id)
    if not timeline_path.exists():
        raise FileNotFoundError("timeline.json missing; run TEMPORAL_ANALYSIS first")
    timeline = Timeline.model_validate(read_json(timeline_path))
    scenes = SceneDetectionResult.model_validate(read_json(ctx.paths.scenes_file(ctx.video_id))).scenes
    vision_path = ctx.paths.vision_file(ctx.video_id)
    analyses = {a.scene_id: a for a in VisionResult.model_validate(read_json(vision_path)).scenes} if vision_path.exists() else {}
    tpath = ctx.paths.transcript_file(ctx.video_id)
    transcript = Transcript.model_validate(read_json(tpath)) if tpath.exists() else None

    generator = TemporalQAGenerator(cfg.qa)
    records = generator.generate(ctx.video_id, ctx.duration, timeline, scenes, analyses, transcript)

    paraphrased = 0
    if cfg.qa.paraphrase and cfg.llm.provider not in ("none", None):
        from video_dataset.llm.base import create_llm_client
        from video_dataset.questions.paraphrase import Paraphraser

        client = ctx.get_model(f"llm:{cfg.llm.provider}:{cfg.llm.model}", lambda: create_llm_client(cfg.llm.provider, cfg.llm.model, cfg.llm.base_url, cfg.llm.api_key_env, float(cfg.llm.timeout_seconds), requests_per_minute=float(cfg.llm.requests_per_minute)))
        paraphrased = Paraphraser(client).paraphrase(records)

    result = QAResult(video_id=ctx.video_id, generator=generator.name, questions=records)
    out = ctx.paths.qa_file(ctx.video_id)
    write_json_atomic(out, result)
    by_type: dict[str, int] = {}
    for r in records:
        by_type[str(r.type)] = by_type.get(str(r.type), 0) + 1
    log.info("%d questions %s (%d paraphrased)", len(records), by_type, paraphrased)
    return StageOutput(artifact_path=str(out), metrics={"questions": len(records), "by_type": by_type, "paraphrased": paraphrased}, message=f"{len(records)} questions generated")
