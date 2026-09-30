"""QA_GENERATION stage."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from video_dataset.pipeline.context import StageOutput, VideoContext
from video_dataset.questions.generator import TemporalQAGenerator
from video_dataset.questions.spacy_generator import SpacyQAGenerator
from video_dataset.schemas.events import Timeline
from video_dataset.schemas.qa import QARecord, QAResult
from video_dataset.schemas.scene import SceneDetectionResult
from video_dataset.schemas.transcript import Transcript
from video_dataset.schemas.vision import VisionResult
from video_dataset.stages import Stage
from video_dataset.utils.ids import qa_id as make_qa_id
from video_dataset.utils.io import read_json, write_json_atomic
from video_dataset.utils.text import normalize_text

GENERATORS = ("template", "spacy", "both")


def merge_question_sets(video_id: str, sets: list[list[QARecord]], max_questions: int) -> list[QARecord]:
    """Concatenate several generators' output: drop duplicate (type, question) pairs, cap the total
    and renumber question_ids so they stay unique and stable."""
    seen: set[str] = set()
    merged: list[QARecord] = []
    for records in sets:
        for r in records:
            key = f"{r.type}|{normalize_text(r.question)}"
            if key in seen:
                continue
            seen.add(key)
            merged.append(r)
    merged = merged[: max(0, int(max_questions))] if max_questions else merged
    return [r.model_copy(update={"question_id": make_qa_id(video_id, n)}) for n, r in enumerate(merged, start=1)]


def select_generators(qa_cfg: Any, get_model: Callable[[str, Callable[[], Any]], Any], log: logging.Logger | logging.LoggerAdapter[Any]) -> list[TemporalQAGenerator | SpacyQAGenerator]:
    """Generators for ``qa.generator`` (template | spacy | both). The spaCy pipeline is loaded once per
    process through ``get_model``; when it cannot be loaded the stage warns and uses templates so a
    batch run never dies on a missing language model."""
    mode = (getattr(qa_cfg, "generator", "template") or "template").lower()
    if mode not in GENERATORS:
        raise ValueError(f"qa.generator must be one of {GENERATORS}, got '{mode}'")
    generators: list[TemporalQAGenerator | SpacyQAGenerator] = []
    if mode in ("template", "both"):
        generators.append(TemporalQAGenerator(qa_cfg))
    if mode in ("spacy", "both"):

        def _load_spacy() -> SpacyQAGenerator:
            gen = SpacyQAGenerator(qa_cfg)
            gen.load()
            return gen

        try:
            generators.append(get_model(f"spacy:{qa_cfg.spacy_model}", _load_spacy))
        except Exception as exc:
            log.warning("spaCy generator unavailable (%s); falling back to template questions", str(exc)[:200])
            if not generators:
                generators.append(TemporalQAGenerator(qa_cfg))
    return generators


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

    generators = select_generators(cfg.qa, ctx.get_model, log)
    log.info("question generators: %s", ", ".join(g.name for g in generators))

    sets = [g.generate(ctx.video_id, ctx.duration, timeline, scenes, analyses, transcript) for g in generators]
    records = sets[0] if len(sets) == 1 else merge_question_sets(ctx.video_id, sets, int(cfg.qa.max_questions))
    generator_name = "+".join(g.name for g in generators)

    paraphrased = 0
    if cfg.qa.paraphrase and cfg.llm.provider not in ("none", None):
        from video_dataset.llm.base import create_llm_client
        from video_dataset.questions.paraphrase import Paraphraser

        client = ctx.get_model(f"llm:{cfg.llm.provider}:{cfg.llm.model}", lambda: create_llm_client(cfg.llm.provider, cfg.llm.model, cfg.llm.base_url, cfg.llm.api_key_env, float(cfg.llm.timeout_seconds), requests_per_minute=float(cfg.llm.requests_per_minute)))
        paraphrased = Paraphraser(client).paraphrase(records)

    result = QAResult(video_id=ctx.video_id, generator=generator_name, questions=records)
    out = ctx.paths.qa_file(ctx.video_id)
    write_json_atomic(out, result)
    by_type: dict[str, int] = {}
    for r in records:
        by_type[str(r.type)] = by_type.get(str(r.type), 0) + 1
    log.info("%d questions %s (%d paraphrased)", len(records), by_type, paraphrased)
    return StageOutput(artifact_path=str(out), metrics={"questions": len(records), "by_type": by_type, "paraphrased": paraphrased, "generator": generator_name}, message=f"{len(records)} questions generated ({generator_name})")
