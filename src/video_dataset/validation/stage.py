"""VALIDATION stage: structural checks + grounding + verification + quality + dedup -> validated/<video_id>.json.

Nothing is silently deleted. Every record gets a status: accepted / review / rejected / duplicate.
"""

from __future__ import annotations

from pathlib import Path

from video_dataset.deduplication.qa import dedupe_questions
from video_dataset.deduplication.records import dedupe_timed_texts
from video_dataset.pipeline.context import StageOutput, VideoContext
from video_dataset.schemas.common import ConfidenceSource
from video_dataset.schemas.events import Event, EventSource, EventType, Timeline
from video_dataset.schemas.ocr import OCRResult
from video_dataset.schemas.qa import QAResult, QAType
from video_dataset.schemas.quality import QualityScore, ValidationInfo, ValidationStatus
from video_dataset.schemas.scene import FrameSamplingResult, SceneDetectionResult
from video_dataset.schemas.transcript import Transcript
from video_dataset.schemas.validated import ValidatedEvent, ValidatedRelation, ValidatedScene, ValidatedVideo
from video_dataset.schemas.vision import VerificationVerdict, VisionResult
from video_dataset.stages import Stage
from video_dataset.utils.device import resolve_device
from video_dataset.utils.io import read_json, write_json_atomic
from video_dataset.utils.parallel import parallel_map, stage_workers
from video_dataset.validation.alignment import AlignmentResult, CLIPCaptionAligner, alignment_text, find_loaded_clip
from video_dataset.validation.checks import (
    check_clips,
    check_files_exist,
    check_frame_sequence,
    check_ids_exist,
    check_non_empty,
    check_timespan,
    confidence_status,
)
from video_dataset.validation.consistency import check_scene_consistency
from video_dataset.validation.grounding import answer_grounding_score, numeric_grounding_score
from video_dataset.validation.quality import event_quality, qa_quality, relation_quality, scene_quality, scene_quality_flags
from video_dataset.validation.verifier import create_verifier
from video_dataset.vision.base import AnalysisContext


def _caption_alignment(ctx: VideoContext, analyses: dict, frames_by_scene: dict[str, list], log) -> dict[str, AlignmentResult]:  # type: ignore[no-untyped-def]
    """CLIP caption-vs-frames ranks for every captioned scene, or {} when disabled / unavailable."""
    vcfg = ctx.config.validation
    raw = vcfg.clip_alignment
    mode = ("on" if raw else "off") if isinstance(raw, bool) else str(raw or "auto").lower()
    if mode in ("off", "false", "0", "none"):
        return {}
    captions = {sid: text for sid, a in analyses.items() if a.summary and frames_by_scene.get(sid) and (text := alignment_text(a, str(vcfg.clip_alignment_text)))}
    if len(captions) < int(vcfg.clip_alignment_min_candidates):
        return {}
    enricher = find_loaded_clip(ctx.models)
    if enricher is None and mode in ("on", "true", "1", "always"):
        def _load():  # type: ignore[no-untyped-def]
            from video_dataset.vision.enrichers.clip import CLIPZeroShotEnricher

            device = resolve_device(ctx.config.vision.device or ctx.config.project.device)
            log.info("loading CLIP %s for caption alignment", ctx.config.vision.clip_model)
            return CLIPZeroShotEnricher(ctx.config.vision.clip_model, device, aesthetic=None)

        try:
            enricher = ctx.get_model(f"clip_aligner:{ctx.config.vision.clip_model}", _load)
        except Exception as exc:
            log.warning("caption alignment skipped: CLIP could not be loaded (%s)", str(exc)[:160])
            return {}
    if enricher is None:
        log.debug("caption alignment skipped: no CLIP model loaded (validation.clip_alignment=auto)")
        return {}
    try:
        return CLIPCaptionAligner(enricher).align({sid: [Path(f.frame_path) for f in frs] for sid, frs in frames_by_scene.items()}, captions, int(vcfg.clip_alignment_max_frames))
    except Exception as exc:
        log.warning("caption alignment failed: %s", str(exc)[:160])
        return {}


# Providers whose descriptions are assembled from measurements and classifier/detector outputs
# (no generative model wrote free text): accepted without a verification signal.
MEASURED_PROVIDERS = frozenset({"heuristic", "cv_models", None})


def _status(issues_hard: list[str], suggested: str) -> ValidationStatus:
    if issues_hard:
        return ValidationStatus.REJECTED
    return ValidationStatus(suggested)


def _load(path: Path, model):  # type: ignore[no-untyped-def]
    return model.model_validate(read_json(path)) if path.exists() else None


def validation_stage(ctx: VideoContext) -> StageOutput:
    log = ctx.logger(Stage.VALIDATION)
    cfg = ctx.config
    vcfg = cfg.validation
    duration = ctx.duration

    scenes = SceneDetectionResult.model_validate(read_json(ctx.paths.scenes_file(ctx.video_id))).scenes
    sampling = FrameSamplingResult.model_validate(read_json(ctx.paths.frames_file(ctx.video_id)))
    vision = _load(ctx.paths.vision_file(ctx.video_id), VisionResult)
    timeline = _load(ctx.paths.timeline_file(ctx.video_id), Timeline) or Timeline(video_id=ctx.video_id, duration=duration)
    qa = _load(ctx.paths.qa_file(ctx.video_id), QAResult) or QAResult(video_id=ctx.video_id, generator="none")
    transcript = _load(ctx.paths.transcript_file(ctx.video_id), Transcript)
    ocr = _load(ctx.paths.ocr_file(ctx.video_id), OCRResult)

    scene_ids = {s.scene_id for s in scenes}
    frames = {f.frame_id: f for f in sampling.frames}
    clips = {c.clip_id: c for c in sampling.clips}
    scans = {s.scene_id: s for s in sampling.scans}
    frames_by_scene: dict[str, list] = {}
    for f in sampling.frames:
        frames_by_scene.setdefault(f.scene_id, []).append(f)
    clips_by_scene: dict[str, list] = {}
    for c in sampling.clips:
        clips_by_scene.setdefault(c.scene_id, []).append(c)
    try:
        video_flags = list(ctx.media_info.quality_flags)
    except FileNotFoundError:
        video_flags = []

    analyzer = ctx.models.get(f"vision:{cfg.vision.provider}:{cfg.vision.model}")
    device = resolve_device(cfg.vision.device or cfg.project.device)
    verifier = ctx.get_model(f"verifier:{vcfg.verifier}", lambda: create_verifier(vcfg, cfg.vision, device, analyzer, cfg.temporal.min_camera_motion, cfg.temporal.min_camera_consistency))

    # ------------------------------------------------------------------ scenes
    v_scenes: list[ValidatedScene] = []
    analyses = {a.scene_id: a for a in (vision.scenes if vision else [])}
    scene_items = []
    for a in analyses.values():
        checks: dict[str, bool | None] = {}
        issues: list[str] = []
        hard: list[str] = []
        ok, msg = check_timespan(a.start_time, a.end_time, duration, allow_instant=False)
        checks["timestamps"] = ok
        if not ok:
            hard.append(msg or "bad timestamps")
        ok, msg = check_ids_exist([a.scene_id], scene_ids, "scene_id")
        checks["scene_exists"] = ok
        if not ok:
            hard.append(msg or "unknown scene")
        ok, msg = check_ids_exist(a.frame_ids, set(frames), "frame_id")
        checks["frames_exist"] = ok
        if not ok:
            (hard if vcfg.reject_on_missing_files else issues).append(msg or "unknown frames")
        ok, msg = check_files_exist([frames[f].frame_path for f in a.frame_ids if f in frames])
        checks["frame_files_exist"] = ok
        if not ok:
            (hard if vcfg.reject_on_missing_files else issues).append(msg or "missing frame files")
        ok, msg = check_non_empty(a.summary, "summary", min_words=4)
        checks["summary_present"] = ok
        if not ok:
            hard.append(msg or "no summary")
        if a.errors:
            issues.append("analyzer errors: " + "; ".join(a.errors)[:200])
        quality = scene_quality(a)
        generative = a.provider not in MEASURED_PROVIDERS
        if generative:
            suggested = confidence_status(a.confidence, vcfg.minimum_confidence, vcfg.review_confidence)
            if a.confidence is None:
                issues.append("no verification signal for generated description")
            elif a.confidence_source == ConfidenceSource.MODEL_SELF_REPORT:
                issues.append("confidence is model self-report, not verified")
                suggested = "review" if suggested == "accepted" and vcfg.verifier != "none" and a.verification is None else suggested
            if a.verification is not None and a.verification.verdict == VerificationVerdict.UNSUPPORTED:
                hard.append("verifier found the description unsupported")
        else:
            # measured annotation (pixel statistics, optical flow, detector + CLIP labels): nothing to
            # hallucinate, so no verification signal is required; a shot without detections simply
            # carries no object confidence. Only a *low* detector score sends it to review.
            suggested = "accepted"
            checks["measurement_derived"] = True
            if a.confidence is not None and a.confidence < vcfg.review_confidence:
                issues.append(f"mean detector confidence {a.confidence:.2f} below review_confidence {vcfg.review_confidence:.2f}")
                suggested = "review"
        if quality.description_quality is not None and quality.description_quality < 0.15:
            issues.append("low description specificity")
            if suggested == "accepted":
                suggested = "review"
        # --- frame sequence: ordered, inside the shot, no large gaps ---------------------------
        if vcfg.frame_sequence_checks:
            ok, msg = check_frame_sequence(
                frames_by_scene.get(a.scene_id, []), a.start_time, a.end_time,
                min_gap_seconds=float(cfg.frame_sampling.max_frame_gap_seconds or 5.0), max_frames=int(cfg.frame_sampling.max_frames_per_scene),
            )
            checks["frame_sequence"] = ok
            if not ok:
                issues.append(msg or "frame sequence problem")
                if suggested == "accepted":
                    suggested = "review"
        # --- clip files: present, readable, matching the shot span -----------------------------
        if vcfg.clip_checks and clips_by_scene.get(a.scene_id):
            missing, corrupt, inexact = check_clips(clips_by_scene[a.scene_id], float(cfg.frame_sampling.clip_tolerance_seconds or 0.25))
            checks["clip_files_exist"] = not missing and not corrupt
            checks["clip_duration_exact"] = not inexact
            if missing:
                (hard if vcfg.reject_on_missing_files else issues).append("missing clip files: " + ", ".join(missing[:3]))
            if corrupt:
                (hard if vcfg.reject_on_missing_files else issues).append("unreadable clip files: " + ", ".join(corrupt[:3]))
            if inexact:
                issues.append("clip length differs from the shot: " + "; ".join(inexact[:3]))
                if suggested == "accepted":
                    suggested = "review"
        # --- measured visual quality -----------------------------------------------------------
        flags = scene_quality_flags(a.measurements, video_flags, cfg.quality)
        a.quality_flags = flags
        checks["visual_quality"] = not flags
        if flags:
            (hard if cfg.quality.reject_flagged else issues).append("visual quality: " + ", ".join(flags))
            if suggested == "accepted":
                suggested = "review"
        # --- cross-field consistency (annotation vs measurement, caption vs camera field) --------
        if vcfg.consistency_checks:
            c_hard, c_soft = check_scene_consistency(a)
            checks["consistent"] = not c_hard
            (hard if vcfg.reject_contradictions else issues).extend("contradiction: " + x for x in c_hard)
            issues.extend("inconsistency: " + x for x in c_soft)
            if (c_hard or c_soft) and suggested == "accepted":
                suggested = "review"
        v_scenes.append(ValidatedScene(analysis=a, validation=ValidationInfo(status=_status(hard, suggested), issues=hard + issues, checks=checks, verifier_score=a.verification.score if a.verification else None), quality=quality))
        if a.summary:
            scene_items.append((a.scene_id, a.summary, a.start_time, a.end_time))
    # --- caption <-> frames alignment (CLIP retrieval rank among the video's own captions) --------
    alignment = _caption_alignment(ctx, analyses, frames_by_scene, log)
    n_misaligned = 0
    for vs in v_scenes:
        res = alignment.get(vs.analysis.scene_id)
        if res is None:
            continue
        ok = not res.misaligned(float(vcfg.clip_alignment_min_rank_fraction), float(vcfg.clip_alignment_min_margin))
        vs.validation.checks["caption_alignment"] = ok
        vs.analysis.enrichments["caption_alignment"] = {"clip_score": res.score, "rank": res.rank, "candidates": res.candidates, "best_other": res.best_other, "best_score": res.best_score, "margin": res.margin}
        if not ok:
            n_misaligned += 1
            vs.validation.issues.append(f"caption ranks {res.rank} of {res.candidates} by CLIP similarity to its own frames (margin {res.margin:.3f}" + (f", best match: {res.best_other})" if res.best_other else ")"))
            if vs.validation.status == ValidationStatus.ACCEPTED:
                vs.validation.status = ValidationStatus.REVIEW
    if alignment:
        log.info("caption alignment: %d of %d scenes rank below the top %.0f%% of their video's captions", n_misaligned, len(alignment), float(vcfg.clip_alignment_min_rank_fraction) * 100)
    # duplicate descriptions: identical text for *different, non-overlapping* scenes is flagged (kept, not exported)
    scene_dups = dedupe_timed_texts(scene_items, cfg.deduplication.near_duplicate_threshold, 0.0, require_overlap=False) if cfg.deduplication.enabled else {}
    for vs in v_scenes:
        if vs.analysis.scene_id in scene_dups and vs.analysis.provider not in MEASURED_PROVIDERS:
            vs.validation.status = ValidationStatus.DUPLICATE
            vs.validation.duplicate_of = scene_dups[vs.analysis.scene_id]
            vs.validation.issues.append("description identical to another scene")

    # ------------------------------------------------------------------ events
    # pass 1: cheap structural checks; pass 2: verifier calls (the only expensive part) on the pool,
    # in timeline order and within the per-video budget; pass 3: status + quality.
    v_events: list[ValidatedEvent] = []
    event_quality_by_id: dict[str, QualityScore] = {}
    n_verified = 0
    budget = int(vcfg.max_vlm_verifications_per_video)
    prepared: list[tuple[Event, dict, list[str], list[str]]] = []
    to_verify: list[tuple[Event, list, AnalysisContext]] = []
    for e in timeline.events:
        checks = {}
        issues = []
        hard = []
        ok, msg = check_timespan(e.start_time, e.end_time, duration)
        checks["timestamps"] = ok
        if not ok:
            hard.append(msg or "bad timestamps")
        ok, msg = check_ids_exist(e.scene_ids, scene_ids, "scene_id")
        checks["scenes_exist"] = ok
        if not ok:
            hard.append(msg or "unknown scene")
        ok, msg = check_ids_exist(e.frame_ids, set(frames), "frame_id")
        checks["frames_exist"] = ok
        if not ok:
            (hard if vcfg.reject_on_missing_files else issues).append(msg or "unknown frames")
        ok, msg = check_ids_exist(e.clip_ids, set(clips), "clip_id")
        checks["clips_exist"] = ok
        if not ok:
            issues.append(msg or "unknown clip")
        ok, msg = check_non_empty(e.event, "event description", min_words=2)
        checks["description_present"] = ok
        if not ok:
            hard.append(msg or "empty event")
        # verification of model-generated claims
        if e.source == EventSource.VISION and e.event_type in (EventType.ACTION, EventType.APPEARANCE) and budget > 0 and not hard:
            if e.verification is None or e.verification.verdict == VerificationVerdict.UNKNOWN:
                sc = e.scene_ids[0] if e.scene_ids else None
                fr = [frames[f] for f in e.frame_ids if f in frames] or (frames_by_scene.get(sc, []) if sc else [])
                to_verify.append((e, fr, AnalysisContext(video_id=ctx.video_id, scan=scans.get(sc) if sc else None)))
                budget -= 1
        prepared.append((e, checks, issues, hard))

    v_workers = stage_workers(ctx, vcfg, len(to_verify)) if getattr(verifier, "parallel_safe", False) else 1
    if to_verify:
        log.info("verifying %d event claims with %s (%d worker%s)", len(to_verify), verifier.name, v_workers, "s" if v_workers != 1 else "")

    def _verify(item: tuple[Event, list, AnalysisContext]):  # type: ignore[no-untyped-def]
        e, fr, context = item
        try:
            return verifier.verify(e.event, fr, context)
        except Exception as exc:
            log.warning("verifier failed on %s: %s", e.event_id, str(exc)[:160])
            return None

    for (e, _fr, _ctx), res in zip(to_verify, parallel_map(_verify, to_verify, v_workers)):
        if res is not None and res.verdict != VerificationVerdict.UNKNOWN:
            e.verification = res
            e.confidence = res.score
            e.confidence_source = ConfidenceSource.VERIFIER
            n_verified += 1

    for e, checks, issues, hard in prepared:
        if e.source == EventSource.VISION and e.event_type in (EventType.ACTION, EventType.APPEARANCE) and not hard:
            if e.verification is not None and e.verification.verdict == VerificationVerdict.UNSUPPORTED:
                hard.append("verifier found the claim unsupported by the frames")
        suggested = confidence_status(e.confidence, vcfg.minimum_confidence, vcfg.review_confidence)
        if e.confidence is None:
            issues.append("confidence unavailable")
        elif e.confidence_source == ConfidenceSource.MODEL_SELF_REPORT:
            issues.append("confidence is model self-report")
            if suggested == "accepted":
                suggested = "review"
        if e.confidence is not None and e.confidence < vcfg.minimum_confidence and e.boundary_precision in ("measured", "asr"):
            issues.append("low confidence temporal boundary")
        q = event_quality(e)
        event_quality_by_id[e.event_id] = q
        v_events.append(ValidatedEvent(event=e, validation=ValidationInfo(status=_status(hard, suggested), issues=hard + issues, checks=checks, verifier_score=e.verification.score if e.verification else None), quality=q))
    if cfg.deduplication.enabled:
        ev_dups = dedupe_timed_texts([(e.event_id, e.event, e.start_time, e.end_time) for e in timeline.events], cfg.deduplication.near_duplicate_threshold, cfg.deduplication.evidence_iou_threshold)
        for ve in v_events:
            if ve.event.event_id in ev_dups:
                ve.validation.status = ValidationStatus.DUPLICATE
                ve.validation.duplicate_of = ev_dups[ve.event.event_id]
    event_status = {ve.event.event_id: ve.validation.status for ve in v_events}
    event_by_id: dict[str, Event] = {e.event_id: e for e in timeline.events}

    # ------------------------------------------------------------------ relations
    v_relations: list[ValidatedRelation] = []
    for r in timeline.relations:
        hard = []
        issues = []
        checks = {}
        ok, msg = check_ids_exist([r.event_a, r.event_b], set(event_by_id), "event_id")
        checks["events_exist"] = ok
        if not ok:
            hard.append(msg or "unknown event")
        if ok and (event_status.get(r.event_a) == ValidationStatus.REJECTED or event_status.get(r.event_b) == ValidationStatus.REJECTED):
            hard.append("references a rejected event")
        suggested = confidence_status(r.confidence, vcfg.minimum_confidence, vcfg.review_confidence)
        if r.confidence_source == ConfidenceSource.MODEL_SELF_REPORT:
            issues.append("inferred by a model without verification")
            suggested = "review" if suggested == "accepted" else suggested
        qualities = [event_quality_by_id[x] for x in (r.event_a, r.event_b) if x in event_quality_by_id]
        v_relations.append(ValidatedRelation(relation=r, validation=ValidationInfo(status=_status(hard, suggested), issues=hard + issues, checks=checks), quality=relation_quality(r.confidence, qualities)))

    # ------------------------------------------------------------------ QA
    for q_rec in qa.questions:
        hard = []
        issues = []
        checks = {}
        ev = q_rec.evidence
        ok, msg = check_timespan(ev.start_time, ev.end_time, duration)
        checks["evidence_timestamps"] = ok
        if not ok:
            hard.append(msg or "bad evidence timestamps")
        if any(t < -0.05 or t > duration + 0.05 for t in ev.timestamps):
            checks["timestamps_in_range"] = False
            hard.append("evidence timestamp outside the video")
        else:
            checks["timestamps_in_range"] = True
        ok, msg = check_ids_exist(ev.scene_ids, scene_ids, "scene_id")
        checks["scenes_exist"] = ok
        if not ok:
            hard.append(msg or "unknown scene")
        ok, msg = check_ids_exist(ev.event_ids, set(event_by_id), "event_id")
        checks["events_exist"] = ok
        if not ok:
            hard.append(msg or "unknown event")
        ok, msg = check_ids_exist(ev.frame_ids, set(frames), "frame_id")
        checks["frames_exist"] = ok
        if not ok:
            (hard if vcfg.reject_on_missing_files else issues).append(msg or "unknown frames")
        ok, msg = check_files_exist([frames[f].frame_path for f in ev.frame_ids if f in frames])
        checks["frame_files_exist"] = ok
        if not ok:
            (hard if vcfg.reject_on_missing_files else issues).append(msg or "missing frame files")
        checks["has_evidence"] = bool(ev.event_ids or ev.scene_ids)
        if not checks["has_evidence"]:
            hard.append("question has no evidence")
        rejected_ev = [x for x in ev.event_ids if event_status.get(x) == ValidationStatus.REJECTED]
        if rejected_ev:
            hard.append("evidence event rejected: " + ", ".join(rejected_ev[:3]))
        # answer grounding against evidence texts
        texts = [event_by_id[x].event for x in ev.event_ids if x in event_by_id]
        texts += [str(event_by_id[x].attributes.get("text", "")) for x in ev.event_ids if x in event_by_id]
        texts += [analyses[s].summary or "" for s in ev.scene_ids if s in analyses]
        if transcript:
            texts.append(transcript.text_between(ev.start_time, ev.end_time))
        if ocr:
            texts += [t.text for t in ocr.tracks if t.first_seen <= ev.end_time and t.last_seen >= ev.start_time]
        grounding = answer_grounding_score(q_rec.answer, texts, (ev.start_time, ev.end_time))
        if q_rec.type in (QAType.DURATION, QAType.EVENT_LOCALIZATION):
            # these answers are numbers derived from the evidence window; judge the numbers, and require
            # that the question itself names the evidence
            q_lex = answer_grounding_score(q_rec.question, texts, None)
            grounding = round(0.7 * numeric_grounding_score(q_rec.answer, (ev.start_time, ev.end_time), duration) + 0.3 * q_lex, 3)
        checks["answer_grounded"] = grounding >= vcfg.grounding_min_overlap
        if not checks["answer_grounded"]:
            hard.append(f"answer not supported by evidence (grounding {grounding:.2f})")
        suggested = confidence_status(q_rec.confidence, vcfg.minimum_confidence, vcfg.review_confidence)
        if q_rec.confidence is None:
            issues.append("evidence confidence unavailable")
        elif q_rec.confidence < vcfg.minimum_confidence:
            issues.append("low confidence evidence")
        qualities = [event_quality_by_id[x] for x in ev.event_ids if x in event_quality_by_id]
        q_rec.quality = qa_quality(q_rec, grounding, qualities)
        q_rec.validation = ValidationInfo(status=_status(hard, suggested), issues=hard + issues, checks=checks)
    if cfg.deduplication.enabled:
        q_dups = dedupe_questions(qa.questions, cfg.deduplication)
        for q_rec in qa.questions:
            if q_rec.question_id in q_dups and q_rec.validation is not None:
                q_rec.validation.status = ValidationStatus.DUPLICATE
                q_rec.validation.duplicate_of = q_dups[q_rec.question_id]

    # ------------------------------------------------------------------ summary + write
    def counts(statuses: list[ValidationStatus]) -> dict[str, int]:
        out = {s.value: 0 for s in ValidationStatus}
        for s in statuses:
            out[str(s)] += 1
        return out

    summary = {
        "scenes": counts([v.validation.status for v in v_scenes]),
        "events": counts([v.validation.status for v in v_events]),
        "relations": counts([v.validation.status for v in v_relations]),
        "qa": counts([q.validation.status for q in qa.questions if q.validation]),
        "events_verified_now": n_verified,
        "verifier": verifier.name,
        "verify_workers": v_workers,
        "scenes_quality_flagged": sum(1 for v in v_scenes if v.analysis.quality_flags),
        "scenes_contradictory": sum(1 for v in v_scenes if v.validation.checks.get("consistent") is False),
        "scenes_caption_misaligned": n_misaligned,
        "video_quality_flags": video_flags,
    }
    validated = ValidatedVideo(video_id=ctx.video_id, duration=duration, scenes=v_scenes, events=v_events, relations=v_relations, qa=qa.questions, summary=summary)
    out = ctx.paths.validated_file(ctx.video_id)
    write_json_atomic(out, validated)
    qs = summary["qa"]
    log.info("QA accepted=%d review=%d rejected=%d duplicate=%d | events %s | scenes %s", qs["accepted"], qs["review"], qs["rejected"], qs["duplicate"], summary["events"], summary["scenes"])
    return StageOutput(artifact_path=str(out), metrics=summary, message=f"{qs['accepted']} accepted / {qs['rejected']} rejected / {qs['review']} review ({qs['duplicate']} duplicates)")
