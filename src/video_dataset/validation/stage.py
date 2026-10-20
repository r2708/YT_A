"""VALIDATION stage: structural checks + grounding + verification + quality + consistency + policy
-> validated/<video_id>.json.

Nothing is silently deleted. Every record gets a status: accepted / review / rejected / duplicate,
assigned by the versioned policy in ``validation.policy`` from named check results. A check that
could not run is recorded as ``not_run`` and is never mistaken for a pass.
"""

from __future__ import annotations

from pathlib import Path

from video_dataset.deduplication.qa import dedupe_questions
from video_dataset.deduplication.records import dedupe_timed_texts
from video_dataset.pipeline.context import StageOutput, VideoContext
from video_dataset.schemas.common import ConfidenceSource
from video_dataset.schemas.events import Event, EventSource, EventType, RelationType, Timeline
from video_dataset.schemas.ocr import OCRResult
from video_dataset.schemas.qa import QAResult, QAType
from video_dataset.schemas.quality import CheckState, QualityScore, ValidationInfo, ValidationStatus
from video_dataset.schemas.scene import FrameSamplingResult, SceneDetectionResult
from video_dataset.schemas.transcript import Transcript
from video_dataset.schemas.validated import ValidatedEvent, ValidatedRelation, ValidatedScene, ValidatedVideo
from video_dataset.schemas.vision import (
    MOVEMENT_TAXONOMY,
    CameraMovement,
    CameraMovementFamily,
    VerificationVerdict,
    VisionResult,
)
from video_dataset.stages import Stage
from video_dataset.utils.device import resolve_device
from video_dataset.utils.io import read_json, write_json_atomic
from video_dataset.utils.parallel import parallel_map, stage_workers
from video_dataset.validation.alignment import (
    AlignmentResult,
    CLIPCaptionAligner,
    alignment_text,
    find_loaded_clip,
)
from video_dataset.validation.checks import (
    check_clips,
    check_files_exist,
    check_frame_sequence,
    check_ids_exist,
    check_non_empty,
    check_timespan,
    confidence_status,
)
from video_dataset.validation.claims import claim_summary, scene_claims
from video_dataset.validation.consistency import check_scene_consistency
from video_dataset.validation.grounding import answer_grounding_score, numeric_grounding_score
from video_dataset.validation.policy import POLICY_VERSION, decide
from video_dataset.validation.qa_checks import recompute_answer
from video_dataset.validation.quality import (
    event_quality,
    qa_quality,
    relation_quality,
    scene_quality,
    scene_quality_flags,
)
from video_dataset.validation.relations import before_cycles, relation_holds
from video_dataset.validation.text_checks import TextChecker, answer_type_for
from video_dataset.validation.verifier import create_verifier
from video_dataset.vision.base import AnalysisContext

NOT_RUN = CheckState.NOT_RUN.value
NOT_APPLICABLE = CheckState.NOT_APPLICABLE.value
ERROR = CheckState.ERROR.value


def _caption_alignment(ctx: VideoContext, analyses: dict, frames_by_scene: dict[str, list], log) -> tuple[dict[str, AlignmentResult], str]:  # type: ignore[no-untyped-def]
    """(CLIP caption-vs-frames ranks for every captioned scene, state): the state says why the
    check produced nothing when it did (disabled / too few captions / no model)."""
    vcfg = ctx.config.validation
    raw = vcfg.clip_alignment
    mode = ("on" if raw else "off") if isinstance(raw, bool) else str(raw or "auto").lower()
    if mode in ("off", "false", "0", "none"):
        return {}, "disabled"
    captions = {sid: text for sid, a in analyses.items() if a.summary and frames_by_scene.get(sid) and (text := alignment_text(a, str(vcfg.clip_alignment_text)))}
    if len(captions) < int(vcfg.clip_alignment_min_candidates):
        return {}, "too_few_candidates"
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
            return {}, "model_unavailable"
    if enricher is None:
        log.debug("caption alignment skipped: no CLIP model loaded (validation.clip_alignment=auto)")
        return {}, "model_unavailable"
    try:
        return CLIPCaptionAligner(enricher).align({sid: [Path(f.frame_path) for f in frs] for sid, frs in frames_by_scene.items()}, captions, int(vcfg.clip_alignment_max_frames)), "ran"
    except Exception as exc:
        log.warning("caption alignment failed: %s", str(exc)[:160])
        return {}, "error"


# Providers whose descriptions are assembled from measurements and classifier/detector outputs
# (no generative model wrote free text): accepted without a verification signal.
MEASURED_PROVIDERS = frozenset({"heuristic", "cv_models", None})


def _load(path: Path, model):  # type: ignore[no-untyped-def]
    return model.model_validate(read_json(path)) if path.exists() else None


def _family(label) -> CameraMovementFamily:  # type: ignore[no-untyped-def]
    try:
        return MOVEMENT_TAXONOMY[CameraMovement(str(label))][0]
    except (ValueError, KeyError):
        return CameraMovementFamily.UNKNOWN


def camera_motion_agreement(a) -> tuple[str, str | None]:  # type: ignore[no-untyped-def]
    """Explicit check: the camera label vs the measured optical-flow label. ``not_applicable`` when
    either side is unknown (nothing to compare), ``failed`` when the families disagree with a
    consistent measurement (zoom vs dolly and tracking vs pan are compatible refinements)."""
    m = a.measurements
    label = a.camera.movement
    if m is None or not m.camera_motion_label or label == CameraMovement.UNKNOWN:
        return NOT_APPLICABLE, None
    fam, flow_fam = _family(label), _family(m.camera_motion_label)
    if fam == flow_fam:
        return CheckState.PASSED.value, None
    compatible = {fam, flow_fam} <= {CameraMovementFamily.ZOOM, CameraMovementFamily.DOLLY} or (fam == CameraMovementFamily.TRACKING and flow_fam in (CameraMovementFamily.PAN, CameraMovementFamily.TILT, CameraMovementFamily.HANDHELD, CameraMovementFamily.COMPLEX))
    if compatible or flow_fam in (CameraMovementFamily.UNKNOWN, CameraMovementFamily.COMPLEX, CameraMovementFamily.HANDHELD) or (m.camera_motion_consistency or 0.0) < 0.6:
        return CheckState.PASSED.value, None
    return CheckState.FAILED.value, f"camera label {label} vs measured {m.camera_motion_label} ({m.camera_motion_consistency:.2f} consistent)"


def _text_checker(ctx: VideoContext) -> TextChecker:
    vcfg = ctx.config.validation
    model = vcfg.nlp_model
    if model == "auto":
        model = ctx.config.qa.spacy_model if ctx.config.qa.generator in ("spacy", "both") else None
    return ctx.get_model(f"text_checker:{model}", lambda: TextChecker.load(model))


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
    text = _text_checker(ctx) if vcfg.text_checks else None
    clip_tol = float(cfg.frame_sampling.clip_tolerance_seconds or 0.25)

    # ------------------------------------------------------------------ scenes
    v_scenes: list[ValidatedScene] = []
    analyses = {a.scene_id: a for a in (vision.scenes if vision else [])}
    scene_items = []
    n_claims = {"supported": 0, "contradicted": 0, "not_checked": 0, "unknown": 0}
    n_suspected_cuts = 0
    for a in analyses.values():
        checks: dict[str, bool | str | None] = {}
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
        semantic = "measured"
        if generative:
            suggested = confidence_status(a.confidence, vcfg.minimum_confidence, vcfg.review_confidence)
            if a.verification is not None and a.verification.verdict != VerificationVerdict.UNKNOWN:
                semantic = f"verifier:{a.verification.verifier or vcfg.verifier}"
                if a.verification.verdict == VerificationVerdict.UNSUPPORTED:
                    hard.append("verifier found the description unsupported")
            else:
                semantic = NOT_RUN  # generated text nobody checked against the frames
                issues.append("no verifier judged the generated description")
            if a.confidence is None:
                issues.append("no verification signal for generated description")
            elif a.confidence_source == ConfidenceSource.MODEL_SELF_REPORT:
                issues.append("confidence is model self-report, not verified")
                suggested = "review" if suggested == "accepted" else suggested
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
        else:
            checks["frame_sequence"] = NOT_RUN
        # --- clip files: present, readable, matching the shot span and its boundaries ---------
        scene_clips = clips_by_scene.get(a.scene_id, [])
        if vcfg.clip_checks and scene_clips:
            missing, corrupt, inexact = check_clips(scene_clips, clip_tol)
            checks["clip_files_exist"] = not missing and not corrupt
            checks["clip_duration_exact"] = not inexact
            if missing:
                (hard if vcfg.reject_on_missing_files else issues).append("missing clip files: " + ", ".join(missing[:3]))
            if corrupt:
                (hard if vcfg.reject_on_missing_files else issues).append("unreadable clip files: " + ", ".join(corrupt[:3]))
            if inexact:
                issues.append("clip length differs from the shot: " + "; ".join(inexact[:3]))
            # clips the copy-mode check skipped on purpose (re-encoded with an accurate seek, duration
            # verified above) are exact by construction: not applicable rather than not run
            states = {("not_applicable" if c.boundary_check == "not_run" and c.codec == "libx264" else c.boundary_check) for c in scene_clips}
            if "failed" in states:
                checks["clip_boundaries"] = False
                bad = [f"{c.clip_id} ({c.boundary_error_seconds:.2f}s)" for c in scene_clips if c.boundary_check == "failed"]
                issues.append("clip starts/ends outside the shot: " + ", ".join(bad[:3]))
            elif states <= {"passed", "not_applicable"}:
                checks["clip_boundaries"] = True if "passed" in states else NOT_APPLICABLE
            elif "error" in states:
                checks["clip_boundaries"] = ERROR
                issues.append("clip boundary could not be matched against the source")
            else:
                checks["clip_boundaries"] = NOT_RUN
        elif scene_clips:
            checks["clip_files_exist"] = checks["clip_duration_exact"] = checks["clip_boundaries"] = NOT_RUN
        else:
            checks["clip_files_exist"] = checks["clip_duration_exact"] = checks["clip_boundaries"] = NOT_APPLICABLE
        # --- measured visual quality -----------------------------------------------------------
        flags = scene_quality_flags(a.measurements, video_flags, cfg.quality, scans.get(a.scene_id), a.end_time - a.start_time)
        a.quality_flags = flags
        checks["visual_quality"] = not flags
        if flags:
            (hard if cfg.quality.reject_flagged else issues).append("visual quality: " + ", ".join(flags))
        # --- cross-field consistency (annotation vs measurement, caption vs camera field) --------
        c_hard: list[str] = []
        c_soft: list[str] = []
        if vcfg.consistency_checks:
            c_hard, c_soft = check_scene_consistency(a)
            checks["consistent"] = not c_hard
            (hard if vcfg.reject_contradictions else issues).extend("contradiction: " + x for x in c_hard)
            issues.extend("inconsistency: " + x for x in c_soft)
        else:
            checks["consistent"] = NOT_RUN
        state, detail = camera_motion_agreement(a)
        checks["camera_motion_agreement"] = state
        if detail:
            issues.append("camera motion disagreement: " + detail)
        # --- text structure: one text must not claim two incompatible camera movements ---------
        if text is not None:
            state, problems = text.caption_consistent([a.summary or "", a.temporal_progression or "", a.visual_style.motion or "", *a.actions])
            checks["text_consistent"] = state
            issues.extend("text contradiction: " + p for p in problems)
        else:
            checks["text_consistent"] = NOT_RUN
        # --- one shot? a large visual change inside the shot may be a missed cut ---------------
        scan = scans.get(a.scene_id)
        if scan is not None and scan.samples:
            peak = max((s.change for s in scan.samples[1:]), default=0.0)
            if peak >= float(vcfg.max_in_shot_change):
                checks["single_shot"] = False
                issues.append(f"visual change of {peak:.2f} inside the shot: possible missed cut, not a single-shot generation target")
                n_suspected_cuts += 1
            else:
                checks["single_shot"] = True
        else:
            checks["single_shot"] = NOT_RUN
        # --- claim-level evidence --------------------------------------------------------------
        if vcfg.claims:
            claims = scene_claims(a, c_hard + c_soft, generative)
            a.enrichments["claims"] = claims
            for k, v in claim_summary(claims).items():
                n_claims[k] = n_claims.get(k, 0) + v
        status, reasons = decide("scene", checks, hard, suggested, issues, semantic)
        v_scenes.append(ValidatedScene(
            analysis=a,
            validation=ValidationInfo(status=status, issues=hard + issues, checks=checks, verifier_score=a.verification.score if a.verification else None, reason_codes=reasons, policy_version=POLICY_VERSION, semantic_verification=semantic),
            quality=quality,
        ))
        if a.summary:
            scene_items.append((a.scene_id, a.summary, a.start_time, a.end_time))
    # --- caption <-> frames alignment (CLIP retrieval rank among the video's own captions) --------
    alignment, alignment_state = _caption_alignment(ctx, analyses, frames_by_scene, log)
    n_misaligned = 0
    for vs in v_scenes:
        res = alignment.get(vs.analysis.scene_id)
        if res is None:
            vs.validation.checks["caption_alignment"] = NOT_RUN if alignment_state in ("model_unavailable", "error") else NOT_APPLICABLE
            continue
        ok = not res.misaligned(float(vcfg.clip_alignment_min_rank_fraction), float(vcfg.clip_alignment_min_margin))
        vs.validation.checks["caption_alignment"] = ok
        vs.analysis.enrichments["caption_alignment"] = {"clip_score": res.score, "rank": res.rank, "candidates": res.candidates, "best_other": res.best_other, "best_score": res.best_score, "margin": res.margin}
        if not ok:
            n_misaligned += 1
            vs.validation.issues.append(f"caption ranks {res.rank} of {res.candidates} by CLIP similarity to its own frames (margin {res.margin:.3f}" + (f", best match: {res.best_other})" if res.best_other else ")"))
            vs.validation.reason_codes.append("failed_check:caption_alignment")
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
            vs.validation.duplicate_method = "text_minhash"
            vs.validation.issues.append("description identical to another scene")
            vs.validation.reason_codes.append("duplicate:text")

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
        checks["clips_exist"] = ok if e.clip_ids else None
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
        semantic = "measured"
        if e.source == EventSource.VISION and e.event_type in (EventType.ACTION, EventType.APPEARANCE) and not hard:
            generative_event = str(e.attributes.get("provider")) not in ("heuristic", "cv_models", "None")
            if e.verification is not None and e.verification.verdict != VerificationVerdict.UNKNOWN:
                semantic = f"verifier:{e.verification.verifier or verifier.name}"
                if e.verification.verdict == VerificationVerdict.UNSUPPORTED:
                    hard.append("verifier found the claim unsupported by the frames")
            elif generative_event:
                semantic = NOT_RUN
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
        status, reasons = decide("event", checks, hard, suggested, issues, semantic)
        v_events.append(ValidatedEvent(event=e, validation=ValidationInfo(status=status, issues=hard + issues, checks=checks, verifier_score=e.verification.score if e.verification else None, reason_codes=reasons, policy_version=POLICY_VERSION, semantic_verification=semantic), quality=q))
    if cfg.deduplication.enabled:
        ev_dups = dedupe_timed_texts([(e.event_id, e.event, e.start_time, e.end_time) for e in timeline.events], cfg.deduplication.near_duplicate_threshold, cfg.deduplication.evidence_iou_threshold)
        for ve in v_events:
            if ve.event.event_id in ev_dups:
                ve.validation.status = ValidationStatus.DUPLICATE
                ve.validation.duplicate_of = ev_dups[ve.event.event_id]
                ve.validation.duplicate_method = "text_minhash+evidence_overlap"
                ve.validation.reason_codes.append("duplicate:text_and_window")
    event_status = {ve.event.event_id: ve.validation.status for ve in v_events}
    event_by_id: dict[str, Event] = {e.event_id: e for e in timeline.events}
    # duplicate events are not exported; QA evidence that names one is pointed at the canonical copy
    # (same text, overlapping window) so no exported question references a missing event
    canonical_event = {ve.event.event_id: ve.validation.duplicate_of for ve in v_events if ve.validation.status == ValidationStatus.DUPLICATE and ve.validation.duplicate_of}
    if canonical_event:
        for q_rec in qa.questions:
            ids = [canonical_event.get(x, x) for x in q_rec.evidence.event_ids]
            q_rec.evidence.event_ids = list(dict.fromkeys(ids))

    # ------------------------------------------------------------------ relations
    v_relations: list[ValidatedRelation] = []
    cycles = before_cycles(timeline.relations) if vcfg.relation_checks else []
    cyclic_events = {eid for cyc in cycles for eid in cyc}
    if cycles:
        log.warning("%d cycle(s) in the BEFORE graph, e.g. %s", len(cycles), " -> ".join(cycles[0][:5]))
    n_relation_violations = 0
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
        if vcfg.relation_checks and ok:
            holds, why = relation_holds(r, event_by_id[r.event_a], event_by_id[r.event_b])
            checks["interval_semantics"] = holds
            if not holds:
                n_relation_violations += 1
                hard.append(f"relation {r.relation} contradicts the event intervals: {why}")
            checks["acyclic"] = not (r.relation == RelationType.BEFORE and r.event_a in cyclic_events and r.event_b in cyclic_events)
            if checks["acyclic"] is False:
                issues.append("part of a cycle in the BEFORE graph")
        elif not vcfg.relation_checks:
            checks["interval_semantics"] = NOT_RUN
        suggested = confidence_status(r.confidence, vcfg.minimum_confidence, vcfg.review_confidence)
        semantic = "measured"
        if r.relation == RelationType.CAUSES or r.evidence_kind == "model_inferred" or r.confidence_source == ConfidenceSource.MODEL_SELF_REPORT:
            semantic = NOT_RUN if r.evidence_kind != "human" else "human"
            issues.append("inferred by a model without verification" if r.evidence_kind != "human" else "human reviewed")
            if semantic == NOT_RUN:
                suggested = "review" if suggested == "accepted" else suggested
        qualities = [event_quality_by_id[x] for x in (r.event_a, r.event_b) if x in event_quality_by_id]
        status, reasons = decide("relation", checks, hard, suggested, issues, semantic)
        v_relations.append(ValidatedRelation(relation=r, validation=ValidationInfo(status=status, issues=hard + issues, checks=checks, reason_codes=reasons, policy_version=POLICY_VERSION, semantic_verification=semantic), quality=relation_quality(r.confidence, qualities)))

    # ------------------------------------------------------------------ QA
    n_recomputed_failed = 0
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
        # question form + answer type (text structure) and independent answer recomputation
        if not q_rec.answer_type:
            q_rec.answer_type = answer_type_for(str(q_rec.type), q_rec.template_id)
        if text is not None:
            state, problems = text.question_form(q_rec.question)
            checks["question_form"] = state
            issues.extend("question form: " + p for p in problems)
            state, problems = text.answer_matches_type(q_rec.answer, q_rec.answer_type)
            checks["answer_type"] = state
            issues.extend("answer type: " + p for p in problems)
        else:
            checks["question_form"] = checks["answer_type"] = NOT_RUN
        if vcfg.recompute_answers:
            state, detail = recompute_answer(q_rec, event_by_id)
            checks["answer_recomputed"] = state
            if state == CheckState.FAILED.value:
                n_recomputed_failed += 1
                hard.append("answer disagrees with the evidence events: " + (detail or ""))
        else:
            checks["answer_recomputed"] = NOT_RUN
        # fine-timing questions need events timed inside the shot, not shot-inherited boundaries
        if q_rec.type in (QAType.DURATION, QAType.EVENT_LOCALIZATION):
            precs = {event_by_id[x].boundary_precision for x in ev.event_ids if x in event_by_id}
            checks["timing_precision"] = not (precs and precs <= {"scene"}) or ("shot" in q_rec.question.lower())
            if checks["timing_precision"] is False:
                issues.append("fine-timing question on a shot-inherited boundary")
        suggested = confidence_status(q_rec.confidence, vcfg.minimum_confidence, vcfg.review_confidence)
        if q_rec.confidence is None:
            issues.append("evidence confidence unavailable")
        elif q_rec.confidence < vcfg.minimum_confidence:
            issues.append("low confidence evidence")
        qualities = [event_quality_by_id[x] for x in ev.event_ids if x in event_quality_by_id]
        q_rec.quality = qa_quality(q_rec, grounding, qualities)
        semantic_states = {ve.validation.semantic_verification for ve in v_events if ve.event.event_id in ev.event_ids}
        semantic = NOT_RUN if NOT_RUN in semantic_states else "measured"
        status, reasons = decide("qa", checks, hard, suggested, issues, semantic)
        q_rec.validation = ValidationInfo(status=status, issues=hard + issues, checks=checks, reason_codes=reasons, policy_version=POLICY_VERSION, semantic_verification=semantic)
    if cfg.deduplication.enabled:
        q_dups = dedupe_questions(qa.questions, cfg.deduplication)
        for q_rec in qa.questions:
            if q_rec.question_id in q_dups and q_rec.validation is not None:
                q_rec.validation.status = ValidationStatus.DUPLICATE
                q_rec.validation.duplicate_of = q_dups[q_rec.question_id]
                q_rec.validation.duplicate_method = "text_minhash+evidence_overlap"
                q_rec.validation.reason_codes.append("duplicate:question")

    # ------------------------------------------------------------------ summary + write
    def counts(statuses: list[ValidationStatus]) -> dict[str, int]:
        out = {s.value: 0 for s in ValidationStatus}
        for s in statuses:
            out[str(s)] += 1
        return out

    def reason_counts(infos: list[ValidationInfo]) -> dict[str, int]:
        out: dict[str, int] = {}
        for info in infos:
            for code in info.reason_codes:
                key = code.split(":", 1)[0] + ":" + code.split(":", 1)[1].split(" ")[0][:40] if ":" in code else code
                out[key] = out.get(key, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1])[:40])

    summary = {
        "scenes": counts([v.validation.status for v in v_scenes]),
        "events": counts([v.validation.status for v in v_events]),
        "relations": counts([v.validation.status for v in v_relations]),
        "qa": counts([q.validation.status for q in qa.questions if q.validation]),
        "events_verified_now": n_verified,
        "verifier": verifier.name,
        "verify_workers": v_workers,
        "text_checker": text.name if text else "disabled",
        "policy_version": POLICY_VERSION,
        "scenes_quality_flagged": sum(1 for v in v_scenes if v.analysis.quality_flags),
        "scenes_contradictory": sum(1 for v in v_scenes if v.validation.checks.get("consistent") is False),
        "scenes_caption_misaligned": n_misaligned,
        "caption_alignment": alignment_state,
        "scenes_suspected_missed_cut": n_suspected_cuts,
        "claims": n_claims,
        "relation_interval_violations": n_relation_violations,
        "relation_before_cycles": len(cycles),
        "qa_answers_recomputed_failed": n_recomputed_failed,
        "reason_codes": {
            "scenes": reason_counts([v.validation for v in v_scenes]),
            "events": reason_counts([v.validation for v in v_events]),
            "qa": reason_counts([q.validation for q in qa.questions if q.validation]),
        },
        "video_quality_flags": video_flags,
    }
    validated = ValidatedVideo(video_id=ctx.video_id, duration=duration, scenes=v_scenes, events=v_events, relations=v_relations, qa=qa.questions, summary=summary)
    out = ctx.paths.validated_file(ctx.video_id)
    write_json_atomic(out, validated)
    qs = summary["qa"]
    log.info("QA accepted=%d review=%d rejected=%d duplicate=%d | events %s | scenes %s", qs["accepted"], qs["review"], qs["rejected"], qs["duplicate"], summary["events"], summary["scenes"])
    return StageOutput(artifact_path=str(out), metrics=summary, message=f"{qs['accepted']} accepted / {qs['rejected']} rejected / {qs['review']} review ({qs['duplicate']} duplicates)")
