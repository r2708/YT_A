"""Turn validated per-video annotations into the exported dataset records."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from video_dataset.schemas.common import ConfidenceSource
from video_dataset.schemas.dataset import (
    ClipCaptionRecord,
    EventRecord,
    FrameCaptionRecord,
    SceneRecord,
    VideoDescriptionRecord,
)
from video_dataset.schemas.ocr import OCRResult
from video_dataset.schemas.qa import QARecord, QAType
from video_dataset.schemas.quality import QualityScore, ValidationInfo, ValidationStatus
from video_dataset.schemas.scene import Clip, Frame, Scene
from video_dataset.schemas.transcript import Transcript
from video_dataset.schemas.validated import ValidatedVideo
from video_dataset.schemas.vision import CameraMovement, SceneAnalysis, ShotType
from video_dataset.utils.ids import record_id
from video_dataset.utils.text import lower_first, sentence
from video_dataset.vision.motion import MOVEMENT_PHRASES

_EXPORTABLE = {ValidationStatus.ACCEPTED, ValidationStatus.REVIEW}


def _rel(path: str | None, base: Path | None) -> str | None:
    if path is None:
        return None
    if base is None:
        return path
    try:
        return str(Path(path).resolve().relative_to(base.resolve()))
    except ValueError:
        return path


_LIGHT_NUMBERS = re.compile(r"\s*\([^)]*\)")  # "dim overall brightness (mean 56/255)" -> "dim overall brightness"

class DatasetBuilder:
    def __init__(self, include_review: bool = True, include_rejected: bool = False, path_base: Path | None = None):
        self.include_review = include_review
        self.include_rejected = include_rejected
        self.path_base = path_base

    def _keep(self, v: ValidationInfo | None) -> bool:
        if v is None:
            return True
        if v.status == ValidationStatus.ACCEPTED:
            return True
        if v.status == ValidationStatus.REVIEW:
            return self.include_review
        if v.status == ValidationStatus.REJECTED:
            return self.include_rejected
        return False  # duplicates never exported

    # ------------------------------------------------------------------ builders
    def frames(self, validated: ValidatedVideo, frames: list[Frame], ocr: OCRResult | None) -> list[FrameCaptionRecord]:
        out: list[FrameCaptionRecord] = []
        ocr_by_frame: dict[str, list[str]] = {}
        if ocr:
            for d in ocr.detections:
                ocr_by_frame.setdefault(d.frame_id, []).append(d.text)
        for vs in validated.scenes:
            if not self._keep(vs.validation):
                continue
            a = vs.analysis
            fa_by_id = {fa.frame_id: fa for fa in a.frame_analyses}
            for f in [fr for fr in frames if fr.scene_id == a.scene_id]:
                fa = fa_by_id.get(f.frame_id)
                if fa and fa.caption:
                    caption, source, objects, actions, conf, conf_src = fa.caption, "frame_analysis", [o.name for o in fa.objects], fa.actions, fa.confidence, fa.confidence_source
                    measurements = fa.measurements or a.measurements
                    camera = fa.camera or a.camera
                else:
                    if not a.summary:
                        continue
                    caption = f"{a.summary.rstrip('.')}. (Frame at {f.timestamp:.1f}s within this shot.)"
                    source = "scene_summary" if a.provider != "heuristic" else "measurement"
                    objects, actions, conf, conf_src = [o.name for o in a.objects], a.actions, a.confidence, a.confidence_source
                    measurements, camera = a.measurements, a.camera
                out.append(
                    FrameCaptionRecord(
                        record_id=record_id("frame", validated.video_id, f.frame_id),
                        video_id=validated.video_id,
                        scene_id=a.scene_id,
                        frame_id=f.frame_id,
                        frame_path=_rel(f.frame_path, self.path_base) or f.frame_path,
                        timestamp=f.timestamp,
                        caption=caption,
                        caption_source=source,
                        objects=objects,
                        actions=actions,
                        camera=camera,
                        environment=a.environment,
                        measurements=measurements,
                        ocr_text=sorted(set(ocr_by_frame.get(f.frame_id, []))),
                        provider=a.provider,
                        confidence=conf,
                        confidence_source=ConfidenceSource(conf_src),
                        quality=vs.quality,
                        validation=vs.validation,
                    )
                )
        return out

    def clips(self, validated: ValidatedVideo, clips: list[Clip], transcript: Transcript | None) -> list[ClipCaptionRecord]:
        out: list[ClipCaptionRecord] = []
        by_scene = {vs.analysis.scene_id: vs for vs in validated.scenes}
        for c in clips:
            vs = by_scene.get(c.scene_id)
            if vs is None or not self._keep(vs.validation) or not vs.analysis.summary:
                continue
            a = vs.analysis
            desc = (a.summary or "").strip()
            if a.temporal_progression:
                desc += " " + sentence(a.temporal_progression)
            if c.duration < (a.end_time - a.start_time) - 0.5:
                desc += f" (This clip covers {c.start_time:.1f}s to {c.end_time:.1f}s of the shot.)"
            out.append(
                ClipCaptionRecord(
                    record_id=record_id("clip", validated.video_id, c.clip_id),
                    video_id=validated.video_id,
                    scene_id=c.scene_id,
                    clip_id=c.clip_id,
                    clip_path=_rel(c.clip_path, self.path_base) or c.clip_path,
                    start_time=c.start_time,
                    end_time=c.end_time,
                    media_duration=c.media_duration,
                    exact=c.exact,
                    description=desc,
                    actions=a.actions,
                    camera=a.camera,
                    transcript=(transcript.text_between(c.start_time, c.end_time) or None) if transcript else None,
                    frame_ids=c.frame_ids,
                    provider=a.provider,
                    confidence=a.confidence,
                    confidence_source=ConfidenceSource(a.confidence_source),
                    quality=vs.quality,
                    validation=vs.validation,
                )
            )
        return out

    def scenes(self, validated: ValidatedVideo, clips: list[Clip], transcript: Transcript | None, ocr: OCRResult | None) -> list[SceneRecord]:
        out: list[SceneRecord] = []
        clip_by_scene = {c.scene_id: c for c in clips if not c.clip_id.count("_") > 1}
        for vs in validated.scenes:
            if not self._keep(vs.validation) or not vs.analysis.summary:
                continue
            a = vs.analysis
            out.append(
                SceneRecord(
                    record_id=record_id("scene", validated.video_id, a.scene_id),
                    video_id=validated.video_id,
                    scene_id=a.scene_id,
                    start=a.start_time,
                    end=a.end_time,
                    duration=round(a.end_time - a.start_time, 3),
                    summary=a.summary or "",
                    environment=a.environment,
                    objects=[o.name for o in a.objects],
                    object_details=[o.model_dump(mode="json") for o in a.objects],
                    people=a.people.model_dump(mode="json") if a.people else None,
                    actions=a.actions,
                    camera=a.camera,
                    visual_style=a.visual_style,
                    temporal_progression=a.temporal_progression,
                    transcript=(transcript.text_between(a.start_time, a.end_time) or None) if transcript else None,
                    ocr_text=[t.text for t in ocr.tracks if a.scene_id in t.scene_ids] if ocr else [],
                    frame_ids=a.frame_ids,
                    clip_id=clip_by_scene[a.scene_id].clip_id if a.scene_id in clip_by_scene else None,
                    measurements=a.measurements,
                    provider=a.provider,
                    confidence=a.confidence,
                    confidence_source=ConfidenceSource(a.confidence_source),
                    quality=vs.quality,
                    validation=vs.validation,
                )
            )
        return out

    def events(self, validated: ValidatedVideo) -> list[EventRecord]:
        out: list[EventRecord] = []
        rel_by_event: dict[str, list[dict[str, Any]]] = {}
        for vr in validated.relations:
            if not self._keep(vr.validation):
                continue
            r = vr.relation
            rel_by_event.setdefault(r.event_a, []).append({"relation": str(r.relation), "event_b": r.event_b, "event_b_text": r.event_b_text, "gap_seconds": r.gap_seconds, "confidence": r.confidence, "relation_id": r.relation_id})
        for ve in validated.events:
            if not self._keep(ve.validation):
                continue
            e = ve.event
            out.append(
                EventRecord(
                    record_id=record_id("event", validated.video_id, e.event_id),
                    video_id=validated.video_id,
                    event_id=e.event_id,
                    event_type=str(e.event_type),
                    start_time=e.start_time,
                    end_time=e.end_time,
                    event=e.event,
                    entities=e.entities,
                    action=e.action,
                    scene_ids=e.scene_ids,
                    frame_ids=e.frame_ids,
                    clip_ids=e.clip_ids,
                    source=str(e.source),
                    confidence=e.confidence,
                    confidence_source=ConfidenceSource(e.confidence_source),
                    relations=rel_by_event.get(e.event_id, []),
                    quality=ve.quality,
                    validation=ve.validation,
                )
            )
        return out

    def qa(self, validated: ValidatedVideo) -> tuple[list[QARecord], list[QARecord]]:
        temporal: list[QARecord] = []
        long_video: list[QARecord] = []
        for q in validated.qa:
            if not self._keep(q.validation):
                continue
            if q.is_long_range or q.type == QAType.LONG_RANGE or (q.type == QAType.MULTI_EVENT and len(q.evidence.scene_ids) >= 3):
                long_video.append(q)
            else:
                temporal.append(q)
        return temporal, long_video

    def video_descriptions(self, validated: ValidatedVideo, scenes: list[Scene], title: str | None) -> list[VideoDescriptionRecord]:
        kept = [vs for vs in validated.scenes if self._keep(vs.validation) and vs.analysis.summary]
        if not kept:
            return []
        analyses = {vs.analysis.scene_id: vs.analysis for vs in kept}
        ordered = [s for s in scenes if s.scene_id in analyses]
        records = []
        # whole video, plus segments of at most ~8 shots / 90 s so descriptions stay usable as generation prompts
        groups: list[list[Scene]] = [ordered]
        seg: list[Scene] = []
        for s in ordered:
            seg.append(s)
            if len(seg) >= 8 or (seg[-1].end_time - seg[0].start_time) >= 90.0:
                groups.append(seg)
                seg = []
        if seg and len(groups) > 1 and len(seg) >= 2:
            groups.append(seg)
        seen_spans: set[tuple[float, float]] = set()
        for gi, group in enumerate(groups):
            span = (group[0].start_time, group[-1].end_time)
            if span in seen_spans:
                continue
            seen_spans.add(span)
            rec = self._describe_group(validated, group, analyses, kept, gi, title)
            if rec:
                records.append(rec)
        return records

    @staticmethod
    def _summarize(values: list[str], total: int, max_items: int = 4) -> str | None:
        """Turn per-shot labels into one phrase: 'dim' -> 'dim throughout'; mixed -> 'mostly dim (5 of 8 shots), dark (3 of 8 shots)'."""
        vals = [v for v in values if v]
        if not vals:
            return None
        counts: dict[str, int] = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if len(ranked) == 1:
            return ranked[0][0] if total <= 1 else f"{ranked[0][0]} throughout"
        parts = [f"{label} ({n} of {total} shots)" for label, n in ranked[:max_items]]
        return "mostly " + parts[0] + (", " + ", ".join(parts[1:]) if len(parts) > 1 else "")

    def _describe_group(self, validated: ValidatedVideo, group: list[Scene], analyses: dict[str, SceneAnalysis], kept, gi: int, title: str | None) -> VideoDescriptionRecord | None:  # type: ignore[no-untyped-def]
        shots = []
        subjects: dict[str, int] = {}
        locations: list[str] = []
        actions: list[str] = []
        cam_labels: list[str] = []
        light_labels: list[str] = []
        light_temps: list[str] = []
        transitions: list[str] = []
        confs: list[float | None] = []
        for s in group:
            a = analyses[s.scene_id]
            for o in a.objects:
                subjects[o.name.lower()] = subjects.get(o.name.lower(), 0) + 1
            if a.people and a.people.count:
                subjects["people"] = subjects.get("people", 0) + a.people.count
            loc = a.environment.location or (str(a.environment.setting) if str(a.environment.setting) != "unknown" else None)
            if loc and loc not in locations:
                locations.append(loc)
            actions += [x for x in a.actions if x not in actions]
            shot = a.camera.shot_type if a.camera.shot_type != ShotType.UNKNOWN else None
            mov = MOVEMENT_PHRASES.get(a.camera.movement) if a.camera.movement != CameraMovement.UNKNOWN else None
            cam = ", ".join(x for x in [f"{shot} shot" if shot else None, mov] if x)
            if cam:
                cam_labels.append(cam)
            # lighting level as a label ("dim"); the per-shot numbers stay in the shot list, not the prompt
            level = a.environment.lighting or (a.measurements.lighting_level if a.measurements else None)
            if not level and a.visual_style.lighting:
                level = _LIGHT_NUMBERS.sub("", a.visual_style.lighting).split(",")[0].strip()
            if level:
                light_labels.append(str(level).strip())
            if a.visual_style.lighting and "color temperature" in a.visual_style.lighting:
                light_temps.append(a.visual_style.lighting.split(",")[-1].strip())
            if s.transition_in in ("cut", "fade"):
                transitions.append(str(s.transition_in))
            confs.append(a.confidence)
            shots.append({"scene_id": s.scene_id, "start": s.start_time, "end": s.end_time, "summary": a.summary, "camera": cam or None, "transition_in": str(s.transition_in), "temporal_progression": a.temporal_progression})
        subject = ", ".join(k for k, _ in sorted(subjects.items(), key=lambda kv: -kv[1])[:3]) or None
        n_shots = len(group)
        cams: list[str] = [c for c in [self._summarize(cam_labels, n_shots)] if c]
        light_summary = self._summarize(light_labels, n_shots)
        temp_summary = self._summarize(light_temps, n_shots, max_items=2)
        lightings: list[str] = [x for x in [light_summary, temp_summary] if x]
        progression = " ".join(f"Shot {i + 1} ({sh['start']:.1f}-{sh['end']:.1f}s): {sh['summary']}" + (f" {sentence(str(sh['temporal_progression']))}" if sh["temporal_progression"] else "") for i, sh in enumerate(shots))
        trans_text = None
        if transitions:
            trans_text = f"{len(transitions)} shot changes ({', '.join(sorted(set(transitions)))})"
        prompt_parts = []
        if subject:
            prompt_parts.append(f"Subject: {subject}.")
        if locations:
            prompt_parts.append(f"Environment: {'; '.join(locations[:4])}.")
        if actions:
            prompt_parts.append("Action: " + "; ".join(lower_first(x) for x in actions[:6]) + ".")
        if cams:
            prompt_parts.append("Camera: " + "; ".join(cams[:5]) + ".")
        if lightings:
            prompt_parts.append("Lighting: " + "; ".join(lightings[:3]) + ".")
        motion_summary = self._summarize([a.visual_style.motion or "" for a in (analyses[s.scene_id] for s in group)], n_shots)
        if motion_summary:
            prompt_parts.append("Motion: " + motion_summary + ".")
        prompt_parts.append("Temporal progression: " + progression)
        if trans_text:
            prompt_parts.append(f"Transitions: {trans_text}.")
        conf = round(min(c for c in confs if c is not None), 3) if confs and all(c is not None for c in confs) else None
        statuses = [vs.validation.status for vs in kept if vs.analysis.scene_id in {s.scene_id for s in group}]
        status = ValidationStatus.REVIEW if any(st == ValidationStatus.REVIEW for st in statuses) else ValidationStatus.ACCEPTED
        return VideoDescriptionRecord(
            record_id=record_id("videodesc", validated.video_id, f"{gi:02d}"),
            video_id=validated.video_id,
            start_time=group[0].start_time,
            end_time=group[-1].end_time,
            scene_ids=[s.scene_id for s in group],
            subject=subject,
            environment="; ".join(locations[:4]) or None,
            action="; ".join(actions[:6]) or None,
            camera="; ".join(cams[:5]) or None,
            composition="; ".join(dict.fromkeys(a.visual_style.composition for a in (analyses[s.scene_id] for s in group) if a.visual_style.composition)) or None,
            lighting="; ".join(lightings[:3]) or None,
            motion=motion_summary,
            temporal_progression=progression,
            transitions=trans_text,
            prompt=" ".join(prompt_parts),
            shots=shots,
            confidence=conf,
            confidence_source=ConfidenceSource.DERIVED_MIN if conf is not None else ConfidenceSource.UNAVAILABLE,
            quality=QualityScore.from_components(grounding=conf, description_quality=None),
            validation=ValidationInfo(status=status, issues=[] if status == ValidationStatus.ACCEPTED else ["contains scenes flagged for review"]),
        )
