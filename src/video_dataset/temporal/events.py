"""Build the event timeline from every modality. Each event keeps its source, supporting frames,
scene ids, boundary precision and a confidence that is either derived from a real signal or None."""

from __future__ import annotations

from video_dataset.config import TemporalConfig
from video_dataset.schemas.common import ConfidenceSource
from video_dataset.schemas.events import Event, EventSource, EventType
from video_dataset.schemas.ocr import OCRResult
from video_dataset.schemas.scene import Frame, Scene, SceneScan, TransitionType
from video_dataset.schemas.transcript import AudioAnalysis, AudioEventCategory, Transcript
from video_dataset.schemas.vision import CameraMovement, SceneAnalysis, Setting, VisionResult
from video_dataset.utils.ids import event_id as make_event_id
from video_dataset.utils.logging import get_logger
from video_dataset.utils.text import content_words, lower_first, sentence
from video_dataset.utils.timecode import clamp, interval_overlap
from video_dataset.vision.motion import MOVEMENT_PHRASES, motion_segments

_PERSON_WORDS = {"person", "people", "man", "woman", "boy", "girl", "child", "children", "someone", "player", "worker", "driver", "rider", "crowd", "group", "he", "she", "they", "figure"}


def _short(text: str | None, limit: int = 90) -> str | None:
    if not text:
        return None
    first = text.strip().split(". ")[0].strip().rstrip(".")
    return first if len(first) <= limit else first[: limit - 1].rstrip() + "…"


def frames_in(frames: list[Frame], start: float, end: float, scene_ids: list[str] | None = None, limit: int = 12) -> list[str]:
    sel = [f for f in frames if start - 1e-3 <= f.timestamp <= end + 1e-3 and (scene_ids is None or f.scene_id in scene_ids)]
    if not sel and scene_ids:
        sel = [f for f in frames if f.scene_id in scene_ids]
    sel.sort(key=lambda f: f.timestamp)
    if limit <= 1 and sel:
        return [sel[len(sel) // 2].frame_id]
    if len(sel) > limit:
        step = (len(sel) - 1) / (limit - 1)
        sel = [sel[int(round(i * step))] for i in range(limit)]
    return [f.frame_id for f in sel]


def entities_for_action(action: str, analysis: SceneAnalysis) -> list[str]:
    low = action.lower()
    ents = [o.name for o in analysis.objects if o.name and o.name.lower() in low]
    words = set(low.replace(",", " ").split())
    if words & _PERSON_WORDS:
        ents.append("person")
    if "camera" in words:
        ents.append("camera")
    if not ents:
        # fall back to the first content word of the action as the subject
        cw = [w for w in action.split() if w.lower() in content_words(action)]
        if cw:
            ents.append(cw[0].lower())
    seen: set[str] = set()
    out = []
    for e in ents:
        if e.lower() not in seen:
            seen.add(e.lower())
            out.append(e)
    return out


def transition_confidence(score: float | None, threshold: float | None) -> float | None:
    """Monotone map of the measured content change at a cut into [0.5, 1]: at threshold -> 0.5, at 2x -> 1.0."""
    if score is None or not threshold:
        return None
    return round(clamp(0.5 + 0.5 * (score - threshold) / threshold, 0.5, 1.0), 3)


log = get_logger("temporal.events")


class EventExtractor:
    def __init__(self, cfg: TemporalConfig, scene_threshold: float | None = None):
        self.cfg = cfg
        self.scene_threshold = scene_threshold

    # ------------------------------------------------------------------ public
    def extract(
        self,
        video_id: str,
        duration: float,
        scenes: list[Scene],
        frames: list[Frame],
        vision: VisionResult | None,
        transcript: Transcript | None,
        ocr: OCRResult | None,
        audio: AudioAnalysis | None,
        scans: list[SceneScan],
    ) -> list[Event]:
        events: list[Event] = []
        analyses = {a.scene_id: a for a in (vision.scenes if vision else [])}
        scan_by_scene = {s.scene_id: s for s in scans}

        events += self._vision_events(video_id, scenes, frames, analyses)
        if self.cfg.include_transitions:
            events += self._transition_events(video_id, scenes, frames, analyses, duration)
        if self.cfg.include_camera_events:
            events += self._camera_events(video_id, scenes, frames, scan_by_scene, analyses)
        if self.cfg.include_speech_events and transcript:
            events += self._speech_events(video_id, transcript, frames, scenes)
        if self.cfg.include_ocr_events and ocr:
            events += self._ocr_events(video_id, ocr, frames, duration)
        if self.cfg.include_audio_events and audio:
            events += self._audio_events(video_id, audio, transcript, frames, scenes, duration)
        if self.cfg.include_state_changes:
            events += self._state_change_events(video_id, scenes, frames, analyses)

        events.sort(key=lambda e: (e.start_time, e.end_time, str(e.event_type)))
        for i, e in enumerate(events, start=1):
            e.event_id = make_event_id(i)
            e.start_time = round(clamp(e.start_time, 0.0, duration), 3)
            e.end_time = round(clamp(max(e.end_time, e.start_time), 0.0, duration), 3)
        return events

    # ------------------------------------------------------------------ sources
    def _vision_events(self, video_id: str, scenes: list[Scene], frames: list[Frame], analyses: dict[str, SceneAnalysis]) -> list[Event]:
        out: list[Event] = []
        prev_objects: set[str] = set()
        for scene in scenes:
            a = analyses.get(scene.scene_id)
            if a is None:
                continue
            fids = frames_in(frames, scene.start_time, scene.end_time, [scene.scene_id])
            conf = a.confidence
            src = ConfidenceSource(a.confidence_source) if conf is not None else ConfidenceSource.UNAVAILABLE
            for action in a.actions:
                if not action or action.lower().startswith("camera "):
                    continue  # camera motion is covered by measured camera events
                out.append(
                    Event(
                        event_id="tmp",
                        video_id=video_id,
                        event_type=EventType.ACTION,
                        start_time=scene.start_time,
                        end_time=scene.end_time,
                        event=sentence(action),
                        entities=entities_for_action(action, a),
                        action=lower_first(action),
                        scene_ids=[scene.scene_id],
                        frame_ids=fids,
                        source=EventSource.VISION,
                        source_ids=[scene.scene_id],
                        confidence=conf,
                        confidence_source=src,
                        boundary_precision="scene",
                        attributes={"provider": a.provider, "model": a.model},
                        verification=a.verification,
                    )
                )
            names = {o.name.lower() for o in a.objects if o.name}
            for name in sorted(names - prev_objects):
                obj = next(o for o in a.objects if o.name.lower() == name)
                desc = " ".join([*obj.attributes[:2], obj.name]) if obj.attributes else obj.name
                loc = f" in the {obj.location}" if obj.location and obj.location.lower() not in {"unknown", "n/a"} else ""
                out.append(
                    Event(
                        event_id="tmp",
                        video_id=video_id,
                        event_type=EventType.APPEARANCE,
                        start_time=scene.start_time,
                        end_time=scene.end_time,
                        event=sentence(f"a {desc} is visible{loc}") if not desc.lower().startswith(("a ", "an ", "the ")) else sentence(f"{desc} is visible{loc}"),
                        entities=[obj.name],
                        action="appears",
                        scene_ids=[scene.scene_id],
                        frame_ids=fids,
                        source=EventSource.VISION,
                        source_ids=[scene.scene_id],
                        confidence=obj.confidence if obj.confidence is not None else conf,
                        confidence_source=(ConfidenceSource(obj.confidence_source) if obj.confidence is not None else src),
                        boundary_precision="scene",
                        attributes={"attributes": obj.attributes, "location": obj.location, "count": obj.count},
                    )
                )
            prev_objects = names
        return out

    def _transition_events(self, video_id: str, scenes: list[Scene], frames: list[Frame], analyses: dict[str, SceneAnalysis], duration: float) -> list[Event]:
        out: list[Event] = []
        for i, scene in enumerate(scenes):
            if scene.transition_in not in (TransitionType.CUT, TransitionType.FADE, TransitionType.ADAPTIVE):
                continue
            prev = scenes[i - 1] if i > 0 else None
            kind = "fade" if scene.transition_in == TransitionType.FADE else "cut"
            a_prev = analyses.get(prev.scene_id) if prev else None
            a_cur = analyses.get(scene.scene_id)
            generative = bool(a_prev and a_cur and a_prev.provider not in ("heuristic", None) and a_cur.provider not in ("heuristic", None))
            before = _short(a_prev.summary) if generative and a_prev else None
            after = _short(a_cur.summary) if generative and a_cur else None
            if before and after:
                text = f"The shot changes by a {kind} from a view of {lower_first(before)} to a view of {lower_first(after)}."
            else:
                detail = ""
                if a_prev and a_cur and a_prev.measurements and a_cur.measurements and a_prev.measurements.lighting_level != a_cur.measurements.lighting_level:
                    detail = f" (lighting goes from {a_prev.measurements.lighting_level} to {a_cur.measurements.lighting_level})"
                text = f"A {kind} to a new shot occurs{detail}."
            out.append(
                Event(
                    event_id="tmp",
                    video_id=video_id,
                    event_type=EventType.TRANSITION,
                    start_time=max(0.0, scene.start_time - 0.05),
                    end_time=min(duration, scene.start_time + 0.05),
                    event=text,
                    entities=["camera"],
                    action=kind,
                    scene_ids=[s for s in ([prev.scene_id] if prev else []) + [scene.scene_id]],
                    frame_ids=(frames_in(frames, prev.start_time, prev.end_time, [prev.scene_id], limit=2)[-1:] if prev else []) + frames_in(frames, scene.start_time, scene.end_time, [scene.scene_id], limit=2)[:1],
                    source=EventSource.SCENE_DETECTOR,
                    source_ids=[scene.scene_id],
                    confidence=transition_confidence(scene.detector_score, self.scene_threshold),
                    confidence_source=ConfidenceSource.MEASUREMENT if scene.detector_score is not None and self.scene_threshold else ConfidenceSource.UNAVAILABLE,
                    boundary_precision="frame",
                    attributes={"transition_type": str(scene.transition_in), "detector_score": scene.detector_score},
                )
            )
        return out

    def _camera_events(self, video_id: str, scenes: list[Scene], frames: list[Frame], scans: dict[str, SceneScan], analyses: dict[str, SceneAnalysis]) -> list[Event]:
        out: list[Event] = []
        for scene in scenes:
            scan = scans.get(scene.scene_id)
            for seg in motion_segments(scan, self.cfg.min_camera_motion, min_run=2, min_consistency=self.cfg.min_camera_consistency):
                if seg.label in (CameraMovement.STATIC, CameraMovement.UNKNOWN):
                    continue
                start = max(scene.start_time, seg.start_time)
                end = min(scene.end_time, seg.end_time)
                if end - start < 0.4:
                    continue
                phrase = MOVEMENT_PHRASES[seg.label]
                out.append(
                    Event(
                        event_id="tmp",
                        video_id=video_id,
                        event_type=EventType.CAMERA,
                        start_time=start,
                        end_time=end,
                        event=sentence(phrase),
                        entities=["camera"],
                        action=str(seg.label),
                        scene_ids=[scene.scene_id],
                        frame_ids=frames_in(frames, start, end, [scene.scene_id], limit=6),
                        source=EventSource.MOTION,
                        source_ids=[scene.scene_id],
                        confidence=seg.consistency,
                        confidence_source=ConfidenceSource.MEASUREMENT,
                        boundary_precision="measured",
                        attributes={"mean_flow_magnitude": seg.mean_magnitude, "samples": seg.n_samples},
                    )
                )
        return out

    def _speech_events(self, video_id: str, transcript: Transcript, frames: list[Frame], scenes: list[Scene]) -> list[Event]:
        out: list[Event] = []
        for seg in transcript.segments:
            text = seg.text.strip()
            if len(text.split()) < self.cfg.speech_min_words:
                continue
            quoted = text if len(text) <= 200 else text[:197].rstrip() + "…"
            scene_ids = seg.scene_ids or [s.scene_id for s in scenes if s.start_time < seg.end and s.end_time > seg.start]
            out.append(
                Event(
                    event_id="tmp",
                    video_id=video_id,
                    event_type=EventType.SPEECH,
                    start_time=seg.start,
                    end_time=max(seg.end, seg.start + 0.2),
                    event=f'A speaker says: "{quoted}"',
                    entities=["speaker"],
                    action="speaking",
                    scene_ids=scene_ids,
                    frame_ids=frames_in(frames, seg.start, seg.end, scene_ids or None, limit=4),
                    source=EventSource.ASR,
                    source_ids=[seg.segment_id],
                    confidence=seg.confidence,
                    confidence_source=ConfidenceSource(seg.confidence_source) if seg.confidence is not None else ConfidenceSource.UNAVAILABLE,
                    boundary_precision="asr",
                    attributes={"text": text, "language": transcript.language},
                )
            )
        return out

    def _ocr_events(self, video_id: str, ocr: OCRResult, frames: list[Frame], duration: float) -> list[Event]:
        out: list[Event] = []
        tracks = ocr.tracks
        if not self.cfg.include_static_overlay_text:
            from video_dataset.ocr.merge import classify_tracks

            total_scenes = len({f.scene_id for f in frames})
            tracks, overlays = classify_tracks(tracks, total_scenes)
            skipped = [t for t in tracks if not t.is_event_worthy]
            if skipped:
                log.info("ocr: %d of %d text tracks are static overlays %s or fragments; not emitted as events", len(skipped), len(tracks), overlays)
        for tr in tracks:
            if not tr.is_event_worthy:
                continue
            end = max(tr.last_seen, tr.first_seen + 0.5)
            out.append(
                Event(
                    event_id="tmp",
                    video_id=video_id,
                    event_type=EventType.TEXT_ON_SCREEN,
                    start_time=tr.first_seen,
                    end_time=min(duration, end),
                    event=f'On-screen text "{tr.text}" is visible.',
                    entities=[tr.text],
                    action="text_appears",
                    scene_ids=tr.scene_ids,
                    frame_ids=tr.frame_ids[:8],
                    source=EventSource.OCR,
                    source_ids=[tr.track_id],
                    confidence=tr.mean_confidence,
                    confidence_source=ConfidenceSource.OCR_SCORE if tr.mean_confidence is not None else ConfidenceSource.UNAVAILABLE,
                    boundary_precision="frame",
                    attributes={"num_detections": tr.num_detections, "bbox": tr.bbox},
                )
            )
        return out

    def _audio_events(self, video_id: str, audio: AudioAnalysis, transcript: Transcript | None, frames: list[Frame], scenes: list[Scene], duration: float) -> list[Event]:
        out: list[Event] = []
        speech = [(s.start, s.end) for s in (transcript.segments if transcript else [])]
        for ae in audio.events:
            if ae.category in (AudioEventCategory.SILENCE, AudioEventCategory.SPEECH):
                continue
            length = max(1e-3, ae.end - ae.start)
            overlap = sum(interval_overlap(ae.start, ae.end, s, e) for s, e in speech)
            if ae.category == AudioEventCategory.NON_SPEECH_SOUND and overlap / length > 0.6:
                continue  # mostly speech; already covered by ASR
            if ae.category == AudioEventCategory.NON_SPEECH_SOUND and duration and length > 0.5 * duration:
                continue  # "sound present" for most of the video says nothing about *when* anything happens
            if ae.category == AudioEventCategory.NON_SPEECH_SOUND:
                text = "Non-speech sound is audible (type not identified)."
                entities = ["sound"]
            else:
                text = f"{ae.label} is audible."
                entities = [ae.label.lower()]
            scene_ids = [s.scene_id for s in scenes if s.start_time < ae.end and s.end_time > ae.start]
            out.append(
                Event(
                    event_id="tmp",
                    video_id=video_id,
                    event_type=EventType.SOUND,
                    start_time=ae.start,
                    end_time=min(duration, ae.end),
                    event=text,
                    entities=entities,
                    action="sound",
                    scene_ids=scene_ids,
                    frame_ids=frames_in(frames, ae.start, ae.end, scene_ids or None, limit=4),
                    source=EventSource.AUDIO,
                    source_ids=[ae.event_id],
                    confidence=ae.score,
                    confidence_source=ConfidenceSource(ae.confidence_source) if ae.score is not None else ConfidenceSource.UNAVAILABLE,
                    boundary_precision="measured",
                    attributes={"category": str(ae.category), "label": ae.label, "speech_overlap": round(overlap / length, 3)},
                )
            )
        return out

    def _state_change_events(self, video_id: str, scenes: list[Scene], frames: list[Frame], analyses: dict[str, SceneAnalysis]) -> list[Event]:
        out: list[Event] = []
        for i in range(1, len(scenes)):
            prev, cur = scenes[i - 1], scenes[i]
            a, b = analyses.get(prev.scene_id), analyses.get(cur.scene_id)
            if a is None or b is None:
                continue
            changes: list[tuple[str, str, str, float | None, ConfidenceSource]] = []
            # model-derived attributes (only when both sides are known)
            for attr, va, vb in (
                ("setting", a.environment.setting, b.environment.setting),
                ("location", a.environment.location, b.environment.location),
                ("time of day", a.environment.time_of_day, b.environment.time_of_day),
                ("weather", a.environment.weather, b.environment.weather),
            ):
                if va and vb and str(va) != str(vb) and str(va) != Setting.UNKNOWN and str(vb) != Setting.UNKNOWN and str(va).lower() != "unknown" and str(vb).lower() != "unknown":
                    confs = [c for c in (a.confidence, b.confidence) if c is not None]
                    conf = round(min(confs), 3) if len(confs) == 2 else None
                    changes.append((attr, str(va), str(vb), conf, ConfidenceSource.DERIVED_MIN if conf is not None else ConfidenceSource.UNAVAILABLE))
            # measured lighting level (brightness) - confidence from the size of the brightness jump
            ma, mb = a.measurements, b.measurements
            if ma and mb and ma.lighting_level and mb.lighting_level and ma.lighting_level != mb.lighting_level and ma.brightness_mean is not None and mb.brightness_mean is not None:
                delta = abs(mb.brightness_mean - ma.brightness_mean)
                changes.append(("lighting", ma.lighting_level, mb.lighting_level, round(clamp(delta / 80.0, 0.0, 1.0), 3), ConfidenceSource.MEASUREMENT))
            for attr, va, vb, conf, src in changes:
                t = cur.start_time
                out.append(
                    Event(
                        event_id="tmp",
                        video_id=video_id,
                        event_type=EventType.STATE_CHANGE,
                        start_time=max(prev.start_time, t - 1.0),
                        end_time=min(cur.end_time, t + 1.0),
                        event=f"The {attr} changes from {va} to {vb}.",
                        entities=[attr],
                        action="changes",
                        scene_ids=[prev.scene_id, cur.scene_id],
                        frame_ids=frames_in(frames, prev.start_time, prev.end_time, [prev.scene_id], limit=1) + frames_in(frames, cur.start_time, cur.end_time, [cur.scene_id], limit=1),
                        source=EventSource.DERIVED,
                        source_ids=[prev.scene_id, cur.scene_id],
                        confidence=conf,
                        confidence_source=src,
                        boundary_precision="frame",
                        attributes={"attribute": attr, "from": va, "to": vb, "boundary_time": t},
                    )
                )
        return out
