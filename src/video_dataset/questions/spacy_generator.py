"""spaCy-based question generation.

Each event sentence ("The red car enters a tunnel.") is dependency-parsed and its subject, root verb,
object and prepositional location are turned into temporally grounded questions ("What does the red
car enter around 6 seconds?", "When does the red car enter a tunnel?"). Speech and on-screen text
additionally get named-entity questions. Every question carries the event's evidence and inherits
its confidence, exactly like the template generator; nothing is asked that the event text and
timestamps do not answer.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from video_dataset.questions.generator import _conf
from video_dataset.questions.ocr_filter import unusable_text_events
from video_dataset.schemas.events import Event, EventType, Timeline
from video_dataset.schemas.qa import Difficulty, Evidence, QARecord, QAType
from video_dataset.schemas.scene import Scene
from video_dataset.schemas.transcript import Transcript
from video_dataset.schemas.vision import SceneAnalysis
from video_dataset.utils.ids import qa_id as make_qa_id
from video_dataset.utils.logging import get_logger
from video_dataset.utils.text import (
    approx_duration_phrase,
    approx_timestamp_phrase,
    lower_first,
    normalize_text,
    sentence,
)

if TYPE_CHECKING:
    from video_dataset.config import QAConfig

log = get_logger("questions.spacy")

INSTALL_HINT = "pip install 'video-dataset-pipeline[qa-spacy]' && python -m spacy download en_core_web_sm"
LOCATION_PREPS = {"in", "at", "on", "near", "inside", "outside", "along", "through", "under", "over", "behind", "beside", "across", "into", "onto", "by"}
ENTITY_WORDS = {"PERSON": "person", "GPE": "place", "LOC": "location", "ORG": "organization", "DATE": "date", "TIME": "time", "MONEY": "amount of money", "PRODUCT": "product", "EVENT": "event", "WORK_OF_ART": "title", "NORP": "group"}


@dataclass
class Parsed:
    subject: str | None = None
    verb: str | None = None  # root verb as written ("enters")
    verb_lemma: str | None = None  # "enter"
    predicate: str | None = None  # everything after the subject: "enters a tunnel"
    predicate_lemma: str | None = None  # "enter a tunnel"
    clause_q: str | None = None  # interrogative clause: "does the red car enter a tunnel" / "is a person visible"
    copula: bool = False
    obj: str | None = None
    location: str | None = None
    entities: list[tuple[str, str]] = field(default_factory=list)  # (text, label)


@dataclass
class _Cand:
    type: QAType
    question: str
    answer: str
    template_id: str
    timestamps: list[float]
    difficulty: Difficulty


def _span_text(token: Any) -> str:
    return token.doc[token.left_edge.i : token.right_edge.i + 1].text.strip()


def _aux_for(root: Any) -> str:
    return {"VBP": "do", "VBD": "did"}.get(root.tag_, "does")


class SpacyQAGenerator:
    """Question generator that uses spaCy dependency parsing and NER."""

    name = "spacy_v1"

    def __init__(self, cfg: QAConfig, model_name: str | None = None):
        self.cfg = cfg
        self.model_name: str = str(model_name or getattr(cfg, "spacy_model", "en_core_web_sm"))
        self._nlp: Any = None

    # ------------------------------------------------------------------ model
    def load(self) -> None:
        if self._nlp is not None:
            return
        try:
            import spacy
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError(f"qa.generator=spacy needs the 'spacy' package; {INSTALL_HINT}") from exc
        try:
            self._nlp = spacy.load(self.model_name)
        except OSError as exc:
            raise RuntimeError(f"spaCy model '{self.model_name}' is not installed; {INSTALL_HINT}") from exc
        log.info("spaCy model %s loaded", self.model_name)

    @classmethod
    def available(cls, model_name: str = "en_core_web_sm") -> bool:
        try:
            import spacy

            return spacy.util.is_package(model_name) or bool(spacy.util.get_package_path(model_name))
        except Exception:
            return False

    # ------------------------------------------------------------------ parsing
    def parse(self, text: str) -> Parsed:
        self.load()
        doc = self._nlp(text.strip())
        out = Parsed(entities=[(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ENTITY_WORDS])
        sents = list(doc.sents) or [doc[:]]
        sent = sents[0]
        root = sent.root
        if root.pos_ not in ("VERB", "AUX"):
            return out
        subj_tok = next((t for t in root.children if t.dep_ == "nsubj"), None)
        if subj_tok is None or subj_tok.pos_ == "PRON" or subj_tok.right_edge.i >= root.i:
            return out  # no usable subject, a pronoun, or an inverted clause
        subject = _span_text(subj_tok)
        out.subject = subject if subj_tok.left_edge.pos_ == "PROPN" or subj_tok.left_edge.ent_type_ else lower_first(subject)
        out.verb = root.text
        out.verb_lemma = root.lemma_
        out.copula = root.lemma_ == "be"
        # predicate: the sentence from the root onward, minus the trailing full stop; tokens between the
        # subject and the root (auxiliaries, negation, adverbs) belong to it too
        after = sent.doc[root.i : sent.end].text.strip().rstrip(".!?").strip()
        pre = sent.doc[subj_tok.right_edge.i + 1 : root.i].text.strip()
        rest = after[len(root.text):].strip()
        out.predicate = " ".join(x for x in (pre, after) if x)
        out.predicate_lemma = " ".join(x for x in (pre, root.lemma_, rest) if x)
        if out.copula:
            out.clause_q = " ".join(x for x in (root.text, out.subject, pre, rest) if x)
        elif pre and pre.split()[0].lower() in ("is", "are", "was", "were", "has", "have", "had", "can", "will", "would", "could", "should", "may", "might", "must"):
            out.clause_q = f"{pre.split()[0]} {out.subject} {' '.join(pre.split()[1:] + [after])}".strip()
        else:
            out.clause_q = f"{_aux_for(root)} {out.subject} {out.predicate_lemma}"
        obj_tok = next((t for t in root.children if t.dep_ in ("dobj", "attr", "oprd")), None)
        if obj_tok is not None and not out.copula:
            out.obj = _span_text(obj_tok)
        for prep in sent:
            if prep.dep_ == "prep" and prep.text.lower() in LOCATION_PREPS and (prep.head == root or prep.head.head == root) and prep.i > root.i:
                pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
                if pobj is not None:
                    out.location = f"{prep.text.lower()} {_span_text(pobj)}"
                    break
        return out

    # ------------------------------------------------------------------ candidates
    def _from_event(self, e: Event) -> list[_Cand]:
        t = round(e.start_time + e.duration / 2.0, 1)
        at = approx_timestamp_phrase(t)
        easy, medium = Difficulty.EASY, Difficulty.MEDIUM
        long_ev = e.duration > 10.0
        out: list[_Cand] = []
        when_answer = f"{approx_timestamp_phrase(e.start_time).capitalize()} (starts at {e.start_time:.1f}s)."

        if e.event_type == EventType.SPEECH:
            quoted = (e.attributes.get("text") or "").strip()
            if quoted:
                out.append(_Cand(QAType.TIMESTAMP, f"What does the speaker say {at}?", sentence(f'the speaker says "{quoted}"'), "spacy_speech_v1", [t], easy))
                for text, label in self.parse(quoted).entities[:2]:
                    out.append(_Cand(QAType.TIMESTAMP, f"Which {ENTITY_WORDS[label]} is mentioned in the speech {at}?", sentence(text), "spacy_speech_entity_v1", [t], medium))
            return out
        if e.event_type == EventType.TEXT_ON_SCREEN:
            text = (e.attributes.get("text") or "").strip() or e.event.rstrip(".")
            out.append(_Cand(QAType.TIMESTAMP, f"What text is shown on screen {at}?", sentence(text), "spacy_ocr_v1", [t], easy))
            for ent, label in self.parse(text).entities[:1]:
                out.append(_Cand(QAType.TIMESTAMP, f"Which {ENTITY_WORDS[label]} appears in the on-screen text {at}?", sentence(ent), "spacy_ocr_entity_v1", [t], medium))
            return out
        if e.event_type == EventType.CAMERA:
            out.append(_Cand(QAType.TIMESTAMP, f"How does the camera move {at}?", sentence(e.event), "spacy_camera_v1", [t], easy))
            return out
        if e.event_type not in (EventType.ACTION, EventType.APPEARANCE, EventType.SOUND):
            return out  # transitions and state changes are covered by the template generator

        if e.event_type == EventType.SOUND and e.attributes.get("label") == "sound_present":
            return out  # unnamed background sound is not an answer
        p = self.parse(e.event)
        if not (p.subject and p.verb and p.predicate and p.clause_q):
            return out
        subj, pred = p.subject, p.predicate
        if not p.copula:
            out.append(_Cand(QAType.TIMESTAMP, f"What does {subj} do {at}?", sentence(f"{subj} {pred}"), "spacy_subject_action_v1", [t], easy))
            if p.obj:
                out.append(_Cand(QAType.TIMESTAMP, f"What does {subj} {p.verb_lemma} {at}?", sentence(p.obj), "spacy_object_v1", [t], medium))
        out.append(_Cand(QAType.EVENT_LOCALIZATION, f"When {p.clause_q}?", when_answer, "spacy_when_v1", [], easy))
        if p.location:
            out.append(_Cand(QAType.TIMESTAMP, f"Where is {subj} {at}?", sentence(f"{subj} is {p.location}"), "spacy_location_v1", [t], medium))
        out.append(_Cand(QAType.TIMESTAMP, f"What {pred} {at}?", sentence(subj), "spacy_who_v1", [t], medium))
        if e.duration >= 1.0:
            q = f"Approximately how long {p.clause_q}?" if e.boundary_precision != "scene" else f"Approximately how long does the shot last in which {subj} {pred}?"
            out.append(_Cand(QAType.DURATION, q, f"{approx_duration_phrase(e.duration).capitalize()} (from {e.start_time:.1f}s to {e.end_time:.1f}s).", "spacy_duration_v1", [], medium if not long_ev else Difficulty.HARD))
        return out

    # ------------------------------------------------------------------ interface
    def generate(
        self,
        video_id: str,
        duration: float,
        timeline: Timeline,
        scenes: list[Scene],
        analyses: dict[str, SceneAnalysis],
        transcript: Transcript | None = None,
    ) -> list[QARecord]:
        """Same signature as ``TemporalQAGenerator.generate`` so the stage can swap generators."""
        self.load()
        rng = random.Random(f"{self.cfg.seed}:spacy:{video_id}")
        skip = unusable_text_events(timeline.events, shape_filter=bool(self.cfg.ocr_filter), min_chars=int(self.cfg.ocr_min_chars), max_repeats=int(self.cfg.ocr_max_repeats))
        events = [e for e in timeline.events if (self.cfg.allow_unscored_evidence or e.confidence is not None) and e.event_id not in skip]
        target = int(round(duration / 60.0 * float(self.cfg.questions_per_minute)))
        target = max(int(self.cfg.min_questions), min(int(self.cfg.max_questions), target))
        per_event = max(1, int(getattr(self.cfg, "spacy_max_per_event", 3)))

        seen: set[str] = set()
        chosen: list[tuple[Event, _Cand]] = []
        for e in events:
            try:
                cands = self._from_event(e)
            except Exception as exc:
                log.warning("spaCy parse failed for %s (%s)", e.event_id, str(exc)[:120])
                continue
            rng.shuffle(cands)
            kept = 0
            for c in cands:
                key = f"{c.type}|{normalize_text(c.question)}"
                if key in seen:
                    continue
                seen.add(key)
                chosen.append((e, c))
                kept += 1
                if kept >= per_event:
                    break
        if len(chosen) > target:
            rng.shuffle(chosen)
            chosen = chosen[:target]
        chosen.sort(key=lambda ec: (ec[0].start_time, ec[0].event_id, str(ec[1].type)))

        records: list[QARecord] = []
        for n, (e, c) in enumerate(chosen, start=1):
            conf, src = _conf([e])
            records.append(
                QARecord(
                    question_id=make_qa_id(video_id, n),
                    video_id=video_id,
                    type=c.type,
                    question=c.question,
                    answer=c.answer,
                    evidence=Evidence(
                        start_time=round(max(0.0, e.start_time), 3),
                        end_time=round(min(duration, max(e.end_time, e.start_time)), 3),
                        timestamps=[round(t, 3) for t in (c.timestamps or [e.start_time])],
                        scene_ids=list(e.scene_ids),
                        event_ids=[e.event_id],
                        frame_ids=list(e.frame_ids[:16]),
                        transcript_segment_ids=list(e.source_ids) if e.event_type == EventType.SPEECH else [],
                        ocr_track_ids=list(e.source_ids) if e.event_type == EventType.TEXT_ON_SCREEN else [],
                    ),
                    difficulty=c.difficulty,
                    confidence=conf,
                    confidence_source=src,
                    generator=self.name,
                    template_id=c.template_id,
                    is_long_range=False,
                )
            )
        log.info("spaCy generated %d questions from %d events (target %d)", len(records), len(events), target)
        return records
