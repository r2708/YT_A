"""Question dedup: identical or near-identical wording counts as a duplicate only when the questions
are about the same evidence. Similar wording about different moments is preserved."""

from __future__ import annotations

from video_dataset.config import DeduplicationConfig
from video_dataset.deduplication.near import NearDuplicateIndex
from video_dataset.schemas.qa import QARecord
from video_dataset.utils.text import jaccard, normalize_text, tokenize
from video_dataset.utils.timecode import interval_iou


def dedupe_questions(records: list[QARecord], cfg: DeduplicationConfig) -> dict[str, str]:
    index = NearDuplicateIndex(cfg.question_similarity_threshold, cfg.num_perm, cfg.shingle_size)
    by_id = {r.question_id: r for r in records}
    kept: list[str] = []
    dups: dict[str, str] = {}
    for r in sorted(records, key=lambda x: x.question_id):
        qn = normalize_text(r.question)
        an = normalize_text(r.answer)
        canon = None
        # exact (question, answer) duplicates regardless of evidence
        for k in kept:
            other = by_id[k]
            if normalize_text(other.question) == qn and normalize_text(other.answer) == an:
                canon = k
                break
        if canon is None:
            for k in index.query(r.question):
                other = by_id[k]
                if other.type != r.type:
                    continue
                same_events = bool(set(other.evidence.event_ids) & set(r.evidence.event_ids)) and set(other.evidence.event_ids) == set(r.evidence.event_ids)
                iou = interval_iou(other.evidence.start_time, other.evidence.end_time, r.evidence.start_time, r.evidence.end_time)
                answers_similar = jaccard(tokenize(other.answer), tokenize(r.answer)) >= 0.6
                if (same_events or iou >= cfg.evidence_iou_threshold) and answers_similar:
                    canon = k
                    break
        if canon is not None:
            dups[r.question_id] = canon
        else:
            kept.append(r.question_id)
            index.add(r.question_id, r.question)
    return dups
