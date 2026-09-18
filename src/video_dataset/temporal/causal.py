"""CAUSES relations are never derived from timestamps alone. This module defines the hook; the default
implementation emits nothing, and the optional LLM implementation labels its output as model-reported."""

from __future__ import annotations

import json
from typing import Protocol

from video_dataset.llm.base import LLMClient
from video_dataset.schemas.common import ConfidenceSource
from video_dataset.schemas.events import Event, EventType, RelationType, TemporalRelation
from video_dataset.utils.logging import get_logger
from video_dataset.vision.parsing import extract_json

log = get_logger("temporal.causal")


class CausalInferencer(Protocol):
    name: str

    def infer(self, video_id: str, events: list[Event], relations: list[TemporalRelation]) -> list[TemporalRelation]: ...


class NullCausalInferencer:
    name = "none"

    def infer(self, video_id: str, events: list[Event], relations: list[TemporalRelation]) -> list[TemporalRelation]:
        return []


_CAUSAL_TYPES = {EventType.ACTION, EventType.STATE_CHANGE, EventType.SOUND, EventType.APPEARANCE}


class LLMCausalInferencer:
    name = "llm"

    def __init__(self, client: LLMClient, max_pairs: int = 40, max_gap: float = 5.0):
        self.client = client
        self.max_pairs = max_pairs
        self.max_gap = max_gap

    def infer(self, video_id: str, events: list[Event], relations: list[TemporalRelation]) -> list[TemporalRelation]:
        by_id = {e.event_id: e for e in events}
        pairs = []
        for r in relations:
            if r.relation != RelationType.BEFORE or r.gap_seconds is None or r.gap_seconds > self.max_gap:
                continue
            a, b = by_id.get(r.event_a), by_id.get(r.event_b)
            if a and b and a.event_type in _CAUSAL_TYPES and b.event_type in _CAUSAL_TYPES:
                pairs.append((a, b))
        pairs = pairs[: self.max_pairs]
        if not pairs:
            return []
        listing = "\n".join(f"{i}. A: {a.event} ({a.start_time:.1f}-{a.end_time:.1f}s) -> B: {b.event} ({b.start_time:.1f}-{b.end_time:.1f}s)" for i, (a, b) in enumerate(pairs))
        prompt = (
            "Below are pairs of events observed in a video, A happening before B. For each pair decide whether A visibly and "
            "directly causes B (not just precedes it). Be conservative: most pairs are NOT causal.\n\n"
            f"{listing}\n\n"
            'Return JSON: {"pairs": [{"index": <int>, "causes": <bool>, "confidence": <0-1>, "rationale": "<short>"}]} and include every index.'
        )
        schema = {
            "type": "object",
            "properties": {"pairs": {"type": "array", "items": {"type": "object", "properties": {"index": {"type": "integer"}, "causes": {"type": "boolean"}, "confidence": {"type": "number"}, "rationale": {"type": "string"}}, "required": ["index", "causes", "confidence", "rationale"], "additionalProperties": False}}},
            "required": ["pairs"],
            "additionalProperties": False,
        }
        try:
            resp = self.client.complete(prompt, json_schema=schema, max_tokens=2000, temperature=0.0)
            data = extract_json(resp.text) or json.loads(resp.text)
        except Exception as exc:
            log.warning("causal inference failed: %s", exc)
            return []
        out: list[TemporalRelation] = []
        for item in data.get("pairs", []):
            try:
                idx = int(item["index"])
                if not item.get("causes") or not (0 <= idx < len(pairs)):
                    continue
                a, b = pairs[idx]
                conf = float(item.get("confidence", 0.0))
            except (KeyError, TypeError, ValueError):
                continue
            out.append(
                TemporalRelation(
                    relation_id="tmp",
                    video_id=video_id,
                    event_a=a.event_id,
                    event_b=b.event_id,
                    relation=RelationType.CAUSES,
                    event_a_text=a.event,
                    event_b_text=b.event,
                    gap_seconds=round(b.start_time - a.end_time, 3),
                    confidence=round(max(0.0, min(1.0, conf)), 3),
                    confidence_source=ConfidenceSource.MODEL_SELF_REPORT,
                    derivation=f"llm_inference:{self.client.model}",
                    rationale=str(item.get("rationale") or "")[:300] or None,
                )
            )
        return out
