"""Pick the caption to train on from what the source already provides, in configured priority
order, before anything is generated."""

from __future__ import annotations

from dataset_factory.schema.canonical import CanonicalRecord


def choose_caption(rec: CanonicalRecord, priority: list[str]) -> tuple[str | None, str | None]:
    """(caption text, source field name) - first non-empty candidate in ``priority``."""
    for field in priority:
        if field == "prompt_variants":
            for i, v in enumerate(rec.prompt_variants):
                if v and v.strip():
                    return v.strip(), f"prompt_variants[{i}]"
            continue
        value = getattr(rec, field, None)
        if isinstance(value, str) and value.strip():
            return value.strip(), field
    return None, None
