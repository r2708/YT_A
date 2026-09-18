from __future__ import annotations

import hashlib

from video_dataset.utils.text import normalize_text


def text_key(text: str) -> str:
    return hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()


def exact_duplicates(items: list[tuple[str, str]]) -> dict[str, str]:
    """items = [(id, text)] -> {duplicate_id: canonical_id} (first occurrence is canonical)."""
    seen: dict[str, str] = {}
    dups: dict[str, str] = {}
    for item_id, text in items:
        k = text_key(text)
        if k in seen:
            dups[item_id] = seen[k]
        else:
            seen[k] = item_id
    return dups
