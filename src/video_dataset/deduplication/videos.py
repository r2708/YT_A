"""Duplicate video detection.

* exact: same YouTube id (blocked at registration) or byte-identical files (sha256 in the state DB)
* near: perceptual fingerprint - the set of dHashes of the low-resolution scan frames. A re-upload,
  re-encode or re-scaled copy of the same video shares most of those hashes; unrelated videos share
  almost none. Similarity = Jaccard overlap of the two hash sets.
"""

from __future__ import annotations

from collections.abc import Iterable

from video_dataset.storage.state_db import StateDB
from video_dataset.utils.imagehash import jaccard_hashes


def duplicate_videos(db: StateDB) -> dict[str, str]:
    """{video_id: canonical_video_id} for videos whose downloaded file hash matches an earlier video."""
    out: dict[str, str] = {}
    with db._lock:
        rows = db._conn.execute("SELECT sha256, video_id FROM file_hashes").fetchall()
    by_hash: dict[str, list[str]] = {}
    for r in rows:
        by_hash.setdefault(r["sha256"], []).append(r["video_id"])
    for _sha, vids in by_hash.items():
        if len(vids) > 1:
            canon = sorted(vids)[0]
            for v in vids:
                if v != canon:
                    out[v] = canon
    return out


def video_fingerprint(hashes: Iterable[str | None]) -> list[str]:
    """Sorted distinct hashes (None dropped)."""
    return sorted({h for h in hashes if h})


def fingerprint_similarity(a: Iterable[str], b: Iterable[str]) -> float:
    return jaccard_hashes(set(a), set(b))


def near_duplicate_videos(fingerprints: dict[str, Iterable[str]], threshold: float = 0.8, min_hashes: int = 8) -> dict[str, str]:
    """{video_id: canonical_video_id} over a set of fingerprints. The canonical video of a group is
    the one with the smallest id, so the mapping is stable across runs. Videos with fewer than
    ``min_hashes`` distinct hashes (very short / static content) are never matched."""
    sets = {vid: set(fp) for vid, fp in fingerprints.items()}
    ids = sorted(v for v, s in sets.items() if len(s) >= min_hashes)
    parent: dict[str, str] = {v: v for v in ids}

    def find(v: str) -> str:
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    # inverted index: hash -> videos, so only videos sharing at least one hash are compared
    index: dict[str, list[str]] = {}
    for v in ids:
        for h in sets[v]:
            index.setdefault(h, []).append(v)
    candidates: dict[str, set[str]] = {}
    for vids in index.values():
        if len(vids) > 1:
            for a in vids:
                candidates.setdefault(a, set()).update(vids)
    for a in ids:
        for b in sorted(candidates.get(a, ())):
            if b <= a:
                continue
            if jaccard_hashes(sets[a], sets[b]) >= threshold:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    out: dict[str, str] = {}
    for v in ids:
        root = find(v)
        if root != v:
            out[v] = root
    return out
