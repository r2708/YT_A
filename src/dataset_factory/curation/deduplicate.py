"""Duplicate grouping. Exact groups (same SHA-256) keep one canonical record; the rest stay in
the manifest, traceable, excluded from approved exports. Perceptual candidates (fingerprint
similarity across *different* source videos) are flagged for review, never deleted."""

from __future__ import annotations

from collections.abc import Iterable

from dataset_factory.config import DedupRules
from dataset_factory.media.fingerprints import fingerprint_similarity
from dataset_factory.schema.canonical import CanonicalRecord


def group_duplicates(records: Iterable[CanonicalRecord], rules: DedupRules) -> dict[str, dict[str, object]]:
    """record_id -> {group, kind, canonical} for every record that belongs to a group."""
    recs = list(records)
    out: dict[str, dict[str, object]] = {}
    if rules.exact:
        by_sha: dict[str, list[CanonicalRecord]] = {}
        for r in recs:
            if r.video_sha256:
                by_sha.setdefault(r.video_sha256, []).append(r)
        for sha, members in by_sha.items():
            if len(members) < 2:
                continue
            members.sort(key=lambda r: r.record_id)
            for i, r in enumerate(members):
                out[r.record_id] = {"group": f"sha:{sha[:16]}", "kind": "exact", "canonical": i == 0}
    if rules.perceptual:
        with_fp = [r for r in recs if r.video_fingerprint and r.record_id not in out]
        # bucket by any shared hash to avoid the quadratic blow-up on large manifests
        index: dict[str, list[int]] = {}
        for i, r in enumerate(with_fp):
            for h in set(r.video_fingerprint):
                index.setdefault(h, []).append(i)
        parent = list(range(len(with_fp)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        seen_pairs: set[tuple[int, int]] = set()
        for bucket in index.values():
            if len(bucket) < 2:
                continue
            for a in bucket:
                for b in bucket:
                    if a >= b or (a, b) in seen_pairs:
                        continue
                    seen_pairs.add((a, b))
                    ra, rb = with_fp[a], with_fp[b]
                    if ra.canonical_video_id == rb.canonical_video_id and ra.original_video_id == rb.original_video_id:
                        continue  # shots of one video legitimately resemble each other
                    if fingerprint_similarity(ra.video_fingerprint, rb.video_fingerprint) >= rules.fingerprint_similarity:
                        parent[find(a)] = find(b)
        groups: dict[int, list[CanonicalRecord]] = {}
        for i, r in enumerate(with_fp):
            groups.setdefault(find(i), []).append(r)
        for members in groups.values():
            if len(members) < 2:
                continue
            members.sort(key=lambda r: r.record_id)
            gid = "near:" + members[0].record_id
            for i, r in enumerate(members):
                out[r.record_id] = {"group": gid, "kind": "near", "canonical": i == 0}
    return out
