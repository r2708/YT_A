"""Duplicate video detection: same YouTube id (blocked at registration) or byte-identical files."""

from __future__ import annotations

from video_dataset.storage.state_db import StateDB


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
