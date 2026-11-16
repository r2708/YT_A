"""Resumable job state for V3 (Change 3). One row per source video in ``<work_dir>/state.sqlite``;
every transition is appended to ``history`` so nothing is skipped silently."""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path
from typing import Any

STATES = (
    "DISCOVERED", "RIGHTS_PENDING", "RIGHTS_APPROVED", "ACQUIRED", "PROBED", "DECODE_VALIDATED", "LEGACY_ANNOTATIONS_MATCHED",
    "CLIPS_EXTRACTED", "TASK_ASSETS_CREATED", "UPLOADED", "REMOTE_ASSETS_VERIFIED", "CLEANUP_ALLOWED", "COMPLETE",
)
FAILED_RETRYABLE = "FAILED_RETRYABLE"
FAILED_REVIEW = "FAILED_REVIEW"
ORDER = {s: i for i, s in enumerate(STATES)}


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


class JobStore:
    def __init__(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = Path(path)
        self.db = sqlite3.connect(str(path), timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                source_video_id TEXT PRIMARY KEY, state TEXT NOT NULL, failure TEXT, reason TEXT,
                attempts INTEGER NOT NULL DEFAULT 0, flags TEXT NOT NULL DEFAULT '{}', updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT, source_video_id TEXT NOT NULL, state TEXT NOT NULL,
                failure TEXT, reason TEXT, at TEXT NOT NULL
            );
            """
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    # ------------------------------------------------------------------ reads
    def get(self, vid: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT * FROM jobs WHERE source_video_id=?", (vid,)).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["flags"] = json.loads(out["flags"] or "{}")
        return out

    def all(self) -> list[dict[str, Any]]:
        rows = self.db.execute("SELECT * FROM jobs ORDER BY source_video_id").fetchall()
        return [{**dict(r), "flags": json.loads(r["flags"] or "{}")} for r in rows]

    def reached(self, vid: str, state: str) -> bool:
        job = self.get(vid)
        return bool(job) and ORDER.get(job["state"], -1) >= ORDER[state] and not job["failure"]  # type: ignore[index]

    def history(self, vid: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.execute("SELECT * FROM history WHERE source_video_id=? ORDER BY id", (vid,))]

    # ------------------------------------------------------------------ writes
    def discover(self, vid: str) -> None:
        if self.get(vid) is None:
            self._write(vid, "DISCOVERED", None, None, 0, {})

    def advance(self, vid: str, state: str, reason: str | None = None, **flags: Any) -> None:
        if state not in ORDER:
            raise ValueError(f"unknown state {state}")
        job = self.get(vid) or {"attempts": 0, "flags": {}}
        self._write(vid, state, None, reason, int(job["attempts"]), {**job["flags"], **flags})

    def fail(self, vid: str, reason: str, retryable: bool) -> None:
        """Keeps the last good state; the failure and its reason sit next to it."""
        job = self.get(vid) or {"state": "DISCOVERED", "attempts": 0, "flags": {}}
        self._write(vid, job["state"], FAILED_RETRYABLE if retryable else FAILED_REVIEW, reason, int(job["attempts"]) + 1, job["flags"])

    def set_flags(self, vid: str, **flags: Any) -> None:
        job = self.get(vid)
        if job is None:
            return
        merged = {**job["flags"], **flags}
        self.db.execute("UPDATE jobs SET flags=?, updated_at=? WHERE source_video_id=?", (json.dumps(merged, sort_keys=True), _now(), vid))
        self.db.commit()

    def reset(self, vid: str, state: str = "DISCOVERED") -> None:
        job = self.get(vid)
        if job is not None:
            self._write(vid, state, None, "reset", 0, {})

    def _write(self, vid: str, state: str, failure: str | None, reason: str | None, attempts: int, flags: dict[str, Any]) -> None:
        now = _now()
        self.db.execute(
            "INSERT INTO jobs(source_video_id, state, failure, reason, attempts, flags, updated_at) VALUES(?,?,?,?,?,?,?) "
            "ON CONFLICT(source_video_id) DO UPDATE SET state=excluded.state, failure=excluded.failure, reason=excluded.reason, "
            "attempts=excluded.attempts, flags=excluded.flags, updated_at=excluded.updated_at",
            (vid, state, failure, reason, attempts, json.dumps(flags, sort_keys=True), now),
        )
        self.db.execute("INSERT INTO history(source_video_id, state, failure, reason, at) VALUES(?,?,?,?,?)", (vid, state, failure, reason, now))
        self.db.commit()

    def summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for job in self.all():
            key = job["failure"] or job["state"]
            out[key] = out.get(key, 0) + 1
        return out
