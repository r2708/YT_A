"""SQLite processing state: which stage of which video is done, running, failed.

The database is the checkpoint system. Every stage marks itself RUNNING -> DONE/FAILED, records
its artifact path and metrics, and the runner skips DONE stages on resume.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from video_dataset.schemas.common import utc_now_iso
from video_dataset.stages import STAGE_ORDER, Stage, StageStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id     TEXT PRIMARY KEY,
    youtube_id   TEXT,
    url          TEXT NOT NULL,
    source_type  TEXT NOT NULL DEFAULT 'youtube',
    title        TEXT,
    duration     REAL,
    status       TEXT NOT NULL DEFAULT 'pending',
    last_error   TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_videos_youtube ON videos(youtube_id);

CREATE TABLE IF NOT EXISTS stages (
    video_id         TEXT NOT NULL,
    stage            TEXT NOT NULL,
    status           TEXT NOT NULL,
    attempts         INTEGER NOT NULL DEFAULT 0,
    started_at       TEXT,
    finished_at      TEXT,
    duration_seconds REAL,
    error            TEXT,
    artifact_path    TEXT,
    metrics_json     TEXT,
    config_hash      TEXT,
    PRIMARY KEY (video_id, stage),
    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS stage_log (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL,
    stage    TEXT NOT NULL,
    level    TEXT NOT NULL,
    message  TEXT NOT NULL,
    ts       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_stage_log_video ON stage_log(video_id);

CREATE TABLE IF NOT EXISTS file_hashes (
    sha256   TEXT PRIMARY KEY,
    video_id TEXT NOT NULL
);
"""


@dataclass
class StageRecord:
    video_id: str
    stage: Stage
    status: StageStatus
    attempts: int = 0
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    error: str | None = None
    artifact_path: str | None = None
    metrics: dict[str, Any] | None = None
    config_hash: str | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> StageRecord:
        return cls(
            video_id=row["video_id"],
            stage=Stage(row["stage"]),
            status=StageStatus(row["status"]),
            attempts=row["attempts"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            duration_seconds=row["duration_seconds"],
            error=row["error"],
            artifact_path=row["artifact_path"],
            metrics=json.loads(row["metrics_json"]) if row["metrics_json"] else None,
            config_hash=row["config_hash"],
        )


class StateDB:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ------------------------------------------------------------------ videos
    def upsert_video(
        self,
        video_id: str,
        url: str,
        youtube_id: str | None = None,
        source_type: str = "youtube",
        title: str | None = None,
        duration: float | None = None,
    ) -> None:
        now = utc_now_iso()
        with self._lock:
            self._conn.execute(
                """INSERT INTO videos (video_id, youtube_id, url, source_type, title, duration, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                   ON CONFLICT(video_id) DO UPDATE SET
                     url = excluded.url,
                     youtube_id = COALESCE(excluded.youtube_id, videos.youtube_id),
                     title = COALESCE(excluded.title, videos.title),
                     duration = COALESCE(excluded.duration, videos.duration),
                     updated_at = excluded.updated_at""",
                (video_id, youtube_id, url, source_type, title, duration, now, now),
            )
            self._conn.commit()

    def update_video_meta(self, video_id: str, title: str | None = None, duration: float | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE videos SET title = COALESCE(?, title), duration = COALESCE(?, duration), updated_at = ? WHERE video_id = ?",
                (title, duration, utc_now_iso(), video_id),
            )
            self._conn.commit()

    def set_video_status(self, video_id: str, status: str, error: str | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE videos SET status = ?, last_error = ?, updated_at = ? WHERE video_id = ?",
                (status, error, utc_now_iso(), video_id),
            )
            self._conn.commit()

    def get_video(self, video_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,)).fetchone()
        return dict(row) if row else None

    def find_by_youtube_id(self, youtube_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM videos WHERE youtube_id = ?", (youtube_id,)).fetchone()
        return dict(row) if row else None

    def list_videos(self, status: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if status:
                rows = self._conn.execute("SELECT * FROM videos WHERE status = ? ORDER BY created_at", (status,)).fetchall()
            else:
                rows = self._conn.execute("SELECT * FROM videos ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]

    def delete_video(self, video_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))
            self._conn.execute("DELETE FROM stage_log WHERE video_id = ?", (video_id,))
            self._conn.execute("DELETE FROM file_hashes WHERE video_id = ?", (video_id,))
            self._conn.commit()

    # ------------------------------------------------------------------ stages
    def get_stage(self, video_id: str, stage: Stage) -> StageRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM stages WHERE video_id = ? AND stage = ?", (video_id, stage.value)
            ).fetchone()
        return StageRecord.from_row(row) if row else None

    def get_stages(self, video_id: str) -> dict[Stage, StageRecord]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM stages WHERE video_id = ?", (video_id,)).fetchall()
        return {Stage(r["stage"]): StageRecord.from_row(r) for r in rows}

    def stage_status(self, video_id: str, stage: Stage) -> StageStatus:
        rec = self.get_stage(video_id, stage)
        return rec.status if rec else StageStatus.PENDING

    def is_done(self, video_id: str, stage: Stage) -> bool:
        return self.stage_status(video_id, stage) == StageStatus.DONE

    def start_stage(self, video_id: str, stage: Stage) -> None:
        now = utc_now_iso()
        with self._lock:
            self._conn.execute(
                """INSERT INTO stages (video_id, stage, status, attempts, started_at, finished_at, error)
                   VALUES (?, ?, 'RUNNING', 1, ?, NULL, NULL)
                   ON CONFLICT(video_id, stage) DO UPDATE SET
                     status = 'RUNNING', attempts = stages.attempts + 1, started_at = excluded.started_at,
                     finished_at = NULL, error = NULL""",
                (video_id, stage.value, now),
            )
            self._conn.execute("UPDATE videos SET status = 'processing', updated_at = ? WHERE video_id = ?", (now, video_id))
            self._conn.commit()

    def finish_stage(
        self,
        video_id: str,
        stage: Stage,
        artifact_path: str | None = None,
        metrics: dict[str, Any] | None = None,
        config_hash: str | None = None,
        duration_seconds: float | None = None,
        status: StageStatus = StageStatus.DONE,
    ) -> None:
        now = utc_now_iso()
        with self._lock:
            self._conn.execute(
                """INSERT INTO stages (video_id, stage, status, attempts, finished_at, duration_seconds, artifact_path, metrics_json, config_hash)
                   VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?)
                   ON CONFLICT(video_id, stage) DO UPDATE SET
                     status = excluded.status, finished_at = excluded.finished_at, duration_seconds = excluded.duration_seconds,
                     artifact_path = excluded.artifact_path, metrics_json = excluded.metrics_json,
                     config_hash = excluded.config_hash, error = NULL""",
                (
                    video_id,
                    stage.value,
                    status.value,
                    now,
                    duration_seconds,
                    artifact_path,
                    json.dumps(metrics, default=str) if metrics is not None else None,
                    config_hash,
                ),
            )
            self._conn.commit()

    def fail_stage(self, video_id: str, stage: Stage, error: str, duration_seconds: float | None = None) -> None:
        now = utc_now_iso()
        with self._lock:
            self._conn.execute(
                """INSERT INTO stages (video_id, stage, status, attempts, finished_at, duration_seconds, error)
                   VALUES (?, ?, 'FAILED', 1, ?, ?, ?)
                   ON CONFLICT(video_id, stage) DO UPDATE SET
                     status = 'FAILED', finished_at = excluded.finished_at,
                     duration_seconds = excluded.duration_seconds, error = excluded.error""",
                (video_id, stage.value, now, duration_seconds, error[:4000]),
            )
            self._conn.execute(
                "UPDATE videos SET status = 'failed', last_error = ?, updated_at = ? WHERE video_id = ?",
                (error[:1000], now, video_id),
            )
            self._conn.commit()

    def reset_stages(self, video_id: str, from_stage: Stage | None = None) -> list[Stage]:
        """Mark a stage and everything downstream PENDING (used by --force / retry)."""
        targets = STAGE_ORDER if from_stage is None else STAGE_ORDER[STAGE_ORDER.index(from_stage) :]
        with self._lock:
            for st in targets:
                self._conn.execute(
                    "UPDATE stages SET status = 'PENDING', error = NULL WHERE video_id = ? AND stage = ?",
                    (video_id, st.value),
                )
            self._conn.execute("UPDATE videos SET status = 'pending', last_error = NULL WHERE video_id = ?", (video_id,))
            self._conn.commit()
        return targets

    def recover_interrupted(self) -> int:
        """Stages left RUNNING by a crashed process become FAILED('interrupted') so they re-run."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE stages SET status = 'FAILED', error = 'interrupted (process stopped while running)' WHERE status = 'RUNNING'"
            )
            self._conn.commit()
            return cur.rowcount

    def failed_stages(self) -> list[StageRecord]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM stages WHERE status = 'FAILED' ORDER BY video_id").fetchall()
        return [StageRecord.from_row(r) for r in rows]

    def first_incomplete_stage(self, video_id: str) -> Stage | None:
        stages = self.get_stages(video_id)
        for st in STAGE_ORDER:
            rec = stages.get(st)
            if rec is None or rec.status != StageStatus.DONE:
                return st
        return None

    # ------------------------------------------------------------------ logging / stats
    def log(self, video_id: str, stage: Stage | str, level: str, message: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO stage_log (video_id, stage, level, message, ts) VALUES (?, ?, ?, ?, ?)",
                (video_id, str(stage), level, message[:4000], utc_now_iso()),
            )
            self._conn.commit()

    def get_log(self, video_id: str, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM stage_log WHERE video_id = ? ORDER BY id DESC LIMIT ?", (video_id, limit)
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def summary(self) -> dict[str, Any]:
        with self._lock:
            videos = self._conn.execute("SELECT status, COUNT(*) AS n FROM videos GROUP BY status").fetchall()
            stages = self._conn.execute("SELECT stage, status, COUNT(*) AS n FROM stages GROUP BY stage, status").fetchall()
            total = self._conn.execute("SELECT COUNT(*) AS n FROM videos").fetchone()["n"]
        per_stage: dict[str, dict[str, int]] = {}
        for r in stages:
            per_stage.setdefault(r["stage"], {})[r["status"]] = r["n"]
        return {
            "videos_total": total,
            "videos_by_status": {r["status"]: r["n"] for r in videos},
            "stages": per_stage,
        }

    # ------------------------------------------------------------------ dedup helpers
    def register_file_hash(self, sha256: str, video_id: str) -> str | None:
        """Returns the video_id that already owns this hash (duplicate) or None if new."""
        with self._lock:
            row = self._conn.execute("SELECT video_id FROM file_hashes WHERE sha256 = ?", (sha256,)).fetchone()
            if row and row["video_id"] != video_id:
                return str(row["video_id"])
            self._conn.execute(
                "INSERT OR REPLACE INTO file_hashes (sha256, video_id) VALUES (?, ?)", (sha256, video_id)
            )
            self._conn.commit()
        return None

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> StateDB:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
