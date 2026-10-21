"""The canonical manifest store: SQLite, one row per record, plus per-step completion markers so
every command is resumable and skips work whose inputs (asset, config section, pipeline version)
have not changed."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dataset_factory.schema.canonical import CanonicalRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    record_id TEXT PRIMARY KEY,
    source_record_hash TEXT,
    canonical_video_id TEXT,
    record_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_records_video ON records(canonical_video_id);
CREATE TABLE IF NOT EXISTS steps (
    record_id TEXT NOT NULL,
    step TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    detail_json TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (record_id, step)
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class StateStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(_SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> StateStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ records
    def upsert(self, rec: CanonicalRecord) -> None:
        self.conn.execute(
            "INSERT INTO records(record_id, source_record_hash, canonical_video_id, record_json, updated_at) VALUES (?,?,?,?,?) "
            "ON CONFLICT(record_id) DO UPDATE SET source_record_hash=excluded.source_record_hash, canonical_video_id=excluded.canonical_video_id, record_json=excluded.record_json, updated_at=excluded.updated_at",
            (rec.record_id, rec.source_record_hash, rec.canonical_video_id, json.dumps(rec.to_row(), ensure_ascii=False), now_utc()),
        )

    def commit(self) -> None:
        self.conn.commit()

    def get(self, record_id: str) -> CanonicalRecord | None:
        row = self.conn.execute("SELECT record_json FROM records WHERE record_id=?", (record_id,)).fetchone()
        return CanonicalRecord.model_validate(json.loads(row[0])) if row else None

    def source_hash(self, record_id: str) -> str | None:
        row = self.conn.execute("SELECT source_record_hash FROM records WHERE record_id=?", (record_id,)).fetchone()
        return row[0] if row else None

    def iter_records(self, limit: int | None = None, video_id: str | None = None) -> Iterator[CanonicalRecord]:
        """Streams rows ordered by record_id; never materialises the whole manifest."""
        sql = "SELECT record_json FROM records"
        params: tuple[Any, ...] = ()
        if video_id:
            sql += " WHERE canonical_video_id=?"
            params = (video_id,)
        sql += " ORDER BY record_id"
        if limit:
            sql += f" LIMIT {int(limit)}"
        cur = self.conn.execute(sql, params)
        while True:
            rows = cur.fetchmany(256)
            if not rows:
                return
            for (js,) in rows:
                yield CanonicalRecord.model_validate(json.loads(js))

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM records").fetchone()[0])

    def video_ids(self) -> list[str]:
        return [r[0] for r in self.conn.execute("SELECT DISTINCT canonical_video_id FROM records WHERE canonical_video_id IS NOT NULL ORDER BY 1")]

    # ------------------------------------------------------------------ steps
    def step_current(self, record_id: str, step: str, input_hash: str) -> bool:
        row = self.conn.execute("SELECT input_hash FROM steps WHERE record_id=? AND step=?", (record_id, step)).fetchone()
        return bool(row) and row[0] == input_hash

    def mark_step(self, record_id: str, step: str, input_hash: str, status: str, detail: dict[str, Any] | None = None) -> None:
        self.conn.execute(
            "INSERT INTO steps(record_id, step, input_hash, status, detail_json, updated_at) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(record_id, step) DO UPDATE SET input_hash=excluded.input_hash, status=excluded.status, detail_json=excluded.detail_json, updated_at=excluded.updated_at",
            (record_id, step, input_hash, status, json.dumps(detail or {}, ensure_ascii=False), now_utc()),
        )

    def step_counts(self, step: str) -> dict[str, int]:
        return {st: n for st, n in self.conn.execute("SELECT status, COUNT(*) FROM steps WHERE step=? GROUP BY status", (step,))}

    # ------------------------------------------------------------------ meta
    def set_meta(self, key: str, value: Any) -> None:
        self.conn.execute("INSERT INTO meta(key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, json.dumps(value, default=str)))
        self.conn.commit()

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default
