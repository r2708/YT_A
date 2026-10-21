"""Persistent provider cache (SQLite). Key = asset fingerprint + input text + operation + template
version + provider + model + settings, so the same question is never asked twice."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from dataset_factory.state import now_utc


def cache_key(*, asset: str | None, text: str | None, operation: str, template_version: str, provider: str, model: str | None, settings: dict[str, Any] | None = None) -> str:
    payload = {"asset": asset, "text": text, "op": operation, "tv": template_version, "p": provider, "m": model, "s": settings or {}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class ProviderCache:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute("CREATE TABLE IF NOT EXISTS entries (key TEXT PRIMARY KEY, provider TEXT, model TEXT, operation TEXT, raw_output TEXT, result_json TEXT, created_at TEXT)")
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT result_json FROM entries WHERE key=?", (key,)).fetchone()
        if row:
            self.hits += 1
            return json.loads(row[0])
        self.misses += 1
        return None

    def put(self, key: str, *, provider: str, model: str | None, operation: str, raw_output: str, result: dict[str, Any]) -> None:
        self.conn.execute("INSERT OR REPLACE INTO entries VALUES (?,?,?,?,?,?,?)", (key, provider, model, operation, raw_output, json.dumps(result, ensure_ascii=False), now_utc()))
        self.conn.commit()

    def size(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0])

    def close(self) -> None:
        self.conn.close()
