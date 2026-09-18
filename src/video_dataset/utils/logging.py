"""Structured logging: rich console output plus a JSON-lines log per video."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_CONFIGURED = False
_LOGGER_NAME = "video_dataset"


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("video_id", "stage", "metrics"):
            val = getattr(record, key, None)
            if val is not None:
                payload[key] = val
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(level: str = "INFO", log_dir: str | Path | None = None, quiet: bool = False) -> logging.Logger:
    """Configure the package logger once. Safe to call repeatedly."""
    global _CONFIGURED
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if _CONFIGURED:
        return logger
    logger.propagate = False
    if not quiet:
        try:
            from rich.logging import RichHandler

            handler: logging.Handler = RichHandler(
                rich_tracebacks=False, show_path=False, markup=False, show_time=True, omit_repeated_times=False
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
        except Exception:  # pragma: no cover - rich always installed
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / "pipeline.log", encoding="utf-8")
        fh.setFormatter(JsonLineFormatter())
        logger.addHandler(fh)
    # Quieten noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "PIL", "numba", "faster_whisper", "yt_dlp", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    if not name:
        return logging.getLogger(_LOGGER_NAME)
    if name.startswith(_LOGGER_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")


class VideoLogAdapter(logging.LoggerAdapter):
    """Injects video_id / stage into every record so the JSON log is filterable."""

    def process(self, msg: Any, kwargs: Any) -> tuple[Any, Any]:
        extra = dict(self.extra or {})
        extra.update(kwargs.pop("extra", {}) or {})
        kwargs["extra"] = extra
        return msg, kwargs


@contextmanager
def video_log_file(video_id: str, log_dir: str | Path) -> Iterator[logging.Handler]:
    """Attach a per-video JSON-lines file handler for the duration of the block."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(Path(log_dir) / f"{video_id}.log", encoding="utf-8")
    handler.setFormatter(JsonLineFormatter())
    handler.addFilter(lambda rec: getattr(rec, "video_id", video_id) == video_id)
    logger = logging.getLogger(_LOGGER_NAME)
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        handler.close()


def stage_logger(video_id: str, stage: str) -> VideoLogAdapter:
    return VideoLogAdapter(get_logger(f"stage.{stage.lower()}"), {"video_id": video_id, "stage": stage})


def stage_line(stage: str, video_id: str, message: str = "", ok: bool | None = True) -> str:
    """Format the human readable progress line: `[SCENE] video_001 ✓ 37 scenes detected`."""
    mark = "✓" if ok else ("✗" if ok is False else "…")
    body = f" {message}" if message else ""
    return f"[{stage}] {video_id} {mark}{body}"
