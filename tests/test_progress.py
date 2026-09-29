"""ProgressLog / formatting helpers used by the terminal progress lines."""

from __future__ import annotations

import logging

from video_dataset.utils.progress import ProgressLog, fmt_bytes, fmt_seconds


def test_fmt_helpers() -> None:
    assert fmt_seconds(None) == "--:--"
    assert fmt_seconds(65) == "01:05"
    assert fmt_seconds(3700) == "1:01:40"
    assert fmt_bytes(512) == "512 B"
    assert fmt_bytes(2.5 * 1024 * 1024) == "2.5 MB"


def test_progress_is_throttled_and_finishes(caplog) -> None:  # type: ignore[no-untyped-def]
    log = logging.getLogger("video_dataset.test.progress")
    log.propagate = True
    with caplog.at_level(logging.INFO, logger="video_dataset.test.progress"):
        p = ProgressLog(log, "OCR", 100, unit="frames", every_seconds=1000)
        for i in range(1, 51):
            p.update(i, extra="x")
        assert len(caplog.records) == 1  # first eligible update logs, the rest are throttled
        assert "OCR 1/100 frames (1%)" in caplog.records[0].message
        p.finish("done")
        assert len(caplog.records) == 2
        assert "100/100 frames (100%)" in caplog.records[-1].message
        p.finish()  # idempotent
        assert len(caplog.records) == 2


def test_progress_duration_unit(caplog) -> None:  # type: ignore[no-untyped-def]
    log = logging.getLogger("video_dataset.test.progress2")
    with caplog.at_level(logging.INFO, logger="video_dataset.test.progress2"):
        p = ProgressLog(log, "transcription", 3600, unit="s", every_seconds=0)
        p.update(90, extra="3 segments")
        assert "01:30/1:00:00 (2%)" in caplog.records[-1].message
        assert "eta" in caplog.records[-1].message


def test_progress_without_total(caplog) -> None:  # type: ignore[no-untyped-def]
    log = logging.getLogger("video_dataset.test.progress3")
    with caplog.at_level(logging.INFO, logger="video_dataset.test.progress3"):
        p = ProgressLog(log, "scan", None, unit="frames", every_seconds=0)
        p.update(7)
        assert caplog.records[-1].message.startswith("scan 7 frames elapsed")
