"""Intra-stage parallel helpers and the worker-count policy."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from video_dataset.utils import parallel as par
from video_dataset.utils.parallel import chunk_evenly, parallel_map, resolve_workers, stage_workers


def test_parallel_map_keeps_order_and_reports_progress():
    seen: list[tuple[int, int]] = []
    threads: set[str] = set()

    def work(x: int) -> int:
        threads.add(threading.current_thread().name)
        time.sleep(0.01 * (5 - x))  # later items finish first
        return x * 10

    out = parallel_map(work, range(5), workers=4, on_done=lambda i, r: seen.append((i, r)))
    assert out == [0, 10, 20, 30, 40]
    assert seen == [(0, 0), (1, 10), (2, 20), (3, 30), (4, 40)]
    assert len(threads) > 1
    assert parallel_map(work, range(3), workers=1) == [0, 10, 20]  # sequential path, same result
    assert parallel_map(work, [], workers=4) == []


def test_parallel_map_propagates_exceptions():
    def boom(x: int) -> int:
        if x == 2:
            raise ValueError("item 2")
        return x

    with pytest.raises(ValueError, match="item 2"):
        parallel_map(boom, range(4), workers=3)


def test_chunk_evenly():
    assert chunk_evenly(list(range(10)), 3) == [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert chunk_evenly(list(range(2)), 5) == [[0], [1]]
    assert chunk_evenly([], 3) == []
    assert [x for c in chunk_evenly(list(range(7)), 4) for x in c] == list(range(7))


def test_resolve_workers_policy(monkeypatch):
    monkeypatch.setattr(par, "auto_workers", lambda cap=4: 4)
    assert resolve_workers(None, 0) == 4  # auto
    assert resolve_workers(None, 0, concurrent=8) == 1  # 8 videos in flight -> one thread each
    assert resolve_workers(None, 0, concurrent=2) == 2
    assert resolve_workers(None, 6) == 6  # explicit pipeline value is taken as is
    assert resolve_workers(2, 6) == 2  # per-stage override wins
    assert resolve_workers(None, 6, n_items=3) == 3  # never more workers than items
    assert resolve_workers(None, 6, n_items=0) == 1


def test_stage_workers_reads_context(monkeypatch):
    monkeypatch.setattr(par, "auto_workers", lambda cap=4: 4)
    ctx = SimpleNamespace(config=SimpleNamespace(pipeline=SimpleNamespace(stage_workers=0, workers=8)), concurrency=1)
    assert stage_workers(ctx, SimpleNamespace(workers=None), 100) == 4  # one video running -> full pool
    ctx.concurrency = 4
    assert stage_workers(ctx, SimpleNamespace(workers=None), 100) == 1
    assert stage_workers(ctx, SimpleNamespace(workers=3), 100) == 3
    ctx.config.pipeline.stage_workers = 2
    ctx.concurrency = 1
    assert stage_workers(ctx, SimpleNamespace(workers=None), 100) == 2
