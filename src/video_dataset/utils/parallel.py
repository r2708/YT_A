"""Intra-stage parallelism helpers.

Stages that loop over many small independent items (clips, frames, scenes, verifications) run those
items on a thread pool. Threads are enough because the heavy work happens outside the GIL: FFmpeg
and ffprobe are subprocesses, OpenCV / ONNX runtime release the GIL, and API calls wait on the
network. Pure-Python loops gain nothing from this and are left sequential.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

T = TypeVar("T")
R = TypeVar("R")

MAX_AUTO_WORKERS = 4  # safe default for an 8 GB laptop; raise pipeline.stage_workers explicitly on bigger boxes


def auto_workers(cap: int = MAX_AUTO_WORKERS) -> int:
    return max(1, min(cap, os.cpu_count() or 1))


def resolve_workers(override: int | None, default: int | None, n_items: int | None = None, concurrent: int = 1) -> int:
    """Per-stage override wins; 0 / None fall back to ``pipeline.stage_workers``; 0 / None there means
    auto, which shares min(4, cores) between the ``concurrent`` videos the runner processes at once.
    Never more workers than items."""
    w = override if override else default
    if not w or int(w) <= 0:
        w = max(1, auto_workers() // max(1, int(concurrent or 1)))
    else:
        w = int(w)
    if n_items is not None:
        w = max(1, min(w, int(n_items)))
    return max(1, w)


def stage_workers(ctx: Any, section: Any, n_items: int | None = None) -> int:
    """Thread count for one stage of one video: ``<section>.workers`` override, else
    ``pipeline.stage_workers``; auto shares the cores between the videos the runner is processing
    concurrently (``ctx.concurrency``)."""
    pipe = getattr(getattr(ctx, "config", None), "pipeline", None)
    return resolve_workers(
        getattr(section, "workers", None), getattr(pipe, "stage_workers", 0), n_items,
        concurrent=int(getattr(ctx, "concurrency", 1) or 1),
    )


def parallel_map(fn: Callable[[T], R], items: Iterable[T], workers: int, on_done: Callable[[int, R], None] | None = None) -> list[R]:
    """``[fn(x) for x in items]`` on ``workers`` threads, results in input order.

    Exceptions propagate exactly as they would in the list comprehension (the first one raised wins),
    so callers keep their existing per-item try/except inside ``fn``. ``on_done(index, result)`` is
    called on the calling thread as results are collected, for progress lines."""
    seq = list(items)
    if workers <= 1 or len(seq) <= 1:
        out: list[R] = []
        for i, x in enumerate(seq):
            r = fn(x)
            if on_done:
                on_done(i, r)
            out.append(r)
        return out
    results: list[R] = []
    with ThreadPoolExecutor(max_workers=min(workers, len(seq))) as pool:
        for i, r in enumerate(pool.map(fn, seq)):
            if on_done:
                on_done(i, r)
            results.append(r)
    return results


def chunk_evenly(seq: list[T], n_chunks: int) -> list[list[T]]:
    """Split a list into at most ``n_chunks`` contiguous, near-equal pieces (no empty chunks)."""
    n_chunks = max(1, min(n_chunks, len(seq)))
    size, extra = divmod(len(seq), n_chunks)
    out: list[list[T]] = []
    start = 0
    for i in range(n_chunks):
        end = start + size + (1 if i < extra else 0)
        out.append(seq[start:end])
        start = end
    return [c for c in out if c]
