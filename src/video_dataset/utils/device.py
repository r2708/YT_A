"""Compute device resolution: CUDA -> MPS -> CPU, with an explicit override."""

from __future__ import annotations

from functools import lru_cache

from video_dataset.utils.logging import get_logger

log = get_logger("device")


@lru_cache(maxsize=8)
def resolve_device(preference: str | None = "auto") -> str:
    pref = (preference or "auto").lower()
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch optional
        log.debug("torch not importable (%s); using cpu", exc)
        return "cpu"
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        log.warning("CUDA requested but not available; falling back to CPU")
        return "cpu"
    if pref == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        log.warning("MPS requested but not available; falling back to CPU")
        return "cpu"
    # auto
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def torch_dtype(device: str, preference: str = "auto"):  # type: ignore[no-untyped-def]
    import torch

    if preference != "auto":
        return getattr(torch, preference)
    if device == "cuda":
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def empty_cache(device: str) -> None:
    try:
        import torch

        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
    except Exception as exc:  # pragma: no cover
        log.debug("empty_cache(%s) failed: %s", device, exc)


def is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "oom" in msg or type(exc).__name__ in {"OutOfMemoryError"}
