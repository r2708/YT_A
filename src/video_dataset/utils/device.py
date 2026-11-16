"""Compute device selection in one place: auto-detection (CUDA on Windows / Linux, MPS on Apple
Silicon, otherwise CPU) with an explicit, validated override. Nothing else in the code base may
hard-code ``cuda`` / ``.cuda()``; every model takes the string this module returns."""

from __future__ import annotations

import os
import platform
import sys
from typing import Any

from video_dataset.utils.logging import get_logger

log = get_logger("device")


ENV_DEVICE = "VIDEO_PIPELINE_DEVICE"  # auto | cpu | cuda | mps - overrides the configured device for this process
ENV_REQUIRE = "REQUIRE_ACCELERATOR"  # true: an unavailable requested accelerator is an error, not a CPU fallback
DEVICES = ("auto", "cpu", "cuda", "mps")

_require_accelerator = False  # set from project.require_accelerator by configure()
_fallback_reason: str | None = None
_logged: set[str] = set()


class AcceleratorUnavailable(RuntimeError):
    """The explicitly requested device is not usable and REQUIRE_ACCELERATOR is on."""


def configure(require_accelerator: bool = False) -> None:
    """Apply ``project.require_accelerator`` (the env var REQUIRE_ACCELERATOR=true does the same)."""
    global _require_accelerator
    _require_accelerator = bool(require_accelerator)


def _require() -> bool:
    return _require_accelerator or os.getenv(ENV_REQUIRE, "false").strip().lower() in ("1", "true", "yes", "on")


def _cuda_available(torch) -> bool:  # type: ignore[no-untyped-def]
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _mps_available(torch) -> bool:  # type: ignore[no-untyped-def]
    mps = getattr(torch.backends, "mps", None)
    try:
        return bool(mps and mps.is_available() and (not hasattr(mps, "is_built") or mps.is_built()))
    except Exception:
        return False


def _auto(torch, system: str) -> str:  # type: ignore[no-untyped-def]
    """Windows / Linux: NVIDIA CUDA, else CPU. macOS: Apple MPS, else CPU. DirectML and other
    backends are never picked automatically (workload-specific; only after they were tested)."""
    if system == "Darwin":
        if _mps_available(torch):
            return "mps"
        return "cuda" if _cuda_available(torch) else "cpu"
    return "cuda" if _cuda_available(torch) else "cpu"


def resolve_device(preference: str | None = "auto") -> str:
    """The device every model / tensor of this process must use. Order of precedence: the env var
    VIDEO_PIPELINE_DEVICE, then ``preference`` (project.device or a per-section override), then
    auto-detection. A requested accelerator that is not usable falls back to CPU with a warning, or
    raises :class:`AcceleratorUnavailable` when REQUIRE_ACCELERATOR / project.require_accelerator is on."""
    global _fallback_reason
    env = os.getenv(ENV_DEVICE, "").strip().lower()
    pref = (env or preference or "auto").strip().lower()
    if pref not in DEVICES:
        raise ValueError(f"device must be one of {', '.join(DEVICES)} (got '{pref}'; check {ENV_DEVICE} / project.device)")
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch optional
        if pref in ("cuda", "mps") and _require():
            raise AcceleratorUnavailable(f"device '{pref}' requested but torch is not importable ({exc})") from exc
        log.debug("torch not importable (%s); using cpu", exc)
        return "cpu"
    system = platform.system()
    if pref == "cpu":
        selected = "cpu"
    elif pref == "auto":
        selected = _auto(torch, system)
    else:
        available = _cuda_available(torch) if pref == "cuda" else _mps_available(torch)
        if available:
            selected = pref
        else:
            reason = f"requested device '{pref}' is not available on {system} (torch {torch.__version__})"
            if _require():
                raise AcceleratorUnavailable(reason + f"; {ENV_REQUIRE}=true")
            _fallback_reason = reason
            if reason not in _logged:
                _logged.add(reason)
                log.warning("%s; falling back to CPU", reason)
            selected = "cpu"
    key = f"{pref}->{selected}"
    if key not in _logged:
        _logged.add(key)
        log.info("device: %s", runtime_info(preference, _selected=selected))
    return selected


def runtime_info(preference: str | None = "auto", _selected: str | None = None) -> dict[str, Any]:
    """The ``runtime`` block of a run manifest: OS, device policy, selected device, GPU name and
    memory where the framework exposes them. Detected values only."""
    env = os.getenv(ENV_DEVICE, "").strip().lower()
    policy = env or (preference or "auto").lower()
    info: dict[str, Any] = {
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "device_policy": policy,
        "device_policy_source": "env" if env else "config",
        "selected_device": None,
        "gpu_name": None,
        "gpu_memory_bytes": None,
        "torch_version": None,
        "cuda_available": False,
        "mps_available": False,
        "require_accelerator": _require(),
        "accelerated_media_decode": False,  # FFmpeg runs in software; hardware decode is never required
        "fallback_reason": None,
    }
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        info.update(selected_device="cpu", fallback_reason=f"torch not importable: {exc}")
        return info
    info.update(torch_version=torch.__version__, cuda_available=_cuda_available(torch), mps_available=_mps_available(torch))
    try:
        selected = _selected or resolve_device(preference)
    except AcceleratorUnavailable as exc:
        info.update(selected_device=None, fallback_reason=str(exc))
        return info
    info["selected_device"] = selected
    if policy in ("cuda", "mps") and selected != policy:
        info["fallback_reason"] = _fallback_reason or f"requested device '{policy}' is not available"
    if selected == "cuda":
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_bytes"] = int(torch.cuda.get_device_properties(0).total_memory)
        except Exception as exc:  # pragma: no cover
            log.debug("could not read CUDA device properties: %s", exc)
    elif selected == "mps":
        info["gpu_name"] = f"Apple {platform.machine()} (MPS)"
        try:
            info["gpu_memory_bytes"] = int(torch.mps.recommended_max_memory())
        except Exception:
            pass
    return info


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
