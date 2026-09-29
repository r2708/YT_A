"""A PID lock so only one pipeline process 'owns' the state DB at a time.

`recover_interrupted()` turns every RUNNING stage into FAILED('interrupted'). That is right after a
crash, but wrong while a batch is still alive in another process: a `status` check from a second
terminal used to clobber the live run's bookkeeping. The lock file records the pid of the process
that is executing stages; a stale lock (dead pid) is ignored and overwritten.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

log = logging.getLogger("video_dataset.runlock")

LOCK_NAME = "run.lock"


def pid_alive(pid: int) -> bool:
    """Portable liveness check. Never uses os.kill on Windows (any signal but CTRL_* terminates there)."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        process_query_limited_information = 0x1000
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return False
        try:
            still_active = 259
            code = ctypes.c_ulong()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return code.value == still_active
            return True
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class RunLock:
    def __init__(self, data_dir: str | Path):
        self.path = Path(data_dir) / LOCK_NAME
        self.pid = os.getpid()
        self.held = False

    def owner(self) -> int | None:
        """Pid recorded in the lock file, or None if there is no (readable) lock."""
        try:
            return int(self.path.read_text().strip())
        except (FileNotFoundError, ValueError, OSError):
            return None

    def live_owner(self) -> int | None:
        """Pid of another *running* process holding the lock; None if free, stale, or ours."""
        pid = self.owner()
        if pid is None or pid == self.pid:
            return None
        return pid if pid_alive(pid) else None

    def acquire(self) -> bool:
        """Take the lock unless a live foreign process holds it. Returns True when held by us."""
        other = self.live_owner()
        if other is not None:
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(str(self.pid))
        except OSError as exc:  # unwritable data dir: proceed without a lock rather than refuse to run
            log.warning("could not write %s: %s", self.path, exc)
            return False
        self.held = True
        return True

    def release(self) -> None:
        if not self.held:
            return
        self.held = False
        if self.owner() == self.pid:
            try:
                self.path.unlink()
            except OSError:
                pass
