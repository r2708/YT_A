"""Exceptions shared across stages.

Anything deriving from NonRetryableError is a deliberate policy decision (URL rejected, video too long,
disk full ...): the runner records it as FAILED immediately instead of burning stage retries on it.
"""

from __future__ import annotations


class NonRetryableError(RuntimeError):
    """A failure that retrying cannot fix."""


class VideoRejected(NonRetryableError):
    """The input violates a configured limit (URL policy, duration, size)."""


class InsufficientDiskSpace(NonRetryableError):
    """The data disk is below the configured free-space floor."""
