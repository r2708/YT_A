"""Automated validation: structural checks, grounding, verification, quality scoring, status assignment."""

from video_dataset.validation.checks import check_timespan
from video_dataset.validation.quality import description_quality_score
from video_dataset.validation.verifier import create_verifier

__all__ = ["check_timespan", "create_verifier", "description_quality_score"]
