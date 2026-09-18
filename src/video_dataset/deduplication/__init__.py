"""Exact and near-duplicate detection for questions, events, descriptions and videos."""

from video_dataset.deduplication.exact import exact_duplicates
from video_dataset.deduplication.near import NearDuplicateIndex, near_duplicates
from video_dataset.deduplication.qa import dedupe_questions
from video_dataset.deduplication.records import dedupe_timed_texts

__all__ = ["NearDuplicateIndex", "dedupe_questions", "dedupe_timed_texts", "exact_duplicates", "near_duplicates"]
