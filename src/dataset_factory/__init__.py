"""dataset_factory: curation of verified Text-to-Video and Image-to-Video training records from
the youtube-video-dataset exports produced by ``video_dataset``.

One canonical, versioned manifest (SQLite + Parquet/JSONL) is the source of truth; T2V and I2V
exports are derived from it by versioned adapters. Everything runs offline by default, one clip at
a time, and never modifies the source records or media.
"""

from __future__ import annotations

__version__ = "0.1.0"
PIPELINE_VERSION = __version__
