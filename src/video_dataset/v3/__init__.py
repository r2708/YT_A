"""V3: visual, training-ready Text-to-Video and Video-to-Video datasets built on top of the existing
annotation dataset (see youtube_dataset_v3_pipeline.md).

    legacy.py      read the existing JSONL / Parquet export without mutating it; inventory + source index
    rights.py      rights manifest: explicit booleans + evidence; nothing is processed without `approved`
    state.py       resumable job states (DISCOVERED ... COMPLETE, FAILED_RETRYABLE / FAILED_REVIEW)
    resolution.py  resolution buckets driven by the `resolution` config section
    media.py       authorized acquisition, probe, decode validation, exact clip extraction, clip metrics
    v2v.py         deterministic source -> target edits and pair validation
    pipeline.py    per-source job: rights gate -> acquire -> match legacy annotations -> clips -> T2V / V2V
    publish.py     aggregate, stage the two repositories, upload, read back + verify checksums
    cleanup.py     delete local media only after the remote copy is verified
    loader.py      one loader for both task datasets with a bounded, crash-safe local cache
"""

V3_SCHEMA_VERSION = "3.0.0"
