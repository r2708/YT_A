# Source audit: raj270898/youtube-video-dataset

Measured from repository metadata and Parquet footers (no record download).

- files: 3531; shards: 6 (shard_0001, shard_0002, shard_0003, shard_0004, shard_0005, shard_0006)
- media files in the repository: 0 -> none: the repository holds annotation records only; clip_path values are relative to the producer's data directory (source.media_root)
- files outside the shards: 349 (examples: cinematic/clips.jsonl, cinematic/clips.parquet, cinematic/dataset.jsonl, cinematic/dataset.parquet, cinematic/events.jsonl)
- viewer rows per config: {'clips': 44190, 'events': 282597, 'frames': 142911, 'long_video_qa': 2655, 'scenes': 43517, 'sequences': 4138, 'temporal_qa': 11236, 'video_descriptions': 6002}
- videos in shard manifests: 326; rights_status: {'absent': 326}; license_status: {'standard': 319, 'creative_commons': 7}
- sample video metadata: {'video_id': 'vid_050e29691327', 'url': 'https://www.youtube.com/watch?v=_YltHpZ1MrI', 'license': None, 'license_status': 'standard', 'rights_status': None, 'channel': None, 'duration': 322.31619}

## clips

- rows (Parquet footers): 44190; viewer: 44190
- per shard: {'shard_0001': 6101, 'shard_0002': 5963, 'shard_0003': 9320, 'shard_0004': 8180, 'shard_0005': 7440, 'shard_0006': 7186}
- columns (45): split, tier, tier_reasons, subsets, canonical_video_id, source_group_id, schema_version, coverage, annotation_scope, generation_target, record_id, record_type, video_id, scene_id, clip_id, clip_path, start_time, end_time, media_duration, exact, actual_start, actual_end, boundary_error_seconds, boundary_check, fps, width, height, description, generation_prompt, prompt_variants, quality_flags, actions, camera, subject_motion, transcript, transcript_segments, transcript_scope, frame_ids, claims, provider, confidence, confidence_source, quality, validation, hard_negatives

## Interpretation

- Only `clips` rows are video-caption candidates; scenes / frames / events / QA are annotations, not training media.
- `clip_path` is relative to the producer's data directory; set `source.media_root` to where those files live. Without it every record is `media_validation_status: not_checked` and nothing is eligible.
- The repository licence covers annotations only; `license_status: standard` videos are `restricted` and never enter approved exports unless rights evidence is supplied.
