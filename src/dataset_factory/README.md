# dataset_factory

Curation of verified **Text-to-Video (T2V)** and **Image-to-Video (I2V)** training records from the
exports that `video_dataset` produces (the `raj270898/youtube-video-dataset` shards or a local
`final/` directory). One canonical, versioned manifest is the source of truth; T2V and I2V exports
are derived from it by versioned adapters. Everything is offline by default, runs one clip at a
time, and never modifies the source records or media.

## Pipeline flow

```
inspect ─► ingest ─► validate ─► extract-frames ─► captions review ─► deduplicate ─► curate ─► export ─► report
 (Hub      (records   (ffprobe +   (I2V start      (reuse source       (sha256 +     (rights,    (canonical,
  metadata  → SQLite   decode +     frame + sha256   captions, rule      perceptual    splits,     wan-t2v,
  only)     manifest)  frame        + timestamp)     checks, optional    candidates)   T2V/I2V     wan-i2v)
                       metrics)                      provider)                         eligibility)
```

Every stage streams from `curated/factory.sqlite`, checkpoints per record and skips records whose
inputs (asset bytes/mtime, the relevant config section, the pipeline version, the source row hash)
have not changed. Interrupt and re-run at any point.

## Setup (macOS, Apple Silicon, 8 GB)

```bash
brew install ffmpeg                      # ffprobe + ffmpeg (required for media checks)
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                         # installs video-dataset and dataset-factory
cp .env.example .env                     # HF_TOKEN only needed to read the Hub dataset
python -m dataset_factory --help
```

Memory notes: records are streamed in Parquet batches (`batch_size`), six downscaled frames per
clip are decoded for the checks, the manifest lives in SQLite, and nothing loads every clip or the
whole dataset into RAM. A full 44k-record ingest from the Hub needs ~2 GB of disk for the Parquet
files and well under 1 GB of RAM.

## Commands

```bash
# 1. audit the source without downloading records -> reports/source_audit.{md,json}
python -m dataset_factory inspect --source raj270898/youtube-video-dataset

# 2. ingest records (never media) - from the Hub, or from a local export
python -m dataset_factory ingest --limit 1000 --shards 1
python -m dataset_factory ingest --source-dir /path/to/final --set source.media_root=/path/to/data

# 3. rule-based media checks (no remote calls), I2V frames, captions, duplicates, eligibility
python -m dataset_factory validate --limit 1000
python -m dataset_factory extract-frames --limit 1000
python -m dataset_factory captions review --limit 100 --max-calls 100 --dry-run
python -m dataset_factory deduplicate
python -m dataset_factory curate

# 4. exports (versioned, atomic) and the report
python -m dataset_factory export --target canonical
python -m dataset_factory export --target wan-t2v
python -m dataset_factory export --target wan-i2v
python -m dataset_factory report
python -m dataset_factory show-config
```

`--config` picks another `dataset.yaml`; `--set a.b.c=value` overrides any key; `--limit` bounds
every stage; `--offline/--online` on `captions review` (the config's `offline: true` forbids any
provider but `none`).

## Where things live

| location | content |
|---|---|
| `config/factory/dataset.yaml` | paths, source, field mapping, rights policy, splits, provider |
| `config/factory/quality_rules.yaml` | media / caption / frame / duplicate thresholds (calibrate on a reviewed sample) |
| `config/factory/export_targets.yaml` | export targets and their packaging / resolution / duration policies |
| `<data_root>/raw/` | downloaded source Parquet + manifests, immutable |
| `<data_root>/curated/factory.sqlite` | the canonical manifest (records + per-step markers) |
| `<data_root>/curated/splits.json` | persisted group -> split, `split_changes.json` when something moved |
| `<data_root>/assets/frames/<record>/` | extracted I2V starting frames (PNG, checksummed) |
| `<data_root>/cache/captions.sqlite` | provider cache |
| `<data_root>/exports/<target>/<version>/` | promoted exports with `release_manifest.json` |
| `<data_root>/reports/summary.{md,json}` | measured counts per status / eligibility / coverage |

`data_root` defaults to `data/factory` (git-ignored) and can point anywhere.

## Source data and media

The Hub repository holds annotation records only (`shard_*/clips.parquet` etc.), **no video or
image files**. `clip_path` values (`clips/<video_id>/clip_NNN.mp4`) are relative to the producer's
data directory. Set `source.media_root` to that directory on the machine that still has the clips;
without it every record is `media_validation_status: not_checked` and nothing becomes eligible.
The factory never downloads from YouTube and never circumvents access controls.

## Canonical schema (v1.0.0)

`schema/canonical.py` is the definition; the key groups:

- identity and provenance: `record_id` (`<config>:<source record id>`), `source_dataset`,
  `source_config`, `source_split`, `source_record_id`, `source_record_hash`, `source_url`,
  `canonical_video_id` (= the producer's `source_group_id`), `original_video_id`, `source_payload`
  (every source field without a canonical home, verbatim)
- media: `video_path`, `video_sha256`, `video_fingerprint`, `start/end/duration_seconds`,
  `width`, `height`, `fps`, `frame_count`, `codec`
- captions: `caption_original`, `generation_prompt_original`, `prompt_variants` (untouched) and
  `caption_final`, `caption_source`, `caption_quality_status`, `caption_quality_score` +
  `caption_quality_score_basis`, `caption_review_method` (`rules` | `provider:<name>` | `human`),
  `caption_model`, `caption_prompt_version`
- I2V: `start_frame_path`, `start_frame_timestamp_seconds`, `start_frame_sha256`,
  `start_frame_width/height`, `start_frame_clip_sha256`, `start_frame_status`, `keyframe_paths`
- structure: `actions` (`{name, start_time_seconds, end_time_seconds, source}`), `subject_motion`,
  `camera_motion`, `shot_type`, `temporal_events`
- verdicts: `media_validation_status` + reasons, `visual_quality_status` + score + reasons,
  `duplicate_group_id` / `duplicate_kind` / `duplicate_canonical`, `source_rights_status` +
  `source_rights_evidence`, `validation_status`, `t2v_eligible` + `t2v_reasons`,
  `i2v_eligible` + `i2v_reasons`, `split`
- versions: `schema_version`, `dataset_version`, `pipeline_version`, `processing_config_hash`,
  `processed_at_utc`

Status vocabulary everywhere: `passed` | `failed` | `needs_review` | `not_checked`. `not_checked`
means the check could not run (asset unavailable); it is never a pass. Rights:
`cleared` | `restricted` | `unknown` | `review_required`; nothing is inferred as `cleared`.

Field mapping: `mapping.aliases` (canonical field -> candidate source fields, first present wins);
overriding one entry keeps the defaults for the rest.

## Rule-based checks vs model review vs human verification

- **Rules** (default, offline): ffprobe/decode integrity, black / frozen frame metrics, caption
  length / placeholders / template residue / duplication / motion terms. They can prove a caption
  is malformed; they cannot prove it is true. `caption_review_method: rules`.
- **Provider review** (optional): a registered adapter returns strictly validated JSON
  (`caption`, `observed`, `assumed`, `issues`, `verdict`, `confidence`). Invalid output never
  replaces a caption; the record goes to review. Results are cached by asset hash + text +
  operation + template version + provider + model + settings. No adapter that sends data off the
  machine is bundled.
- **Human**: supply rights evidence (`rights.evidence_file`) and, for captions, set
  `caption_review_method: human` through your own labelling flow; the schema keeps them apart.

## Rights and provenance

The dataset repository's licence covers the annotations, not the videos. The platform's declared
licence is recorded as evidence (`platform_declared:<status>`) and mapped through
`rights.status_map`; by default `standard` -> `restricted`, Creative Commons declarations ->
`review_required`, absence -> `review_required`. Only `cleared` records (explicit evidence in
`rights.evidence_file`) enter approved exports. `rights.export_override` can allow more; every
release manifest then carries a `WARNING` and the count of overridden records.

## Exports

Each target writes to a staging directory, verifies every referenced path, writes
`release_manifest.json` (schema/dataset/pipeline versions, config hash, counts, adapter, per-file
checksums, manifest checksum) and then promotes atomically. An existing version is never replaced
without `--overwrite`.

- `canonical`: `approved|review|rejected.{parquet,jsonl}` with the full record
- `wan-t2v`: `data.jsonl` rows `{record_id, video, caption, duration_seconds, width, height, fps,
  split, source_record_id, canonical_video_id}`; `videos/` hard-linked (or copied / referenced)
- `wan-i2v`: the same plus `image`, `image_timestamp_seconds`, `image_width/height`, `image_sha256`

These are the documented generic layouts, **not** a verified native format of any Wan trainer.
Add a trainer adapter in `exports/` after inspecting that trainer's loader, with a fixture test
that loads one sample.

## Extending

- caption provider: implement `review(ReviewRequest) -> ReviewResult` and call
  `captions.providers.register_provider("name", factory)`; set `provider.name` and `offline: false`
- export adapter: add a module beside `wan_t2v.py`, a `kind`, and branch in `pipeline.export`
- source: `source/` already handles Hub shards and local exports; add another reader that yields
  `(record_type, row, video_meta)` tuples

## Resuming, inspecting, publishing

- re-run any command: completed, unchanged work is skipped (`skipped_current` in the output)
- rejected / review records: `export --target canonical` -> `rejected.jsonl` / `review.jsonl`
  with `*_reasons` on every record; `report` summarises why records are ineligible
- regenerate reports: `python -m dataset_factory report`
- publish: `export --target ...` with `exports.version` set, then ship the promoted directory
  with its `release_manifest.json`

Training (Wan or otherwise) is outside this package: it runs on a GPU machine against the
promoted export directory.
