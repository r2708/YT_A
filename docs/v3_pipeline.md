# V3: visual T2V / V2V datasets

Implements `youtube_dataset_v3_pipeline.md`. The annotation pipeline and the existing Hub
repository are unchanged; V3 adds a second flow that turns *rights-approved* sources into real
training clips. Code: `src/video_dataset/v3/`, config: `resolution:` and `v3:` in
`config/default.yaml`, tests: `tests/test_v3.py`.

## Changing the resolution

One value in `config/default.yaml`:

```yaml
resolution:
  target: 720        # short edge in pixels: 1080, 720, 480 ...
```

The download cap, the canonical transcode, re-encoded clips and the V3 training clips follow it.
`download_max: 1080` downloads 1080p and downscales clips to `target`. One-off:
`video-dataset run urls.txt --set resolution.target=1080`. A source below the target is kept at its
native size in its own bucket (`native_480_576`, `lowres_360`) and never labelled as the target;
below `experimental_min` it is excluded.

## Workflow

```bash
video-dataset v3 inventory        # read-only: legacy_inventory.json, schema report, source index, revision
video-dataset v3 rights-init      # one entry per source, all metadata_only, every permission false
#   edit data/v3/rights/rights_manifest.jsonl - see below
video-dataset v3 rights-check     # who is eligible, which "approved" entries are incomplete
video-dataset v3 run --limit 10   # acquire -> probe -> decode check -> match -> clips -> T2V + V2V
video-dataset v3 status
video-dataset v3 publish --dry-run   # stage data/v3/staging/{t2v,v2v}; inspect before anything leaves the machine
video-dataset v3 publish --shell     # optional: create both repos with card + schemas only
video-dataset v3 publish             # upload, read back, verify checksums
video-dataset v3 loader-check        # read batches through the unified loader
video-dataset v3 cleanup             # delete local media of verified sources only
```

## Rights manifest

A source is processed only when its entry has `rights_status: approved` **and**
`download_authorized`, `ml_training_allowed`, `derivative_processing_allowed` are literally `true`,
and `creator_or_rightsholder`, `source_license`, `evidence_uri` (a file next to the manifest, or a
URL), `verified_at`, `verified_by` are filled. The pipeline never writes `approved`. A YouTube
"Creative Commons" label is recorded as `platform_declared_license` and is evidence to check, not a
permission. `redistribution_allowed: false` keeps the media out of a public repository.
`local_path` points at a file supplied by the rightsholder (used instead of a download).
`data/v3/` is git-ignored: the manifest and evidence stay local.

## What is produced

- T2V record: legacy `generation_prompt` + a frame-exact clip (`target_media` with sha256),
  resolution block, `annotation_origin: reused_legacy`. Legacy clips must be `accepted`,
  `single_shot`, boundary-checked and 1-30 s (`v3.clips`).
- V2V record: source clip (referenced in the T2V repo, not copied) + instruction + transformed
  target. Transforms are deterministic FFmpeg edits: `cinematic_grade`, `film_grain`, `monochrome`,
  `slow_motion`. Each pair is measured (target differs, layout preserved, timing as declared).
- Only `accepted` records reach `data/<split>.jsonl`; everything else is listed with its reason in
  `quality/<task>_quality_report.json`. Splits are the legacy per-source-group splits.

## Cleanup gate

`cleanup.require_verified_upload: true` (default). A video with an approved rights entry keeps its
source and clips after EXPORT; they are removed by `v3 cleanup` once `v3 publish` set
`metadata_upload_verified` and `visual_asset_upload_verified`. Videos without an approved entry are
metadata-only and are cleaned as before.

## No AI models

`project.no_models: true` (default). The work is done by four tools:

| tool | what it does here |
|---|---|
| FFmpeg | download muxing, transcode, frame-exact clips, V2V edits, decode validation, probing |
| OpenCV | frame sampling, optical flow / camera motion, brightness / colour / sharpness measurements, perceptual hashes, V2V pair checks |
| PySceneDetect | shot boundaries, and the V3 single-shot check on every extracted clip |
| spaCy | question generation (`qa.generator: both`), caption / question text checks, the V3 prompt check |

Whisper (speech-to-text) and OCR stay on through `project.no_models_except: [transcription, ocr]`, so
records keep their transcript and on-screen text. Object detection (YOLO), CLIP and any VLM / LLM are
off, so records carry no detected object classes or aesthetic score; a run refuses to start if one of
them is switched on while `no_models` is true. spaCy's small English
pipeline (`en_core_web_sm`) is a statistical tagger / parser; it is the one learned component.

## Device

`project.device: auto`: CUDA on Windows / Linux, MPS on Apple Silicon, otherwise CPU. Override for
one run with `VIDEO_PIPELINE_DEVICE=cpu|cuda|mps|auto`; `REQUIRE_ACCELERATOR=true` fails instead of
falling back. The selected device is written to `manifest.json -> environment.runtime` and to every
V3 `records.json`.

## Not implemented (on purpose)

- Generative V2V edits (day-to-night, weather, object changes): they need a generation model.
- Latent exports and model-specific frame sampling: no target model / VAE is selected yet.
- Re-running annotation models on reacquired media: annotations are reused; a source whose
  duration no longer matches goes to `FAILED_REVIEW` instead.
- A training loop: `v3/loader.py` hands verified local files to a trainer.
- DirectML and FFmpeg hardware decoding.
