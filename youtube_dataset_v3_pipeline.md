# YouTube Video Dataset V3 — Pipeline Change Specification

**Prepared:** 15 November 2026  
**Purpose:** Define the exact changes needed to evolve the existing metadata-only YouTube video dataset into separate, visual, training-ready Text-to-Video (T2V) and Video-to-Video (V2V) datasets while preserving the existing annotations.

---

## 1. Executive summary

The current processing pipeline is effectively **stateless with respect to media**:

1. A YouTube URL is submitted.
2. The video (and possibly audio) is downloaded temporarily.
3. Frames, clips and other intermediate files are created for analysis.
4. Descriptions, camera annotations, scene/event data, prompts and other outputs are written to JSONL/Parquet.
5. The temporary video, audio, frame and clip files are deleted.
6. The metadata is uploaded to the existing Hugging Face dataset.

This design is suitable for a metadata/annotation dataset, but it does **not** produce the visual training targets required by a video-generation model.

Because the original MP4s, frames and clips have been deleted, the missing visual assets cannot be reconstructed from JSONL/Parquet alone. They must be reacquired from an authorized source and processed again. The good news is that the existing annotations should be reused wherever they can be reliably matched to the reacquired source video. Do not rerun every annotation model by default.

### Target outcome

Maintain three logical dataset layers:

1. **Existing repository — Cinematic Intelligence / annotations:** preserve the current dataset and its historical records. Do not overwrite it.
2. **New T2V repository:** contain prompts paired with actual target video sequences (or compatible video latents), plus model-ready metadata.
3. **New V2V repository:** contain an actual source video, an editing instruction and the corresponding transformed target video (or compatible source/target latents).

Use one training application/dataloader to read both T2V and V2V datasets. Stream or stage only the samples needed for the current batch; do not download the full repositories to the PC.

> **Important storage clarification:** Separate Hugging Face repositories improve organization, permissions and versioning. They do not create separate account-wide storage quotas. Streaming reduces local disk use; it does not make remotely stored video files free or exempt them from Hugging Face storage policies.

---

## 2. Confirmed project constraints and decisions

| Area | Decision / current constraint |
|---|---|
| Existing data | JSONL and Parquet annotations are available on Hugging Face. |
| Original media | Local downloaded MP4s, audio, frames and clips were deleted at the end of the previous pipeline. |
| Reprocessing | Visual assets must be reacquired and processed again. Existing annotations should be mapped and reused when valid. |
| Dataset organization | Keep the existing annotation repository; create separate T2V and V2V Hugging Face dataset repositories. |
| Training access | One training pipeline should load both task datasets. Use remote streaming/bounded temporary staging. |
| Preferred visual resolution | 720p when the source can provide it; otherwise retain the best acceptable native resolution. Do not upscale low-resolution footage and label it as native 720p. |
| Primary resolution fallback | 720p preferred; native 480p/540p/576p accepted in a lower-resolution bucket; 360p may be an experimental/low-quality bucket; below 360p excluded from the primary visual-generation dataset. |
| Model dependency | Keep annotations/media manifests model-independent. Produce model-specific latent exports only after selecting the target model/VAE and preprocessing configuration. |
| Rights | Reprocess and publish only media for which downloading, processing, ML training, transformation and the intended redistribution are authorized. |

---

## 3. Critical rights and source-media gate

Do **not** automatically redownload every URL from the previous manifest.

A publicly viewable YouTube video is not automatically permission to download, transform, redistribute or use its frames/video for ML training. YouTube's Terms restrict downloading and other uses unless expressly authorized by the service, permitted by the applicable terms/law or supported by the necessary permissions. YouTube's help page identifies the Standard YouTube licence as the default and Creative Commons Attribution as a separate licence option. A CC BY label still requires checking that the uploader had the rights to license all material in the video and that the intended use satisfies the licence conditions.

### Required rights workflow

For every original `video_id` / URL, set one of these statuses:

| Status | Meaning | Pipeline action |
|---|---|---|
| `approved` | Evidence supports the planned download, processing, ML-training and storage/redistribution uses. | Eligible for visual processing. |
| `permission_pending` | Permission has been requested but not received/verified. | Do not download or use for visual training yet. |
| `metadata_only` | Existing annotation can be retained for an independently permitted metadata use, but visual training rights are not established. | Keep out of T2V/V2V visual training. |
| `rejected` | Rights are absent, unclear or prohibit the planned use. | Do not reacquire or use as visual training data. |
| `expired_or_revoked` | Permission/licence no longer covers the planned use. | Block new processing; follow the applicable rights/removal process. |

The licence decision must cover the whole audiovisual work, not only the video container. Check music, stock footage, performers, logos, artwork, third-party clips and other embedded material. Store the actual permission or licence evidence and the checks performed.

**Do not assume that converting a video into frames, a latent tensor, a compressed file or an embedding removes copyright or licensing obligations.** Private storage does not itself grant rights to use the material.

---

## 4. Target architecture

### 4.1 Repositories

Recommended repository names (rename if desired, but keep the roles separate):

| Hugging Face repository | Role | What belongs there |
|---|---|---|
| `raj270898/youtube-video-dataset` | Existing annotation foundation | Current JSONL/Parquet, cinematic metadata, provenance, schemas and reports. Preserve existing versions. |
| `raj270898/youtube-t2v-dataset` | T2V training dataset | T2V train/validation/test manifests, prompts, actual target clips or model-compatible latent references, quality results and rights references. |
| `raj270898/youtube-v2v-dataset` | V2V training dataset | V2V train/validation/test manifests, editing instructions, source/target pair references, target clips or latent references, alignment and quality results. |

The T2V and V2V repositories are separate **task datasets**, not separate storage quotas.

### 4.2 Avoid duplicate source media

Use a canonical `asset_id` and a `media_manifest` to identify each physical media asset exactly once. A V2V record may reference a source asset stored in the T2V repository if that is its canonical location, while its edited target is stored in the V2V repository. If the source is not present in T2V, it may be stored in V2V or in a future shared asset repository. Do not upload duplicate copies simply because the same asset is used by both tasks.

Each cross-repository reference must include:

- `repo_id`
- `path` or stable asset URL
- `asset_id`
- `sha256`
- media version/revision where applicable
- rights evidence ID

The training loader must resolve and verify the referenced asset before using it.

### 4.3 High-level data flow

```text
Existing Hugging Face annotation repository (read-only input)
                         |
                         v
              URL + annotation inventory
                         |
                         v
               Rights/provenance gate
                         |
              only approved source URLs
                         |
                         v
            Authorized media reacquisition
                         |
                         v
       Probe -> decode -> scene/clip timestamp match
                         |
             +-----------+-----------+
             |                       |
             v                       v
       T2V preparation          V2V preparation
       prompt + target video    source + instruction +
                               transformed target video
             |                       |
             v                       v
       T2V Hugging Face        V2V Hugging Face
          dataset                dataset
             \                       /
              \                     /
               v                   v
               One unified training pipeline
                remote read + bounded local cache
```

---

## 5. Pipeline changes — exact modules to add or modify

### Change 1 — Freeze and inventory the existing dataset

**Do this before changing code or uploading V3.**

1. Record the exact existing dataset revision/commit.
2. Download only the small metadata files needed for auditing; do not clone/download the full dataset unless a complete local copy is specifically required.
3. Inventory every JSONL/Parquet file and its schema.
4. Record row counts, schema versions, checksums and source IDs.
5. Create a read-only V2/V2.3.0 snapshot or release reference.
6. Do not rewrite existing IDs, annotations or historical records during V3 processing.

Output:

- `legacy_inventory.json`
- `legacy_schema_report.json`
- `legacy_source_index.parquet`
- `legacy_dataset_revision.txt`

Build a canonical source index keyed by `source_video_id` / original YouTube video ID. It should connect video-level, clip-level, scene-level, frame-level and event-level records.

### Change 2 — Add a rights and provenance gate

Before reacquiring any media, process every URL into a rights manifest. The pipeline must stop for records without an approved status.

Minimum fields:

```json
{
  "source_video_id": "source_000001",
  "source_url": "https://www.youtube.com/watch?v=...",
  "creator_or_rightsholder": "verified creator/rightsholder",
  "source_license": "CC-BY-4.0 or written permission reference",
  "rights_status": "approved",
  "download_authorized": true,
  "ml_training_allowed": true,
  "derivative_processing_allowed": true,
  "redistribution_allowed": true,
  "commercial_use_allowed": false,
  "attribution_text": "required attribution, if applicable",
  "evidence_uri": "rights/permission_000001.pdf",
  "verified_at": "YYYY-MM-DD",
  "verified_by": "reviewer or process"
}
```

Use explicit booleans and evidence; do not infer permission from a public URL, a successful download, or a generic statement that the video is "free to use". If rights do not permit redistributing media publicly, do not upload the media to a public dataset repository. Obtain legal advice for uncertain cases.

### Change 3 — Add a resumable media acquisition stage

For each approved source:

1. Reacquire the source through a method authorized by the rightsholder and applicable service terms. Do not bypass DRM, access controls or platform restrictions.
2. Save the temporary source under a deterministic job directory.
3. Compute SHA-256 immediately after acquisition.
4. Use `ffprobe` (or an equivalent media probe) to record duration, dimensions, frame rate, codecs, pixel format, audio streams and rotation metadata.
5. Verify that the complete file decodes.
6. Match legacy clip/scene/frame timestamps to the new source using `source_video_id`, timestamps and content checks where possible.
7. Store processing state in a checkpoint database/manifest so interrupted jobs can resume without repeating completed steps.
8. Never mark media as processed until the required output files have been uploaded and independently verified.

Recommended job states:

```text
DISCOVERED
  -> RIGHTS_PENDING
  -> RIGHTS_APPROVED
  -> ACQUIRED
  -> PROBED
  -> DECODE_VALIDATED
  -> LEGACY_ANNOTATIONS_MATCHED
  -> CLIPS_EXTRACTED
  -> TASK_ASSETS_CREATED
  -> UPLOADED
  -> REMOTE_ASSETS_VERIFIED
  -> CLEANUP_ALLOWED
  -> COMPLETE
```

Failures should move to a retryable `FAILED_RETRYABLE` or a human-review `FAILED_REVIEW` state. Store the failure reason and never silently skip the record.

### Change 4 — Reuse old annotations; regenerate only missing or invalid fields

For a reacquired source video:

- Reuse the existing scene, event, camera, motion and caption annotations if their source ID and time ranges match and they pass validation.
- Re-extract frames only when actual visual frame assets are required.
- Re-run visual models only for missing annotations, mismatched timestamps, low-confidence records or annotations that fail source-video verification.
- Improve `generation_prompt` using the existing description plus verified visual details. Do not invent visual details that are not present in the source.
- Add provenance fields showing whether an annotation was reused, regenerated or human-corrected.

Suggested fields:

```json
{
  "annotation_origin": "reused_legacy",
  "legacy_dataset_revision": "commit-or-tag",
  "annotation_validation": "passed",
  "visual_grounding_status": "human_or_model_verified",
  "last_validated_against_asset": "asset_000001"
}
```

This reduces unnecessary GPU/CPU processing and preserves the value of the existing annotation work.

### Change 5 — Add clip extraction and frame/sequence retention

Use the legacy clip timestamps where valid. Re-extract the actual media segment; a JSON `clip_path` string is not a clip asset.

For each clip:

- Save the actual video file or model-ready representation.
- Keep exact start/end timestamps and source-video ID.
- Preserve a stable clip ID.
- Record frame timestamps/indices.
- Validate duration and frame count.
- Keep temporal ordering.
- Avoid extracting and uploading every possible frame; retain the frames required by the selected model, QA, preview and reproducibility policy.

For video generation, a sequence of frames alone is not automatically equivalent to a video training target. T2V requires a visual target sequence (video or the appropriate model-specific representation). V2V requires both source and transformed target visual sequences.

---

## 6. Resolution, aspect ratio, frame rate and duration policy

### 6.1 Resolution policy

Use the best available source, with 720p as the preferred target for models/configurations that support it.

| Best available source | Processing rule | Training bucket |
|---|---|---|
| 4K / 2K / 1080p | Downscale to model's 720p target, preserving aspect ratio. | `720p` |
| 720p | Keep or perform only model-required preprocessing. | `720p` |
| 576p / 540p / 480p | Keep native resolution; do not upscale and label as 720p. Use a compatible lower-resolution model/configuration. | `native_480_576` |
| 360p | Keep only in an explicitly separate experimental bucket if visual quality and model support justify it. | `lowres_360` |
| Below 360p | Exclude from primary T2V/V2V visual-generation training; keep metadata separately if useful and permitted. | `excluded_lowres` |

This is a dataset policy, not a guarantee that every generation model accepts every resolution. Final width, height, frame count and latent shape must satisfy the chosen model's preprocessing/VAE requirements.

**Do not upscale 480p footage to 720p and call it native 720p.** Upscaling can be used only if intentionally tested as a separate preprocessing variant and clearly marked as upscaled.

### 6.2 Aspect ratio

- Preserve the original aspect ratio.
- Do not stretch 16:9, 9:16, 1:1 or other footage into one shape.
- Use model-supported aspect-ratio buckets or deliberate crop/pad policies.
- Record whether a crop, pad or resize was applied and retain the original dimensions.

### 6.3 FPS and temporal sampling

Do not blindly convert every source to 24 FPS. Record the true source FPS and timestamps first. Then apply a model-specific frame sampling policy (frame count, sampling interval and target FPS) for the selected architecture. Keep timestamps so motion labels can be checked against the sampled frames.

### 6.4 Duration

Use clip lengths that the selected model and training implementation support. Store original clip duration and model training window duration separately. If long clips are divided into windows, retain their relationship through `parent_clip_id`, start/end timestamps and shared `source_group_id`.

### 6.5 Required resolution metadata

```json
{
  "source_width": 1920,
  "source_height": 1080,
  "source_fps": 30.0,
  "training_width": 1280,
  "training_height": 720,
  "training_fps": 24.0,
  "resolution_bucket": "720p",
  "resolution_action": "downscaled",
  "aspect_ratio_preserved": true,
  "upscaled": false,
  "crop_applied": false,
  "frame_sampling_profile": "model_profile_v1"
}
```

The example values are illustrative. `training_fps` and dimensions must be filled from the actual model preprocessing profile, not hardcoded for every model.

---

## 7. T2V dataset requirements

A T2V record must pair a detailed, visually grounded text prompt with an actual target video sequence or compatible target latent. A prompt plus a missing video path is not a valid T2V sample.

### 7.1 T2V record structure

```json
{
  "sample_id": "t2v_000001",
  "task": "text_to_video",
  "source_group_id": "source_group_0001",
  "source_video_id": "source_000001",
  "clip_id": "clip_000001",
  "prompt": "A cinematic aerial tracking shot of a red car driving along a winding mountain road at sunset.",
  "target_asset_id": "asset_000001",
  "target_media": {
    "repo_id": "raj270898/youtube-t2v-dataset",
    "path": "media/720p/clip_000001.mp4",
    "sha256": "..."
  },
  "timeline": [
    {
      "start_seconds": 0.0,
      "end_seconds": 4.0,
      "description": "The car enters the winding road."
    },
    {
      "start_seconds": 4.0,
      "end_seconds": 8.0,
      "description": "The camera follows the car through the curve."
    }
  ],
  "cinematography": {
    "shot_type": "wide",
    "camera_angle": "aerial",
    "camera_movement": "tracking",
    "camera_speed": "slow",
    "lighting": "golden_hour"
  },
  "technical_profile": "wan_or_other_model_profile_v1",
  "rights_evidence_id": "rights_000001",
  "quality_status": "accepted",
  "split": "train"
}
```

### 7.2 T2V quality rules

- Prompt describes what is visibly present and how it moves.
- Camera movement and subject movement labels match the actual clip.
- No generated hallucinated objects or actions are added to the caption.
- Clip decodes and has valid duration/frame sequence.
- No corrupted, black, frozen or severely blurred sequences unless intentionally labelled.
- Duplicate and near-duplicate clips are detected.
- Rights are cleared for the actual video asset and planned use.

---

## 8. V2V dataset requirements

A V2V record must contain **three linked components**:

1. Actual source video.
2. Editing instruction.
3. Actual transformed target video.

A source video and an instruction alone are not V2V supervision. A target that is unrelated to the source is not a valid pair.

### 8.1 V2V transformations to build first

Start with transformations that can be evaluated and aligned reliably:

- Color grading / cinematic look.
- Day-to-night or lighting changes.
- Weather/environment changes.
- Film grain or other controlled visual style changes.
- Slow-motion or frame-rate transformations (with explicit temporal metadata).

More complex edits such as object addition/removal, identity changes, scene replacement and large environment transformations should be introduced only when the output can be reliably validated. Generated targets must be rights-cleared for the applicable training and redistribution use.

### 8.2 V2V record structure

```json
{
  "sample_id": "v2v_000001",
  "task": "video_to_video",
  "source_group_id": "source_group_0001",
  "source_asset_id": "asset_source_0001",
  "source_media": {
    "repo_id": "raj270898/youtube-t2v-dataset",
    "path": "media/720p/source_0001.mp4",
    "sha256": "..."
  },
  "instruction": "Transform the daylight scene into a cinematic nighttime environment while preserving the car, road layout and camera trajectory.",
  "target_asset_id": "asset_target_0001",
  "target_media": {
    "repo_id": "raj270898/youtube-v2v-dataset",
    "path": "media/720p/target_0001.mp4",
    "sha256": "..."
  },
  "preserve": [
    "subject_identity",
    "camera_trajectory",
    "motion_timing",
    "road_geometry"
  ],
  "modify": [
    "lighting",
    "time_of_day",
    "reflections",
    "environment_color"
  ],
  "temporal_alignment": {
    "source_start_seconds": 0.0,
    "target_start_seconds": 0.0,
    "duration_seconds": 8.0,
    "alignment_status": "verified"
  },
  "rights_evidence_id": "rights_000001",
  "quality_status": "accepted",
  "split": "train"
}
```

### 8.3 V2V quality rules

- The source and target are a real, documented pair.
- The instruction accurately describes the change made.
- Attributes in `preserve` remain stable to the degree the task requires.
- Attributes in `modify` visibly change.
- Frame count, duration and time alignment are validated.
- A human or reliable evaluation process checks edit accuracy and unwanted artifacts.
- The target video is not simply a duplicate of the source with a different instruction.

---

## 9. Common media manifest

Use one canonical asset registry so the same source clip is not uploaded twice just because it appears in two task datasets.

```json
{
  "asset_id": "asset_000001",
  "source_video_id": "source_000001",
  "clip_id": "clip_000001",
  "media_type": "video",
  "repo_id": "raj270898/youtube-t2v-dataset",
  "path": "media/720p/clip_000001.mp4",
  "sha256": "...",
  "bytes": 123456789,
  "duration_seconds": 8.0,
  "source_width": 1920,
  "source_height": 1080,
  "training_width": 1280,
  "training_height": 720,
  "source_fps": 30.0,
  "training_fps": 24.0,
  "resolution_bucket": "720p",
  "rights_evidence_id": "rights_000001",
  "decode_status": "passed",
  "upload_status": "verified"
}
```

Use the same `asset_id` and checksum in every T2V/V2V reference. If a V2V source asset is shared from the T2V repository, reference it; do not copy it into V2V. Store the transformed target in V2V.

For model latents, create a separate manifest containing:

- `asset_id`
- `model_id` and model revision
- VAE/encoder name and revision
- preprocessing profile/version
- latent file URI and checksum
- latent shape, dtype and layout
- source media checksum

Never treat latents as model-independent. A latent produced by one model's VAE may not be compatible with another model.

---

## 10. Hugging Face repository layout

Recommended logical layout:

```text
youtube-video-dataset/                 # existing; preserve
  existing annotations and reports


youtube-t2v-dataset/
  README.md
  dataset_info.json
  schemas/
    t2v_schema.json
    media_schema.json
  manifests/
    media_manifest.jsonl
    rights_manifest.jsonl
  data/
    train.jsonl or train.parquet
    validation.jsonl or validation.parquet
    test.jsonl or test.parquet
  media/
    720p/...
    native_480_576/...
  quality/
    t2v_quality_report.json


youtube-v2v-dataset/
  README.md
  dataset_info.json
  schemas/
    v2v_schema.json
    media_schema.json
  manifests/
    pair_manifest.jsonl
    rights_manifest.jsonl
  data/
    train.jsonl or train.parquet
    validation.jsonl or validation.parquet
    test.jsonl or test.parquet
  media/
    720p/...
    native_480_576/...
  quality/
    v2v_quality_report.json
```

The manifests can reference a canonical asset stored in either task repository. If asset reuse becomes complex, introduce a third shared asset repository later; it is not mandatory for the first pilot.

### Storage and visibility cautions

- Splitting repositories does not multiply storage quota.
- Public Hub storage is best-effort and subject to platform policy; private storage has explicit plan limits.
- Confirm the current account storage limits before uploading large media collections.
- Public visibility is not a substitute for media rights. Do not publish restricted videos, frames or derived assets.
- Use Hugging Face LFS/Xet-supported upload paths and resumable uploads as appropriate; verify the final remote revision and checksums.
- Avoid committing local cache directories, original unapproved media, temporary frames, audio or model checkpoints to either repository.

---

## 11. Training with both repositories

### 11.1 Desired training behaviour

One training application should:

1. Load T2V metadata as an iterable/streaming dataset.
2. Load V2V metadata as an iterable/streaming dataset.
3. Resolve each record's media asset reference.
4. Fetch or stage only the media needed for the current batch.
5. Decode the clip / load compatible latents.
6. Train using the correct task conditioning and loss.
7. Remove temporary batch files when safe.
8. Keep only a bounded cache of recently used assets.

Streaming records does not guarantee that every video decoder can seek directly from a remote object. Video decoding may need a seekable local temporary file or a supported range-capable backend. Test the exact loader with the chosen storage format before scaling up.

### 11.2 Cache and temporary-file policy

- Configure explicit Hugging Face and media cache directories.
- Apply a configurable cache-size limit and least-recently-used (LRU) eviction.
- Stage only the clips or latent files needed by active batches.
- Store active batch files in a temporary directory.
- Delete a temporary file only after the batch has been consumed and no worker still needs it.
- Make cache cleanup safe to rerun after crashes.
- Never delete the only verified copy of an asset from remote storage.
- Monitor peak cache usage, temporary-file usage, model-weight usage and checkpoint usage during pilot runs.
- Do not assume remote dataset streaming removes model memory, compute or checkpoint-storage requirements.

### 11.3 T2V/V2V sampling

Do not simply alternate one T2V record and one V2V record forever. Add configurable task sampling weights, per-task batch sizes and task-specific preprocessing. Track T2V and V2V losses separately. Use a shared model only if the chosen architecture supports both types of conditioning; otherwise use separate adapters or compatible model branches.

## 12. Train/validation/test split policy

Split by original `source_group_id` / source video **before** generating derived clips or V2V edits.

Do not randomly split individual clips from the same source video, because near-identical scenes or different transformations of one source could leak between training and evaluation.

Requirements:

- All clips and transformations from one source group belong to one split.
- V2V source and target always remain in the same split.
- All derived versions of a source stay in the same split.
- Record the split assignment in a central manifest.
- Deduplicate before final split assignment, then re-run duplicate/leakage checks after export.

Suggested starting split proportions: 90% train, 5% validation, 5% test, subject to source counts and diversity. These are initial proportions, not a mandatory rule.

---

## 13. Quality assurance gates

A record is not training-ready merely because its JSON parses or its annotation score is high.

### Required gates for every media asset

- [ ] Rights status approved with evidence ID.
- [ ] Media URI resolves and download/read succeeds.
- [ ] SHA-256 matches the manifest.
- [ ] Media probe succeeds.
- [ ] Full or sampled decode succeeds without corruption.
- [ ] Duration and frame timestamps are consistent.
- [ ] Width, height, aspect ratio and FPS are recorded.
- [ ] Resize/crop/fps transformations are documented.
- [ ] No accidental duplicate or near-duplicate leakage.

### Additional T2V gates

- [ ] Prompt accurately describes the target video.
- [ ] Camera and subject motion annotations match the clip.
- [ ] Clip is visually useful and not dominated by blur, black frames or compression artifacts.
- [ ] Video/latent shape matches the model preprocessing profile.

### Additional V2V gates

- [ ] Source and target are genuinely paired.
- [ ] Editing instruction matches the actual transformation.
- [ ] Temporal alignment is verified.
- [ ] Preserved attributes remain consistent.
- [ ] Modified attributes change as requested.
- [ ] Unwanted artifacts are reviewed.

### Training-readiness status

Use explicit states such as:

- `pending`
- `rights_blocked`
- `asset_missing`
- `decode_failed`
- `annotation_mismatch`
- `pair_invalid`
- `needs_human_review`
- `accepted`
- `rejected`

Only `accepted` records enter a training split.

---

## 14. Migration plan and implementation order

Do not reprocess every historical URL in one large job.

### Phase 0 — Freeze and audit

- Freeze the current dataset revision.
- Inventory annotations, source IDs, timestamps and schemas.
- Generate the rights manifest.
- Identify which URLs can legally be reacquired and used.

**Deliverables:** legacy inventory, source index, rights report, V3 schemas.

### Phase 1 — Build a small end-to-end pilot

Use approximately 10–25 rights-cleared source videos. The goal is to validate the pipeline, not to build a competitive model dataset.

- Reacquire media through an authorized source.
- Reuse matching annotations.
- Create a small number of T2V clips.
- Create genuine V2V source/target pairs.
- Upload to the two new repositories.
- Read them back through the planned streaming loader.
- Measure local cache, temporary storage and peak GPU/CPU memory.

**Go/no-go:** every sample must resolve, decode, match its annotation and pass rights/quality gates.

### Phase 2 — Validate at a larger sample size

Expand to approximately 50–100 eligible videos only after the pilot succeeds. Check diversity, near-duplicates, camera movement balance, resolution buckets, transformation quality and storage consumption.

### Phase 3 — Scale selectively

Reprocess only sources that:

- have verified rights;
- provide useful visual diversity;
- have sufficient visual quality;
- map to valuable annotations; and
- fit the storage/compute budget.

Do not make the number of processed URLs the only success metric. A smaller, clean, legally usable and well-aligned dataset is better than a larger dataset with missing media, invalid labels or unclear rights.

### Phase 4 — Training-readiness release

Publish versioned T2V and V2V releases with:

- dataset card and intended use;
- source/licence/attribution summary;
- schemas and manifests;
- split-generation method;
- media validation report;
- duplicate/leakage report;
- resolution/FPS/duration distributions;
- limitations and known exclusions;
- exact model preprocessing profile for latent exports, if applicable.

---

## 15. What should change in the existing codebase

Suggested modules; adapt names to the current project structure.

```text
pipeline/
  legacy_reader/             # Read existing JSONL/Parquet without mutation
  source_inventory/          # Build source_video_id -> annotation mapping
  rights_gate/               # Require explicit approved rights evidence
  media_acquisition/         # Authorized, resumable source acquisition
  media_probe/               # ffprobe, checksum, decode validation
  annotation_matcher/        # Match old clip/scene/frame timestamps
  clip_extractor/            # Extract actual media clips
  frame_extractor/           # Extract only required frame sequences
  resolution_normalizer/     # 720p preference + native fallback policy
  caption_validator/         # Ground prompts against the actual clip
  t2v_builder/               # Prompt + target-video records
  v2v_builder/               # Source + instruction + target pairs
  latent_encoder/             # Optional model-specific VAE encoding
  media_manifest/            # Canonical asset registry and cross-repo refs
  quality_gates/              # Rights, decode, pair, temporal, duplicate checks
  hf_publisher/               # Publish versioned metadata/media artifacts
  remote_verifier/            # Read back remote files and verify checksums
  cleanup/                    # Delete temp files only after remote verification
  training/                   # Unified streaming T2V/V2V dataloader

configs/
  acquisition.yaml
  resolution_policy.yaml
  model_profiles/
  t2v.yaml
  v2v.yaml
  storage_cache.yaml
```

### Mandatory change to current cleanup logic

**Old behaviour:** delete all media immediately after JSONL/Parquet upload.

**New behaviour:** cleanup is conditional and happens only after the selected assets have been uploaded to the intended destination, downloaded/read back successfully, checksum-verified, and recorded in the manifest.

Use two independent completion flags:

- `metadata_upload_verified`
- `visual_asset_upload_verified`

Only when all required outputs for that job are verified may `cleanup_allowed=true` be set. For records that are metadata-only, explicitly record that no visual training asset was produced; do not mark them T2V/V2V-ready.

---

## 16. Recommended resolution policy implementation

```python
from dataclasses import dataclass

@dataclass
class ResolutionDecision:
    bucket: str
    action: str
    target_width: int
    target_height: int
    upscaled: bool
    include_in_primary_training: bool


def choose_resolution(width: int, height: int) -> ResolutionDecision:
    """Choose a dataset bucket while preserving the source aspect ratio.

    This is a dataset policy only. A model-specific preprocessing profile
    must still validate the final dimensions, frame count and latent shape.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive")

    short_edge = min(width, height)

    # 720p class: downscale so the shorter edge is at most 720,
    # while maintaining aspect ratio. Do not upscale smaller footage.
    if short_edge >= 720:
        scale = 720 / short_edge
        target_w = round(width * scale)
        target_h = round(height * scale)
        return ResolutionDecision(
            bucket="720p",
            action="downscale_or_keep",
            target_width=target_w,
            target_height=target_h,
            upscaled=False,
            include_in_primary_training=True,
        )

    if short_edge >= 480:
        return ResolutionDecision(
            bucket="native_480_576",
            action="keep_native",
            target_width=width,
            target_height=height,
            upscaled=False,
            include_in_primary_training=True,
        )

    if short_edge >= 360:
        return ResolutionDecision(
            bucket="lowres_360",
            action="keep_native_experimental",
            target_width=width,
            target_height=height,
            upscaled=False,
            include_in_primary_training=False,
        )

    return ResolutionDecision(
        bucket="excluded_lowres",
        action="exclude_from_primary_visual_training",
        target_width=width,
        target_height=height,
        upscaled=False,
        include_in_primary_training=False,
    )
```

**Implementation note:** the example uses the shorter edge to classify resolution. Validate this policy against landscape, portrait and unusual aspect ratios. The selected model's official preprocessing code takes precedence; some models require specific dimensions or aspect-ratio buckets rather than arbitrary 720-short-edge sizes.

---

## 17. Automatic hardware detection — GPU when supported, otherwise CPU

The pipeline must select its compute device automatically at runtime. It must not hardcode `cuda`, assume that every Windows machine has an NVIDIA GPU, or assume that every Mac has a usable GPU backend.

### Required behaviour

| Operating system | Detection order | Behaviour |
|---|---|---|
| Windows | Supported NVIDIA CUDA GPU → optional supported DirectML backend → CPU | Use CUDA when PyTorch detects a compatible NVIDIA GPU. DirectML may be enabled as an explicitly tested optional backend for compatible workloads. Otherwise use CPU. |
| macOS | Supported Apple Silicon MPS backend → CPU | Use PyTorch MPS when it is available and the selected model/operators support it. If MPS is unavailable (for example, an unsupported Mac, Intel Mac, incompatible PyTorch build or unsupported operation), use CPU. |
| Linux / other | Supported CUDA backend → CPU | Use CUDA when available; otherwise use CPU. Add other accelerators only after they are tested and explicitly supported. |

**Interpretation of “GPU available”:** the GPU must be usable by the selected framework and workload, not merely present in the computer. A GPU listed by the operating system may lack a compatible driver, runtime, memory capacity or required operator support.

### Device-selection requirements

1. Default to `auto` detection.
2. Allow an explicit environment/config override (`auto`, `cpu`, and supported backend names such as `cuda` or `mps`) for debugging and reproducibility.
3. Validate an explicitly requested device. If it is unavailable, log a clear warning and either fall back to CPU or fail when `REQUIRE_ACCELERATOR=true` is set.
4. Log OS, Python version, framework version, detected accelerator, selected device, GPU name and available device memory where the framework exposes them.
5. Pass the selected device into every model, tensor, encoder and inference/training component. Do not scatter hardcoded `.cuda()` calls through the codebase.
6. Keep device selection separate from media decoding. FFmpeg hardware decoding/encoding is an optional, independently detected optimization; the pipeline must still work with software decoding on CPU.
7. Do not silently move a workload between devices mid-run. If an out-of-memory error occurs, apply the configured batch-size/frame-count fallback or stop cleanly and record the failure.
8. Record the selected device and software environment in each processing-run manifest so outputs are reproducible.

### Suggested implementation

Create `runtime/device.py` and centralize device selection there. This PyTorch example supports CUDA on Windows/Linux, MPS on supported Macs, and CPU fallback. It deliberately does not assume DirectML is interchangeable with CUDA; DirectML can be added as a separately tested optional backend if the chosen model supports it.

```python
# runtime/device.py
import os
import platform
import warnings

import torch


def select_device():
    requested = os.getenv("VIDEO_PIPELINE_DEVICE", "auto").strip().lower()
    require_accelerator = os.getenv("REQUIRE_ACCELERATOR", "false").lower() == "true"
    system = platform.system()

    if requested == "cpu":
        selected = torch.device("cpu")
    elif requested == "cuda":
        if torch.cuda.is_available():
            selected = torch.device("cuda")
        else:
            selected = None
    elif requested == "mps":
        mps_available = (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        )
        selected = torch.device("mps") if mps_available else None
    elif requested == "auto":
        selected = None
        if system == "Darwin":
            mps_available = (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
                and torch.backends.mps.is_built()
            )
            if mps_available:
                selected = torch.device("mps")
            elif torch.cuda.is_available():
                # Unusual on macOS, but retain a generic supported-backend check.
                selected = torch.device("cuda")
        else:
            if torch.cuda.is_available():
                selected = torch.device("cuda")

        if selected is None:
            selected = torch.device("cpu")
    else:
        raise ValueError(
            "VIDEO_PIPELINE_DEVICE must be one of: auto, cpu, cuda, mps"
        )

    if selected is None:
        message = (
            f"Requested device '{requested}' is unavailable on {system}."
        )
        if require_accelerator:
            raise RuntimeError(message + " REQUIRE_ACCELERATOR=true.")
        warnings.warn(message + " Falling back to CPU.")
        selected = torch.device("cpu")

    info = {
        "os": system,
        "device": str(selected),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ),
    }

    if selected.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(selected)
        info["gpu_memory_bytes"] = torch.cuda.get_device_properties(selected).total_memory

    print(f"[device] Runtime configuration: {info}")
    return selected, info


DEVICE, DEVICE_INFO = select_device()
```

Usage in processing/training code:

```python
from runtime.device import DEVICE

model = model.to(DEVICE)
frames = frames.to(DEVICE)

# Avoid hardcoded model.cuda() / tensor.cuda().
```

Example environment overrides:

```bash
# Automatic detection (default)
VIDEO_PIPELINE_DEVICE=auto

# Force CPU for testing or unsupported model operations
VIDEO_PIPELINE_DEVICE=cpu

# Explicitly request a supported backend
VIDEO_PIPELINE_DEVICE=cuda
VIDEO_PIPELINE_DEVICE=mps
```

On Windows PowerShell, set variables for the current session with `$env:VIDEO_PIPELINE_DEVICE="auto"`. On macOS/Linux, use `export VIDEO_PIPELINE_DEVICE=auto`.

### Important model-compatibility caveats

- CUDA is the preferred acceleration path for many video-generation training stacks, but it requires a compatible NVIDIA GPU, driver, CUDA-enabled PyTorch build and sufficient VRAM.
- Apple Silicon Macs can use MPS for supported PyTorch operations. MPS is not the same as CUDA, and some video-generation repositories, custom CUDA kernels, attention implementations or VAE components may not support it.
- If a model cannot run on MPS, CPU fallback may permit preprocessing or small tests, but full video-generation training on CPU may be impractically slow or unsupported. The pipeline must report this clearly rather than claiming training succeeded.
- Windows systems with AMD/Intel GPUs may have optional acceleration paths such as DirectML, but support is workload-specific. Do not select DirectML automatically until the complete model and operators have passed tests on it.
- Keep FFmpeg hardware acceleration as a separate capability check. If unavailable, use CPU decoding/encoding; do not treat a failed FFmpeg GPU path as a reason to fail the entire media pipeline.

### Processing-run manifest fields

Add these fields to every run-level manifest (and optionally reference the run ID from each sample):

```json
{
  "runtime": {
    "os": "Windows",
    "device_policy": "auto",
    "selected_device": "cuda",
    "gpu_name": "NVIDIA GPU model",
    "torch_version": "installed-version",
    "accelerated_media_decode": false,
    "fallback_reason": null
  }
}
```

Use the actual detected values. Do not hardcode the example GPU name or versions.

### Hardware detection acceptance tests

- [ ] Windows with a supported NVIDIA GPU selects CUDA.
- [ ] Windows without a supported GPU selects CPU.
- [ ] macOS with supported Apple Silicon MPS selects MPS when the model supports it.
- [ ] macOS without usable MPS/GPU selects CPU.
- [ ] Explicit `cpu` override works on every OS.
- [ ] An unavailable explicit accelerator produces a clear warning and CPU fallback, or a clear failure when `REQUIRE_ACCELERATOR=true`.
- [ ] Device and framework information is written to the run manifest.
- [ ] Media processing still works when FFmpeg hardware acceleration is unavailable.
- [ ] The selected model's actual operations are smoke-tested on the selected device before a long run begins.

## 18. Final acceptance checklist for V3

V3 is ready for a first training experiment only when all of the following are true:

- [ ] Existing annotation dataset is preserved and versioned.
- [ ] Every source has a documented rights decision.
- [ ] No restricted/pending source enters visual training.
- [ ] Media can be reacquired from an authorized source.
- [ ] Existing annotations are matched to the correct source video and timestamps.
- [ ] Actual T2V target videos or compatible target latents exist.
- [ ] Actual V2V source/target pairs and instructions exist.
- [ ] Resolution, FPS, duration and aspect ratio are recorded.
- [ ] Video decoding and checksum checks pass.
- [ ] T2V captions are grounded in target footage.
- [ ] V2V transformations and temporal alignment are validated.
- [ ] Splits are source-video/group-based with no leakage.
- [ ] Cross-repository asset references resolve correctly.
- [ ] Streaming works without full dataset download.
- [ ] Automatic GPU/CPU detection works on supported Windows and macOS configurations.
- [ ] The selected compute device and runtime versions are recorded in the run manifest.
- [ ] Local cache has a hard size limit and cleanup recovery.
- [ ] Remote uploads are read back and checksum-verified before local cleanup.
- [ ] Dataset cards document rights, limitations, preprocessing and intended use.

---

## 19. Immediate next actions

1. **Freeze the existing Hugging Face repository** and record its current revision.
2. Export an inventory of all original source URLs, IDs, timestamps and annotation files.
3. Build and review the rights manifest. Do not redownload all URLs automatically.
4. Create the T2V and V2V repository shells with schemas and dataset cards; do not upload media yet.
5. Implement the new cleanup gate and local media audit/resume system.
6. Select 10–25 cleared videos for the end-to-end pilot.
7. Verify streaming, storage, sample quality and model compatibility before scaling.

**The most important pipeline change:** preserve approved visual training assets (or the exact model-compatible representations you need) and verify their remote upload before deleting temporary files. The current metadata-only pipeline should remain available, but it must no longer be considered a complete T2V/V2V visual dataset pipeline.

---

## 20. References

The following official documentation should be checked again when implementation begins because platform terms and model support can change:

1. [Hugging Face Hub storage limits](https://huggingface.co/docs/hub/main/en/storage-limits) — public/private storage policy and account tiers.
2. [Hugging Face Datasets: Load video data](https://huggingface.co/docs/datasets/en/video_load) — video columns, decoding and frame access. Video support is documented as experimental; pin and test dependency versions.
3. [Hugging Face Datasets: Streaming](https://huggingface.co/docs/datasets/stream) — streaming dataset access. Verify how the selected media backend handles remote video fetches and caching.
4. [Wan 2.1 official repository](https://github.com/Wan-Video/Wan2.1) — example of model-specific resolution support and VACE video-editing requirements. Do not treat Wan settings as universal settings for every model.
5. [YouTube Terms of Service](https://www.youtube.com/t/terms) — restrictions on downloading, modifying and using content.
6. [YouTube Help: Licence types](https://support.google.com/youtube/answer/2797468) — Standard YouTube licence and Creative Commons Attribution licence information.

*This specification is an engineering plan, not legal advice. Obtain appropriate rights/permissions and legal review where needed.*
