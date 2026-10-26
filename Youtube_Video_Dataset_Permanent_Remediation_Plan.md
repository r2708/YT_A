# YouTube Video Dataset — Permanent Remediation Plan

**Dataset:** [`raj270898/youtube-video-dataset`](https://huggingface.co/datasets/raj270898/youtube-video-dataset)  
**Prepared:** 2026-10-26  
**Purpose:** A durable, engineering-focused plan to resolve the dataset's metadata, integrity, annotation-quality, evaluation, loading, media, and licensing issues.

> **Scope and status:** This document specifies what should change and how to verify it. It does not modify the Hugging Face repository or claim that fixes have already been applied. Issues labelled **confirmed from current public artifacts** are directly supported by the public README, file listing, statistics, manifest, or viewer. Issues labelled **needs full-data validation** require running checks across the actual source files and/or video media before they can be called dataset-wide defects.

## Executive summary

The permanent fix is to make publication a deterministic build process—not a collection of manually updated files. There should be one canonical annotation source, one versioned schema, one explicit split registry, one reproducible export pipeline, and one release validator. The README, manifests, statistics, size metadata, per-type Parquet files, combined exports, and viewer-facing exports must all be generated from the same final release snapshot.

**Do not manually patch only `statistics.json` or the README.** That would hide the symptoms while leaving the next release vulnerable to the same drift.

### P0 release blockers

1. Reconcile the dataset's actual row counts with `statistics.json`, the manifest, and the README metadata.
2. Remove all machine-local absolute paths from published artifacts.
3. Decide and enforce one canonical file layout; prevent duplicate files from being ingested twice.
4. Fix or explicitly mark empty `long_video_qa` support.
5. Make split assignment and split integrity reproducible and testable.
6. Prevent unverified, rejected, or review-only annotations from being represented as independently verified high-quality data.
7. Confirm media availability and rights before making the relevant content available for download or commercial use.

## Current public evidence to reconcile

The public repository currently exposes the following clues:

- The README YAML still includes `size_categories: 1K<n<10K`, while the live Hugging Face dataset page displays the broader `1M - 10M` category. Recompute the exact release count and set metadata to the correct category rather than hard-coding a guessed total.
- The public `statistics.json` reports 36 videos, 4,390 clips, 4,287 scenes, 19,802 frames, 44,798 events, 1,591 temporal QA rows, and **0** long-video QA rows.
- The public manifest contains absolute paths beginning with `/Users/rajkhajanchi/Desktop/YT_A/...` and reports `verifier: heuristic` with `events_verified_now: 0` in a visible video entry.
- The repository contains both `shard_*/` directories and root-level Parquet/JSONL files, plus `per_video/`. The README's loader metadata points to `shard_*/...`, so the intended canonical source and whether these copies overlap must be explicitly defined.
- The README advertises multi-scene/long-video QA and broad quality claims, while the published statistics report no long-video QA rows and many review-tier records.
- The public viewer has shown one selected subset/split at a time. A viewer showing only `train` is not sufficient evidence by itself that all underlying records use only the train split; confirm the actual public config and files through the Dataset Viewer API and the exported artifacts.

Sources checked:

- [Dataset page and live viewer](https://huggingface.co/datasets/raj270898/youtube-video-dataset)
- [README.md](https://huggingface.co/datasets/raj270898/youtube-video-dataset/blob/main/README.md)
- [Repository file listing](https://huggingface.co/datasets/raj270898/youtube-video-dataset/tree/main)
- [statistics.json](https://huggingface.co/datasets/raj270898/youtube-video-dataset/blob/main/statistics.json)
- [manifest.json](https://huggingface.co/datasets/raj270898/youtube-video-dataset/blob/main/manifest.json)
- [splits.json](https://huggingface.co/datasets/raj270898/youtube-video-dataset/blob/main/splits.json)
- [Hugging Face Dataset Viewer API documentation](https://huggingface.co/docs/dataset-viewer/quick_start)

---

## 1. Permanent architecture: one source of truth

Adopt this release flow:

```text
Raw source/media + annotation pipeline outputs
                  |
                  v
      Canonical per-video records
                  |
                  v
 Schema migration + normalization + ID checks
                  |
                  v
 Quality validation + media/timestamp verification
                  |
                  v
 Source-video grouping + leak-resistant split assignment
                  |
                  v
 Deterministic export builder (Parquet / JSONL / manifest)
                  |
                  v
 Recompute statistics from final exported records
                  |
                  v
 Release gate: integrity + quality + license + load tests
                  |
                  v
 Publish one immutable release revision on Hugging Face
```

### Repository layout to standardize

Use one canonical layout and explicitly document which files are primary versus convenience exports. A workable target layout is:

```text
README.md
LICENSE
CITATION.cff
schemas/
  current.schema.json
  migrations/
    2.1.0-to-2.2.0.py
splits.json
manifest.json                 # portable paths/URIs only
statistics.json               # generated, never manually edited
shard_0001/
  clips.parquet
  scenes.parquet
  frames.parquet
  events.parquet
  temporal_qa.parquet
  long_video_qa.parquet
  video_descriptions.parquet
...
exports/                      # optional, if aggregate exports are retained
  dataset.parquet
  dataset.jsonl
viewer/                        # optional slim, flattened view
  clips.parquet
  temporal_qa.parquet
```

`per_video/` may be kept as a generation artifact or a documented data source, but it must not silently compete with the published shard files as another loader input. If root-level per-type Parquet files are retained, label them as either **canonical** or **derived convenience exports** and make sure the loaders never ingest both the root-level files and the same records from `shard_*/`.

**Important counting rule:** A combined `dataset.parquet` is an aggregate view. Do not add its row count to the row counts of its component files. Do not count both Parquet and JSONL mirrors as separate examples.

---

## 2. Issue-by-issue permanent solutions

Severity indicates impact if unresolved. “Permanent solution” describes the process/code change needed, not a claim the repair has already been made.

### 1. Wrong size category in the dataset card

**Severity:** High for discoverability/metadata  
**Evidence status:** Confirmed in current public README metadata.

**Problem:** The README metadata declares `size_categories: 1K<n<10K`, while the live dataset page displays `1M - 10M`.

**Permanent solution:**

1. The export builder computes a release-level count using only canonical published records and a declared counting rule.
2. Generate the Hugging Face `size_categories` value from that exact count, using Hugging Face's accepted size-category syntax.
3. Keep the exact counts by record type in `statistics.json`; the size category is only a bucket, not a substitute for exact statistics.
4. Add a CI check that fails if the README size category does not match the computed canonical record count.

**Acceptance criteria:** The category is derived from the same final files published in the release; no manual count is used; the release validator tests it.

### 2. Counts and `statistics.json` are outdated or inconsistent

**Severity:** Critical  
**Evidence status:** Confirmed; public `statistics.json` reports 36 videos and much smaller record counts than the live page/repository indications.

**Permanent solution:**

- Build all statistics from final exported records—not cached pipeline databases, checkpoint files, a previous release, or hand-entered values.
- Include `dataset_revision`, `generated_at_utc`, `schema_version`, `counting_basis`, per-type row counts, per-split counts, per-tier counts, per-status counts, unique-video count, unique-source-group count, media availability counts, and rejected/duplicate counts.
- Recompute statistics after all filtering/deduplication and immediately before publication.
- Generate a checksum for each exported file and record its relative path, byte size, row count, schema version, and SHA-256 in a release manifest.
- Treat statistics as generated output; never hand-edit them.

**Acceptance criteria:** Independently recounting the canonical files reproduces all reported counts exactly; generated statistics reference the release revision/hash; aggregate exports are not double-counted.

### 3. Absolute local filesystem paths leak into `manifest.json` and statistics

**Severity:** High  
**Evidence status:** Confirmed in the public manifest.

**Problem:** Fields contain paths beginning with `/Users/rajkhajanchi/Desktop/YT_A/...`. These are not portable and disclose a local development directory structure.

**Permanent solution:**

- Store repository-relative paths such as `per_video/vid_.../clips.jsonl`, or stable publicly accessible URLs/URIs where an artifact is actually hosted.
- Keep local paths only in private pipeline state, never in published manifests.
- Add a release-blocking path scan that rejects POSIX home-directory paths (`/Users/`, `/home/`), Windows drive paths (`C:\...`), workstation-specific temp folders, and unresolved absolute paths.
- Resolve every published relative path against the release root and assert it exists before export.

**Acceptance criteria:** No public metadata includes machine-local absolute paths; every local relative path resolves inside the release package; every external URL passes a configurable availability check.

### 4. Dataset file layout and loader `configs` disagree or are ambiguous

**Severity:** Critical for reproducibility  
**Evidence status:** Confirmed that the repository has `shard_*/`, root-level Parquet/JSONL, and `per_video/`, while README metadata uses `shard_*/...` globs. Whether these copies are duplicates must be tested.

**Permanent solution:**

1. Pick one canonical loader input for each config. Prefer one explicit data source, for example `shard_*/clips.parquet`, where each record belongs to exactly one shard.
2. Either remove root-level duplicates from the published loader path or clearly move/document them as aggregate convenience exports. Do not use broad globs that include both canonical shards and aggregate exports.
3. Remove references to files such as `sequences.parquet` if they are not part of the published release, or generate and validate them if they are promised.
4. Generate README file listings/config declarations from the release manifest rather than maintaining two unrelated lists.
5. Add a duplicate-ID test across all shards and an aggregate-vs-component consistency test.

**Acceptance criteria:** Every canonical record appears exactly once in its configured dataset; each declared config loads; no config silently selects missing files or double-loads aggregate records.

### 5. Dataset Viewer failures/timeouts with large nested schemas

**Severity:** High for usability  
**Evidence status:** User-reported; must distinguish dataset-caused complexity from Hugging Face service issues.

**Permanent solution:**

- Keep the complete rich schema for training/research downloads, but create a separate **viewer projection** with a small, stable set of display fields.
- Flatten or stringify especially complex nested fields for viewer presentation, for example `camera_movement`, `quality_overall`, `validation_status`, `transcript_scope`, and a JSON text preview for `hard_negatives`.
- Keep binary video and frame media out of the tabular viewer rows; use stable media references and separate optional download locations.
- Reduce unnecessary per-row fields in the viewer-only export. Do not delete useful data from the canonical training export just to make the viewer fast.
- Test `/splits`, `/size`, `/first-rows`, and `/rows` endpoints for every advertised config after each release; record any Hugging Face service failures separately from dataset schema failures.

**Acceptance criteria:** All advertised configs can load small previews; the viewer projection has a documented schema; full nested data remains available in the canonical Parquet files.

### 6. High share of bronze/review records

**Severity:** High  
**Evidence status:** Current statistics show large review counts for clips, frames, scenes, and video descriptions.

**Permanent solution:**

- Separate **validation status** (`accepted`, `review`, `rejected`, `duplicate`) from **quality tier** (`gold`, `silver`, `bronze`). They answer different questions.
- Define explicit, versioned tier criteria. For example:
  - `gold`: accepted, evidence-backed, required grounding and temporal checks pass, no blocking contradictions, task-specific fields complete.
  - `silver`: accepted and usable for the stated task; some non-critical optional measurements may be absent.
  - `bronze`: incomplete or weakly verified; excluded from recommended training splits by default.
- Keep `review` records out of the default high-quality training config. If they are retained for exploration, label the config accordingly, e.g. `clips_all` versus `clips_accepted`.
- Create failure reason codes instead of generic `status_review`: missing evidence, low confidence, boundary unverified, transcript mismatch, contradiction, media missing, and so on.
- Track acceptance rate by pipeline stage and fix the upstream stage causing repeated failures instead of lowering acceptance thresholds to inflate counts.

**Acceptance criteria:** Each row's status/tier is reproducible from explicit rules; release card reports per-tier/per-status counts; default training configs do not accidentally mix review/rejected records.

### 7. Null or incomplete quality fields

**Severity:** High  
**Evidence status:** Null grounding/temporal fields and unavailable confidence values appear in public examples.

**Permanent solution:**

- Define required versus optional fields **per record type and per task** in a versioned JSON Schema.
- Unknown values must be `null` (or a formally documented unknown enum), never a fabricated score or default zero. For every missing required measure, add a machine-readable reason.
- Use explicit confidence provenance, such as `detector_score`, `asr_logprob`, `measurement`, `human_verified`, `derived_min`, `heuristic`, or `unavailable`.
- Do not assign a numerical confidence score when the provider returned none. A missing confidence is not zero confidence and is not a high confidence.
- Add completeness summaries by field and record type. Fail release only for task-critical missing fields; issue warnings for optional fields.

**Acceptance criteria:** Required fields are present and type-valid; all nulls in task-critical fields are handled by policy; confidence provenance matches the source and actual computation.

### 8. Transcript truncation, repetition, and poor segmentation

**Severity:** High  
**Evidence status:** Needs full-data validation.

**Permanent solution:**

- Preserve the raw ASR output as an immutable provenance artifact and store cleaned/normalized transcript separately.
- Store each segment with `start`, `end`, `text`, optional `speaker_id`, and source confidence. Require `0 <= start <= end <= media_duration` within a documented frame/sample tolerance.
- Label scope explicitly: `clip`, `scene`, or `video`. Never populate a clip-level transcript from a full-video transcript without marking it as inherited context.
- When deriving clip text from word/segment timestamps, select only segments overlapping the clip interval. Make overlap policy explicit and deterministic.
- Detect exact repeated segments, near-duplicate consecutive segments, large unexplained gaps, truncated words at clip boundaries, non-monotonic timestamps, and segments outside the source duration.
- Compare a sample against the source audio; ASR text can be mechanically well-formed and still be wrong. Store language/model/version and processing configuration.

**Acceptance criteria:** Every clip transcript's scope is explicit; segments have valid timestamps; repetitions and clipping are reported; sampled ASR accuracy is measured; unresolved failures are labelled `review` rather than silently accepted.

### 9. `long_video_qa` is empty or nearly empty

**Severity:** Critical if long-video QA is a promised capability  
**Evidence status:** Confirmed by current `statistics.json` (`long_video_qa: 0`) and zero-byte JSONL entry in the repository listing.

**Permanent solution:**

- Choose one of two honest release states:
  1. **Supported and populated:** Generate long-video QA from evidence spanning multiple scenes/events, validate answers/evidence, and publish non-empty records; or
  2. **Not yet supported:** Remove the config and capability claims from the card until the data exists.
- Do not create questions simply to hit a numeric target. Generate from valid event relations and require evidence spanning at least two distinct scenes for any question labelled cross-scene.
- Validate temporal relations, answer derivation, source event IDs, evidence timestamps, and whether the supplied evidence is sufficient to answer the question.
- Do not ship an empty placeholder as if it were a functioning training subset.

**Acceptance criteria:** If advertised, the subset has non-zero, schema-valid, validated rows and documented task coverage; otherwise README, config metadata, and stats all explicitly mark it unavailable.

### 10. Template-heavy or weak hard negatives

**Severity:** High for contrastive training quality  
**Evidence status:** Requires complete hard-negative audit.

**Permanent solution:**

- Every negative must contain `kind`, `source_ids`, a rule/method identifier, the changed property, the intended contrast, and validation status.
- Define a negative as valid only when it is false under the source evidence and the positive is true under the same evidence window. For an `attribute_swap`, only the intended attribute should change; do not allow unintended semantic changes.
- Reject negatives that are merely unrelated/distant examples unless the task explicitly trains retrieval discrimination. Mark these as `other_event` rather than a hard negative if they do not form a close contrast.
- Score negative utility by semantic similarity plus evidence-based falsity, and sample-audit the positives and negatives against the video.
- Measure negative diversity per kind, repeated templates, and rate of negatives that a reviewer judges genuinely challenging.

**Acceptance criteria:** All published negatives have traceable provenance; a sampled review shows that negatives are wrong for the claimed reason; repeated templates and invalid negatives are measured, not assumed acceptable.

### 11. Train/validation/test splits are not consistently represented publicly

**Severity:** High  
**Evidence status:** The current viewer has shown `train` only in the observed config, while `splits.json` describes fractions for validation/test and the statistics report train/test video counts. The full split state must be reconciled.

**Permanent solution:**

- Treat `splits.json` as the source of truth for source-video grouping, generated deterministically from a fixed seed/salt and an explicit split algorithm.
- Build each published config with real `train`, `validation`, and `test` splits—or document that a config intentionally exposes only a single train split and provide a separate multi-split config.
- Assign every derived clip, scene, frame, event, QA row, and video description according to its parent source video/source group.
- Add a release check that compares split labels in all record files with `splits.json` and the public Dataset Viewer `/splits` response.
- Ensure the test split is large and diverse enough for the benchmark. With only a few source videos, label results as a pilot evaluation and do not make broad generalization claims.

**Acceptance criteria:** Every record has exactly one valid split; split registry and rows agree; no source group appears across splits; published loaders expose the documented split structure.

### 12. Overly wide nested schema

**Severity:** Medium–High  
**Evidence status:** Schema complexity is visible in the viewer; impact varies by library/workflow.

**Permanent solution:**

- Keep a normalized top-level core: `record_id`, `record_type`, `video_id`, `source_group_id`, `split`, `schema_version`, timestamps, validation status, tier, and stable evidence references.
- Keep complex typed objects in documented struct columns in canonical Parquet; do not flatten everything into dozens of redundant fields.
- Provide optional task-specific exports (`clips`, `temporal_qa`, etc.) and a compact viewer projection. Include a payload JSON only when needed for heterogeneous fields, with clear schema versioning.
- Avoid maintaining several subtly different copies of the same field. Define a single authoritative path per value and generate compatibility aliases only when needed.
- Test the canonical data with Hugging Face `datasets`, PyArrow, and at least one alternative reader such as Polars.

**Acceptance criteria:** Canonical data retains semantics and types; supported loaders work; viewer columns are understandable; schema duplication and undocumented aliases are removed.

### 13. Clip/media file availability is unclear

**Severity:** High for users who need the actual media  
**Evidence status:** Some rows include relative `clip_path` references; full reachability and hosting coverage need validation.

**Permanent solution:**

- For every media reference, record `media_status` (`hosted`, `external_only`, `restricted`, `missing`, `not_redistributable`) and a stable URI or relative path where appropriate.
- Validate each published local media path. If clips are not included in the public release, state that clearly and provide the legal/source retrieval workflow where allowed.
- Avoid storing a file path that implies a downloadable video exists when only a YouTube URL or private local file is available.
- Track checksums, codec, duration, dimensions, sample rate where relevant, and source-video association for hosted media.
- Never mirror or redistribute media unless rights and platform terms permit it.

**Acceptance criteria:** Every media reference has an honest availability state; no published `clip_path` silently points to a missing private file; the card accurately describes which media bytes users can access.

### 14. Inconsistent schema versions (`2.1.0`, `2.2.0`, etc.)

**Severity:** Medium–High  
**Evidence status:** Multiple schema versions are present in public rows/commit history.

**Permanent solution:**

- Pick one schema version for the released dataset. Preserve source-version provenance separately if necessary.
- Implement explicit, tested migrations from each supported old schema version to the current schema. Migrations must be deterministic and idempotent.
- Validate all migrated rows against the current schema before export. A row that cannot be safely migrated should be quarantined and reported, not partially coerced without a warning.
- Publish the schema and migration changelog. Increment major version for breaking changes, minor for backward-compatible fields, and patch for non-semantic corrections.

**Acceptance criteria:** Every canonical public row declares the release schema version; migration tests cover old versions; no mixed schema silently reaches a supposedly uniform release.

### 15. Confidence provenance is incomplete or contradictory

**Severity:** High  
**Evidence status:** Public examples include `provider: cv_models`, `confidence: null`, and `confidence_source: unavailable`; camera sub-objects may contain separate measured confidence values.

**Permanent solution:**

- Separate record-level confidence from component-level confidence. A camera movement score must not be silently reused as overall description confidence.
- Require `confidence_source` whenever a numeric confidence is present. If no confidence exists, use `confidence: null`, `confidence_source: unavailable`, and a reason code.
- Specify score range and semantics (for example, calibrated probability vs model score vs heuristic confidence). Do not call every 0–1 model output a probability.
- If a value is derived from multiple component scores, store the formula/method and version, such as `derived_min:v1`; do not overwrite raw component scores.

**Acceptance criteria:** Every confidence has an interpretable source; missing scores are explicit; field semantics are documented and consistent across record types.

### 16. Temporal fields and `exact: true` can disagree with verification state

**Severity:** High  
**Evidence status:** Public examples show `exact: true` together with `boundary_check: not_run` and null `actual_start`, `actual_end`, and boundary error.

**Permanent solution:**

- Redefine `exact` to mean an explicit, measured condition, not “timestamps were requested.”
- Prefer clearer fields such as `boundary_verified: true/false`, `boundary_check_status: passed/failed/not_run`, `actual_start`, `actual_end`, and `boundary_error_seconds`.
- Compute expected duration as `end_time - start_time`; compare with the extracted media's duration from `ffprobe` or equivalent. Define tolerance using frame duration and time-base precision; document the policy.
- Set `boundary_verified=true` only when the boundary check ran and passed. If not run, do not export `exact=true`; use `false` or `null` under the schema's defined meaning.
- Validate `start_time >= 0`, `end_time > start_time`, `end_time <= source_duration + tolerance`, actual boundaries within source duration, and duration agreement.

**Acceptance criteria:** No record claims verified/exact boundaries without the required measurements and a passed check; duration mismatches are quantified and either corrected or flagged for review.

### 17. Video/source counts do not agree across registered, processed, and exported data

**Severity:** High  
**Evidence status:** Current statistics report 36 completed videos while the manifest contains a larger video registry; exact lifecycle semantics need reconciliation.

**Permanent solution:**

- Track lifecycle statuses independently: `registered`, `downloaded`, `media_validated`, `processed`, `annotated`, `exported`, `accepted`, `failed`, `skipped`, `rights_blocked`.
- Generate counts for every status from the per-video manifest. Do not use a single field `videos` to mean different pipeline stages.
- Require each exported row's `video_id` to exist in the manifest and be export-eligible.
- Publish both `registered_video_count` and `exported_source_video_count` so the difference is explicit.

**Acceptance criteria:** All counts can be recomputed from the manifest; every difference between lifecycle stages is explained by status and reason.

### 18. Dataset evaluation has too few independent source videos

**Severity:** High  
**Evidence status:** Current statistics report three test videos; this is a very small benchmark, even if split assignment is correct.

**Permanent solution:**

- Grow the source-video pool and split by source group before deriving related records.
- Stratify splits by content/domain, duration, language, scene/event categories, and relevant quality tiers without moving related clips across splits.
- Keep a locked test set that is not used for prompt tuning, threshold tuning, or annotation rule development.
- Report sample size, number of source videos, content mix, uncertainty/confidence intervals, and task-specific metrics.
- If expanding is not possible, label evaluation as a small pilot set and avoid claiming broad generalization.

**Acceptance criteria:** The benchmark has a documented split policy and sufficient source-video diversity for the claims made; test data remains untouched during development.

### 19. QA task mix is narrow and few examples are labelled hard

**Severity:** Medium–High  
**Evidence status:** Current statistics show 1,591 temporal QA rows, with 1,427 timestamp questions and 15 labelled hard. These are descriptive counts, not automatic proof of poor quality.

**Permanent solution:**

- Define a task taxonomy before generation: timestamp localization, duration, event ordering, before/after, state change, causal/temporal relation where justified, cross-scene linking, entity continuity, and evidence selection.
- Generate questions only when source annotations contain enough evidence; do not invent question classes that the source video cannot support.
- Use automated balance reporting and review targets, not quotas that encourage low-quality synthetic data.
- Calibrate difficulty through a documented rubric or pilot model/human evaluation. “Hard” should require multi-step evidence, multiple event references, or a nontrivial temporal comparison—not simply longer question text.
- Validate the answer against evidence, event IDs, timestamps, and source media.

**Acceptance criteria:** QA coverage is reported by task/difficulty; answer/evidence checks pass; difficult examples are genuinely multi-step and audited.

### 20. Event distribution is heavily weighted toward on-screen text

**Severity:** Medium–High depending on intended use  
**Evidence status:** Current statistics report 25,915 `text_on_screen` events and 983 `action` events out of 44,798 total events.

**Permanent solution:**

- First decide the dataset's actual intended task mix. A text-heavy dataset may be appropriate for OCR/video-text tasks but should not be represented as balanced action understanding without evidence.
- Report event distributions by source video, duration, split, and event type—not just global counts.
- If action understanding is an objective, source or annotate more varied meaningful actions and use the same evidence/acceptance rules as other event types.
- Avoid rebalancing by duplicating rare labels; collect additional independent content or explicitly provide task-specific balanced subsets.

**Acceptance criteria:** Dataset card accurately reflects event composition; advertised tasks have sufficient examples; diversity and source independence are reported.

### 21. Non-speech sound-event coverage is very low

**Severity:** High for audio tasks; lower for video-only tasks  
**Evidence status:** Current statistics report 23 sound events versus 3,094 speech events.

**Permanent solution:**

- If audio understanding is an objective, define labels and annotation requirements for non-speech sounds, silence, music, ambience, and audiovisual synchronization as appropriate.
- Keep audio event detection distinct from speech transcripts; speech transcripts are not a substitute for sound-event labels.
- Record audio availability, sampling properties, event timestamps, annotator/model provenance, and confidence.
- If audio is not in scope, remove/qualify claims that imply broad audio-event coverage.

**Acceptance criteria:** Audio task claims match the exported annotation coverage; coverage is measured by event type and independently spot-checked.

### 22. Camera-motion, subject-motion, prompts, and descriptions may contradict each other

**Severity:** High  
**Evidence status:** Requires cross-field and video-based validation; examples in the viewer motivate the check but do not establish an error rate.

**Permanent solution:**

- Define a single canonical camera-motion enum and direction convention (`pan_left`, `pan_right`, `tilt_up`, `tilt_down`, `zoom_in`, `zoom_out`, `static`, `unknown`). Map prose to this enum in a validation step.
- Keep camera motion separate from subject motion. For example, a subject moving left in frame is not proof that the camera pans left.
- Run rule-based consistency checks across `camera.movement`, `camera.movement_direction`, `subject_motion`, `actions`, `description`, `generation_prompt`, and hard negatives.
- Use linguistic extraction only to flag candidate contradictions; do not automatically rewrite labels from prose without checking the video/evidence.
- Maintain a sample-based human/video review queue for ambiguous motion labels.

**Acceptance criteria:** Contradictions are flagged and resolved or marked uncertain; movement labels follow a documented vocabulary and direction convention.

### 23. Overall quality score can be high with missing grounding/temporal metrics

**Severity:** Medium–High  
**Evidence status:** Public examples show `overall` computed from only `description_quality` while grounding and temporal accuracy are null.

**Permanent solution:**

- Define score names by meaning, such as `description_quality`, `grounding_score`, `temporal_accuracy_score`, and `task_usable_score`.
- Do not label an average or partial component as a universal `overall` without explaining its formula. Preferred approach: `overall_score: null` if required components are absent, plus `available_component_scores` and `score_coverage`.
- If partial scoring is intentionally supported, include `components_used`, `components_missing`, formula version, and score coverage; ensure users can filter records by required component coverage.
- Tier assignment must consider status and required task-specific measures, not just a high score from one component.

**Acceptance criteria:** Every score is interpretable; no row can appear fully grounded/temporally accurate solely because description quality is high; formulas are versioned.

### 24. Transcript scope is unclear at clip level

**Severity:** High  
**Evidence status:** Needs full-data validation; current examples repeat video-level narrative context across clips.

**Permanent solution:**

- Use `transcript_scope` as a required enum (`clip`, `scene`, `video`, `inherited_context`, `none`).
- For clip-specific fields, derive transcript segments from timestamps that overlap the clip interval; preserve inherited full-video transcript separately.
- Prevent QA generation from using text outside the question's allowed evidence scope unless the task explicitly requires cross-video context.
- Add a leakage check that detects answer text from events or transcript portions outside the intended evidence interval.

**Acceptance criteria:** Scope is explicit and accurate; temporal QA evidence and context are compatible with the task; no hidden future/elsewhere transcript content leaks answers into clip-level tasks.

### 25. Source-video split leakage has not been independently excluded

**Severity:** High  
**Evidence status:** Documentation describes source-level splits, but end-to-end integrity needs a release check.

**Permanent solution:**

- Group by a canonical source group (`source_group_id` / canonical video ID), not only by row ID or processed clip ID.
- Deduplicate and cluster exact/near-duplicate source media before assigning splits. Reuploads or lightly edited copies may need to share a group.
- Derive all related rows from the parent video's split; never independently random-split clips, frames, events, and QA.
- Publish a split audit containing unique groups per split, overlap counts, duplicate cluster overlap, and any exclusions.

**Acceptance criteria:** No source group or known duplicate cluster crosses splits; every derived record inherits the parent split; the check runs automatically before every release.

### 26. Licensing/provenance of source YouTube videos, transcripts, frames, and clips is unclear

**Severity:** High for redistribution/commercial use  
**Evidence status:** README declares MIT while also saying content is sourced from YouTube and users should respect original licenses. Rights require a legal/provenance review.

**Permanent solution:**

- Separate the license for original annotations/code from rights associated with source video/audio, transcripts, thumbnails, and extracted frames. A repository-wide MIT declaration must not imply rights the dataset publisher does not possess.
- Track provenance per source: source URL/ID, retrieval date, applicable license/terms when known, rights basis, media redistribution permission, attribution requirements, and takedown status.
- Publish only assets that can lawfully be distributed. For restricted media, consider metadata-only records plus source links, subject to applicable law and platform terms; do not include restricted media bytes in the release.
- Add an exclusion path for rights-uncertain or takedown-requested sources and regenerate dependent records/statistics after removal.
- Obtain qualified legal review before making commercial-use or unrestricted redistribution claims.

**Acceptance criteria:** Every distributed asset has a documented rights basis; the dataset card distinguishes annotation license from source-media rights; restricted assets are excluded or appropriately handled.

### 27. Too many schema fields and aliases make the dataset cumbersome

**Severity:** Medium–High  
**Evidence status:** Schema complexity is visible in the public viewer; exact user impact varies by consumer.

**Permanent solution:**

- Keep a canonical typed schema for training and a compact task-specific projection for common loading workflows.
- Mark fields as required/optional/deprecated. Avoid silent aliases and duplicate representations of the same concept.
- Supply minimal examples for each config and tested loader snippets for Hugging Face `datasets`, PyArrow, and Polars.
- Add schema contract tests that compare actual Parquet schema, JSON Schema, and README examples.

**Acceptance criteria:** Schema documentation matches every published file; old/deprecated fields are migration-managed; recommended loaders work on a small clean install.

---

## 3. Required automated audit checks

Create a command such as `python tools/audit_release.py --root dist --strict`. The command should return a non-zero exit code if a release blocker is found. It should emit machine-readable `audit_report.json` and a short human-readable summary.

### Integrity checks

- Every `record_id` is unique within its canonical record type and follows its documented format.
- Every `video_id`, `scene_id`, `clip_id`, `frame_id`, and `event_id` reference resolves to a valid parent record where required.
- Every record uses the current schema version after migrations.
- No duplicate canonical record is included across shards unless duplication is an explicitly documented feature.
- Every per-type row count, split count, status count, tier count, and unique-video count matches recomputation.
- The combined export equals the documented union of canonical per-type records; aggregate files are not counted twice.
- `README.md` config paths match files in the release manifest and all advertised configs load.

### Temporal and media checks

- `start_time >= 0`; `end_time > start_time`; `end_time` does not exceed the source duration beyond a documented tolerance.
- `media_duration` matches `end_time - start_time` within the documented tolerance.
- `boundary_check_status=passed` requires actual boundary measurements and a computed error within tolerance.
- `exact=true` is forbidden when boundary verification was not run.
- Every frame/clip media reference has a declared availability state and resolves if advertised as included.
- Transcript segment timestamps are monotonic or otherwise valid under documented overlap rules and fall within source duration.
- Clip-scope transcript content is derived from segments overlapping the clip interval.

### Quality and label checks

- Required quality values are present for the targeted task or the record is not accepted into that task's high-quality config.
- Every confidence score has a documented source and semantics; missing values remain missing, not fabricated.
- `overall_score` coverage and formula are internally consistent with component fields.
- Camera/subject movement conflicts are reported; do not automatically “fix” an ambiguity without evidence.
- Each hard negative has a valid kind, provenance, source IDs, and a check that it is false under the same evidence scope.
- `accepted`, `review`, `rejected`, and `duplicate` statuses and `gold`, `silver`, `bronze` tiers meet explicit, versioned rules.

### Split and license checks

- All records inherit the split of their canonical source group.
- Train/validation/test source-group intersections are empty.
- Exact/near-duplicate groups do not cross splits unless explicitly approved and documented.
- All included media has a rights/availability state; records with prohibited or unresolved distribution rights cannot include restricted media bytes.
- Published metadata includes no workstation-specific absolute paths.

---

## 4. Proposed scoring and acceptance policy

Use task-specific acceptance rather than one global score to decide if a row is usable.

### Example tier policy

| Tier | Minimum policy | Default recommended for supervised training? |
|---|---|---|
| Gold | Accepted; all task-required evidence checks passed; no unresolved critical contradictions; temporal/audio/visual labels verified to the standard claimed by the release | Yes |
| Silver | Accepted; task-required fields present; automated checks pass; optional measurements may be unavailable but are disclosed | Yes, depending on task |
| Bronze | Incomplete, low-confidence, partially checked, or otherwise useful only with caveats | No, unless the user intentionally opts in |
| Review | Unresolved issue or insufficient evidence | No |
| Rejected | Failed hard validation or known bad annotation | Never in the default release config |

Do not make `gold` a target percentage. Gold is a result of meeting criteria, not a quota.

### Score policy

- Keep component scores separate.
- Record formula/version and `components_used` for derived scores.
- If a required component is missing, either set the task-usable aggregate to null or explicitly mark it as partial; do not silently present it as fully measured.
- Report component completeness as a percentage by record type.

---

## 5. Fixing order and implementation phases

### Phase 1 — Stop metadata drift and prevent bad exports

1. Create one canonical release manifest and define which file pattern is authoritative.
2. Remove machine-local paths from export generation.
3. Regenerate `statistics.json`, README size metadata, file listings, and config declarations from actual final outputs.
4. Resolve the `long_video_qa` empty-subset contradiction.
5. Run loader smoke tests on every declared config.

**Exit condition:** Exact row counts reconcile; no local path leaks; configs are valid; empty/planned datasets are honestly documented.

### Phase 2 — Enforce data integrity

6. Normalize all records to one schema version.
7. Validate IDs, parent references, per-type counts, media references, and duplicates.
8. Validate time intervals/durations and prohibit unsupported `exact=true` claims.
9. Establish reproducible source-group splits and verify no leakage.
10. Generate the machine-readable audit report and fail publication on blockers.

**Exit condition:** The strict audit passes and produces reproducible statistics for the release revision.

### Phase 3 — Improve annotation quality

11. Define required/optional fields and tier/status rules by record type and task.
12. Audit transcripts, camera/subject motion, quality score completeness, and hard negatives.
13. Create a gold/silver training export and keep review/bronze data clearly separated.
14. Review a stratified human sample for captions, event labels, timestamps, transcripts, and QA answers.
15. Publish accuracy/error estimates with sample size and confidence intervals.

**Exit condition:** Release notes disclose what has been human-checked, what was automated, and the measured error rates from sampled review.

### Phase 4 — Improve dataset usefulness and claim accuracy

16. Expand and diversify temporal QA where the source material supports it.
17. Add long-video QA only after it is generated and validated; otherwise remove the claim.
18. Increase action/audio event coverage only if those remain target tasks.
19. Improve the viewer projection and publish tested load examples.
20. Resolve media rights and update license/provenance documentation.

**Exit condition:** Each advertised capability is supported by a populated, validated subset and a clear rights statement.

---

## 6. Release acceptance checklist

A new dataset revision must not be published as a quality release unless all applicable checks pass.

- [ ] Exact row counts computed from final canonical files.
- [ ] README size category derived from exact canonical count.
- [ ] `statistics.json` regenerated from current release outputs.
- [ ] Manifest records include a revision/hash, portable paths, file sizes, checksums, and counts.
- [ ] No local absolute filesystem paths remain in published artifacts.
- [ ] README configs resolve only to the intended canonical files; no duplicate ingestion.
- [ ] Every declared config loads successfully using supported tooling.
- [ ] Empty subsets are removed from claims/configs or explicitly marked as unavailable.
- [ ] Schema migration completed; all exported rows pass current schema validation.
- [ ] All IDs and cross-file references resolve.
- [ ] All accepted temporal records pass required boundary checks.
- [ ] `exact=true` is never emitted without completed boundary verification.
- [ ] Transcript scope is explicit and segment timing passes checks.
- [ ] Confidence/quality score provenance is consistent; partial scores are visibly partial.
- [ ] Review/rejected rows are excluded from recommended high-quality config by default.
- [ ] Hard negatives have traceable construction and falsity evidence.
- [ ] Split counts and source-group integrity pass; no source leakage is detected.
- [ ] Media availability/rights status is set for every media reference.
- [ ] License and provenance text have been reviewed before redistribution/commercial claims.
- [ ] Human review sample and its methodology/results are recorded.
- [ ] Public viewer `/splits`, `/size`, `/first-rows`, and row-preview endpoints checked after publication.

---

## 7. Practical implementation notes

### Recommended generated artifacts

- `statistics.json`: counts, quality distributions, split distributions, completeness metrics, dataset revision, generation timestamp, and counting rule.
- `manifest.json`: source groups/videos and per-file metadata using portable paths/URIs only.
- `audit_report.json`: per-check pass/fail/warning, counts, sample failing record IDs, and script version.
- `schemas/current.schema.json`: current schema contract.
- `schemas/migrations/`: deterministic migrations from prior versions.
- `splits.json`: canonical source-group split mapping and algorithm/version/seed.
- `README.md`: generated or verified config/file list and accurate descriptions of available subsets.

### CI / release command design

The actual script names may differ, but the release process should expose stable commands comparable to:

```bash
# 1. Produce normalized release artifacts from canonical records
python tools/build_release.py --input data/final --output dist

# 2. Validate all rows, references, media metadata, time bounds, splits and license flags
python tools/audit_release.py --root dist --strict

# 3. Recompute stats and verify that README/manifest match the output
python tools/check_release_metadata.py --root dist

# 4. Test loading every published config
python tools/smoke_test_loaders.py --root dist

# 5. Publish only if every required command exits successfully
```

These commands are a recommended interface to implement; they should not be treated as existing commands until added to the pipeline repository.

### Do not do these things

- Do not just edit the displayed row count by hand.
- Do not lower acceptance thresholds just to increase the gold/silver count.
- Do not set missing confidence/grounding values to `1.0`, `0`, or another placeholder.
- Do not say “verified” when only schema/heuristic validation ran.
- Do not randomly split derived clips when multiple rows come from the same source video.
- Do not load both shard files and aggregate files as if they were independent records.
- Do not advertise long-video QA or broad audio understanding while the associated subset/annotations are absent or insufficient.
- Do not imply that an MIT license on annotations grants redistribution rights to underlying YouTube media.

---

## Final recommendation

The most durable approach is a **single, versioned, reproducible release pipeline with mandatory validation gates**. Fix the release metadata and layout first; then enforce schema, timing, split, quality, and rights checks; finally improve the content mix and measure annotation accuracy. This prevents the same classes of issues from recurring with future dataset growth.

The public artifacts already demonstrate metadata/path inconsistencies and an empty long-video QA subset. Other issues—especially transcript correctness, camera-motion accuracy, hard-negative validity, near-duplicate leakage, and true visual grounding—must be resolved by running the audit across all underlying rows and the accessible media, not by editing metadata alone.
