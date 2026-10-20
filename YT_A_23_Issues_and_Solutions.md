# YT_A Video Dataset Pipeline: 23 Major Issues and Recommended Solutions

**Project:** [r2708/YT_A](https://github.com/r2708/YT_A)  
**Related dataset:** [raj270898/youtube-video-dataset on Hugging Face](https://huggingface.co/datasets/raj270898/youtube-video-dataset)  
**Purpose:** Improve the pipeline that converts YouTube or local videos into structured, time-grounded data for video understanding, video-language models, and video-generation training.


## Implementation status (2026-10-19)

Everything below was implemented with the installed, measurement-only tools (no LLM / VLM, no API).
Tests: `tests/test_issue_fixes.py` (22 tests, including an end-to-end run whose export must pass
`video-dataset release-check`). Schema version is now `2.1.0`; the exported Parquet schema changed,
so every Hub shard must be re-pushed after the next run (`scripts/hf_refresh.py`, no `--only`).

| # | Issue | What landed | Where |
|---|---|---|---|
| 1 | Prompt / video mismatch | claim-level evidence per scene (`claims`: field, value, source, status, evidence frames); generated text that no verifier judged is `semantic_verification: not_run` and can never be accepted or strict | `validation/claims.py`, `validation/stage.py` |
| 2 | Heuristic-only semantic validation | explicit check states (`passed` / `failed` / `not_run` / `not_applicable` / `error`), `semantic_verification` per record, per-video `coverage`; measured providers are `measured`, generative ones need a verifier | `schemas/quality.py`, `validation/policy.py`, `dataset/coverage.py` |
| 3 | Camera-motion conflicts | explicit `camera_motion_agreement` check (label family vs optical-flow family, `not_applicable` when unknown), camera and subject motion stay separate, disagreement -> review | `validation/stage.py` |
| 4 | Clip boundaries | first / last clip frames located in the source by perceptual hash -> `actual_start` / `actual_end` / `boundary_error_seconds` / `boundary_check`; half-open interval convention documented; validation check `clip_boundaries` | `frame_sampling/boundary.py`, `frame_sampling/clips.py`, `schemas/common.py` |
| 5 | Missing / stale media references | `final/media_manifest.jsonl` + `statistics.json["media_references"]`, media gate in `release-check`, upload refuses broken references when `include_media` | `dataset/media_manifest.py`, `dataset/release.py`, `dataset/upload.py` |
| 6 | spaCy text layer | text-structure checks: caption contradictions, question form, answer type (spaCy for verbs when available, rules otherwise), text normalisation | `validation/text_checks.py` |
| 7 | Split leakage | `source_group_id` (near-duplicate cluster) on every record, `splits.json["groups"]`, `split_leaks` + release gate | `dataset/curation.py`, `dataset/release.py` |
| 8 | Multi-shot prompts | `annotation_scope` / `generation_target` on every record; a large in-shot change -> `not_for_generation` + review; `export.strict.generation_targets` | `schemas/dataset.py`, `validation/stage.py` |
| 9 | Transcript alignment | `transcript_segments` with absolute + relative times, word clipping, `partial`, `transcript_scope`; `speaker_visible` always null; `Transcript.status` | `transcription/align.py`, `dataset/builders.py` |
| 10 | Event timing | `start_uncertainty_ms` / `end_uncertainty_ms` from the timing method; tracked subject motion times actions inside the shot (`measured`); fine-timing QA on shot boundaries flagged | `temporal/events.py` |
| 11 | Relation verification | `relation_holds` per relation type, BEFORE-graph cycle detection, `evidence_kind`, model-inferred CAUSES capped at review | `validation/relations.py` |
| 12 | Hard negatives | `verification_status` + `evidence_interval` on every negative, positive / negative statistics, gate that no negative equals its positive | `schemas/curation.py`, `dataset/negatives.py` |
| 13 | QA quality | `answer_type`, question-form / answer-type checks, independent answer recomputation from the evidence events (disagreement rejects) | `validation/qa_checks.py` |
| 14 | Confidence calibration | per-source score distributions in statistics, `calibration.status: uncalibrated` until labels exist, `video-dataset calibrate` (accuracy, Brier, ECE, false-accept / reject, recommended per-source thresholds) | `dataset/calibration.py`, `dataset/stats.py` |
| 15 | Fail-closed policy | versioned policy engine with required checks per record type and `reason_codes`; strict subset | `validation/policy.py` |
| 16 | Technical quality | `frozen`, `low_fps`, `extreme_aspect_ratio`, `frame_count_mismatch` flags, `quality.policy_version` | `validation/quality.py`, `preprocessing/stage.py` |
| 17 | Duration / sampling | `duration_<band>` subsets, duration distributions per record type, sampling strategy in the manifest | `dataset/curation.py`, `dataset/stats.py` |
| 18 | Schema integrity | `schema_version` on every record, schema + reference + JSONL/Parquet gates | `dataset/release.py` |
| 19 | Deduplication | `duplicate_method` on validation info, duplicate reason codes, strict subset excludes near-duplicate videos | `validation/stage.py`, `validation/policy.py` |
| 20 | Rights / provenance | `rights_status`, `rights_<status>` subsets, `export.require_rights`, `final/provenance.jsonl`, `video-dataset remove-source` with tombstones | `dataset/curation.py`, `dataset/takedown.py` |
| 21 | Optional-stage coverage | `coverage` per record / manifest, coverage rates in statistics, `no_audio` vs `not_run` | `dataset/coverage.py` |
| 22 | Reproducibility | `environment` manifest (commit, python, platform, ffmpeg, packages, models, config hashes), `pipeline.rerun_on_config_change` | `utils/environment.py`, `pipeline/runner.py` |
| 23 | Real-world quality proof | `release-check` report with quality metrics, review queue + human labels loop feeding calibration, end-to-end test | `dataset/release.py`, `dataset/review.py` |

Still a human task (the code only prepares it): building the gold benchmark of real videos. Use
`video-dataset review-queue` to collect candidates, label them, and `calibrate` to measure.

## 1. Scope and how to use this document

This is an engineering improvement backlog, not a claim that all 23 items are confirmed software defects. The repository README describes several existing safeguards, including a validation stage, record statuses, deduplication, per-video checkpoints, dataset statistics, and synthetic end-to-end tests. Each item below should therefore be checked against the current implementation and real output before changing code.

The recommendations prioritize **correct and traceable data over volume**. The pipeline should never mark data as trustworthy just because a model returned valid JSON or a heuristic score passed a threshold.

### Recommended status labels

- **Confirmed defect:** a reproducible test or real record demonstrates failure.
- **Risk / needs verification:** the design can fail in a known way; verify whether current code already prevents it.
- **Improvement:** strengthens reliability, measurement, or maintainability even if there is no current bug.

### High-level priorities

| Priority | Focus | Why it comes first |
|---|---|---|
| P0 | Prompt-to-video alignment, semantic verification, timestamps, media references, camera-motion consistency | Incorrect core annotations poison training data. |
| P1 | Transcript/event alignment, temporal reasoning, QA and hard-negative correctness, duplicate-safe splits | These errors can create misleading labels and evaluation leakage. |
| P2 | spaCy preprocessing, calibrated quality scores, human review, technical-quality policies | These improve annotation coverage and make quality decisions more defensible. |
| P3 | Provenance, reproducibility, regression benchmarks, reports, operations | These make dataset releases auditable and repeatable at scale. |

---

# 2. The 23 issues and their best solutions

## Issue 1 — Prompt or description does not accurately match the actual video

**Status:** Core risk; verify on real clips.  
**Severity:** Critical (P0)

### Problem
A generated caption or `generation_prompt` may mention the wrong object, action, environment, camera behavior, lighting, or event. Text can sound plausible while not being supported by the frames. Text-only checks cannot prove that the text is visually correct.

### Why it matters
For video-generation training, mismatched prompt/video pairs teach the model the wrong relationship between language and visual output. For video understanding, the same mismatch produces incorrect labels and unreliable evaluation results.

### Recommended solution
1. Keep the generated description as a **candidate annotation**, not a final truth.
2. Verify each important claim against multiple frames sampled across the clip—not only the first frame.
3. Use a capable video-language model (VLM), or an image VLM over timestamped frame samples, to check objects, actions, setting, and camera claims.
4. Use CLIP-style text-image similarity only as a supporting signal. Similarity is not proof that every claim is correct and should not be the sole acceptance gate.
5. Store evidence for each important claim: supporting frame IDs, timestamps, verifier result, verifier model/version, and any contradictions.
6. If the model cannot verify a claim, remove it, mark it as unknown, or send the record to review. Do not guess.

### Implementation guidance
Use a structured annotation schema with separate fields for `subject`, `action`, `environment`, `camera`, `lighting`, and `temporal_progression`. Ask the verifier to return claim-level results such as `supported`, `partially_supported`, `unsupported`, and `unknown`. Require evidence timestamps for temporal claims.

### Acceptance criteria
- A human-labelled benchmark measures precision for objects, actions, camera claims, and scene attributes separately.
- Unsupported claims do not pass as `accepted`.
- The exported record allows an auditor to locate the frames used as evidence.

---

## Issue 2 — Semantic validation relies too heavily on heuristics

**Status:** Risk / needs verification for every configured provider.  
**Severity:** Critical (P0)

### Problem
Brightness, sharpness, optical flow, frame changes, lexical overlap, and other low-level checks can validate technical properties, but they cannot reliably establish scene meaning. A default or fallback heuristic vision provider may measure motion without knowing what action occurred.

### Why it matters
A pipeline can generate technically plausible metadata while falsely claiming that a person runs, an object falls, or a camera zooms. A score such as `0.8` can look authoritative even if it measures something unrelated to semantic truth.

### Recommended solution
1. Separate **technical validation** from **semantic validation**. Neither should substitute for the other.
2. Use the heuristic provider for image measurements and motion signals only.
3. For semantic records, require a VLM-capable verifier or explicitly mark semantic verification as unavailable.
4. Combine independent evidence: VLM verification, object/action detection where suitable, timestamps, OCR/ASR evidence where relevant, and deterministic checks.
5. Treat model disagreement as a review signal instead of averaging incompatible scores into a false sense of certainty.
6. Do not silently treat `provider: none`, a mock provider, or a heuristic-only run as semantically verified.

### Implementation guidance
Add an explicit field such as `validation.capabilities_available` and record which validators ran. Introduce a policy matrix: for example, a record requiring semantic action labels cannot be `accepted` when no semantic verifier ran.

### Acceptance criteria
- Tests prove that a heuristic-only configuration cannot claim semantic verification.
- Output distinguishes `not_checked` from `passed`.
- Validation reports show counts for each verifier/provider and the failure reasons.

---

## Issue 3 — Camera-motion labels conflict with measured motion

**Status:** Risk / needs verification.  
**Severity:** High (P0)

### Problem
A record may state `camera.movement = tracking` while measurements suggest `zoom_in`, `pan_left`, or no reliable camera motion. Optical flow can also confuse camera movement with moving subjects, parallax, or a moving background.

### Why it matters
Camera motion is a meaningful condition in cinematic and video-generation datasets. Conflicting labels can teach a model the wrong framing or motion behavior.

### Recommended solution
1. Keep **camera motion** separate from **subject motion**.
2. Estimate global motion from background features or a robust global-motion model; do not classify camera movement from raw optical-flow magnitude alone.
3. Detect zoom-like motion separately from pan/tilt/translation where possible; retain `unknown` if signals are ambiguous.
4. Ask the VLM for a camera-motion label, then compare it with measured motion and the supporting frame sequence.
5. Add a consistency rule: material disagreement should produce `review`, not automatically select one label.
6. Represent uncertainty explicitly, for example `camera_motion: unknown` plus a reason, rather than forcing a label.

### Implementation guidance
Define an agreed label ontology (`static`, `pan_left`, `pan_right`, `tilt_up`, `tilt_down`, `zoom_in`, `zoom_out`, `tracking`, `handheld`, `unknown`) and document that labels may overlap only where allowed. If multiple simultaneous movements are supported, use a list with clear definitions rather than an ambiguous single string.

### Acceptance criteria
- A labelled camera-motion test set includes static shots, camera moves, moving subjects, and mixed motion.
- Camera and subject motion are stored separately.
- Conflicts are visible in `validation.checks` and cannot pass unnoticed.

---

## Issue 4 — Clip boundaries and extracted media may not match intended timestamps

**Status:** Risk / needs verification, especially for stream-copy cutting.  
**Severity:** Critical (P0)

### Problem
A clip requested for time `start_time` to `end_time` may begin or end at a nearby keyframe, include extra content, or have a different duration from the metadata. Frame-rate conversion, variable frame rate, timestamp offsets, and inaccurate seeking can also cause boundary drift.

### Why it matters
The description, labels, transcript, and event timings can all refer to a different moment than the exported media actually contains. This invalidates temporal supervision.

### Recommended solution
1. Establish a single canonical timeline from `ffprobe`/FFmpeg metadata and use it throughout the pipeline.
2. Keep requested timestamps and verified actual timestamps separate.
3. For boundary-critical training clips, prefer accurate trimming with re-encoding rather than assuming stream-copy cuts are frame-exact. A faster stream-copy path may be used only where its tolerance is acceptable and verified.
4. After cutting, probe the output duration, frame timestamps, frame count, and audio duration.
5. Decode frames near the beginning and end of every clip and verify they fall inside the intended segment.
6. Define tolerances per use case, such as a maximum boundary error in milliseconds or frames. Send clips outside tolerance to `review` or `rejected`.
7. Make all clip intervals half-open or inclusive in one documented way, and use that convention everywhere.

### Implementation guidance
Store `requested_start`, `requested_end`, `actual_start`, `actual_end`, `duration_error_ms`, extraction method, FFmpeg version, and validation result. Re-check the output rather than trusting the command that created it.

### Acceptance criteria
- Synthetic tests verify cuts at known timestamps, including non-keyframe boundaries and variable frame rate where supported.
- Every exported clip passes duration and boundary checks.
- A failed clip is not exported as accepted merely because FFmpeg exited successfully.

---

## Issue 5 — Exported records can reference missing or stale media files

**Status:** Risk / needs verification across cleanup, export, and packaging.  
**Severity:** Critical (P0)

### Problem
JSONL or Parquet records may reference a frame or clip that was deleted, moved, renamed, or omitted from a published bundle. A path can exist locally but not in the distributed dataset.

### Why it matters
Training jobs fail at load time, dataset consumers get incomplete samples, and it becomes difficult to reproduce an annotation or verify a claim.

### Recommended solution
1. Add a final **referential-integrity validator** after export and before publication.
2. For every `media_path`, resolve the path relative to a declared dataset root and check existence, readability, and expected media type.
3. Store a content hash for each referenced asset and include it in the manifest.
4. Make cleanup aware of the export contract: either retain all referenced media or generate a package that includes every referenced asset before deleting intermediates.
5. Run a fresh-process test against the final package—not only against the working directory.
6. If media are intentionally external, use a documented stable URI plus checksum instead of a misleading local path.

### Implementation guidance
Create an export manifest with one row per record and referenced asset. Validate that every reference resolves and that there are no unreferenced assets unless explicitly permitted. Fail publication if any required reference is missing.

### Acceptance criteria
- `package --include-media` or equivalent produces a self-contained bundle whose references all resolve.
- A test that removes a referenced clip causes validation to fail.
- The dataset manifest reports the number of broken and verified references; publication requires zero broken required references.

---

## Issue 6 — spaCy is not used as a complete NLP preprocessing and consistency stage

**Status:** Improvement / needs verification against current stage wiring.  
**Severity:** High (P1)

### Problem
spaCy may be used mainly for question generation, while prompt cleanup, grammatical normalization, action extraction, entity consistency, and text validation remain ad hoc or absent. At the same time, spaCy cannot see the video and should not independently decide that an action happened.

### Why it matters
Inconsistent wording and weak event representations reduce search quality, increase duplicate descriptions, and create QA that is grammatically or semantically defective.

### Recommended solution
Use spaCy as a **text-structure and consistency layer**, not a visual truth detector:
1. Normalize whitespace, Unicode, punctuation, and common formatting artifacts while preserving the raw original text.
2. Extract candidate entities, noun phrases, verbs, dependency relations, tense, and negation.
3. Convert candidate event wording into a normalized representation such as subject–predicate–object plus modifiers, where possible.
4. Detect contradictions within text fields, such as “camera remains static” alongside “camera rapidly pans.”
5. Validate QA question grammar and ensure answer types match the question (person/object/action/time/location).
6. Compare extracted text claims against VLM-grounded visual annotations, keeping uncertain matches for review.
7. Keep language-specific processing configurable. Do not apply English-only rules blindly to Hindi, Hinglish, or other languages.

### Implementation guidance
Add a dedicated `TEXT_PREPROCESSING` or `NLP_VALIDATION` stage with versioned outputs. Store the raw text, normalized text, spaCy model name/version, extracted predicates/entities, and rule results. spaCy-derived annotations should be labelled as text-derived candidates, not verified visual facts.

### Acceptance criteria
- Unit tests cover punctuation, negation, tense, pronouns, multiple actions, multilingual text, and malformed captions.
- The NLP stage can be rerun independently when models or rules change.
- NLP checks improve measurable QA quality without introducing unverified visual labels.

---

## Issue 7 — Similar clips from the same source can leak across train, validation, and test splits

**Status:** Risk / needs verification of split assignment and near-duplicate handling.  
**Severity:** Critical (P1)

### Problem
Randomly splitting individual clip records can put adjacent shots, overlapping clips, re-encodes, or near-identical segments from one source video into different splits. The model may effectively see test content during training.

### Why it matters
Validation metrics look stronger than true generalization. This is especially damaging for video QA, retrieval, and prompt-to-video evaluation.

### Recommended solution
1. Assign splits at the **source-video group level**, not at the individual record level.
2. Keep every scene, clip, frame, event, transcript segment, and QA record derived from a source video in the same split.
3. Detect identical source videos using stable source IDs and file hashes. Detect near duplicates using perceptual hashes and/or embeddings before assigning splits.
4. Group alternate uploads or edited copies where feasible, using conservative similarity rules and manual review for borderline matches.
5. Generate and freeze a versioned split manifest. Do not silently reshuffle on each export.
6. Check that no group or known duplicate pair crosses split boundaries.

### Implementation guidance
Use a stable group key such as `source_group_id`, distinct from `video_id` and `record_id`. Apply splitting after duplicate clustering and before record-level export. For evaluation releases, document the split strategy and seed/salt.

### Acceptance criteria
- An automated test proves no source group crosses splits.
- Duplicate detection is run before split assignment.
- Split membership remains stable for an unchanged dataset version.

---

## Issue 8 — A long-video description may combine multiple shots into one training prompt

**Status:** Risk for generation-oriented records; verify intended target format.  
**Severity:** High (P1)

### Problem
A video-level or multi-shot description may include several subjects, locations, actions, and camera movements. If it is attached to one short clip, the prompt no longer describes a single visual target.

### Why it matters
For prompt-conditioned video generation, a prompt should have a clear relationship to the target clip. Combining separate scenes into one prompt creates a contradictory or impossible target, unless the task explicitly asks for multi-shot generation.

### Recommended solution
1. Separate **clip-level generation captions** from **scene summaries** and **whole-video summaries**.
2. For a single-shot training target, describe only the clip's visible content and time range.
3. For multi-shot or long-video generation, represent the prompt as a sequence of shot-level instructions with explicit order, duration, and transition metadata.
4. Detect scene changes inside a proposed generation clip. Split the target or label it explicitly as a multi-shot clip.
5. Ensure a record's prompt scope matches its media scope.

### Implementation guidance
Add fields such as `annotation_scope: frame|clip|scene|video` and `generation_target: single_shot|multi_shot|not_for_generation`. Make export filters explicit so a training job cannot accidentally mix incompatible targets.

### Acceptance criteria
- Single-shot generation records contain no unsupported cross-scene details.
- Multi-shot records have ordered shot instructions and aligned timestamps.
- Training export can select one target type without hand-written data cleanup.

---

## Issue 9 — Transcript segments do not align with the actual clip or visual event

**Status:** Risk / needs verification.  
**Severity:** High (P1)

### Problem
A transcript from the full video may be attached to a shorter clip without clipping word timestamps to the segment boundaries. Speech may also be dubbed, off-screen, translated, or unrelated to the visible action. ASR may misrecognize names, accents, or multilingual speech.

### Why it matters
Speech-derived QA may claim that a visible person said something when the audio does not support speaker identity, or may place dialogue in the wrong time segment.

### Recommended solution
1. Store ASR words or segments with start/end timestamps and model confidence where available.
2. When producing a clip record, select only transcript segments that overlap the clip interval and shift timestamps to the clip-relative timeline when required.
3. Preserve both absolute source-video times and clip-relative times with clear field names.
4. Distinguish `speech_content`, `speaker_id`, and `visible_person_match`. Do not infer that a visible person is speaking without synchronization evidence.
5. Mark dubbed narration, off-screen speech, and uncertain speaker attribution separately where detectable.
6. Run language/script checks and review low-confidence, unusually short, or anomalous transcripts.
7. Do not use transcript content as proof of a visual event unless there is direct supporting visual evidence.

### Implementation guidance
Maintain one canonical transcript in source-video coordinates. Derive clip transcript views from it instead of repeatedly transcribing each clip independently unless clip-specific ASR is intentionally needed. Record language, ASR model/version, timestamps, and confidence.

### Acceptance criteria
- No clip contains transcript words outside its declared source interval, except documented context padding.
- Absolute and relative timestamps are not mixed.
- Visual speaker attribution requires an independent check or remains unknown.

---

## Issue 10 — Event timestamps are inherited from shot boundaries rather than measured action timing

**Status:** Risk / needs verification.  
**Severity:** High (P1)

### Problem
A shot can last several seconds and contain several actions. Assigning an event the full shot interval—or estimating its timing from a scene boundary—does not precisely localize when the action starts or finishes.

### Why it matters
Temporal QA such as “When does the person pick up the cup?” or duration questions becomes inaccurate. Event boundaries are central labels for temporal reasoning.

### Recommended solution
1. Keep **shot boundaries** and **event boundaries** as separate concepts.
2. Sample frames or short frame sequences around the event and estimate action start/end within the shot.
3. Use temporal models or frame-level action analysis where the task requires fine timing.
4. Use an uncertainty interval where exact boundaries are ambiguous.
5. Record the evidence frames and the temporal-resolution limits of the method.
6. If an event cannot be timed at the required resolution, use a coarse label or exclude it from fine-grained temporal tasks.

### Implementation guidance
Represent an event as `start_time`, `end_time`, `temporal_precision`, `start_uncertainty_ms`, and `end_uncertainty_ms` where applicable. Never imply millisecond precision if the source sampling cadence only supports coarse estimates.

### Acceptance criteria
- A manually timed benchmark reports event start/end error.
- Shot boundary times are not automatically treated as action boundaries.
- QA questions requiring fine timing are generated only from events with adequate timing precision.

---

## Issue 11 — Temporal relations and causal claims are not adequately verified

**Status:** Risk / needs verification.  
**Severity:** High (P1)

### Problem
Relations such as `BEFORE`, `DURING`, `OVERLAPS`, `CAUSES`, or “because” may be generated from approximate timestamps or narrative assumptions. Temporal order is measurable; causation often is not directly visible in a video.

### Why it matters
False causal labels are more damaging than ordinary caption errors because they teach a model to invent explanations. A relation can also be mathematically inconsistent with event intervals.

### Recommended solution
1. Validate interval relations deterministically from event start/end values. For example, a `BEFORE` relation must satisfy the documented interval semantics.
2. Use a single documented interval-relation ontology and test all permitted relations.
3. Distinguish **observed temporal ordering** from **inferred causal explanation**.
4. Default causal inference to disabled or review-only unless supported by a task-specific evaluation protocol.
5. Require explicit evidence for every relation and record whether it was measured, model-inferred, or human-reviewed.
6. Detect cycles or contradictory relations where the ontology says they are impossible.

### Implementation guidance
Add relation validation separate from relation generation. A causal model may propose a relation, but it should not make the relation true simply by producing confident language. Consider a graph consistency pass before export.

### Acceptance criteria
- Unit tests cover boundary equality, overlap, containment, ordering, and invalid intervals.
- Causal claims have a provenance/source field and do not pass under a temporal-only verifier.
- Contradictory event graphs are flagged.

---

## Issue 12 — Hard negatives can be confused with positive training labels

**Status:** Risk / needs verification in exports and training adapters.  
**Severity:** High (P1)

### Problem
A plausible but incorrect caption or answer intended as a hard negative may be exported in a field or file that downstream code treats as a valid positive. Negation may also be lost when text is normalized.

### Why it matters
The model may be trained to generate or accept the wrong caption, undermining the purpose of hard-negative supervision.

### Recommended solution
1. Store hard negatives in a clearly typed field, separate from the positive caption/answer.
2. Include a reason for the negative, such as wrong object, action, direction, temporal order, or camera motion.
3. Validate each negative against the media and the positive record; do not assume any generated alternative is actually false.
4. Ensure negative records reference the same evidence interval as the associated positive, unless the task explicitly contrasts different clips.
5. Keep negatives out of ordinary positive-caption exports by default. Require training adapters to opt in explicitly.
6. Preserve negation and contradiction signals during text preprocessing.

### Implementation guidance
Use a schema such as `hard_negatives: [{text, negative_type, evidence_interval, verification_status}]`. Make dataset statistics report positive and negative counts separately.

### Acceptance criteria
- Positive and negative data are distinguishable without parsing text.
- Every exported negative has a documented type and validation state.
- Tests prove that the default positive-training export does not accidentally include negatives as targets.

---

## Issue 13 — Generated QA has grammatical, answerability, or evidence-grounding problems

**Status:** Risk / needs verification with a human-reviewed QA benchmark.  
**Severity:** High (P1)

### Problem
Template- or NLP-generated questions can be awkward, ambiguous, redundant, or impossible to answer from the video. A question may ask “who” when the evidence only identifies an object, or ask about an action that the annotation does not support.

### Why it matters
Unanswerable QA teaches models to guess. Repeated or malformed questions add volume but not useful supervision.

### Recommended solution
1. Generate QA only from validated events and evidence, with the target answer type explicit.
2. Add deterministic checks for question form, answer type, empty answers, invalid timestamps, unsupported entities, and duplicates.
3. Ask an independent verifier to answer the question from the evidence clip/frames without seeing the proposed answer, then compare the result.
4. Reject or review questions where the verifier cannot answer, disagrees materially, or needs information outside the evidence interval.
5. Measure grammar and answerability separately from visual grounding.
6. Keep template, spaCy, and LLM-generated QA sources identifiable so quality differences can be measured.

### Implementation guidance
Each QA record should include `question`, `answer`, `answer_type`, `evidence_ids`, `evidence_interval`, `generator`, `generator_version`, `verification_result`, and `validation_status`. Avoid using only lexical overlap as evidence that an answer is correct.

### Acceptance criteria
- A human-labelled sample estimates answerability and answer correctness.
- QA checks include both positive examples and deliberately unanswerable examples.
- A question cannot be accepted without traceable evidence.

---

## Issue 14 — Confidence scores are not calibrated to actual accuracy

**Status:** Improvement / verify score semantics across all providers.  
**Severity:** High (P1)

### Problem
Scores produced by different models, verifiers, or heuristics may have different meanings. A model's self-reported confidence, ASR probability, CLIP similarity, verifier score, and measured optical-flow consistency are not interchangeable probabilities of correctness.

### Why it matters
A global threshold can accept unreliable items or reject good ones. A number such as `0.9` may be interpreted as “90% likely to be correct” even when it has never been calibrated.

### Recommended solution
1. Record `confidence_source`, model/provider/version, and the precise meaning of each score.
2. Do not compare scores from different sources as if they shared a common scale.
3. Create a human-labelled calibration set stratified by annotation type and difficulty.
4. Measure calibration with appropriate metrics such as reliability diagrams, expected calibration error (ECE), Brier score, precision/recall, and false-accept rate.
5. Set thresholds per annotation/task and optimize for the cost of false acceptance, not for dataset size.
6. Store a missing confidence as `null` or `unavailable`; never substitute zero or an invented score.

### Implementation guidance
Keep raw model scores and final policy decisions separate. Consider an overall quality score only as a summary, with component scores and `components_used` exposed. A low overall score should not hide a critical failure such as a missing media file.

### Acceptance criteria
- Each released score has documented semantics.
- Thresholds are selected using labelled evaluation data.
- Reports include false-accept and false-reject rates by annotation type.

---

## Issue 15 — Acceptance policy may treat missing validation as a pass, or mix incompatible checks

**Status:** Risk / verify actual status-assignment logic.  
**Severity:** Critical (P0)

### Problem
Optional stages can be skipped or unavailable. If policy logic confuses “not checked” with “passed,” or calculates a broad score from only the checks that happened to run, incomplete records may be marked `accepted`.

### Why it matters
The dataset's acceptance label becomes meaningless: records can pass precisely because the relevant validation component was unavailable.

### Recommended solution
1. Define required checks by record type. A generation clip, temporal event, frame caption, and QA pair have different required validations.
2. Use explicit states such as `passed`, `failed`, `not_run`, `not_applicable`, and `error` for each check.
3. Make acceptance policy fail closed for critical checks: missing required semantic validation, invalid timestamps, or missing referenced media should block acceptance.
4. Keep `accepted`, `review`, `rejected`, and `duplicate` semantics documented and mutually exclusive.
5. Store machine-readable rejection reasons rather than only one overall status.
6. Ensure review records cannot enter a strict training release accidentally.

### Implementation guidance
Build a small policy engine that consumes named check results and a versioned rule configuration. Do not let individual validators make final status decisions independently. Record `policy_version` in every validated record.

### Acceptance criteria
- Tests show that each missing required check results in `review` or `rejected`, never a silent pass.
- Every status has a documented downstream export behavior.
- A strict training export contains only records meeting the policy for their record type.

---

## Issue 16 — Video technical-quality checks do not cover all defects relevant to training

**Status:** Improvement / verify current quality features against intended use.  
**Severity:** Medium–High (P2)

### Problem
Brightness, contrast, sharpness, and basic motion metrics do not fully characterize usability. A clip can be technically decodable but still be corrupted, frozen, almost entirely black, severely blurred, overloaded with watermarks, or too low-resolution for the intended training task.

### Why it matters
Low-quality clips waste compute and can bias the model toward artifacts. However, automatically discarding all low-light or blurry footage can also remove valid cinematic examples, so quality must be task-specific.

### Recommended solution
1. Run decoding and integrity checks first: frame decode failures, zero/incorrect duration, truncated streams, invalid timestamps, missing audio/video streams, and inconsistent metadata.
2. Measure duplicate/frozen frames, blur, brightness distribution, resolution, frame rate, compression artifacts, and extreme aspect ratios.
3. Use thresholds by target task rather than one universal “good video” score.
4. Preserve useful but stylistically unusual content—night scenes, intentional blur, grain, slow motion—when it matches the intended dataset.
5. Store each measurement and the policy decision separately.
6. Flag severe technical failures for rejection; route borderline or task-dependent cases to review.

### Implementation guidance
Produce a `technical_quality` object with individual metrics, threshold policy version, and findings. Avoid hiding all measurements inside a single score.

### Acceptance criteria
- The technical-quality benchmark includes corrupted, frozen, low-light, blurry, low-resolution, and intentionally stylized clips.
- Rejections are explainable through named checks.
- Policies can be configured for different dataset subsets.

---

## Issue 17 — Minimum clip duration and sampling density may not fit the task

**Status:** Improvement / verify dataset distribution.  
**Severity:** Medium–High (P2)

### Problem
Very short clips may not contain enough context to establish an action; overly long clips may contain multiple unrelated events. A fixed frame-sampling rate may miss fast motion or waste compute on static sections.

### Why it matters
A clip needs sufficient context for its annotation, but there is no single ideal duration for every task. A captioning dataset and a fine-grained action-timing dataset have different needs.

### Recommended solution
1. Measure the distribution of clip duration, frame rate, number of shots, action count, and event coverage.
2. Define duration bands for task-specific exports, not one global duration cutoff.
3. Require adequate visual context for an action label, and mark action labels as uncertain when the relevant start/end is not visible.
4. Use adaptive sampling: combine regular temporal samples with extra samples around scene changes, motion peaks, or candidate event boundaries.
5. Preserve original sampling timestamps and sampling strategy in metadata.
6. Evaluate missed-event rate as sampling density changes.

### Implementation guidance
Add configurable `min_duration`, `max_duration`, and task-specific policy profiles. If a clip is divided artificially due to a maximum scene duration, label it as an artificial split and do not treat the split as a true scene cut.

### Acceptance criteria
- Dataset statistics show the duration distribution by record type and target task.
- Fast actions are tested to ensure the sampling strategy detects them sufficiently often.
- Duration decisions and artificial splits are traceable.

---

## Issue 18 — Record schemas and cross-record references may be inconsistent

**Status:** Risk / improvement; verify current schema validators and exports.  
**Severity:** High (P1)

### Problem
A record may have an unknown `video_id`, invalid `scene_id`, missing `record_id`, incorrect timestamp order, an incompatible enum, a malformed nested field, or a media path that does not match the declared record type. JSONL and Parquet can also interpret null/mixed fields differently.

### Why it matters
The dataset may load but contain logically broken references, or one export format may disagree with another. Such issues often appear downstream when data is too expensive to reprocess.

### Recommended solution
1. Use a versioned typed schema for every record type and validate before writing.
2. Enforce invariants such as `start_time < end_time`, `0 <= start_time`, stable IDs, allowed enum values, valid confidence ranges, and parent-record existence.
3. Validate relationships among videos, scenes, frames, clips, events, transcripts, and QA records.
4. Use explicit nullability and stable Parquet types; do not rely on type inference from JSONL samples.
5. Add schema migration/versioning rules instead of silently changing field meaning.
6. Run a final integrity pass across the full export and compare JSONL vs Parquet record counts and identifiers.

### Implementation guidance
Store `schema_version` on records or in a versioned dataset manifest. Validate both individual objects and dataset-level invariants. Treat a schema validation failure as a blocking error for strict releases.

### Acceptance criteria
- Invalid timestamps, missing parents, and wrong field types fail tests.
- JSONL and Parquet exports have matching record IDs and counts.
- Every released dataset states its schema version.

---

## Issue 19 — Deduplication may miss semantic duplicates or remove useful variants

**Status:** Improvement / verify duplicate clustering on real data.  
**Severity:** Medium–High (P2)

### Problem
Exact hashes detect byte-identical files, but re-encoded or resized videos can look different at the byte level. Text near-duplicate checks can miss semantically identical captions, while aggressive similarity thresholds can wrongly merge different actions or useful camera variants.

### Why it matters
Duplicates inflate dataset size, bias splits and evaluation, and over-represent popular scenes. Over-aggressive deduplication can remove valid diversity.

### Recommended solution
Use a tiered deduplication system:
1. Exact file and content hashes for identical assets.
2. Perceptual hashes or sampled-frame fingerprints for resized/re-encoded video.
3. Temporal overlap and source IDs for overlapping clips from the same video.
4. Text embeddings or normalized text similarity for candidate caption/QA duplicates.
5. Human review for borderline near-duplicate clusters.
6. Keep deduplication decisions and representative IDs rather than destructively erasing records.
7. Run deduplication before split assignment and again on final records for type-specific duplication.

### Implementation guidance
Store `duplicate_group_id`, `duplicate_of`, similarity method, score, and decision source. Distinguish redundant duplicates from legitimate alternate views, edits, resolutions, and differently labelled tasks.

### Acceptance criteria
- Evaluation reports precision and recall of duplicate detection on a labelled sample.
- Related duplicate clusters cannot cross splits.
- Every removed/excluded duplicate is traceable to a retained representative.

---

## Issue 20 — License, provenance, and usage rights are not sufficiently enforced end to end

**Status:** Improvement / compliance-critical for public or commercial releases.  
**Severity:** High (P3)

### Problem
Recording a source URL or YouTube license field is not, by itself, proof that downloading, redistributing, training on, or publishing the media is allowed. Rights may differ for the video, audio, music, people shown, and third-party material inside the clip.

### Why it matters
A technically high-quality dataset can still be unsuitable for release or commercial training. Missing provenance makes later removal and audit difficult.

### Recommended solution
1. Define a documented source-eligibility policy before download and processing.
2. Store source URL, platform ID, uploader/channel where permitted, retrieval time, declared license, license evidence, access restrictions, and source hash.
3. Treat unknown or restricted rights as a separate status; do not automatically equate “publicly viewable” with “free to redistribute.”
4. Track rights at the media/asset level and propagate restrictions to every derived frame, clip, transcript, and annotation.
5. Provide a takedown/deletion mechanism that finds all derived artifacts and exports for a source asset.
6. Review privacy and identity rules, including not identifying people from video unless there is a valid and explicit reason under the applicable policy.
7. Get appropriate legal review for the intended use and jurisdiction; this document is not legal advice.

### Implementation guidance
Add a provenance manifest and a release gate. Use `rights_status` values such as `verified`, `restricted`, `unknown`, and `removed`, with clear downstream export rules. Do not expose credentials or unnecessary personal data in metadata.

### Acceptance criteria
- Every released media asset has a traceable source and rights status.
- Assets with unknown/restricted status are excluded from releases that require verified rights.
- A source can be removed from working data and all exports using its provenance ID.

---

## Issue 21 — Optional dependencies and skipped stages can silently reduce annotation coverage

**Status:** Risk / needs verification of configuration and export policy.  
**Severity:** Medium–High (P2)

### Problem
Optional stages such as transcription, OCR, vision verification, or audio event detection may be missing, disabled, or fail for a particular video. Continuing can be appropriate for batch robustness, but the resulting record may have less evidence than expected.

### Why it matters
A dataset assembled from mixed configurations can have inconsistent annotation depth while appearing uniform. A missing OCR/transcript stage may be mistaken for “no text” or “no speech.”

### Recommended solution
1. Record for each video which stages were requested, executed, skipped, failed, or unavailable.
2. Distinguish `no_speech_detected` from `transcription_not_run`, and `no_text_detected` from `ocr_not_run`.
3. Define minimum required modalities for each dataset subset or task.
4. Keep the batch running when safe, but prevent incomplete records from entering subsets that require the missing modality.
5. Include dependency/model availability in the run manifest and final statistics.
6. Add explicit diagnostics for fallback provider activation.

### Implementation guidance
Use a per-record `coverage` object or normalized stage manifest. Dataset filters should select records by capability/coverage, not merely by overall status.

### Acceptance criteria
- Reports show coverage rates for vision, audio, ASR, OCR, temporal verification, and media checks.
- A failed optional stage is not misreported as a valid negative finding.
- Strict subsets exclude records missing required modalities.

---

## Issue 22 — Model, dependency, and configuration changes can make datasets irreproducible

**Status:** Improvement / verify lock files and metadata completeness.  
**Severity:** Medium–High (P3)

### Problem
The same source video can yield different annotations when Python packages, FFmpeg, model versions, prompt templates, model weights, or configuration defaults change. Without sufficient provenance, the pipeline cannot reliably explain or reproduce those differences.

### Why it matters
Regression investigation becomes guesswork, and regenerating a dataset release may silently alter labels or split membership.

### Recommended solution
1. Pin production dependency versions and record the Python version, OS/platform, FFmpeg/ffprobe versions, and optional provider versions.
2. Record model repository/name, exact revision or checksum, inference settings, prompt-template version, and provider/API model identifier where available.
3. Hash the effective configuration for each stage and store it with the stage checkpoint and output artifacts.
4. Version annotation schemas, validation rules, thresholds, split manifests, and dataset releases.
5. When a stage's effective inputs change, invalidate and rerun only the dependent stages; do not silently reuse stale outputs.
6. Store enough metadata for reproducibility without exposing API keys, cookies, or other secrets.

### Implementation guidance
Create a run-level manifest containing the code commit, configuration hash, dependency lockfile hash, model identifiers, stage statuses, start/end times, and source manifest. Separate immutable raw source metadata from regenerable derived artifacts.

### Acceptance criteria
- A release can be traced to a code commit and exact configuration.
- Changing a relevant config/model version triggers the expected reprocessing.
- No secrets are written to manifests or logs.

---

## Issue 23 — Tests and reports do not yet prove real-world annotation quality

**Status:** Improvement; verify breadth of actual tests and evaluation datasets.  
**Severity:** High (P3; foundational for safely shipping changes)

### Problem
Synthetic end-to-end tests are useful for testing known cuts, timestamps, text overlays, and pipeline wiring, but they cannot represent the variety of real YouTube videos: fast actions, camera shake, occlusion, multiple languages, music, narration, subtitles, transitions, low-quality footage, and model hallucinations.

### Why it matters
A green test suite can prove that the pipeline runs without proving that the dataset is correct. Without a benchmark, new models or rules can reduce label quality while improving speed or volume.

### Recommended solution
1. Build a **gold evaluation set** of real videos with human-checked scene/clip boundaries, captions, camera motion, actions, transcript timing, and QA answerability.
2. Cover diverse genres, languages, lengths, frame rates, video quality, action speeds, shot types, and difficult cases.
3. Measure quality by task: boundary error, caption claim precision, action-label precision/recall, camera-motion accuracy, transcript timestamp error, QA answerability/correctness, duplicate detection quality, media-reference integrity, and acceptance false-positive rate.
4. Keep synthetic tests for deterministic mechanics and add real-video regression tests for semantics.
5. Compare model/provider versions against the same frozen benchmark before changing default configuration.
6. Generate a release report with data counts, coverage, error rates, rejection reasons, score distributions, duplicate statistics, duration distributions, split balance, provenance status, and cost/runtime.
7. Introduce a human-review queue for uncertain or contradictory examples and feed reviewed corrections back into the benchmark.
8. Define release gates, such as zero broken media references, no split leakage detected, and minimum precision targets for critical labels.

### Implementation guidance
Version the benchmark and reviewer guidelines. Use double annotation on a subset to estimate disagreement. Keep benchmark videos separate from training data and prevent accidental leakage into model-development experiments.

### Acceptance criteria
- Every dataset release has a machine-generated quality report.
- CI checks deterministic invariants; scheduled or release-time evaluation checks semantic quality.
- A model/provider upgrade is not promoted solely because it is faster or generates more records.

---

# 3. Recommended target architecture

The most reliable design is a **staged, evidence-first pipeline**. Every stage should write versioned artifacts, record what it did, and make downstream decisions from explicit results rather than implicit assumptions.

```text
Source URL / Local File
        |
        v
[1. Source Eligibility + Provenance]
        |
        v
[2. Download + Media Integrity / ffprobe]
        |
        v
[3. Canonicalize Timeline + Audio/Video Metadata]
        |
        v
[4. Scene Detection + Boundary Verification]
        |
        +--------------------+
        |                    |
        v                    v
[5. Frame/Clip Sampling]  [6. Audio Extraction + ASR]
        |                    |
        v                    v
[7. Vision/VLM Annotation] [8. OCR + Audio Events]
        |                    |
        +---------+----------+
                  v
        [9. spaCy Text Normalization /
         Candidate Event & Consistency Checks]
                  |
                  v
        [10. Temporal Event/Relation Validation]
                  |
                  v
        [11. Claim-Level Evidence Verification]
                  |
                  v
        [12. Typed Schema + Referential Integrity]
                  |
                  v
        [13. Deduplication + Source-Group Splits]
                  |
                  v
        [14. Versioned Acceptance Policy]
             /          |          \
         accepted      review     rejected/duplicate
             |            |
             v            v
        [Strict Export] [Human Review + Corrections]
             |            |
             +------<-----+
                  |
                  v
        [15. Media Packaging + Final Integrity Check]
                  |
                  v
        [16. Quality Report + Dataset Release Manifest]
```

## Architectural rules

1. **Evidence before status:** generate annotations, collect evidence, validate, then assign status.
2. **Do not conflate layers:** media integrity, technical quality, text quality, semantic grounding, temporal accuracy, rights, and schema integrity are separate checks.
3. **Keep raw and derived data:** never overwrite source metadata or silently destroy the only copy of an intermediate needed to reproduce a result.
4. **Record uncertainty:** `unknown` is a valid value. Do not force labels where the evidence is insufficient.
5. **Keep stage outputs reproducible:** record model/configuration versions and hashes.
6. **Make release policy explicit:** a strict training export should include only records that meet the relevant requirements for their record type and intended task.

---

# 4. Suggested implementation roadmap

## Phase 1 — Protect dataset correctness first

- [ ] Add claim-level prompt/video verification (Issue 1).
- [ ] Ensure heuristic-only providers cannot imply semantic verification (Issue 2).
- [ ] Add camera-motion conflict checks (Issue 3).
- [ ] Verify actual clip boundaries and durations after extraction (Issue 4).
- [ ] Add full media-reference integrity checks before export/package (Issue 5).
- [ ] Make validation policy fail closed for missing required checks (Issue 15).
- [ ] Add schema and cross-record integrity tests (Issue 18).

**Exit gate:** no known critical integrity issue can pass into the strict accepted dataset unnoticed.

## Phase 2 — Fix alignment and temporal supervision

- [ ] Add dedicated spaCy preprocessing and text-consistency checks (Issue 6).
- [ ] Split by source-video groups after duplicate clustering (Issue 7).
- [ ] Separate single-shot prompts from multi-shot/video summaries (Issue 8).
- [ ] Align transcript segments with absolute and clip-relative timestamps (Issue 9).
- [ ] Validate event timing independently from shot boundaries (Issue 10).
- [ ] Add deterministic relation consistency and limit causal claims (Issue 11).
- [ ] Separate verified hard negatives from positive labels (Issue 12).
- [ ] Evaluate QA grammar, answerability, and evidence correctness (Issue 13).

**Exit gate:** temporal and QA records are traceable to the correct interval and cannot silently mix positive, negative, or unverified content.

## Phase 3 — Calibrate quality and curation

- [ ] Calibrate confidence by task and source (Issue 14).
- [ ] Expand technical-quality checks with task-specific policies (Issue 16).
- [ ] Review clip-duration and adaptive-sampling strategy (Issue 17).
- [ ] Evaluate duplicate detection on labelled near-duplicate examples (Issue 19).
- [ ] Report missing/failed optional-stage coverage (Issue 21).

**Exit gate:** quality decisions are measurable, explainable, and appropriate to the intended training task.

## Phase 4 — Production readiness and release governance

- [ ] Add rights/provenance enforcement and source-removal tracking (Issue 20).
- [ ] Pin dependencies and version code, models, configurations, and schemas (Issue 22).
- [ ] Create a human-labelled real-video benchmark and release report (Issue 23).
- [ ] Validate the complete packaged dataset in a clean environment.
- [ ] Publish a dataset card describing intended use, limitations, collection policy, schema, and known biases.

**Exit gate:** each release is reproducible, auditable, tested on real examples, and accompanied by clear quality and provenance information.

---

# 5. Recommended strict-release acceptance policy

A record should be included in the strict training export only when all required checks for its record type pass. A suggested policy is:

| Check | Strict-release rule |
|---|---|
| Source eligibility / rights | Must meet the release's documented rights policy. |
| Media integrity | Must decode successfully; required media references must exist. |
| Schema and timestamps | Must pass schema validation and timestamp invariants. |
| Semantic grounding | Required claims must be supported by evidence from the relevant media interval. |
| Temporal correctness | Required timestamps/relations must pass the precision policy for that task. |
| Required modality coverage | Required ASR/OCR/vision checks must have run successfully; “not run” is not “passed.” |
| Duplicate handling | Must have a documented deduplication outcome and source-group split. |
| Positive/negative role | Positive targets and hard negatives must remain distinct. |
| Confidence / quality | Must meet thresholds calibrated on a labelled benchmark, with critical checks enforced independently. |
| Reproducibility | Release manifest must record code, schemas, configuration, model identifiers, and source provenance. |

Use `review` for uncertain or conflicting records and `rejected` for clear failures. A rejected record can remain in internal audit data, but should not enter a strict positive training release. The exact thresholds should be measured and then selected for the intended task; they should not be chosen only to maximize the number of accepted records.

---

# 6. Quality metrics to track on every dataset release

At minimum, publish these metrics in `statistics.json` or a separate quality report:

- **Integrity:** decode failures, broken media paths, missing parent references, schema-validation failures, timestamp violations.
- **Semantic quality:** claim-level precision by subject, action, environment, camera, lighting, and temporal progression.
- **Temporal quality:** clip boundary error, event start/end error, transcript timestamp error, relation inconsistency rate.
- **QA quality:** human-estimated answerability, answer correctness, evidence sufficiency, grammar/ambiguity failure rate.
- **Curation:** accepted/review/rejected/duplicate counts, reasons for review/rejection, duplicate-cluster sizes.
- **Coverage:** percentage of records with vision verification, ASR, OCR, audio events, and required media evidence.
- **Dataset composition:** duration distribution, resolution/frame-rate distribution, language distribution, video genre/source distribution, split sizes, and class/action imbalance.
- **Governance:** rights-status distribution, missing provenance count, removed-source count, release version and generation timestamp.
- **Operations:** processing time, cost per source minute, failure/retry rates, model/provider usage, and stage-level throughput.

Do not report a single overall quality score without the component metrics and the validation policy that produced it.

---

# 7. Final recommendation

Do not start by adding more models or generating more records. First make it impossible for critical media, timestamp, schema, or evidence failures to pass silently. Then benchmark prompt/video alignment and temporal correctness on real videos. After those foundations are reliable, improve spaCy-based text processing, QA generation, deduplication, quality calibration, and release automation.

The target outcome is not simply a larger dataset. It is a dataset where each label has a clear scope, each important claim has evidence, each timestamp refers to the right media, uncertain records are handled honestly, and every release can be validated and reproduced.

---

## References

- Repository and documented architecture: https://github.com/r2708/YT_A
- Related Hugging Face dataset: https://huggingface.co/datasets/raj270898/youtube-video-dataset
- Hugging Face video dataset documentation: https://huggingface.co/docs/datasets/en/video_dataset

**Review note:** This backlog is based on the repository’s published architecture/configuration and the issues discussed in the project review. Before implementation, reproduce each suspected failure against the current code and a real sample; preserve fixes already present and add regression tests for every confirmed issue.
