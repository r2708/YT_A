# video-dataset-pipeline

Turn YouTube videos (or local video files) into **temporally grounded, multimodal training data**
for LLMs, video-understanding models and video-generation models.

The pipeline downloads a video, detects shots, samples frames intelligently, transcribes speech,
reads on-screen text, analyses every shot with a swappable vision backend, builds a **timeline of
events with interval relations**, generates **evidence-linked temporal question/answer pairs**,
validates and de-duplicates everything, and exports JSONL + Parquet datasets. Processing is
checkpointed per stage in SQLite, so a batch of 1,000 videos that dies at video 347 resumes
exactly where it stopped.

```
URL -> DOWNLOAD -> PREPROCESS -> SCENE_DETECTION -> FRAME_EXTRACTION -> AUDIO -> TRANSCRIPTION
    -> OCR -> VISION_ANALYSIS -> TEMPORAL_ANALYSIS -> QA_GENERATION -> VALIDATION -> EXPORT
```

> Only process videos you have the right to download and analyse. The downloader stores public
> video metadata (title, channel, duration, license) and nothing about viewers or commenters.

---

## Contents

1. [Architecture](#architecture)
2. [Installation](#installation)
3. [GPU setup](#gpu-setup)
4. [Model setup](#model-setup)
5. [Quick start](#quick-start)
6. [CLI](#cli)
7. [Input formats](#input-formats)
8. [Configuration](#configuration)
9. [Outputs and dataset schemas](#outputs-and-dataset-schemas)
10. [Checkpoint / resume system](#checkpoint--resume-system)
11. [Confidence, quality and hallucination control](#confidence-quality-and-hallucination-control)
12. [Performance](#performance)
13. [Troubleshooting](#troubleshooting)
14. [Development](#development)
15. [Implementation status](#implementation-status)

---

## Architecture

```
video-dataset-pipeline/
├── config/
│   ├── default.yaml          # every knob, documented
│   ├── models.yaml           # named model presets (vision / ASR / LLM / OCR)
│   └── pipeline.yaml         # stage order, retries, workers
├── data/                     # created on first run (see layout below)
├── scripts/
│   ├── make_test_video.py    # synthetic test video with known cuts, pan, text, fade
│   └── show_examples.py      # print example records from data/final
├── src/video_dataset/
│   ├── cli.py                # `video-dataset` Typer CLI
│   ├── config.py             # layered, typed configuration (Pydantic)
│   ├── stages.py             # Stage enum + order
│   ├── schemas/              # Pydantic schemas for every artifact and dataset record
│   ├── storage/              # SQLite state DB (checkpoints) + on-disk path layout
│   ├── pipeline/             # VideoContext, resumable runner
│   ├── downloader/           # yt-dlp source, local-file source
│   ├── preprocessing/        # ffprobe, canonical video.mp4, 16 kHz audio.wav
│   ├── scene_detection/      # PySceneDetect cuts + fades, artificial-split marking
│   ├── frame_sampling/       # low-res scan (change + optical flow) -> selection -> extraction, clips
│   ├── audio/                # chunked WAV reading, energy / AudioSet sound events
│   ├── transcription/        # faster-whisper, transformers Whisper, mock
│   ├── ocr/                  # RapidOCR / EasyOCR / PaddleOCR + cross-frame merge
│   ├── vision/               # VisionAnalyzer interface: heuristic, HF VLM, Anthropic, OpenAI-compatible, mock, CLIP enricher
│   ├── llm/                  # text/image LLM clients used by API vision, paraphrase, causal inference
│   ├── temporal/             # event extraction, interval relations, causal hook, timeline
│   ├── questions/            # temporal QA generator (8 categories), optional paraphrase
│   ├── validation/           # checks, grounding, verifiers, quality scores, status assignment
│   ├── deduplication/        # exact / MinHash near-dup / evidence-aware QA dedup / duplicate videos
│   └── dataset/              # record builders, JSONL + Parquet export, statistics, aggregation
└── tests/                    # unit tests + end-to-end integration test on a synthetic video
```

### Data flow and hierarchy

Every record is traceable to the video and its timestamps:

```
VIDEO (metadata.json, media_info.json)
 ├── SCENES        scenes/<video_id>.json          shot boundaries (cut / fade / start / split)
 │    ├── FRAMES   frames/<video_id>/scene_XXX/    change- and motion-aware samples + scan data
 │    ├── CLIPS    clips/<video_id>/clip_XXX.mp4
 │    └── ANALYSIS annotations/<video_id>/scenes.json   environment, objects, people, actions, camera, style, measurements
 ├── AUDIO         audio/<video_id>.wav + .events.json  speech (transcript) + non-speech sound segments
 ├── TRANSCRIPT    transcripts/<video_id>.json
 ├── OCR           ocr/<video_id>.json               detections + merged text tracks
 ├── TIMELINE      annotations/<video_id>/timeline.json  events + temporal relations
 ├── QA            qa/<video_id>.json
 ├── VALIDATED     validated/<video_id>.json         status + issues + quality per record
 └── EXPORT        final/per_video/<video_id>/*.jsonl  -> final/*.jsonl, dataset.parquet, statistics.json
```

### Swappable components

Every model-dependent step sits behind a small interface with a factory, so a component can be
replaced by changing config, not code:

| Interface | Implementations | Config key |
|---|---|---|
| `VisionAnalyzer` (`analyze_frame`, `analyze_scene`, `analyze_clip`, `verify`) | `heuristic` (measurement only), `hf` (any transformers image-text-to-text model: SmolVLM2, Qwen2-VL/2.5-VL, LLaVA-OneVision…), `anthropic` (Claude), `openai_compatible` (vLLM, Ollama, LM Studio, OpenAI), `mock` | `vision.provider` |
| `Transcriber` | `faster_whisper`, `transformers`, `mock`, `none` | `transcription.provider` |
| `OCREngine` | `rapidocr`, `easyocr`, `paddleocr`, `mock`, `none` | `ocr.provider` |
| `AudioEventDetector` | `energy` (never guesses a sound type), `transformers` (AudioSet classifier), `none` | `audio_events.provider` |
| `SceneDetector` | `pyscenedetect` (content / adaptive / threshold + fades), fixed-interval fallback | `scene_detection.detector` |
| `Verifier` | `heuristic` (optical-flow checks), `vlm`, `composite`, `none` | `validation.verifier` |
| `LLMClient` (text + images) | `anthropic`, `openai_compatible`, `mock` | `llm.provider`, used by `qa.paraphrase`, `temporal.causal_inference` |
| `CausalInferencer` | `none` (default), `llm` | `temporal.causal_inference` |
| Enrichers | `clip` zero-shot attribute scores | `vision.enrichers` |

---

## Installation

Requirements: Python 3.11+, FFmpeg (with ffprobe), ~2 GB disk for models, more for videos.

```bash
# FFmpeg
brew install ffmpeg            # macOS
sudo apt install ffmpeg        # Debian / Ubuntu

# Project
git clone <this repo> video-dataset-pipeline && cd video-dataset-pipeline
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[asr,ocr,dev]"                # core + faster-whisper + RapidOCR + tests
# optional extras
pip install -e ".[vision-local]"               # torch + transformers for local VLMs / CLIP / AudioSet
pip install -e ".[vision-anthropic]"           # Claude API vision backend
pip install -e ".[vision-openai]"              # OpenAI-compatible servers (vLLM, Ollama, ...)
pip install -e ".[all]"

video-dataset doctor                           # verify ffmpeg, GPU, libraries
```

If `ffmpeg` is not on `PATH`, set `project.ffmpeg_path` / `project.ffprobe_path` in config or
install `imageio-ffmpeg` (its bundled binary is used as a fallback).

## GPU setup

`project.device: auto` picks **CUDA -> Apple MPS -> CPU**. You can force a device globally or per component
(`vision.device`, `transcription.device`).

| Component | CUDA | Apple Silicon (MPS) | CPU |
|---|---|---|---|
| faster-whisper (CTranslate2) | float16 | falls back to CPU int8 (CTranslate2 has no MPS backend) | int8 |
| transformers Whisper (`transcription.provider: transformers`) | yes | yes | yes |
| Local VLM (`vision.provider: hf`) | yes, fp16 | yes, fp16 | yes, fp32 (slow) |
| CLIP enricher / AudioSet classifier | yes | yes | yes |
| PySceneDetect, optical-flow scan, OCR (RapidOCR, ONNX) | CPU | CPU | CPU |

Install the PyTorch build matching your CUDA version first (see pytorch.org), then the extras above.
On OOM the local VLM automatically halves the number of images per request and retries.

## Model setup

Nothing is downloaded until a stage needs it. Presets in `config/models.yaml` cover common choices:

```bash
# tiny local VLM (≈0.5 GB) - works on CPU/MPS/CUDA
video-dataset run urls.txt --set vision.preset=smolvlm2_256m
# Qwen2.5-VL 3B / 7B locally (needs a GPU with 8 / 16+ GB)
video-dataset run urls.txt --set vision.preset=qwen2_5_vl_3b
# Claude (set ANTHROPIC_API_KEY)
video-dataset run urls.txt --set vision.preset=claude
# any OpenAI-compatible VLM server (vLLM / Ollama / LM Studio)
video-dataset run urls.txt --set vision.preset=vllm_qwen2_5_vl --set vision.base_url=http://localhost:8000/v1
# ASR
video-dataset run urls.txt --set transcription.model=large-v3        # faster-whisper sizes: tiny base small medium large-v3
video-dataset run urls.txt --set transcription.preset=mps            # transformers Whisper on Apple GPU
# AudioSet sound classification instead of the energy-only detector
video-dataset run urls.txt --set audio_events.provider=transformers
# CLIP zero-shot enrichment (indoor/outdoor, day/night, shot type, people present)
video-dataset run urls.txt --set vision.enrichers='[clip]'
```

The default vision provider is `heuristic`: it needs no model and only reports **measured** facts
(lighting level, colour temperature, dominant colours, contrast, sharpness, motion magnitude, camera
movement from optical flow). It is the always-available fallback and its measurements are attached to
every scene regardless of which generative backend you use.

API keys go in `.env` (copy `.env.example`) or the environment; they are never written to outputs.

## Quick start

```bash
# one video, whole pipeline, dataset written to data/final/
video-dataset run "https://www.youtube.com/watch?v=SkVqJ1SGeL0"

# a list of URLs (one per line, # comments allowed), 4 videos in parallel for the CPU stages
video-dataset run urls.txt --workers 4

# a directory of URL files and/or local videos
video-dataset run data/input/

# status of every video and stage, then details for one
video-dataset status
video-dataset status vid_1a2b3c4d5e6f

# re-run failed stages, re-validate, re-export
video-dataset retry-failed
video-dataset validate
video-dataset export && video-dataset stats

# look at what came out
python scripts/show_examples.py data/final -n 2
```

Run the same command again and every finished stage is skipped (`[SCENE] vid_… ✓ cached`).

## CLI

```
video-dataset run SOURCE [--workers N] [--force-from STAGE] [--until STAGE] [--no-aggregate]
video-dataset download URL... [--process]
video-dataset resume [--workers N] [--until STAGE] [--include-rejected] [--no-aggregate]
video-dataset process VIDEO_ID [--force-from STAGE] [--until STAGE]
video-dataset status [VIDEO_ID] [--json] [--watch SECONDS]
video-dataset retry-failed [--until STAGE]
video-dataset validate [VIDEO_ID]
video-dataset export [--output DIR] [--rerun]
video-dataset package [-o bundle.zip] [--include-media]
video-dataset upload [--force] [--dry-run] [--status]
video-dataset stats
video-dataset stages
video-dataset show-config
video-dataset doctor
video-dataset clean [VIDEO_ID ...] [--files] [--intermediate] [--orphans] [--failed] [--logs] [--report] [--dry-run] [--yes]
```

Global options (before the command): `--config FILE`, `--set key.path=value` (repeatable),
`--data-dir DIR`, `--log-level LEVEL`. Example:

```bash
video-dataset --config config/prod.yaml --set qa.questions_per_minute=8 run urls.txt
```

`--force-from VISION_ANALYSIS` resets that stage and everything after it for the selected videos;
`--until FRAME_EXTRACTION` stops early (handy for inspecting frames before spending model time).

Progress lines look like:

```
[DOWNLOAD] vid_0a3f9c2e8b71 ✓ source.mp4 (41.2 MB) (6.3s)
[PREPROCESS] vid_0a3f9c2e8b71 ✓ 1280x720 @ 24.00 fps, 150.0s, audio=yes (1.1s)
[SCENE] vid_0a3f9c2e8b71 ✓ 37 scenes detected (4.8s)
[FRAMES] vid_0a3f9c2e8b71 ✓ 214 frames extracted, 37 clips (21.0s)
[AUDIO] vid_0a3f9c2e8b71 ✓ 9 audio segments (energy) (0.4s)
[ASR] vid_0a3f9c2e8b71 ✓ 12 transcript segments (18.9s)
[OCR] vid_0a3f9c2e8b71 ✓ 31 text detections (6 unique) (9.7s)
[VISION] vid_0a3f9c2e8b71 ✓ 37 scenes analyzed (heuristic) (3.2s)
[TEMPORAL] vid_0a3f9c2e8b71 ✓ 94 events, 321 relations (0.3s)
[QA] vid_0a3f9c2e8b71 ✓ 13 questions generated (0.1s)
[VALIDATION] vid_0a3f9c2e8b71 ✓ 9 accepted / 0 rejected / 4 review (0 duplicates) (0.2s)
[EXPORT] vid_0a3f9c2e8b71 ✓ frames=214, clips=37, scenes=37, events=94, temporal_qa=11, long_video_qa=2, video_descriptions=6 (0.2s)
```

Every video also gets a JSON-lines log at `data/logs/<video_id>.log`, and `data/logs/pipeline.log`
holds the whole run.

### Monitoring a batch

With more than one input, `run` prints one line per finished video with the running totals and an
ETA, and the final summary table shows the time spent per video:

```
[cpu 12/120] vid_0a3f9c2e8b71 ok  failed=1  elapsed=07:41  eta=1:09:12
[model 3/119] vid_77ab1c9d0e42 ok  failed=1  elapsed=21:05  eta=13:35:40
```

From another terminal, `video-dataset status --watch 5` redraws the checkpoint table every 5 s
(the RUNNING stage is highlighted; the footer counts finished videos and running stages), and
`video-dataset status --json` feeds dashboards or scripts. `video-dataset clean --report` shows
how much disk each data sub-directory uses.

## Input formats

* a YouTube URL in any common form (`watch?v=`, `youtu.be/`, `shorts/`, `embed/`, playlist parameters are ignored) or a bare 11-character id
* a local video file (`.mp4 .mkv .webm .mov .avi .m4v …`)
* a text file with one input per line (`#` comments and blank lines ignored; CSV whose first column is a URL also works)
* a directory containing any number of such files and/or videos

The same YouTube video reached through different URL variants gets the same `video_id`
(`vid_` + 12 hex chars derived from the YouTube id), so it is never downloaded twice. Byte-identical
files under different ids are reported as duplicates in the download metrics.

## Configuration

Layers, later wins: `config/default.yaml` -> `config/models.yaml` -> `config/pipeline.yaml` ->
`--config my.yaml` -> environment (`VIDEO_DATASET_DATA_DIR`, `VIDEO_DATASET_DEVICE`,
`VIDEO_DATASET_LOG_LEVEL`, `YT_DLP_COOKIES`) -> `--set key=value`.

The important knobs (see `config/default.yaml` for all of them with comments):

```yaml
download:        { max_resolution: 1080, retries: 3, concurrency: 2, cookies_file: null }
scene_detection: { detector: content, threshold: 27.0, fade_threshold: 12.0, min_scene_len_seconds: 0.6, max_scene_duration: 60.0 }
frame_sampling:  { min_frames_per_scene: 3, max_frames_per_scene: 20, motion_aware: true, scene_boundary_frames: true,
                   analysis_fps: 4.0, change_threshold: 0.12, extract_clips: true, clip_codec: copy }
transcription:   { provider: faster_whisper, model: base, language: auto, vad_filter: true, word_timestamps: true }
audio_events:    { provider: energy, model: MIT/ast-finetuned-audioset-10-10-0.4593, min_score: 0.3 }
ocr:             { provider: rapidocr, min_confidence: 0.5, max_frames_per_scene: 6, merge_similarity: 0.85 }
vision:          { provider: heuristic, preset: null, model: null, max_images_per_request: 6, image_max_side: 768,
                   frame_captions: false, verify_with_model: true, enrichers: [] }
llm:             { provider: none, model: null }        # text helper for qa.paraphrase / temporal.causal_inference
temporal:        { min_camera_consistency: 0.6, min_camera_motion: 0.35, max_relation_neighbors: 6, long_range_min_gap_seconds: 20 }
qa:              { questions_per_minute: 5, min_questions: 8, max_questions: 400, type_weights: {...}, paraphrase: false }
validation:      { minimum_confidence: 0.75, review_confidence: 0.5, verifier: heuristic, grounding_min_overlap: 0.3 }
deduplication:   { near_duplicate_threshold: 0.9, question_similarity_threshold: 0.92, evidence_iou_threshold: 0.5 }
export:          { formats: [jsonl, parquet], include_review: true, include_rejected: false }
pipeline:        { stage_retries: 1, continue_on_error: true, workers: 1, model_stages_sequential: true }
```

`max_scene_duration` splits very long shots for analysis; those boundaries are marked
`transition_in: split` and are **never** treated as cuts by the event/relation logic.

### Cleanup, dataset growth and upload

```yaml
cleanup:
  after_export: media          # none | media | all - see "Cleanup and disk management"
export:
  merge_existing: true         # rebuilding final/ keeps records of earlier videos whose per_video export is gone
upload:
  provider: none               # huggingface: push final/ to the Hub in shards once it reaches threshold_mb
  repo_id: null
  threshold_mb: 1024
```

### Input policy and resource limits

```yaml
download:
  allowed_domains: []          # e.g. [youtube.com, youtu.be]; empty = any http(s) site yt-dlp supports
limits:
  max_duration_seconds: 7200   # 2 h; longer videos are rejected before a byte is downloaded (null = no limit)
  max_file_size_gb: null       # reject downloads whose reported size exceeds this
  min_free_disk_gb: 5          # DOWNLOAD / PREPROCESS / FRAME_EXTRACTION refuse to start below this
  disk_check_stages: [DOWNLOAD, PREPROCESS, FRAME_EXTRACTION]
vision:
  requests_per_minute: 0       # API providers: cap calls per minute (token bucket shared per provider+model)
llm:
  requests_per_minute: 0
```

* **URL validation** happens before anything is handed to yt-dlp. Inputs must be `http(s)` URLs
  with a public, fully qualified host: embedded credentials, private/loopback/link-local IPs,
  `localhost`, control characters and over-long strings are dropped when the input list is read,
  and `download.allowed_domains` (suffix match, so `youtube.com` covers `www.` and `m.`) is
  enforced in the DOWNLOAD stage. Rejected inputs are recorded as `rejected` in `download.json`.
* **Duration and size limits** are checked from yt-dlp's metadata through its `match_filter` hook,
  so an over-long or over-sized video is skipped without downloading it; local files are checked
  again after probing in PREPROCESS. Rejections are not retried and show up in
  `video-dataset status` as `FAILED` with the reason.
* **Disk space** is checked against `limits.min_free_disk_gb` before every stage listed in
  `disk_check_stages`, and a download additionally has to fit its reported size on top of the floor.
* **API rate limiting** applies to the `anthropic` / `openai_compatible` vision and LLM providers.
  The bucket is shared process-wide per provider+model, so scene analysis, causal inference and
  paraphrasing calling the same API draw from one quota; the SDKs' own retries still handle 429s.

## Outputs and dataset schemas

`video-dataset export` merges the per-video exports into `data/final/`:

```
dataset.jsonl              EVERY record in one file; each line carries record_type (frame | clip | scene | event | temporal_qa | long_video_qa | video_description)
dataset.parquet            the same records as normalized rows (one file, payload column holds the full JSON)
frames.jsonl               frame  -> caption (+ objects, camera, environment, measurements, OCR text)
clips.jsonl                clip   -> description (+ actions, camera, transcript)
scenes.jsonl               scene  -> full structured description
events.jsonl               time range -> event (+ outgoing temporal relations)
temporal_qa.jsonl          temporal question + answer + evidence
long_video_qa.jsonl        multi-scene / long-range reasoning question + answer + evidence
video_descriptions.jsonl   generation-oriented description (subject, environment, action, camera, lighting, motion, progression, transitions)
dataset.parquet            normalized rows for every record (record_type, video_id, times, text, answer, confidence, status, payload JSON)
statistics.json            dataset statistics (also printed by `video-dataset stats`)
manifest.json              per-video counts and file locations
```

For a single upload use `dataset.jsonl` (or `dataset.parquet`). Media files are referenced by paths
relative to `data/` (`frames/<video_id>/…jpg`, `clips/<video_id>/…mp4`); to ship them together run
`video-dataset package --include-media -o bundle.zip`, which writes one zip containing the dataset
files plus every referenced frame and clip in the same relative layout. Set `export.per_type_jsonl: false`
if you only want the combined file.

Records with status `accepted` are always exported; `review` records are exported by default
(flagged, `export.include_review`), `rejected` and `duplicate` records stay in `data/validated/` and are
excluded. Nothing is deleted.

### Scene record (`scenes.jsonl`)

```json
{
  "record_id": "scene_0a3f9c2e8b71_scene_003",
  "video_id": "vid_0a3f9c2e8b71",
  "scene_id": "scene_003",
  "start": 31.4, "end": 48.9, "duration": 17.5,
  "summary": "A red sports car drives along a winding mountain road surrounded by dense forest...",
  "environment": {"location": "mountain road", "setting": "outdoor", "weather": "clear", "lighting": "bright", "time_of_day": "day", "background": "forested hills"},
  "objects": ["red sports car", "road", "trees"],
  "object_details": [{"name": "red sports car", "attributes": ["red", "convertible"], "location": "center", "count": 1, "confidence": null, "confidence_source": "unavailable"}],
  "people": null,
  "actions": ["a red sports car drives toward the camera", "camera tracks the vehicle"],
  "camera": {"shot_type": "wide", "camera_angle": "eye_level", "movement": "tracking", "zoom": null, "stability": null, "is_aerial": null, "confidence": 0.82, "confidence_source": "measurement"},
  "visual_style": {"composition": "car centred on the road, leading lines from the asphalt", "lighting": "hard sunlight from the upper left", "color": "saturated greens and a red accent", "depth_of_field": "deep", "framing": "wide", "perspective": "low, near road level", "motion": "smooth forward tracking", "transitions": null},
  "temporal_progression": "The car grows larger as it approaches; the road bends left in the last frames.",
  "transcript": "We are entering the mountains now.",
  "ocr_text": [],
  "frame_ids": ["frame_003_000", "frame_003_001", "frame_003_002"],
  "clip_id": "clip_003",
  "measurements": {"brightness_mean": 141.2, "contrast": 0.41, "saturation_mean": 118.6, "dominant_colors": ["#4f7a2b", "#8d8f93", "#b0201c"], "color_temperature": "neutral", "edge_density": 0.071, "sharpness": 312.5, "motion_magnitude": 1.84, "camera_motion_label": "zoom_in", "camera_motion_consistency": 0.82, "brightness_trend": "stable", "lighting_level": "normal"},
  "provider": "api:anthropic", "confidence": 0.91, "confidence_source": "verifier",
  "quality": {"grounding": 0.91, "temporal_accuracy": null, "description_quality": 0.78, "overall": 0.845, "components_used": ["grounding", "description_quality"], "notes": []},
  "validation": {"status": "accepted", "issues": [], "checks": {"timestamps": true, "scene_exists": true, "frames_exist": true, "frame_files_exist": true, "summary_present": true}, "verifier_score": 0.91, "duplicate_of": null}
}
```

### Event record (`events.jsonl`)

```json
{
  "record_id": "event_0a3f9c2e8b71_event_0014",
  "video_id": "vid_0a3f9c2e8b71", "event_id": "event_0014",
  "event_type": "action", "start_time": 42.1, "end_time": 48.7,
  "event": "The car enters a tunnel.",
  "entities": ["car", "tunnel"], "action": "enters a tunnel",
  "scene_ids": ["scene_004"], "frame_ids": ["frame_004_000", "frame_004_001"], "clip_ids": [],
  "source": "vision", "confidence": 0.93, "confidence_source": "verifier",
  "relations": [
    {"relation": "BEFORE", "event_b": "event_0016", "event_b_text": "The camera tilts up.", "gap_seconds": 2.3, "confidence": 0.81, "relation_id": "rel_00041"},
    {"relation": "CHANGES_TO", "event_b": "event_0017", "event_b_text": "The car stops.", "gap_seconds": 0.4, "confidence": 0.9, "relation_id": "rel_00043"}
  ],
  "quality": {"grounding": 0.93, "temporal_accuracy": null, "description_quality": 0.52, "overall": 0.725, "components_used": ["grounding", "description_quality"], "notes": ["boundaries inherited from the shot; exact action timing inside the shot unmeasured"]},
  "validation": {"status": "accepted", "issues": [], "checks": {"timestamps": true, "scenes_exist": true, "frames_exist": true, "clips_exist": true, "description_present": true}, "verifier_score": 0.93, "duplicate_of": null}
}
```

Event types: `action`, `appearance`, `transition`, `camera`, `speech`, `text_on_screen`, `sound`,
`state_change`. Relation types: `BEFORE AFTER DURING OVERLAPS STARTS ENDS CONTINUES INTERRUPTS CAUSES CHANGES_TO`
(`CAUSES` only from the optional LLM inferencer, always flagged as model-reported).

### Temporal QA record (`temporal_qa.jsonl`, `long_video_qa.jsonl`)

```json
{
  "question_id": "qa_0a3f9c2e8b71_000123",
  "video_id": "vid_0a3f9c2e8b71",
  "type": "temporal_ordering",
  "question": "Which happens first: the car enters a tunnel, or the camera tilts up?",
  "answer": "The car enters a tunnel happens first.",
  "evidence": {"start_time": 42.1, "end_time": 53.0, "timestamps": [42.1, 51.0], "scene_ids": ["scene_004", "scene_005"], "event_ids": ["event_0014", "event_0016"], "frame_ids": ["frame_004_000", "frame_005_001"], "relation_ids": ["rel_00041"], "transcript_segment_ids": [], "ocr_track_ids": []},
  "difficulty": "medium",
  "confidence": 0.81, "confidence_source": "derived_min",
  "generator": "template_v1", "template_id": "ordering_v1",
  "options": ["the camera tilts up", "the car enters a tunnel"],
  "is_long_range": false,
  "quality": {"grounding": 0.95, "temporal_accuracy": null, "description_quality": 0.4, "overall": 0.675, "components_used": ["grounding", "description_quality"], "notes": []},
  "validation": {"status": "accepted", "issues": [], "checks": {"evidence_timestamps": true, "timestamps_in_range": true, "scenes_exist": true, "events_exist": true, "frames_exist": true, "frame_files_exist": true, "has_evidence": true, "answer_grounded": true}, "verifier_score": null, "duplicate_of": null}
}
```

Question types: `timestamp`, `before_after`, `temporal_ordering`, `duration`, `event_localization`,
`state_change`, `multi_event`, `long_range`. Every question is built from concrete timeline events, so
the answer is grounded by construction and the evidence block lists the exact events, scenes,
timestamps and frames that support it. Long-range questions are generated only when two events far
apart in time share an entity.

### Frame / clip / video description records

`frames.jsonl` rows carry `frame_path`, `timestamp`, `caption`, `caption_source`
(`frame_analysis` when the vision backend captioned the frame itself, `scene_summary` or `measurement`
otherwise), objects, camera, environment, measurements and OCR text. `clips.jsonl` rows carry
`clip_path`, the time range, a description that includes the shot's temporal progression, actions,
camera and transcript. `video_descriptions.jsonl` rows are assembled generation prompts:

```json
{
  "record_id": "videodesc_0a3f9c2e8b71_00", "video_id": "vid_0a3f9c2e8b71",
  "start_time": 0.0, "end_time": 150.0, "scene_ids": ["scene_001", "..."],
  "subject": "llama, penguin", "environment": "snowy mountain slope; ice cave",
  "action": "a llama slides down the slope; a penguin waddles toward the llama",
  "camera": "wide shot, the camera pans left; medium shot, the camera is static",
  "lighting": "bright, cool daylight", "motion": "moderate motion; the camera pans left",
  "temporal_progression": "Shot 1 (0.0-4.2s): ... Shot 2 (4.2-9.8s): ...",
  "transitions": "36 shot changes (cut, fade)",
  "prompt": "Subject: llama, penguin. Environment: snowy mountain slope; ice cave. Action: ... Camera: ... Lighting: ... Temporal progression: Shot 1 (0.0-4.2s): ...",
  "shots": [{"scene_id": "scene_001", "start": 0.0, "end": 4.2, "summary": "...", "camera": "wide shot, the camera pans left", "transition_in": "start", "temporal_progression": null}]
}
```

### Parquet

`dataset.parquet` has one row per exported record with columns
`record_id, record_type, video_id, scene_id, start_time, end_time, text, answer, qa_type, difficulty,
confidence, confidence_source, validation_status, quality_overall, media_path, payload` where
`payload` is the full JSON record. Filter by `record_type` and `validation_status` to build training splits.

## Checkpoint / resume system

State lives in `data/state.db` (SQLite, WAL mode):

* `videos` - one row per video (id, url, title, duration, status)
* `stages` - `(video_id, stage) -> status, attempts, timing, error, artifact path, metrics, config hash`
* `stage_log` - per-stage log lines; `file_hashes` - duplicate video detection

Stages move `PENDING -> RUNNING -> DONE | FAILED | SKIPPED`. On start-up, stages left `RUNNING` by a
killed process are marked `FAILED (interrupted)` and re-run. `DONE` stages are skipped unless
`--force-from` is used; if a stage's config section changed since it ran, the status line says so.
The VISION_ANALYSIS stage additionally checkpoints after every scene, so a long video interrupted
half-way resumes at the next scene.

`video-dataset status VIDEO_ID` shows the checkpoint table:

```
DOWNLOAD          DONE    {"status": "done", "size_bytes": 43210987}
PREPROCESS        DONE    {"duration": 150.0, "fps": 24.0, ...}
SCENE_DETECTION   DONE    {"scenes": 37, "fades": 1, "splits": 0}
FRAME_EXTRACTION  DONE    {"frames": 214, "clips": 37}
AUDIO             DONE
TRANSCRIPTION     DONE    {"segments": 12, "words": 61, "language": "en"}
OCR               DONE
VISION_ANALYSIS   RUNNING
TEMPORAL_ANALYSIS PENDING
...
```

Optional stages (`AUDIO`, `TRANSCRIPTION`, `OCR`) may fail without stopping the video; later stages
simply work without those inputs. A failure in any other stage stops that video only; the batch
continues and `video-dataset retry-failed` re-runs from the first failed stage.

### What happens after a crash or shutdown

Say `video-dataset run urls.txt` was started with 10 URLs and the machine went down while video 4
was in `FRAME_EXTRACTION`. Every URL was registered in `state.db` at start-up, every finished stage
was committed the moment it finished (WAL journal, atomic JSON writes, `.part` downloads that
yt-dlp resumes), so on the next start:

```bash
video-dataset resume            # or: video-dataset run urls.txt  (same result, needs the file)
```

* videos 1-3: every stage `DONE` -> reused, nothing re-runs (they are `done`, their after-export
  cleanup already happened);
* video 4: `DOWNLOAD`..`SCENE_DETECTION` reused; the `FRAME_EXTRACTION` row left `RUNNING` is marked
  `FAILED (interrupted)` at start-up and re-run from scratch (a `VISION_ANALYSIS` interruption
  resumes at the next un-analysed scene instead);
* videos 5-10: never started, so they are downloaded and processed now. `resume` rebuilds their
  input from the URL stored at registration, which is why the URL file is optional.

`resume` runs the same batch logic as `run` (parallel CPU stages, sequential model stages,
progress lines, aggregation and the automatic upload at the end). Videos that were *rejected* by a
limit (URL policy, duration, size) are skipped unless `--include-rejected`, because they would fail
the same way. `status` shows where each video stopped; `retry-failed` is the narrower variant that
only touches videos with a `FAILED` stage.

## Confidence, quality and hallucination control

The pipeline never invents a confidence. Every score carries a `confidence_source`:

| source | meaning |
|---|---|
| `verifier` | an explicit verification pass judged the claim against frames (VLM verifier, or optical-flow check for camera claims) |
| `measurement` | derived from a measured signal (optical-flow consistency, content-change at a cut, brightness delta) |
| `asr_logprob` | `exp(avg token log-prob) * (1 - no_speech_prob)` from the ASR decoder |
| `ocr_score`, `detector_score` | recognizer / classifier probability |
| `derived_min` | minimum over the confidences of the supporting records (relations, QA) |
| `model_self_report` | the generating model's own estimate - a weak signal, flagged for review |
| `unavailable` | no reliable estimate exists; the value is `null` |

Grounding flow for generative annotations:

```
frames -> VisionAnalyzer.analyze_scene -> candidate annotation
       -> VisionAnalyzer.verify(summary, frames) -> supported / partially / unsupported / unknown
       -> confidence := verifier score  -> VALIDATION: accepted (>= 0.75) / review (>= 0.5) / rejected
```

Additional guards: prompts demand observable facts only and forbid identifying people; measured camera
motion is passed to the model as a hint and used to verify camera claims; QA answers are checked
lexically against their evidence (`grounding_min_overlap`); records with `null` confidence go to
`review`, never silently to `accepted`. Quality scores (`grounding`, `temporal_accuracy`,
`description_quality`, `overall`) list which components were actually available in `components_used`.

## Performance

* The video is never loaded into RAM: FFmpeg streams a downscaled 4 fps analysis stream for the scan
  pass, frames are extracted by sequential seeking, audio is read in windows, clips are cut by FFmpeg.
* `--workers N` runs DOWNLOAD..AUDIO for N videos concurrently; model stages run sequentially so each
  model loads once per process (`pipeline.model_stages_sequential`).
* `frame_sampling.analysis_fps`, `analysis_width`, `max_frames_per_scene`, `vision.max_images_per_request`
  and `vision.image_max_side` are the main cost levers. `ocr.max_frames_per_scene` bounds OCR time.
* `frame_sampling.clip_codec: auto` (default) stream-copies each clip, measures the file and re-encodes
  only the clips whose length is off by more than `clip_tolerance_seconds` (stream copy can only start at
  a keyframe, so on sparse-keyframe sources such as AV1 a 1.5 s scene can come out as 6 s of footage).
  `libx264` always re-encodes; `copy` never does but flags such clips `exact: false` in `clips.jsonl`.
  Every clip record carries `media_duration`, `exact` and `codec`.
* `vision.frame_captions: false` by default; enable it for per-frame captions from the VLM (N calls per scene).
* Local VLMs back off on OOM by halving images per request; API clients retry with exponential backoff.
* Model downloads are cached by Hugging Face (`HF_HOME`) and CTranslate2.

Rough numbers on an Apple M2 (8 GB) for a 2.5-minute 720p video with the default heuristic vision:
download ≈ 6 s, scene detection ≈ 5 s, scan + frames + clips ≈ 20 s, faster-whisper `base` on CPU ≈ 20 s,
RapidOCR ≈ 10 s, everything else < 5 s.

## Cleanup and disk management

Per minute of 1080p video the pipeline keeps roughly 15-60 MB of original download, a same-size
canonical `video.mp4` (a hard link when no transcode was needed, so it costs nothing extra), ~2 MB
of 16 kHz WAV, 1-5 MB of JPEG frames and a copy of every scene as a clip. Everything except the
exported dataset in `data/final/` can be regenerated by re-running the stage that produced it.

### Automatic cleanup after export

Once a video's EXPORT stage is DONE its records live in `data/final/per_video/<video_id>/`, and the
runner deletes the working files it no longer needs, controlled by `cleanup.after_export`:

| Level | Deleted when the video reaches `done` | Kept |
|---|---|---|
| `none` | nothing | everything |
| `media` (default) | original `source.*`, canonical `video.mp4`, `audio.wav` | all JSON artifacts, so `--force-from VALIDATION` and `export --rerun` still work |
| `all` | also `downloads/<id>/` metadata, scenes, audio events, transcript, OCR, annotations, QA, validated | frames and clips (referenced by the dataset), the per-video export, the log |

The step is logged as `[CLEANUP] vid_... ✓ removed 3 working file(s), freed 412.7 MB` and in the
video's `stage_log`. A cleaned video stays `done` in the checkpoint DB: running the same URL again
reuses every checkpoint and does not re-download. To re-process it from scratch, `clean VIDEO_ID`
(forget state) and run again; to redo a stage whose inputs were deleted, `--force-from DOWNLOAD`.

### The final dataset only grows

`video-dataset run` / `export` rebuild `data/final/*.jsonl`, `dataset.jsonl` and `dataset.parquet`
from every `per_video/` export. With `export.merge_existing: true` (default) the rebuild also
keeps the records already in the final files for videos whose per-video export folder is gone, so
adding new videos, or re-running with a cleaned `per_video/`, never drops earlier data. A video
that is re-exported in the current run replaces its own earlier records (matched by video id and
record id), so the dataset never contains two versions of one video. `statistics.json` and
`manifest.json` are recomputed over the merged set.

### Automatic upload to the Hugging Face Hub

Set a repo and a write token and the pipeline pushes the dataset to the Hub on its own, one shard
at a time, as soon as the local `final/` directory reaches `upload.threshold_mb` (default 1024 MB):

```bash
echo "HF_TOKEN=hf_..." >> .env                   # a *write* token; .env is git-ignored and loaded automatically
video-dataset run urls.txt --set upload.provider=huggingface --set upload.repo_id=my-org/my-video-dataset
video-dataset upload --status                   # current shard size, uploaded shards
video-dataset upload --force                    # push the current shard now, whatever its size
video-dataset upload --dry-run --force          # show what would go
```

```yaml
upload:
  provider: huggingface
  repo_id: my-org/my-video-dataset   # created as a private dataset repo on first upload
  threshold_mb: 1024                 # 1 GB
  include_media: false               # true also uploads the frames and clips (paths stay relative to the shard)
  path_in_repo: ""                   # optional prefix, e.g. v1
  after_upload: archive              # archive: keep a local copy under final/uploaded/shard_NNNN | delete: free disk
```

How it works:

1. After every aggregation (`run`, `export`) the current shard is measured: the dataset files in
   `final/` plus `per_video/` (and the referenced frames and clips when `include_media` is on).
2. At or above the threshold the shard is hard-linked into a staging folder and uploaded to
   `repo_id/<path_in_repo>/shard_NNNN/` in one commit (with retries). A dataset card with
   `configs` for every record type is added to the repo the first time, so the Hub viewer works.
3. On success the local shard is rotated: `archive` moves the files to `final/uploaded/shard_NNNN/`
   (nothing is deleted), `delete` removes them and, with `include_media`, their frames and clips.
   `final/` is empty again and the next videos start `shard_NNNN+1`. `final/uploads.json` records
   every shard, its videos and the commit URL.
4. If the upload fails, nothing is rotated: the shard stays in `final/` and is retried on the next
   run or with `video-dataset upload --force`.

The Hub dataset therefore only grows and no shard is ever uploaded twice. Media paths in the records
are relative to the producer's data directory; with `include_media` they resolve inside the shard.
The token is only read from the environment (`upload.token_env`) and never written to config, logs
or the dataset.

### Manual cleanup

`video-dataset clean` plans first and deletes only after showing the plan (add `--dry-run` to only
look, `--yes` to skip the prompt):

| Command | What goes | When to use it |
|---|---|---|
| `clean --report` | nothing; prints size per sub-directory and free space | before deciding what to remove |
| `clean --intermediate` | original `source.*` downloads (when a separate `video.mp4` exists) and `audio.wav` for videos whose EXPORT is DONE | reclaim the bulk of the space while keeping the dataset and resumability |
| `clean --failed` | every artifact of videos whose DOWNLOAD or PREPROCESS failed (partial downloads, rejected inputs) | after a batch with many unavailable/rejected videos |
| `clean --orphans` | artifacts of video ids that are no longer in `state.db` | after `clean VIDEO_ID` without `--files`, or a rebuilt database |
| `clean --logs` | `data/logs/*.log` | logs are append-only and grow with every run |
| `clean VIDEO_ID` | the video's checkpoint state only (files stay; it is re-processed from scratch next run) | reset one video |
| `clean VIDEO_ID --files` | state **and** every artifact of that video | remove a video completely (re-export afterwards: `video-dataset export`) |

What each option keeps and why:

* **Frames, clips, scenes, transcripts, OCR, annotations, QA** are referenced by the exported
  records (paths inside `frames.jsonl`, `clips.jsonl`, ...) or needed to resume later stages, so no
  bulk option removes them. Use `clean VIDEO_ID --files` for those.
* After `--intermediate`, `--force-from TRANSCRIPTION` on that video will fail because `audio.wav`
  is gone; run `--force-from PREPROCESS` instead (it re-extracts the WAV from `video.mp4`).
* The exported dataset (`data/final/`, including `per_video/`) is never touched by `clean` or the
  automatic cleanup. Delete it by hand or re-export.
* `data/state.db` is the checkpoint database; deleting it makes every artifact an orphan. Prefer
  `clean VIDEO_ID` or `--force-from`.

Manual equivalents when you are outside the CLI:

```bash
video-dataset clean --report                         # where does the space go?
video-dataset clean --intermediate --failed --dry-run
video-dataset clean --intermediate --failed --yes
rm -rf data/frames/vid_XXXX data/clips/vid_XXXX      # same as clean vid_XXXX --files for those dirs
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ffmpeg not found` | install FFmpeg or set `project.ffmpeg_path`; `video-dataset doctor` shows what is detected |
| yt-dlp `Sign in to confirm you're not a bot` / 403 | update yt-dlp (`pip install -U yt-dlp`); for videos you are authorized to access, point `download.cookies_file` (or `YT_DLP_COOKIES`) at an exported cookies file |
| `This video is unavailable` | recorded as `UNAVAILABLE`, the batch continues; check `video-dataset status VIDEO_ID` |
| Only one scene detected | lower `scene_detection.threshold` (e.g. 20) or use `detector: adaptive`; long single shots are split at `max_scene_duration` |
| Too many scenes (flashing content) | raise `threshold`, raise `min_scene_len_seconds` |
| ASR is slow on macOS | CTranslate2 has no MPS backend; use `transcription.preset=mps` (transformers Whisper on MPS) or a smaller model |
| Hindi / Hinglish speech comes out as low-confidence Urdu-script text, most speech QA rejected | the `base` model is too small for Hindi. Use `--set transcription.model=medium` (or `large-v3`), pin `--set transcription.language=hi`, and bias the script with `--set transcription.initial_prompt="नमस्ते, आज हम बात करेंगे"`; then `video-dataset run urls.txt --force-from TRANSCRIPTION` |
| VLM output is not JSON | parsed tolerantly; the raw text becomes the summary and `errors: ["unparseable_json"]` is recorded, so the scene lands in `review`. In our smoke test SmolVLM2-256M echoed the schema keys instead of filling them (≈40 s per scene on an M2 CPU) - use a 2B+ model (`smolvlm2_2b`, `qwen2_5_vl_3b`) or an API backend, which use JSON-schema constrained output |
| CUDA out of memory | lower `vision.max_images_per_request` / `image_max_side`, use a smaller model, or set `vision.torch_dtype: float16` |
| Everything is `review` | with the heuristic provider generative descriptions do not exist, so QA built on speech/OCR/measured events is scored by their signals; add a verifier-capable vision provider (`vlm`/`composite`) to raise confidence |
| Re-run a stage with new settings | `video-dataset run urls.txt --force-from QA_GENERATION` |

## Development

```bash
pip install -e ".[dev]"
ruff check src tests scripts
mypy
python -m pytest -q             # includes the end-to-end integration test on a synthetic video (~40 s)
python -m pytest -q -m "not slow"   # fast unit tests only
python -m pytest --cov --cov-report=term --cov-report=html   # coverage report in htmlcov/index.html
python scripts/make_test_video.py data/input/synthetic_test.mp4   # the synthetic video used by the tests
python scripts/benchmark.py --seconds 60                          # per-stage timing on a synthetic clip
```

Use `python -m pytest` rather than a bare `pytest` when the virtualenv was created with
`--system-site-packages`: the system `pytest` binary runs with the system interpreter and cannot
import the editable install.

### Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request: `ruff` + `mypy`, then the test
suite with coverage on Python 3.11 and 3.12 (FFmpeg and RapidOCR installed, everything else
offline). The coverage percentage lands in the job summary and `coverage.xml` / `htmlcov/` are
uploaded as artifacts. On `main` (and via *Run workflow*) a third job runs the benchmark on a 30 s
synthetic clip and uploads `benchmark.json` so per-stage timings can be compared across commits.

### Performance benchmarks

`scripts/benchmark.py` runs the whole pipeline with the offline components (mock ASR, heuristic
vision, energy audio events, RapidOCR when installed), reads the per-stage times the runner
checkpoints in SQLite and prints seconds per stage plus the realtime factor (processing seconds per
video second):

```
python scripts/benchmark.py --seconds 120                 # tile the synthetic clip to ~2 min
python scripts/benchmark.py --video my.mp4 --set ocr.provider=none
python scripts/benchmark.py --output bench.json --markdown
```

Numbers depend on the machine and the providers; compare runs of the same config. Model-backed
providers (`faster_whisper`, `hf`, API backends) can be benchmarked the same way with `--set`.

The integration test builds a 14-second synthetic video with three hard cuts, a fade, a scrolling
texture (camera pan), on-screen text and a tone/silence soundtrack, runs every stage with the mock
transcriber, RapidOCR, heuristic vision and the energy audio detector, exports and aggregates the
dataset, and then verifies that a second run reuses every checkpoint.

## Implementation status

Fully implemented and tested offline:

* download (yt-dlp + local files), preprocessing, PySceneDetect scene detection with fades and marked splits
* streaming scan pass with change + optical-flow statistics, change/motion-aware frame selection, frame extraction, clip cutting
* chunked audio analysis (energy / silence), faster-whisper + transformers ASR adapters with derived confidence
* RapidOCR / EasyOCR / PaddleOCR adapters and cross-frame text tracking
* heuristic (measurement-only) vision analyzer with camera-motion labelling and claim verification
* event extraction from every modality, interval-algebra relations (+ CONTINUES / INTERRUPTS / CHANGES_TO rules), timeline
* template-based temporal QA for all eight categories with evidence and derived confidence
* validation, grounding, quality scores, evidence-aware deduplication, JSONL + Parquet export, statistics
* SQLite checkpointing, per-scene resume inside vision analysis, retries, per-video logs, Typer CLI

Implemented but dependent on an external model/API/network (exercised only via mocks in the test suite):

* `vision.provider: hf` (local transformers VLMs), `anthropic`, `openai_compatible` - real generative descriptions, objects, people, actions, camera and style, plus model verification
* `audio_events.provider: transformers` (AudioSet classifier), `vision.enrichers: [clip]`
* `qa.paraphrase` and `temporal.causal_inference: llm` (need `llm.provider`)
* `transcription.provider: faster_whisper` downloads its model on first use (verified in the sample run: the `base` model ran on CPU int8; the sample video has no dialogue, so 0 segments was the correct result)

Smoke-tested on real frames: `vision.preset=smolvlm2_256m` loads and generates through the `hf` adapter on CPU, but the 256M model does not follow the JSON schema, so its scenes fall back to measured annotations and land in `review`. Larger local models or the API backends are needed for semantic annotations.

Without a generative vision backend the pipeline is fully functional but object/people/action
annotations come only from speech, OCR, sound and measured signals; scene summaries are factual
measurement descriptions. Plug in a VLM to get semantic descriptions - the rest of the pipeline is unchanged.
