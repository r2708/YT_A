---
license: mit
task_categories:
- video-classification
- question-answering
- text-generation
- visual-question-answering
language:
- en
tags:
- video
- temporal-reasoning
- multimodal
- question-answering
- video-language
- youtube
pretty_name: YouTube Video Dataset
size_categories:
- 1K<n<10K
configs:
- config_name: frames
  data_files: "shard_*/frames.parquet"
- config_name: clips
  data_files: "shard_*/clips.parquet"
- config_name: scenes
  data_files: "shard_*/scenes.parquet"
- config_name: events
  data_files: "shard_*/events.parquet"
- config_name: temporal_qa
  data_files: "shard_*/temporal_qa.parquet"
- config_name: long_video_qa
  data_files: "shard_*/long_video_qa.parquet"
- config_name: video_descriptions
  data_files: "shard_*/video_descriptions.parquet"
---

# YouTube Video Dataset

A temporally-grounded, multimodal dataset extracted from YouTube videos for training video-language models, temporal reasoning systems, and video question-answering models.

##  Dataset Contents

This dataset contains structured, temporally-grounded multimodal data extracted from YouTube videos:

### **What's Included:**

**🎬 Scene & Shot Analysis**
- Shot boundaries with transition detection (cuts, fades, dissolves)
- Scene-level descriptions with environment, lighting, and composition details
- Camera movement tracking (pans, tilts, zooms, tracking shots)
- Visual style analysis (color grading, framing, perspective)

**🖼️ Frame & Clip Data**
- Intelligently sampled frames from each scene (motion-aware sampling)
- Frame-level captions and visual descriptions
- Per-scene video clips with exact temporal boundaries
- Image measurements (brightness, contrast, dominant colors, sharpness)

**🎤 Audio & Transcription**
- Speech transcripts with word-level timestamps
- Audio event detection (non-speech sounds, silence segments)
- Speaker actions and dialogue attribution

**📝 On-Screen Text (OCR)**
- Detected text with spatial locations and confidence scores
- Cross-frame text tracking (persistent overlays, titles, captions)
- Distinction between static overlays (watermarks) and dynamic text

**⏱️ Temporal Events & Relations**
- Timeline of actions, appearances, state changes, and camera movements
- Temporal relations between events (BEFORE, DURING, OVERLAPS, CAUSES, etc.)
- Long-range event connections across multiple scenes
- Speech and text events synchronized with visual events

**❓ Question-Answer Pairs**
- Evidence-grounded temporal QA (timestamp, duration, ordering, localization)
- Multi-event reasoning questions spanning multiple scenes
- Before/after and state-change questions
- Long-range reasoning for events separated by time

**✅ Quality Metadata**
- Confidence scores for all detected elements, each with its provenance (`confidence_source`)
- Quality assessments (grounding, temporal accuracy, description quality)
- Validation status (accepted, review, rejected)
- Deduplication flags and similarity tracking

**🧭 Curation (every record)**
- `split`: train / validation / test, assigned per source video so no video leaks across splits
- `tier`: gold / silver / bronze from measured signals only, with `tier_reasons`
- `subsets`: named subsets such as `cinematic` (also written to `cinematic/`)
- `hard_negatives` on QA, scene and clip records: rule-built wrong answers / captions (swapped order, shifted time, wrong duration, opposite camera movement or lighting, caption of a measurably different shot)

**🎬 Camera Taxonomy (every provider, one vocabulary)**
- `camera.camera_movement`: static | pan | tilt | dolly | tracking | orbit | crane | zoom | handheld | fpv | drone | complex, with `movement_direction` (left / right / up / down / in / out / forward / backward / clockwise / counterclockwise) and measured `movement_speed`
- `camera.shot_size`, `camera_angle`, `camera_height`, `camera_distance`, `stabilization`
- `lens_type`, `focal_length_mm`, `depth_of_field`, `focus_type`, `perspective`: VLM estimates only (`lens_confidence_source`); null for measurement-only providers, never invented

**✍️ Raw Description + Generation Prompt**
- `summary` / `description`: the raw analysis text
- `generation_prompt` (scene + clip records): the same shot as a text-to-video prompt, assembled from the structured fields only

**🔗 Hierarchy**: video → clip (`clip_id`, `frame_ids`) → frame (`clip_id`); with media included, `clips/metadata.jsonl` and `frames/metadata.jsonl` follow the Hub videofolder / imagefolder convention

**🎥 Measured Composition & Motion** (`cv_models` provider)
- Object position (3x3 grid), scale class and area fraction from detector boxes
- Camera speed from optical flow; subject motion separated from camera motion (tracked / panned past / moves left ... / approaches)
- Aesthetic score from the LAION aesthetic predictor on CLIP embeddings (~1..10)

## 📁 Files

Each record type comes as `<type>.parquet` (explicit schema, what loaders should read) and `<type>.jsonl` (same records, human readable).

- **`temporal_qa`** - Temporal question-answer pairs with evidence and hard-negative answers
- **`long_video_qa`** - Multi-scene reasoning questions
- **`scenes`** - Scene-level descriptions and analysis
- **`frames`** - Frame-level captions and metadata
- **`clips`** - Video clip descriptions
- **`events`** - Timeline events with temporal relations
- **`video_descriptions`** - Video-level descriptions
- **`dataset.jsonl`** - All records combined (single file, each line tagged with `record_type`)
- **`dataset.parquet`** - One normalized row per record (text, answer, times, split, tier, payload JSON)
- **`cinematic/`** - The cinematic subset in the same layout
- **`splits.json`**, **`schemas.json`** - Video ids per split; JSON Schema of every record type

```python
from datasets import load_dataset
qa = load_dataset("parquet", data_files="temporal_qa.parquet", split="train")
train_qa = qa.filter(lambda r: r["split"] == "train" and r["tier"] != "bronze")
```

Read the Parquet files rather than the JSONL files with `load_dataset("json", ...)`: JSON type inference types a column that is null in one file as `null` and then cannot load another file where it is filled.

## 🎯 Use Cases

- **Training video-language models (VLMs)** - Multimodal understanding
- **Video temporal reasoning** - Time-based question answering
- **Action recognition and event detection** - Understanding video events
- **Video captioning** - Generate descriptions from video
- **Video question-answering systems** - QA on video content
- **Multi-modal retrieval and search** - Search across video modalities
- **Long-form video comprehension** - Understanding longer videos

## 📖 Data Format

### Temporal QA Example

```json
{
  "question_id": "qa_02b6e3996553_000001",
  "video_id": "vid_02b6e3996553",
  "type": "before_after",
  "question": "What happens immediately before the speaker says 'In a world where we're surrounded by light'?",
  "answer": "On-screen text 'REILIN JOEY' is visible.",
  "evidence": {
    "start_time": 0.75,
    "end_time": 17.63,
    "timestamps": [0.75, 3.82],
    "scene_ids": ["scene_001", "scene_002"],
    "event_ids": ["event_0002", "event_0007"],
    "frame_ids": ["frame_001_001", "frame_001_002"]
  },
  "difficulty": "medium",
  "confidence": 0.739,
  "confidence_source": "derived_min",
  "quality": {
    "grounding": 1.0,
    "temporal_accuracy": 0.739,
    "description_quality": 0.112,
    "overall": 0.617
  },
  "validation": {
    "status": "review",
    "issues": ["low confidence evidence"]
  },
  "hard_negatives": [
    {"text": "The speaker says 'In a world where we're surrounded by light'.", "kind": "other_event", "source_ids": ["event_0009"], "note": "happens at 21.4s, outside the evidence window 0.8-17.6s"}
  ],
  "split": "train",
  "tier": "bronze",
  "tier_reasons": ["status_review"],
  "subsets": []
}
```

### Scene Example

```json
{
  "record_id": "scene_0a3f9c2e8b71_scene_003",
  "record_type": "scene",
  "video_id": "vid_0a3f9c2e8b71",
  "scene_id": "scene_003",
  "start_time": 31.4,
  "end_time": 48.9,
  "duration": 17.5,
  "summary": "A red sports car drives along a winding mountain road...",
  "environment": {
    "location": "mountain road",
    "setting": "outdoor",
    "lighting": "bright",
    "time_of_day": "day"
  },
  "objects": ["red sports car", "road", "trees"],
  "object_details": [{"name": "car", "count": 1, "position": "center", "scale": "medium", "area_fraction": 0.12, "frame_fraction": 1.0, "confidence": 0.84, "confidence_source": "detector_score"}],
  "actions": ["car drives toward camera", "camera tracks vehicle"],
  "camera": {
    "shot_type": "wide",
    "camera_angle": "eye_level",
    "movement": "tracking",
    "speed": "moderate",
    "tracked_subject": "car"
  },
  "subject_motion": {"subject": "car", "label": "tracked_by_camera", "camera_relation": "tracked", "relative_velocity": [-0.17, 0.0], "n_frames": 6, "confidence": 0.9},
  "transcript": "We are entering the mountains now.",
  "measurements": {
    "brightness_mean": 141.2,
    "contrast": 0.41,
    "motion_magnitude": 1.84,
    "camera_speed": "moderate",
    "aesthetic_score": 6.2
  },
  "split": "train",
  "tier": "silver",
  "tier_reasons": ["no_grounding_component"],
  "subsets": ["cinematic"],
  "hard_negatives": [{"text": "...the camera is static...", "kind": "attribute_swap", "source_ids": ["scene_003"], "note": "camera measured tracking, negative claims static"}]
}
```

## 🔬 Data Quality

All records include:
- **Confidence scores** with their provenance (`confidence_source`: verifier, measurement, detector score, ASR log-prob, ...)
- **Validation status** (accepted/review/rejected)
- **Evidence grounding** - All answers are backed by concrete evidence
- **Quality metrics** - Grounding, temporal accuracy, description quality
- **Tier** - gold (accepted, grounded, high quality), silver (accepted, measured quality >= 0.6), bronze (everything else exported); `tier_reasons` explains the gap
- **Split** - per source video, so evaluation never sees scenes of a training video

Questions are generated from timeline events (not hallucinated), ensuring high-quality, verifiable question-answer pairs.

## 🏗️ Generation Process

Data extracted using a 12-stage pipeline:
1. **Download** - Video acquisition
2. **Preprocess** - Format standardization
3. **Scene Detection** - Shot boundary detection
4. **Frame Extraction** - Motion-aware sampling
5. **Audio Analysis** - Sound event detection
6. **Transcription** - Speech-to-text (Whisper)
7. **OCR** - On-screen text extraction
8. **Vision Analysis** - Scene understanding
9. **Temporal Analysis** - Event timeline construction
10. **QA Generation** - Question-answer pair creation
11. **Validation** - Quality checks and grounding verification
12. **Export** - Dataset compilation

## 📄 License

Dataset sourced from public YouTube videos. Please respect original content licenses and use responsibly for research and educational purposes.

## 🔗 Citation

If you use this dataset, please cite:

```bibtex
@dataset{youtube_video_dataset,
  title={YouTube Video Dataset},
  author={raj270898},
  year={2026},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/datasets/raj270898/youtube-video-dataset}}
}
```

## 📧 Contact

For questions or issues, please open an issue on the dataset repository.
