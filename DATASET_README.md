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
- Confidence scores for all detected elements
- Quality assessments (grounding, temporal accuracy, description quality)
- Validation status (accepted, review, rejected)
- Deduplication flags and similarity tracking

## 📁 Files

- **`temporal_qa.jsonl`** - Temporal question-answer pairs with evidence
- **`long_video_qa.jsonl`** - Multi-scene reasoning questions
- **`scenes.jsonl`** - Scene-level descriptions and analysis
- **`frames.jsonl`** - Frame-level captions and metadata
- **`clips.jsonl`** - Video clip descriptions
- **`events.jsonl`** - Timeline events with temporal relations
- **`video_descriptions.jsonl`** - Video-level descriptions
- **`dataset.jsonl`** - All records combined (single file)
- **`dataset.parquet`** - Parquet format for efficient querying

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
  "quality": {
    "grounding": 1.0,
    "temporal_accuracy": 0.739,
    "description_quality": 0.112,
    "overall": 0.617
  },
  "validation": {
    "status": "review",
    "issues": ["low confidence evidence"]
  }
}
```

### Scene Example

```json
{
  "record_id": "scene_0a3f9c2e8b71_scene_003",
  "video_id": "vid_0a3f9c2e8b71",
  "scene_id": "scene_003",
  "start": 31.4,
  "end": 48.9,
  "duration": 17.5,
  "summary": "A red sports car drives along a winding mountain road...",
  "environment": {
    "location": "mountain road",
    "setting": "outdoor",
    "lighting": "bright",
    "time_of_day": "day"
  },
  "objects": ["red sports car", "road", "trees"],
  "actions": ["car drives toward camera", "camera tracks vehicle"],
  "camera": {
    "shot_type": "wide",
    "camera_angle": "eye_level",
    "movement": "tracking"
  },
  "transcript": "We are entering the mountains now.",
  "measurements": {
    "brightness_mean": 141.2,
    "contrast": 0.41,
    "motion_magnitude": 1.84
  }
}
```

## 🔬 Data Quality

All records include:
- **Confidence scores** (0.5-1.0) for reliability assessment
- **Validation status** (accepted/review/rejected)
- **Evidence grounding** - All answers are backed by concrete evidence
- **Quality metrics** - Grounding, temporal accuracy, description quality

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
