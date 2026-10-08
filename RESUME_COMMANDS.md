# Pipeline Resume Commands

## If Pipeline Shuts Down or Crashes

**What Gets Resumed:**
- ✗ Failed videos - Retries from failed stage
- 🔄 Processing videos - Continues from last checkpoint
- ✅ Done videos - Skipped (already complete)

### Basic Resume Command
```bash
cd /Users/rajkhajanchi/Desktop/YT_A && source .venv/bin/activate && video-dataset resume --workers 12
```

### Resume with Same Settings (cv_models + spacy)
```bash
cd /Users/rajkhajanchi/Desktop/YT_A && source .venv/bin/activate && video-dataset resume --set vision.provider=cv_models --set qa.generator=spacy --workers 12
```

### Resume Only Failed Videos
```bash
cd /Users/rajkhajanchi/Desktop/YT_A && source .venv/bin/activate && video-dataset retry-failed --workers 12
```

### Resume with Specific Settings
```bash
cd /Users/rajkhajanchi/Desktop/YT_A && source .venv/bin/activate && video-dataset resume \
  --set vision.provider=cv_models \
  --set qa.generator=spacy \
  --workers 12
```

## Check Status Before Resume

### View All Video Status
```bash
video-dataset status
```

### View Specific Video Status
```bash
video-dataset status vid_XXXXX
```

### Watch Status (Auto-refresh every 5 seconds)
```bash
video-dataset status --watch 5
```

## Force Re-run From Specific Stage

### Force from VISION_ANALYSIS stage
```bash
video-dataset resume --force-from VISION_ANALYSIS --workers 12
```

### Force from TEMPORAL_ANALYSIS stage
```bash
video-dataset resume --force-from TEMPORAL_ANALYSIS --workers 12
```

### Available stages to force from:
- DOWNLOAD
- PREPROCESS
- SCENE_DETECTION
- FRAME_EXTRACTION
- AUDIO
- TRANSCRIPTION
- OCR
- VISION_ANALYSIS
- TEMPORAL_ANALYSIS
- QA_GENERATION
- VALIDATION
- EXPORT

## Process Specific Video

### Process single video by ID
```bash
video-dataset process vid_6bc5eb7fef06 --set vision.provider=cv_models --set qa.generator=spacy
```

### Process and force from specific stage
```bash
video-dataset process vid_6bc5eb7fef06 --force-from VISION_ANALYSIS
```

## Re-export After Changes

### Re-export all validated data
```bash
video-dataset export --rerun
```

### Re-validate all videos
```bash
video-dataset validate
```

## Clean Up Failed/Incomplete Videos

### View what can be cleaned
```bash
video-dataset clean --report
```

### Clean failed videos only (dry-run first)
```bash
video-dataset clean --failed --dry-run
```

### Actually clean failed videos
```bash
video-dataset clean --failed --yes
```

### Clean orphaned files
```bash
video-dataset clean --orphans --yes
```

## Current Pipeline Run Command

### Sample URLs (3 videos)
```bash
video-dataset run data/input/sample_urls.txt \
  --set vision.provider=cv_models \
  --set qa.generator=spacy \
  --workers 12
```

### Main Cinematic Dataset
```bash
video-dataset run cinematic_youtube_urls_unique_final.txt \
  --set vision.provider=cv_models \
  --set qa.generator=spacy \
  --workers 12
```

## Emergency Commands

### Kill all Python processes (if stuck)
```bash
pkill -9 python3
```

### Reset database for specific video
```python
python3 << 'EOF'
import sqlite3
from pathlib import Path

db_path = Path("data/state.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

vid = "vid_XXXXX"  # Replace with actual video ID

cursor.execute("DELETE FROM videos WHERE video_id = ?", (vid,))
cursor.execute("DELETE FROM stages WHERE video_id = ?", (vid,))

conn.commit()
conn.close()
print(f"✅ Reset {vid}")
EOF
```

### Check disk space
```bash
df -h .
```

### Check data folder sizes
```bash
du -sh data/*
```

## Tips

1. **Always use resume** instead of run if pipeline was interrupted
2. **Check status first** to see what needs to be done
3. **Use --workers 12** for parallel processing of CPU stages
4. **Model stages run sequentially** by default (ASR, OCR, VISION, etc.)
5. **Pipeline auto-saves checkpoints** - safe to interrupt and resume

## Quick Reference

| Command | Purpose |
|---------|---------|
| `video-dataset resume` | Continue from where pipeline stopped |
| `video-dataset retry-failed` | Retry only failed videos |
| `video-dataset status` | Check progress |
| `video-dataset export` | Re-export dataset |
| `video-dataset validate` | Re-validate all data |
| `video-dataset clean --failed --yes` | Remove failed video data |
