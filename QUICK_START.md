# 🚀 Quick Start Guide - YT_A Pipeline

Simplest way to process YouTube videos with automatic cleanup!

---

## ⚡ One-Command Processing (Recommended)

### Step 1: Add YouTube URLs
Edit `data/input/sample_urls.txt`:
```txt
# Add your YouTube URLs here (one per line):
https://www.youtube.com/watch?v=VIDEO_ID_1
https://www.youtube.com/watch?v=VIDEO_ID_2
https://youtu.be/VIDEO_ID_3
```

### Step 2: Run Auto-Process Script
```bash
# Process all videos + automatic cleanup
./process_and_cleanup.sh 4

# Or with default 2 workers:
./process_and_cleanup.sh
```

**That's it!** The script will:
1. ✅ Process all videos
2. ✅ Generate datasets
3. ✅ Automatically delete intermediate files
4. ✅ Save **96% disk space**

---

## 📊 What You Get

After processing completes, you'll have:
```
data/final/
├── dataset.parquet          # Complete dataset
├── frames.jsonl            # Frame descriptions
├── clips.jsonl             # Clip descriptions
├── scenes.jsonl            # Scene analysis
├── events.jsonl            # Timeline events
├── temporal_qa.jsonl       # Temporal Q&A
├── long_video_qa.jsonl     # Long-range Q&A
├── video_descriptions.jsonl # Video descriptions
└── statistics.json         # Summary stats
```

**Storage per video**: Only ~10 MB (vs 270 MB before cleanup!)

---

## 🔄 Processing More Videos

### Option 1: Add to existing file
```bash
# Edit data/input/sample_urls.txt
# Add more URLs
# Run again:
./process_and_cleanup.sh 4
```

### Option 2: Create new batch file
```bash
# Create new file
echo "https://www.youtube.com/watch?v=NEW_VIDEO" > data/input/batch_2.txt

# Process specific file
source .venv/bin/activate
video-dataset run data/input/batch_2.txt --workers 4

# Manual cleanup after
./cleanup_intermediate.sh
```

---

## 📁 File Structure

```
YT_A/
├── process_and_cleanup.sh    # ⭐ Main script (auto-process + cleanup)
├── cleanup_intermediate.sh   # Manual cleanup script (if needed)
├── data/
│   ├── input/
│   │   └── sample_urls.txt   # 👈 Add your URLs here
│   └── final/                # 👈 Your datasets (after processing)
└── config/
    └── storage_optimized.yaml # Optional: Save even more space
```

---

## ⚙️ Advanced Usage

### Use Storage-Optimized Config (Save ~40% more)
```bash
source .venv/bin/activate
video-dataset run data/input/sample_urls.txt \
  --config config/storage_optimized.yaml \
  --workers 4

# Then manual cleanup
./cleanup_intermediate.sh
```

### Check Status During Processing
```bash
# Open new terminal
cd /Users/rajkhajanchi/Desktop/YT_A
source .venv/bin/activate
video-dataset status
```

### View Results
```bash
source .venv/bin/activate

# Show sample records
python scripts/show_examples.py data/final -n 5

# Show statistics
video-dataset stats
```

### Archive Dataset
```bash
# Create compressed archive
tar -czf my_dataset_$(date +%Y%m%d).tar.gz data/final/

# Check size
ls -lh my_dataset_*.tar.gz
```

---

## 💾 Storage Estimates

| Videos | With Auto-Cleanup | Without Cleanup |
|--------|-------------------|-----------------|
| 1      | 10 MB            | 270 MB          |
| 10     | 100 MB           | 2.7 GB          |
| 50     | 500 MB           | 13.5 GB         |
| 100    | 1 GB             | 27 GB           |

---

## 🔍 Troubleshooting

### "No URLs found"
- Check `data/input/sample_urls.txt` has valid URLs
- Remove `#` from URL lines

### "Virtual environment not found"
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[asr,ocr,dev]"
```

### Pipeline fails
```bash
# Check status
video-dataset status

# Retry failed videos
video-dataset retry-failed
```

### Want to keep intermediate files?
Use manual pipeline instead:
```bash
source .venv/bin/activate
video-dataset run data/input/sample_urls.txt --workers 4
# Don't run cleanup script
```

---

## 📝 Summary

**Simplest Workflow:**
1. Add URLs to `data/input/sample_urls.txt`
2. Run `./process_and_cleanup.sh 4`
3. Get datasets from `data/final/`
4. Repeat!

**That's it!** 🎉

---

## 📚 Full Documentation

For detailed information, see:
- `README.md` - Complete pipeline documentation
- `config/default.yaml` - All configuration options
- `config/storage_optimized.yaml` - Space-saving settings
