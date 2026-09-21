#!/bin/bash
# Automatic Video Processing with Auto-Cleanup
# Usage: ./process_and_cleanup.sh [workers]
# Example: ./process_and_cleanup.sh 4

cd "$(dirname "$0")"

# Configuration
WORKERS=${1:-2}  # Default 2 workers if not specified
URL_FILE="data/input/sample_urls.txt"

echo "🚀 YT_A Automated Pipeline with Auto-Cleanup"
echo "=============================================="
echo ""
echo "📋 Configuration:"
echo "   URL File: $URL_FILE"
echo "   Workers:  $WORKERS"
echo ""

# Check if URL file exists and has URLs
if [ ! -f "$URL_FILE" ]; then
    echo "❌ Error: $URL_FILE not found!"
    exit 1
fi

# Count non-empty, non-comment lines
URL_COUNT=$(grep -v "^#" "$URL_FILE" | grep -v "^$" | wc -l | tr -d ' ')
if [ "$URL_COUNT" -lt 1 ]; then
    echo "❌ Error: No URLs found in $URL_FILE"
    echo "   Add YouTube URLs (one per line) and try again."
    exit 1
fi

echo "✅ Found $URL_COUNT video(s) to process"
echo ""

# Activate virtual environment
if [ ! -d ".venv" ]; then
    echo "❌ Error: Virtual environment not found!"
    echo "   Run: python3 -m venv .venv && source .venv/bin/activate && pip install -e ."
    exit 1
fi

echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Check disk space before starting
FREE_SPACE=$(df -h . | awk 'NR==2 {print $4}')
echo "💾 Available disk space: $FREE_SPACE"
echo ""

# Estimate required space (270 MB per video)
REQUIRED_GB=$(echo "scale=1; $URL_COUNT * 0.27" | bc)
echo "⚠️  Estimated space needed: ${REQUIRED_GB} GB (will be cleaned up after)"
echo ""

# Confirm before starting
read -p "▶️  Start processing $URL_COUNT video(s)? (Y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    echo "❌ Processing cancelled"
    exit 0
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📹 STAGE 1: Processing Videos"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Start processing
START_TIME=$(date +%s)
video-dataset run "$URL_FILE" --workers "$WORKERS"
PIPELINE_EXIT=$?

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION / 60))

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ STAGE 1 Complete (${DURATION_MIN} minutes)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if pipeline succeeded
if [ $PIPELINE_EXIT -ne 0 ]; then
    echo "⚠️  Pipeline had errors (exit code: $PIPELINE_EXIT)"
    echo ""
    read -p "❓ Continue with cleanup anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Cleanup cancelled. Check errors with: video-dataset status"
        exit $PIPELINE_EXIT
    fi
fi

# Show status
echo "📊 Pipeline Status:"
video-dataset status
echo ""

# Check if final datasets exist
if [ ! -d "data/final" ] || [ -z "$(ls -A data/final 2>/dev/null)" ]; then
    echo "⚠️  Warning: data/final/ is empty or missing!"
    echo "   Export may have failed. Skipping cleanup for safety."
    exit 1
fi

FINAL_COUNT=$(find data/final -type f 2>/dev/null | wc -l | tr -d ' ')
echo "✅ Final dataset ready: $FINAL_COUNT files in data/final/"
echo ""

# Storage before cleanup
STORAGE_BEFORE=$(du -sh data/ 2>/dev/null | cut -f1)
echo "💾 Storage before cleanup: $STORAGE_BEFORE"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🗑️  STAGE 2: Auto-Cleanup (Freeing Space)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Automatic cleanup (no confirmation needed)
echo "🗑️  Deleting intermediate files..."

# Delete intermediate directories (preserve .gitkeep)
find data/downloads -type f ! -name '.gitkeep' -delete 2>/dev/null
find data/audio -type f ! -name '.gitkeep' -delete 2>/dev/null
find data/clips -mindepth 1 ! -name '.gitkeep' -delete 2>/dev/null
find data/clips -type d -empty -delete 2>/dev/null
find data/frames -mindepth 1 ! -name '.gitkeep' -delete 2>/dev/null
find data/frames -type d -empty -delete 2>/dev/null
find data/transcripts -type f ! -name '.gitkeep' -delete 2>/dev/null
find data/ocr -type f ! -name '.gitkeep' -delete 2>/dev/null
find data/annotations -mindepth 1 ! -name '.gitkeep' -delete 2>/dev/null
find data/annotations -type d -empty -delete 2>/dev/null
find data/qa -type f ! -name '.gitkeep' -delete 2>/dev/null
find data/validated -type f ! -name '.gitkeep' -delete 2>/dev/null
find data/scenes -type f ! -name '.gitkeep' -delete 2>/dev/null
find data/logs -type f -delete 2>/dev/null

# Storage after cleanup
STORAGE_AFTER=$(du -sh data/ 2>/dev/null | cut -f1)

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ STAGE 2 Complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "📊 Final Results:"
echo "   ├─ Videos processed:  $URL_COUNT"
echo "   ├─ Storage before:    $STORAGE_BEFORE"
echo "   ├─ Storage after:     $STORAGE_AFTER"
echo "   ├─ Processing time:   ${DURATION_MIN} minutes"
echo "   └─ Final datasets:    data/final/ ($FINAL_COUNT files)"
echo ""

echo "✅ All Done! Your dataset is ready at: data/final/"
echo ""

# Optional: Show dataset stats
echo "📈 Dataset Statistics:"
video-dataset stats
echo ""

echo "💡 Next Steps:"
echo "   1. View results:    python scripts/show_examples.py data/final -n 3"
echo "   2. Archive dataset: tar -czf dataset_$(date +%Y%m%d).tar.gz data/final/"
echo "   3. Process more:    Add URLs to $URL_FILE and run ./process_and_cleanup.sh again"
echo ""

echo "🎉 Process Complete!"
