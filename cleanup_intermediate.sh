#!/bin/bash
# Cleanup intermediate files while preserving final datasets
# Created: 2026-09-18
# Usage: ./cleanup_intermediate.sh

cd "$(dirname "$0")"

echo "🗑️  YT_A Storage Cleanup Script"
echo "================================"
echo ""

# Calculate space before
echo "📊 Calculating current storage..."
BEFORE=$(du -sh data/ 2>/dev/null | cut -f1)
echo "   Before: $BEFORE"
echo ""

# Safety check - ensure final/ exists
if [ ! -d "data/final" ]; then
    echo "⚠️  WARNING: data/final/ doesn't exist!"
    echo "   Run 'video-dataset export' first to create final datasets"
    exit 1
fi

# Count files in final/
FINAL_COUNT=$(find data/final -type f 2>/dev/null | wc -l | tr -d ' ')
if [ "$FINAL_COUNT" -lt 1 ]; then
    echo "⚠️  WARNING: data/final/ is empty!"
    echo "   Run 'video-dataset export' first"
    exit 1
fi

echo "✅ Found $FINAL_COUNT files in data/final/"
echo ""

# Ask for confirmation
read -p "❓ Delete intermediate files? This will FREE UP SPACE but you cannot reprocess. (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cleanup cancelled"
    exit 0
fi

echo ""
echo "🗑️  Deleting intermediate files..."
echo ""

# Delete intermediate directories (but keep .gitkeep)
echo "   - Deleting downloads..."
find data/downloads -type f ! -name '.gitkeep' -delete 2>/dev/null

echo "   - Deleting audio..."
find data/audio -type f ! -name '.gitkeep' -delete 2>/dev/null

echo "   - Deleting clips..."
find data/clips -mindepth 1 ! -name '.gitkeep' -delete 2>/dev/null
find data/clips -type d -empty -delete 2>/dev/null

echo "   - Deleting frames..."
find data/frames -mindepth 1 ! -name '.gitkeep' -delete 2>/dev/null
find data/frames -type d -empty -delete 2>/dev/null

echo "   - Deleting transcripts..."
find data/transcripts -type f ! -name '.gitkeep' -delete 2>/dev/null

echo "   - Deleting OCR..."
find data/ocr -type f ! -name '.gitkeep' -delete 2>/dev/null

echo "   - Deleting annotations..."
find data/annotations -mindepth 1 ! -name '.gitkeep' -delete 2>/dev/null
find data/annotations -type d -empty -delete 2>/dev/null

echo "   - Deleting QA..."
find data/qa -type f ! -name '.gitkeep' -delete 2>/dev/null

echo "   - Deleting validated..."
find data/validated -type f ! -name '.gitkeep' -delete 2>/dev/null

echo "   - Deleting scenes..."
find data/scenes -type f ! -name '.gitkeep' -delete 2>/dev/null

echo "   - Deleting logs..."
find data/logs -type f -delete 2>/dev/null

echo ""

# Calculate space after
AFTER=$(du -sh data/ 2>/dev/null | cut -f1)
echo "📊 Final storage:"
echo "   After:  $AFTER"
echo ""

# Show what's kept
echo "✅ Preserved:"
echo "   - data/final/          (datasets: JSONL, Parquet, statistics)"
echo "   - data/input/          (URL lists)"
echo "   - data/state.db        (checkpoint database)"
echo ""

echo "🎯 Storage optimization complete!"
echo ""
echo "💡 To archive final datasets:"
echo "   tar -czf dataset_$(date +%Y%m%d).tar.gz data/final/"
echo ""
echo "💡 To process more videos:"
echo "   video-dataset run urls.txt --workers 4"
echo ""
