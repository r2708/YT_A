# Quick Start - YT_A Pipeline

Process YouTube videos into a dataset; the pipeline frees the working files on its own.

## 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[asr,ocr,dev]"
```

## 2. Add URLs

Edit `data/input/sample_urls.txt`, one URL per line (lines starting with `#` are ignored):

```txt
https://www.youtube.com/watch?v=VIDEO_ID_1
https://youtu.be/VIDEO_ID_2
```

## 3. Run

```bash
source .venv/bin/activate
video-dataset run data/input/sample_urls.txt --workers 2
```

That is the whole workflow. As each video reaches its EXPORT stage the runner deletes what it no
longer needs (`cleanup.after_export: media` plus `cleanup.frames_and_clips: true` in
`config/default.yaml`): the download, `video.mp4`, `audio.wav`, the `frames/<id>/scene_NNN`
folders and `clips/<id>/`. The per-video export, the log and the checkpoint state stay, so the
same URL is never processed twice. Add more URLs and run the same command again.

## What you get

```
data/final/
├── dataset.parquet           # complete dataset
├── frames.jsonl              # frame descriptions
├── clips.jsonl               # clip descriptions
├── scenes.jsonl              # scene analysis
├── events.jsonl              # timeline events
├── temporal_qa.jsonl         # temporal Q&A
├── long_video_qa.jsonl       # long-range Q&A
├── video_descriptions.jsonl  # video descriptions
├── statistics.json           # summary stats
└── per_video/<video_id>/     # the export of each video (rebuilding final/ starts from these)
```

## Everyday commands

```bash
video-dataset status                 # where each video is
video-dataset resume                 # continue an interrupted run (no URL file needed)
video-dataset retry-failed           # re-run only videos with a FAILED stage
video-dataset stats                  # dataset statistics
python scripts/show_examples.py data/final -n 5
video-dataset clean --report         # disk usage per data sub-directory
video-dataset clean --failed --logs  # free space held by failed videos and old logs
```

## Keeping the media

Set `cleanup.frames_and_clips: false` (or pass `--set cleanup.frames_and_clips=false`) when the
records' `frame_path` / `clip_path` must resolve on this machine. `cleanup.after_export: none`
keeps every working file. With `upload.include_media: true` the frames and clips are kept until
the shard has been uploaded.

## Archive

```bash
tar -czf dataset_$(date +%Y%m%d).tar.gz data/final/
```

## Troubleshooting

* **Pipeline fails**: `video-dataset status`, then `video-dataset retry-failed`.
* **Redo a finished video**: `video-dataset clean VIDEO_ID` forgets its state; the next run
  downloads and processes it again. `--force-from DOWNLOAD` redoes it in place.
* **Virtual environment missing**: repeat step 1.

Full documentation: `README.md`, all options: `config/default.yaml`,
space-saving preset: `config/storage_optimized.yaml`.
