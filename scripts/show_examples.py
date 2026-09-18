#!/usr/bin/env python
"""Print a few example records from each exported JSONL file (for eyeballing dataset quality)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FILES = ["scenes", "events", "temporal_qa", "long_video_qa", "frames", "clips", "video_descriptions"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("final_dir", nargs="?", default="data/final")
    ap.add_argument("-n", type=int, default=2, help="records per file")
    ap.add_argument("--full", action="store_true", help="print full records instead of a compact view")
    args = ap.parse_args()
    final = Path(args.final_dir)
    for name in FILES:
        path = final / f"{name}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(f"\n=== {name}.jsonl ({len(rows)} records) ===")
        for row in rows[: args.n]:
            if args.full:
                print(json.dumps(row, indent=2, ensure_ascii=False))
                continue
            compact = {k: v for k, v in row.items() if k in {
                "record_id", "question_id", "scene_id", "event_id", "start", "end", "start_time", "end_time", "timestamp",
                "summary", "caption", "description", "event", "question", "answer", "type", "difficulty", "confidence",
                "confidence_source", "entities", "camera", "environment", "evidence", "prompt", "event_type", "actions",
            }}
            if row.get("validation"):
                compact["validation_status"] = row["validation"]["status"]
                if row["validation"].get("issues"):
                    compact["issues"] = row["validation"]["issues"]
            if row.get("quality"):
                compact["quality_overall"] = row["quality"].get("overall")
            print(json.dumps(compact, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
