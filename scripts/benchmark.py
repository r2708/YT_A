#!/usr/bin/env python
"""Per-stage performance benchmark on a synthetic video (no network, no model downloads).

Runs the full pipeline with the same offline components the test suite uses (mock ASR, heuristic
vision, energy audio events, RapidOCR when installed), reads the per-stage wall-clock times the
runner checkpoints in SQLite and reports seconds per stage plus the realtime factor
(seconds of processing per second of video). Use it to compare machines, configs or commits.

    python scripts/benchmark.py                      # 14 s synthetic clip, prints a table
    python scripts/benchmark.py --seconds 120        # tile the clip to ~2 minutes
    python scripts/benchmark.py --video my.mp4       # benchmark on your own file
    python scripts/benchmark.py --set vision.provider=mock --set ocr.provider=none
    python scripts/benchmark.py --output bench.json --markdown   # machine-readable + summary
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from make_test_video import DURATION, make_test_video  # noqa: E402

from video_dataset import __version__  # noqa: E402
from video_dataset.config import load_config  # noqa: E402
from video_dataset.pipeline.runner import PipelineRunner  # noqa: E402
from video_dataset.stages import STAGE_ORDER  # noqa: E402
from video_dataset.utils.urls import classify_input  # noqa: E402


def _tile(src: Path, seconds: float, out: Path) -> Path:
    """Concatenate the synthetic clip with itself until it is at least `seconds` long."""
    reps = max(1, int(seconds // DURATION) + (1 if seconds % DURATION else 0))
    if reps == 1:
        return src
    lst = out.with_suffix(".txt")
    lst.write_text("".join(f"file '{src.as_posix()}'\n" for _ in range(reps)))
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)], check=True)
    return out


def _machine() -> dict[str, str]:
    info = {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["device"] = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    except Exception:
        info["torch"] = "n/a"
    return info


def run_benchmark(video: Path, overrides: dict[str, str], data_dir: Path) -> dict:
    cfg = load_config(None, {"project.data_dir": str(data_dir), "project.log_level": "WARNING", **overrides}, config_dir=ROOT / "config")
    runner = PipelineRunner(cfg)
    try:
        item = classify_input(str(video))
        assert item is not None, video
        registered = runner.register_inputs([item])
        t0 = time.perf_counter()
        results = runner.run_batch(registered)
        total = time.perf_counter() - t0
        vid = registered[0][0]
        stages = runner.db.get_stages(vid)
        duration = float(runner.db.get_video(vid)["duration"] or 0.0)  # type: ignore[index]
        per_stage = {}
        for st in STAGE_ORDER:
            rec = stages.get(st)
            if rec is None:
                continue
            per_stage[st.value] = {
                "status": rec.status.value,
                "seconds": round(rec.duration_seconds or 0.0, 3),
                "realtime_factor": round((rec.duration_seconds or 0.0) / duration, 4) if duration else None,
                "metrics": rec.metrics or {},
            }
        return {
            "version": __version__,
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "video": str(video),
            "video_seconds": duration,
            "completed": results[0].completed,
            "error": results[0].error,
            "total_seconds": round(total, 3),
            "total_realtime_factor": round(total / duration, 4) if duration else None,
            "stages": per_stage,
            "config": {"vision": cfg.vision.provider, "transcription": cfg.transcription.provider, "ocr": cfg.ocr.provider, "audio_events": cfg.audio_events.provider},
            "machine": _machine(),
        }
    finally:
        runner.close()


def format_table(report: dict, markdown: bool = False) -> str:
    rows = [("stage", "status", "seconds", "x realtime")]
    for name, st in report["stages"].items():
        rt = st["realtime_factor"]
        rows.append((name, st["status"], f"{st['seconds']:.2f}", f"{rt:.3f}" if rt is not None else "-"))
    rows.append(("TOTAL", "ok" if report["completed"] else "FAILED", f"{report['total_seconds']:.2f}", f"{report['total_realtime_factor']:.3f}" if report["total_realtime_factor"] else "-"))
    if markdown:
        out = [f"### Benchmark {report['version']} on {report['machine']['platform']} ({report['video_seconds']:.0f} s video)", "", "| " + " | ".join(rows[0]) + " |", "|---|---|---:|---:|"]
        out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
        return "\n".join(out)
    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    lines = ["  ".join(c.ljust(w) if i < 2 else c.rjust(w) for i, (c, w) in enumerate(zip(r, widths, strict=True))) for r in rows]
    header = f"video-dataset {report['version']}  {report['machine']['platform']}  video={report['video_seconds']:.0f}s  providers={report['config']}"
    return "\n".join([header, *lines])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, help="benchmark this file instead of the synthetic clip")
    ap.add_argument("--seconds", type=float, default=DURATION, help="tile the synthetic clip to about this length")
    ap.add_argument("--set", action="append", default=[], metavar="key=value", help="config override (repeatable)")
    ap.add_argument("--output", type=Path, help="write the JSON report here")
    ap.add_argument("--markdown", action="store_true", help="print a Markdown table (for CI summaries)")
    ap.add_argument("--keep", action="store_true", help="keep the temporary data directory")
    args = ap.parse_args()

    overrides = {"transcription.provider": "mock", "audio_events.provider": "energy", "vision.provider": "heuristic", "pipeline.stage_retries": "0"}
    try:
        import rapidocr_onnxruntime  # noqa: F401
    except Exception:
        overrides["ocr.provider"] = "none"
    for kv in args.set:
        k, _, v = kv.partition("=")
        overrides[k.strip()] = v.strip()

    tmp = Path(tempfile.mkdtemp(prefix="vd-bench-"))
    try:
        if args.video:
            video = args.video.resolve()
        else:
            base = make_test_video(tmp / "synthetic.mp4")
            video = _tile(base, args.seconds, tmp / f"synthetic_{int(args.seconds)}s.mp4")
        report = run_benchmark(video, overrides, tmp / "data")
        print(format_table(report, markdown=args.markdown))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, default=str))
            print(f"\nreport written to {args.output}", file=sys.stderr)
        return 0 if report["completed"] else 1
    finally:
        if args.keep:
            print(f"data kept in {tmp}", file=sys.stderr)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
