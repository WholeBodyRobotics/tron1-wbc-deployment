#!/usr/bin/env python3
"""Summarize whether a WBC dry-run sustains its 50 Hz deadline."""

import argparse
import json
from pathlib import Path

import numpy as np


TARGET_PERIOD_MS = 20.0


def percentile_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def latest_diagnostic(root):
    candidates = sorted(
        root.glob("runtime_logs/*/wbc_diagnostics.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No diagnostics found under {root / 'runtime_logs'}")
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "diagnostic",
        nargs="?",
        type=Path,
        help="wbc_diagnostics.jsonl; defaults to the newest runtime log",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=2.0,
        help="Exclude this initial interval from timing statistics.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    path = (
        args.diagnostic.expanduser().resolve()
        if args.diagnostic is not None
        else latest_diagnostic(root)
    )
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError(f"No records in {path}")

    last_elapsed = float(rows[-1].get("elapsed", 0.0))
    rows = [
        row
        for row in rows
        if float(row.get("elapsed", 0.0)) >= args.warmup_seconds
    ]
    if len(rows) < 100:
        raise RuntimeError(
            f"Only {len(rows)} post-warmup records in {path}; "
            "run the dry-run for at least 10 seconds"
        )

    elapsed = np.asarray([float(row["elapsed"]) for row in rows])
    duration = float(elapsed[-1] - elapsed[0])
    achieved_hz = (len(rows) - 1) / duration if duration > 0.0 else 0.0

    metrics = {}
    for name in (
        "inference_ms",
        "state_read_ms",
        "publish_ms",
        "loop_work_ms",
        "loop_dt_ms",
        "odom_age_ms",
        "odom_source_age_ms",
    ):
        values = [
            float(row[name])
            for row in rows
            if isinstance(row.get(name), (int, float))
            and np.isfinite(row[name])
        ]
        if values:
            metrics[name] = percentile_summary(values)

    loop_work = np.asarray(
        [float(row["loop_work_ms"]) for row in rows], dtype=np.float64
    )
    loop_dt = np.asarray(
        [
            float(row["loop_dt_ms"])
            for row in rows
            if isinstance(row.get("loop_dt_ms"), (int, float))
            and np.isfinite(row["loop_dt_ms"])
        ],
        dtype=np.float64,
    )
    deadline_miss_ratio = float(np.mean(loop_work > TARGET_PERIOD_MS))
    scheduling_late_ratio = float(np.mean(loop_dt > 22.0))

    failures = []
    if achieved_hz < 47.5:
        failures.append(f"achieved rate {achieved_hz:.2f} Hz < 47.5 Hz")
    if metrics["loop_work_ms"]["p95"] >= 15.0:
        failures.append(
            f"loop work p95 {metrics['loop_work_ms']['p95']:.2f} ms >= 15 ms"
        )
    if metrics["loop_work_ms"]["p99"] >= TARGET_PERIOD_MS:
        failures.append(
            f"loop work p99 {metrics['loop_work_ms']['p99']:.2f} ms >= 20 ms"
        )
    if deadline_miss_ratio > 0.01:
        failures.append(
            f"deadline miss ratio {deadline_miss_ratio * 100.0:.2f}% > 1%"
        )
    if scheduling_late_ratio > 0.02:
        failures.append(
            f"late scheduling ratio {scheduling_late_ratio * 100.0:.2f}% > 2%"
        )

    print(f"diagnostic={path}")
    print(
        f"records={len(rows)} measured_duration={duration:.2f}s "
        f"full_run_elapsed={last_elapsed:.2f}s achieved_rate={achieved_hz:.2f}Hz"
    )
    for name, stats in metrics.items():
        print(
            f"{name}: median={stats['median']:.2f} "
            f"p95={stats['p95']:.2f} p99={stats['p99']:.2f} "
            f"max={stats['max']:.2f}"
        )
    print(
        f"deadline_miss_ratio={deadline_miss_ratio * 100.0:.2f}% "
        f"scheduling_late_ratio={scheduling_late_ratio * 100.0:.2f}%"
    )

    inference_share = (
        metrics["inference_ms"]["p95"]
        / max(metrics["loop_work_ms"]["p95"], 1.0e-9)
    )
    if failures:
        print("CPU_RESULT=FAIL")
        for failure in failures:
            print(f"  - {failure}")
        if inference_share >= 0.60:
            print(
                "bottleneck=ONNX inference; a faster CPU or CUDA-enabled "
                "onnxruntime may help"
            )
        else:
            print(
                "bottleneck=system/ROS/state I/O scheduling; moving only ONNX "
                "to a 5090 is unlikely to fix all misses"
            )
        raise SystemExit(1)

    print("CPU_RESULT=PASS")
    print(
        f"50Hz headroom at p95="
        f"{TARGET_PERIOD_MS - metrics['loop_work_ms']['p95']:.2f}ms"
    )


if __name__ == "__main__":
    main()
