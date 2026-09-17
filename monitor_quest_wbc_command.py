#!/usr/bin/env python3
"""Read-only monitor for the laptop-side Quest -> WBC command file."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def fmt(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(value):+.5f}" for value in values) + "]"


def direction(position_delta: np.ndarray, rotation_delta: np.ndarray) -> str:
    names = ("x", "y", "z", "roll", "pitch", "yaw")
    values = np.concatenate((position_delta, rotation_delta))
    active = [f"{name}{'+' if value > 0 else '-'}" for name, value in zip(names, values) if abs(value) > 1e-7]
    return " ".join(active) if active else "hold"


def read_json(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command-file", type=Path, default=Path("/tmp/quest_wbc_command.json"))
    parser.add_argument("--poll-hz", type=float, default=100.0)
    args = parser.parse_args()
    if args.poll_hz <= 0.0:
        raise ValueError("--poll-hz must be positive")

    path = args.command_file.expanduser().resolve()
    previous_pose: np.ndarray | None = None
    previous_sequence: int | None = None
    last_mtime_ns: int | None = None
    print(f"[quest-monitor] read-only; watching {path}", flush=True)

    while True:
        try:
            mtime_ns = path.stat().st_mtime_ns
        except FileNotFoundError:
            time.sleep(1.0 / args.poll_hz)
            continue
        if mtime_ns == last_mtime_ns:
            time.sleep(1.0 / args.poll_hz)
            continue
        last_mtime_ns = mtime_ns
        payload = read_json(path)
        if payload is None:
            continue
        if payload.get("estop", False):
            print(f"[quest-monitor] E-STOP: {payload.get('reason', 'unknown')}", flush=True)
            continue
        poses = np.asarray(payload.get("poses", []), dtype=np.float64)
        if poses.shape != (1, 6) or not np.isfinite(poses).all():
            continue
        pose = poses[0]
        sequence = int(payload.get("sequence", -1))
        # Heartbeats retain the same sequence and target; print target changes only.
        if previous_sequence == sequence:
            continue
        if previous_pose is None:
            print(f"seq={sequence} initial_target(rotvec)={fmt(pose)}", flush=True)
        else:
            delta_xyz = pose[:3] - previous_pose[:3]
            delta_rotvec = (
                Rotation.from_rotvec(pose[3:])
                * Rotation.from_rotvec(previous_pose[3:]).inv()
            ).as_rotvec()
            delta_rpy = Rotation.from_rotvec(delta_rotvec).as_euler("xyz")
            pos_norm = float(np.linalg.norm(delta_xyz))
            angle = float(np.linalg.norm(delta_rotvec))
            print(
                f"seq={sequence} dir={direction(delta_xyz, delta_rpy)}\n"
                f"  delta_xyz_m={fmt(delta_xyz)} delta_rpy_rad={fmt(delta_rpy)} "
                f"(|dp|={pos_norm:.5f}m |dR|={math.degrees(angle):.3f}deg)\n"
                f"  target_next = target_prev + delta\n"
                f"  xyz: {fmt(previous_pose[:3])} + {fmt(delta_xyz)} = {fmt(pose[:3])}\n"
                f"  target_next_rotvec={fmt(pose[3:])}",
                flush=True,
            )
        previous_pose = pose.copy()
        previous_sequence = sequence


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[quest-monitor] stopped")
