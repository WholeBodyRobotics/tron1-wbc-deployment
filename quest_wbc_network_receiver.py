#!/usr/bin/env python3
"""Receive Quest commands over TCP and expose them to the local WBC file API."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def atomic_write_json(path: Path, payload: dict) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_last_json(path: Path) -> dict | None:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            end = stream.tell()
            if end <= 0:
                return None
            size = min(end, 131072)
            stream.seek(end - size)
            lines = stream.read().splitlines()
        for line in reversed(lines):
            if line.strip():
                return json.loads(line)
    except (OSError, json.JSONDecodeError):
        return None
    return None


def newest_diagnostic(log_root: Path) -> Path | None:
    candidates = list(log_root.glob("20*/wbc_diagnostics.jsonl"))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime_ns)


def estop_payload(reason: str) -> dict:
    return {
        "source": "quest-network-receiver",
        "frame": "arm_base_gripper_base_link",
        "stamp": time.time(),
        "estop": True,
        "reason": reason,
    }


def validated_command(
    envelope: dict,
    previous_pose: np.ndarray | None,
    previous_sequence: int | None,
    max_position_step: float,
    max_angle_step: float,
) -> tuple[dict, np.ndarray | None, int | None]:
    if envelope.get("type") != "command" or not isinstance(envelope.get("payload"), dict):
        raise ValueError("expected a command envelope")
    payload = dict(envelope["payload"])
    if bool(payload.get("estop", False)):
        return estop_payload(str(payload.get("reason", "remote Quest E-stop"))), previous_pose, previous_sequence
    if payload.get("frame") != "arm_base_gripper_base_link":
        raise ValueError("unexpected command frame")
    if payload.get("rotation_representation") != "rotvec":
        raise ValueError("rotation_representation must be rotvec")
    sequence = int(payload["sequence"])
    if previous_sequence is not None and sequence < previous_sequence:
        raise ValueError(f"sequence moved backwards: {sequence} < {previous_sequence}")
    poses = np.asarray(payload.get("poses", []), dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 6 or len(poses) != 1:
        raise ValueError(f"Quest command must contain one 6D pose, got {poses.shape}")
    if not np.isfinite(poses).all():
        raise ValueError("pose contains non-finite values")
    pose = poses[0]
    if previous_pose is not None and sequence != previous_sequence:
        position_step = float(np.linalg.norm(pose[:3] - previous_pose[:3]))
        rotation_step = float(
            np.linalg.norm(
                (Rotation.from_rotvec(pose[3:]) * Rotation.from_rotvec(previous_pose[3:]).inv()).as_rotvec()
            )
        )
        if position_step > max_position_step:
            raise ValueError(f"position step {position_step:.4f}m exceeds {max_position_step:.4f}m")
        if rotation_step > max_angle_step:
            raise ValueError(f"rotation step {rotation_step:.4f}rad exceeds {max_angle_step:.4f}rad")
    now = time.time()
    payload["source"] = "quest-network"
    payload["stamp"] = now
    payload["timestamps"] = [now]
    payload["poses"] = [pose.tolist()]
    payload["estop"] = False
    trigger = float(payload.get("gripper_trigger", 0.0))
    payload["gripper_trigger"] = float(np.clip(trigger, 0.0, 1.0))
    return payload, pose.copy(), sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--allowed-client", default="192.168.31.85")
    parser.add_argument("--command-file", type=Path, default=Path("/tmp/quest_wbc_command.json"))
    parser.add_argument("--log-root", type=Path, default=Path(__file__).resolve().parent / "runtime_logs")
    parser.add_argument("--watchdog-timeout", type=float, default=0.35)
    parser.add_argument("--state-rate", type=float, default=20.0)
    parser.add_argument("--max-position-step", type=float, default=0.05)
    parser.add_argument("--max-angle-step", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.watchdog_timeout <= 0.0 or args.state_rate <= 0.0:
        raise ValueError("timeouts and rates must be positive")
    command_file = args.command_file.expanduser().resolve()
    log_root = args.log_root.expanduser().resolve()
    # Never feed a latched E-stop from an earlier run into a newly starting
    # WBC.  With no command file, --require-diffusion-command holds output
    # until the first fresh Quest heartbeat arrives.
    command_file.unlink(missing_ok=True)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.bind, args.port))
    server.listen(1)
    print(f"[quest-rx] listening on {args.bind}:{args.port}, allowed={args.allowed_client}", flush=True)
    try:
        while True:
            connection, address = server.accept()
            peer = address[0]
            if args.allowed_client and peer != args.allowed_client:
                print(f"[quest-rx] rejected client {peer}", flush=True)
                connection.close()
                continue
            print(f"[quest-rx] connected: {peer}", flush=True)
            connection.settimeout(0.02)
            buffer = b""
            last_command = time.monotonic()
            last_state_send = 0.0
            previous_pose = None
            previous_sequence = None
            watchdog_latched = False
            try:
                while True:
                    try:
                        chunk = connection.recv(65536)
                        if not chunk:
                            raise ConnectionError("client disconnected")
                        buffer += chunk
                        if len(buffer) > 1048576:
                            raise ValueError("receive buffer exceeded 1 MiB")
                    except socket.timeout:
                        pass
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        if not line.strip():
                            continue
                        envelope = json.loads(line)
                        payload, previous_pose, previous_sequence = validated_command(
                            envelope,
                            previous_pose,
                            previous_sequence,
                            args.max_position_step,
                            args.max_angle_step,
                        )
                        atomic_write_json(command_file, payload)
                        last_command = time.monotonic()
                        watchdog_latched = bool(payload.get("estop", False))
                    now_mono = time.monotonic()
                    if now_mono - last_command > args.watchdog_timeout and not watchdog_latched:
                        atomic_write_json(command_file, estop_payload("Quest network watchdog timeout"))
                        watchdog_latched = True
                        print("[quest-rx] watchdog E-stop latched", flush=True)
                    if now_mono - last_state_send >= 1.0 / args.state_rate:
                        diagnostic = newest_diagnostic(log_root)
                        record = None if diagnostic is None else read_last_json(diagnostic)
                        message = {
                            "type": "state",
                            "record": record,
                            "server_time": time.time(),
                        }
                        connection.sendall((json.dumps(message, separators=(",", ":")) + "\n").encode())
                        last_state_send = now_mono
            except (ConnectionError, OSError, json.JSONDecodeError, ValueError) as exc:
                atomic_write_json(command_file, estop_payload(f"Quest connection stopped: {exc}"))
                print(f"[quest-rx] disconnected/E-stop: {exc}", flush=True)
            finally:
                connection.close()
    except KeyboardInterrupt:
        print("\n[quest-rx] user stop", flush=True)
    finally:
        atomic_write_json(command_file, estop_payload("receiver exited"))
        server.close()


if __name__ == "__main__":
    main()
