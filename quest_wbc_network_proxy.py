#!/usr/bin/env python3
"""Mirror the local Quest file API to a remote WBC receiver over TCP."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path


def atomic_write_line(path: Path, payload: dict) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.31.142")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--command-file", type=Path, default=Path("/tmp/quest_wbc_command.json"))
    parser.add_argument("--state-file", type=Path, default=Path("/tmp/quest_wbc_remote_state.jsonl"))
    parser.add_argument("--reconnect-delay", type=float, default=1.0)
    return parser.parse_args()


def run_connection(args: argparse.Namespace) -> None:
    command_file = args.command_file.expanduser().resolve()
    state_file = args.state_file.expanduser().resolve()
    sock = socket.create_connection((args.host, args.port), timeout=3.0)
    sock.settimeout(0.02)
    print(f"[quest-proxy] connected to {args.host}:{args.port}", flush=True)
    buffer = b""
    # Never replay a command that predates this TCP connection. After a
    # disconnect the receiver latches E-stop; recovery requires fresh state,
    # a released Grip, and a newly generated command anchored at the WBC's
    # configured target.
    try:
        last_mtime = command_file.stat().st_mtime_ns
    except FileNotFoundError:
        last_mtime = None
    try:
        while True:
            try:
                mtime = command_file.stat().st_mtime_ns
                if mtime != last_mtime:
                    with command_file.open("r", encoding="utf-8") as stream:
                        payload = json.load(stream)
                    envelope = {"type": "command", "payload": payload}
                    sock.sendall((json.dumps(envelope, separators=(",", ":")) + "\n").encode())
                    last_mtime = mtime
            except FileNotFoundError:
                pass
            try:
                chunk = sock.recv(65536)
                if not chunk:
                    raise ConnectionError("receiver disconnected")
                buffer += chunk
            except socket.timeout:
                pass
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line.strip():
                    continue
                message = json.loads(line)
                if message.get("type") == "state" and isinstance(message.get("record"), dict):
                    atomic_write_line(state_file, message["record"])
            time.sleep(0.005)
    finally:
        sock.close()


def main() -> None:
    args = parse_args()
    try:
        while True:
            try:
                run_connection(args)
            except (OSError, ConnectionError, json.JSONDecodeError) as exc:
                print(f"[quest-proxy] link down: {exc}; reconnecting", flush=True)
                time.sleep(args.reconnect_delay)
    except KeyboardInterrupt:
        print("\n[quest-proxy] user stop", flush=True)


if __name__ == "__main__":
    main()
