#!/usr/bin/env python3
"""Live WBC pose monitor and manual EE target panel."""

import argparse
import json
import math
import os
import tempfile
import time
import tkinter as tk
from tkinter import ttk


DEFAULT_POSE = [0.15, 0.0, 1.0, 0.0, 0.0, 0.0]
POSE_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")
FIELDS = [
    ("x", -1.0, 1.0, 0.005),
    ("y", -1.0, 1.0, 0.005),
    ("z", 0.0, 2.0, 0.005),
    ("roll", -3.14, 3.14, 0.01),
    ("pitch", -3.14, 3.14, 0.01),
    ("yaw", -3.14, 3.14, 0.01),
]


def write_command(path, pose, command_frame, estop=False):
    data = {
        "stamp": time.time(),
        "pose": [float(v) for v in pose],
        "position": [float(v) for v in pose[:3]],
        "rpy": [float(v) for v in pose[3:6]],
        "command_frame": str(command_frame),
        "estop": bool(estop),
    }
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".ee_command_", suffix=".json", dir=directory
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as command_file:
            json.dump(data, command_file, separators=(",", ":"))
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


class JsonlTailReader:
    """Incrementally read complete JSON objects appended to a JSONL file."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.inode = None
        self.offset = 0
        self.partial = ""
        self.latest = None

    def read_latest(self):
        try:
            stat = os.stat(self.path)
            inode = (stat.st_dev, stat.st_ino)
            if inode != self.inode or stat.st_size < self.offset:
                self.inode = inode
                self.offset = 0
                self.partial = ""
                self.latest = None
            with open(self.path, "r", encoding="utf-8") as state_file:
                state_file.seek(self.offset)
                chunk = state_file.read()
                self.offset = state_file.tell()
        except (FileNotFoundError, OSError):
            return self.latest

        if not chunk:
            return self.latest
        lines = (self.partial + chunk).split("\n")
        self.partial = lines.pop()
        for line in lines:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                self.latest = value
        return self.latest


def pose_from_record(record, command_frame):
    if not isinstance(record, dict):
        return None, None
    ee_key = "command_ee_world" if command_frame == "world" else "command_ee_base"
    ee_pose = record.get(ee_key)
    base_pose = record.get("arm_base_world", record.get("base_world"))

    def valid_pose(value):
        if not isinstance(value, (list, tuple)) or len(value) != 6:
            return None
        pose = [float(item) for item in value]
        return pose if all(math.isfinite(item) for item in pose) else None

    return valid_pose(ee_pose), valid_pose(base_pose)


class EeCommandPanel:
    REFRESH_MS = 100

    def __init__(
        self,
        root,
        output_path,
        state_log,
        initial_pose,
        command_frame,
        ee_frame,
    ):
        self.root = root
        self.output_path = output_path
        self.command_frame = command_frame
        self.ee_frame = ee_frame
        self.initial_pose = [float(value) for value in initial_pose]
        self.target_vars = [
            tk.StringVar(value=f"{float(value):.4f}") for value in initial_pose
        ]
        self.ee_vars = [tk.StringVar(value="—") for _ in POSE_NAMES]
        self.base_vars = [tk.StringVar(value="—") for _ in POSE_NAMES]
        self.latest_ee_pose = None
        self.state_reader = JsonlTailReader(state_log)
        self.status_var = tk.StringVar(value="Waiting for WBC state…")
        self.connection_var = tk.StringVar(value="● STARTING")

        self.root.title("TRON1 WBC · EE Control")
        self.root.geometry("940x500")
        self.root.minsize(820, 450)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        style = ttk.Style()
        style.configure("PoseValue.TLabel", font=("TkFixedFont", 11, "bold"))
        style.configure("Section.TLabelframe.Label", font=("TkDefaultFont", 11, "bold"))

        container = ttk.Frame(root, padding=14)
        container.grid(row=0, column=0, sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        header = ttk.Frame(container)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(
            header,
            text="TRON1 Whole-Body Control",
            font=("TkDefaultFont", 16, "bold"),
        ).pack(side="left")
        self.connection_label = tk.Label(
            header,
            textvariable=self.connection_var,
            fg="#9a6700",
            font=("TkDefaultFont", 10, "bold"),
        )
        self.connection_label.pack(side="right")

        live = ttk.Frame(container)
        live.grid(row=1, column=0, sticky="ew")
        live.columnconfigure(0, weight=1)
        live.columnconfigure(1, weight=1)
        self._build_pose_card(
            live,
            column=0,
            title=f"Current EE · {ee_frame} ({command_frame})",
            variables=self.ee_vars,
        )
        self._build_pose_card(
            live,
            column=1,
            title="Current Base · arm_base (world)",
            variables=self.base_vars,
        )

        target = ttk.LabelFrame(
            container,
            text=f" New EE target · {ee_frame} ({command_frame}) ",
            padding=12,
            style="Section.TLabelframe",
        )
        target.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        for column, name in enumerate(POSE_NAMES):
            target.columnconfigure(column, weight=1)
            unit = "m" if column < 3 else "rad"
            ttk.Label(target, text=f"{name} ({unit})").grid(
                row=0, column=column, sticky="w", padx=4
            )
            _, low, high, step = FIELDS[column]
            ttk.Spinbox(
                target,
                from_=low,
                to=high,
                increment=step,
                textvariable=self.target_vars[column],
                width=11,
                font=("TkFixedFont", 11),
            ).grid(row=1, column=column, sticky="ew", padx=4, pady=(4, 0))

        buttons = ttk.Frame(container)
        buttons.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        ttk.Button(
            buttons, text="Apply XYZ / RPY", command=self.publish
        ).pack(side="left")
        ttk.Button(
            buttons, text="Use current EE", command=self.use_current_ee
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            buttons, text="Startup target", command=self.reset
        ).pack(side="left", padx=(8, 0))
        self.estop_button = tk.Button(
            buttons,
            text="EMERGENCY STOP",
            command=self.emergency_stop,
            bg="#b42318",
            fg="white",
            activebackground="#d92d20",
            activeforeground="white",
            font=("TkDefaultFont", 10, "bold"),
            padx=16,
            pady=4,
        )
        self.estop_button.pack(side="right")

        ttk.Separator(container).grid(row=4, column=0, sticky="ew", pady=(14, 8))
        ttk.Label(container, textvariable=self.status_var).grid(
            row=5, column=0, sticky="w"
        )
        ttk.Label(
            container,
            text=(
                "Position: metres · Orientation: radians (roll, pitch, yaw) · "
                "Keyboard and page commands may be used together"
            ),
            foreground="#667085",
        ).grid(row=6, column=0, sticky="w", pady=(5, 0))

        self.publish()
        self.root.after(self.REFRESH_MS, self.refresh_state)

    def _build_pose_card(self, parent, column, title, variables):
        card = ttk.LabelFrame(
            parent, text=f" {title} ", padding=12, style="Section.TLabelframe"
        )
        card.grid(
            row=0,
            column=column,
            sticky="nsew",
            padx=((0, 7) if column == 0 else (7, 0)),
        )
        for index, (name, variable) in enumerate(zip(POSE_NAMES, variables)):
            row = index // 3
            local_column = (index % 3) * 2
            card.columnconfigure(local_column + 1, weight=1)
            ttk.Label(card, text=f"{name}:").grid(
                row=row, column=local_column, sticky="e", padx=(5, 4), pady=5
            )
            ttk.Label(
                card,
                textvariable=variable,
                style="PoseValue.TLabel",
                width=10,
            ).grid(
                row=row,
                column=local_column + 1,
                sticky="w",
                padx=(0, 8),
                pady=5,
            )

    def target_pose(self):
        pose = []
        for (name, low, high, _), variable in zip(FIELDS, self.target_vars):
            try:
                value = float(variable.get())
            except ValueError as exc:
                raise ValueError(f"{name} is not a number") from exc
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < low or value > high:
                raise ValueError(f"{name} must be in [{low}, {high}]")
            pose.append(value)
        return pose

    def publish(self):
        try:
            pose = self.target_pose()
            write_command(
                self.output_path,
                pose,
                command_frame=self.command_frame,
                estop=False,
            )
        except (OSError, ValueError) as exc:
            self.status_var.set(f"Target not sent: {exc}")
            return
        self.status_var.set(
            "Target sent · "
            + " ".join(
                f"{name}={value:.4f}" for name, value in zip(POSE_NAMES, pose)
            )
        )

    def use_current_ee(self):
        if self.latest_ee_pose is None:
            self.status_var.set("Current EE is not available yet")
            return
        for variable, value in zip(self.target_vars, self.latest_ee_pose):
            variable.set(f"{value:.4f}")
        self.status_var.set("Current EE copied into the target fields; press Apply to send")

    def reset(self):
        for variable, value in zip(self.target_vars, self.initial_pose):
            variable.set(f"{value:.4f}")
        self.publish()

    def emergency_stop(self):
        try:
            pose = self.target_pose()
            write_command(
                self.output_path,
                pose,
                command_frame=self.command_frame,
                estop=True,
            )
        except (OSError, ValueError) as exc:
            self.status_var.set(f"E-stop request failed: {exc}")
            return
        self.status_var.set("EMERGENCY STOP REQUESTED — restart the stack to resume")
        self.estop_button.configure(state="disabled", text="E-STOP SENT")

    def refresh_state(self):
        record = self.state_reader.read_latest()
        ee_pose, base_pose = pose_from_record(record, self.command_frame)
        self.latest_ee_pose = ee_pose
        self._set_pose_values(self.ee_vars, ee_pose)
        self._set_pose_values(self.base_vars, base_pose)

        wall_time = record.get("wall_time", 0.0) if isinstance(record, dict) else 0.0
        age = time.time() - float(wall_time) if wall_time else math.inf
        if ee_pose is not None and base_pose is not None and -1.0 <= age <= 1.0:
            self.connection_var.set(f"● LIVE  {age * 1000:.0f} ms")
            self.connection_label.configure(fg="#067647")
        elif record is not None:
            self.connection_var.set("● STALE")
            self.connection_label.configure(fg="#b54708")
        else:
            self.connection_var.set("● WAITING")
            self.connection_label.configure(fg="#9a6700")
        self.root.after(self.REFRESH_MS, self.refresh_state)

    @staticmethod
    def _set_pose_values(variables, pose):
        for variable, value in zip(variables, pose or [None] * 6):
            variable.set("—" if value is None else f"{value:+.4f}")

    def close(self):
        self.root.destroy()


def parse_pose(text):
    values = [float(value) for value in str(text).replace(",", " ").split()]
    if len(values) != 6:
        raise argparse.ArgumentTypeError("expected 6 values: x y z roll pitch yaw")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/ee_command.json")
    parser.add_argument("--state-log", required=True)
    parser.add_argument("--pose", type=parse_pose, default=DEFAULT_POSE)
    parser.add_argument("--command-frame", choices=("world", "base"), default="world")
    parser.add_argument("--ee-frame", choices=("eef_link", "j6"), default="j6")
    args = parser.parse_args()

    root = tk.Tk()
    EeCommandPanel(
        root,
        output_path=args.out,
        state_log=args.state_log,
        initial_pose=args.pose,
        command_frame=args.command_frame,
        ee_frame=args.ee_frame,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
