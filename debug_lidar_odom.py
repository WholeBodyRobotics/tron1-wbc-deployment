#!/usr/bin/env python3
import argparse
import csv
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from read_lidar_odom import POSITION_OFFSET, Ros2OdomReader, parse_vec3


def vec(values, precision=4):
    return np.array2string(
        np.asarray(values),
        precision=precision,
        suppress_small=True,
        separator=",",
    )


class OdomFrameLogger:
    CSV_FIELDS = [
        "seq",
        "stamp",
        "elapsed",
        "dt",
        "raw_x",
        "raw_y",
        "raw_z",
        "raw_roll",
        "raw_pitch",
        "raw_yaw",
        "world_x",
        "world_y",
        "world_z",
        "world_roll",
        "world_pitch",
        "world_yaw",
        "delta_x",
        "delta_y",
        "delta_z",
        "delta_rot_x",
        "delta_rot_y",
        "delta_rot_z",
        "step_distance",
        "speed",
    ]

    def __init__(self, csv_path=None, max_rate=0.0, full=False):
        self.first_stamp = None
        self.first_position = None
        self.first_rotation = None
        self.previous_stamp = None
        self.previous_position = None
        self.last_print_time = 0.0
        self.max_rate = max(0.0, float(max_rate))
        self.full = bool(full)
        self.seq = 0
        self.csv_file = None
        self.csv_writer = None

        if csv_path:
            path = os.path.abspath(os.path.expanduser(csv_path))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.csv_file = open(path, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.DictWriter(
                self.csv_file, fieldnames=self.CSV_FIELDS
            )
            self.csv_writer.writeheader()
            print(f"csv={path}")

    def close(self):
        if self.csv_file is not None:
            self.csv_file.flush()
            self.csv_file.close()
            self.csv_file = None

    def log(self, data):
        stamp = float(data["stamp"])
        raw_position = np.asarray(data["raw_position"], dtype=np.float64)
        raw_rpy = np.asarray(data["old_rpy"], dtype=np.float64)
        world_pose = np.asarray(data["world_base"], dtype=np.float64)
        world_position = world_pose[:3]
        world_rotation = R.from_euler("xyz", world_pose[3:6])

        if self.first_stamp is None:
            self.first_stamp = stamp
            self.first_position = world_position.copy()
            self.first_rotation = world_rotation

        elapsed = stamp - self.first_stamp
        delta_position = world_position - self.first_position
        delta_rotvec = (world_rotation * self.first_rotation.inv()).as_rotvec()

        if self.previous_stamp is None:
            dt = 0.0
            step_distance = 0.0
            speed = 0.0
        else:
            dt = max(0.0, stamp - self.previous_stamp)
            step_distance = float(
                np.linalg.norm(world_position - self.previous_position)
            )
            speed = step_distance / dt if dt > 1.0e-9 else 0.0

        self.previous_stamp = stamp
        self.previous_position = world_position.copy()
        self.seq += 1

        row = {
            "seq": self.seq,
            "stamp": stamp,
            "elapsed": elapsed,
            "dt": dt,
            "raw_x": raw_position[0],
            "raw_y": raw_position[1],
            "raw_z": raw_position[2],
            "raw_roll": raw_rpy[0],
            "raw_pitch": raw_rpy[1],
            "raw_yaw": raw_rpy[2],
            "world_x": world_pose[0],
            "world_y": world_pose[1],
            "world_z": world_pose[2],
            "world_roll": world_pose[3],
            "world_pitch": world_pose[4],
            "world_yaw": world_pose[5],
            "delta_x": delta_position[0],
            "delta_y": delta_position[1],
            "delta_z": delta_position[2],
            "delta_rot_x": delta_rotvec[0],
            "delta_rot_y": delta_rotvec[1],
            "delta_rot_z": delta_rotvec[2],
            "step_distance": step_distance,
            "speed": speed,
        }

        if self.csv_writer is not None:
            self.csv_writer.writerow(row)
            self.csv_file.flush()

        now = time.monotonic()
        if self.max_rate > 0.0:
            min_period = 1.0 / self.max_rate
            if now - self.last_print_time < min_period:
                return
        self.last_print_time = now

        print(
            f"frame={self.seq:06d} t={elapsed:8.3f}s dt={dt:7.4f}s "
            f"raw_xyz={vec(raw_position)} raw_rpy={vec(raw_rpy)} "
            f"world_xyz={vec(world_position)} world_rpy={vec(world_pose[3:6])} "
            f"delta_xyz={vec(delta_position)} "
            f"delta_rotvec={vec(delta_rotvec)} "
            f"step={step_distance:.5f}m speed={speed:.4f}m/s",
            flush=True,
        )

        if self.full:
            print(
                f"  raw_quat_xyzw={vec(data['raw_odom_quat_xyzw'])} "
                f"world_quat_xyzw={vec(data['orientation_xyzw'])} "
                f"mapped_xyz={vec(data['mapped_position'])}",
                flush=True,
            )


class DebugRos2OdomReader(Ros2OdomReader):
    def __init__(self, logger, **kwargs):
        self.frame_logger = logger
        super().__init__(**kwargs)

    def _odom_cb(self, msg):
        super()._odom_cb(msg)
        data = self.read()
        if data.get("ok", False):
            self.frame_logger.log(data)


def main():
    parser = argparse.ArgumentParser(
        description="Print every transformed ROS2 odometry frame."
    )
    parser.add_argument("--topic", default="/Odometry")
    parser.add_argument(
        "--position-offset",
        type=parse_vec3,
        default=POSITION_OFFSET.tolist(),
    )
    parser.add_argument(
        "--ideal-rpy",
        type=parse_vec3,
        default=[0.0, 0.0, 0.0],
    )
    parser.add_argument(
        "--max-rate",
        type=float,
        default=0.0,
        help="Maximum terminal print rate. Zero prints every callback.",
    )
    parser.add_argument("--csv", default="", help="Optional CSV output path.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also print raw/world quaternions and mapped position.",
    )
    args = parser.parse_args()

    logger = OdomFrameLogger(
        csv_path=args.csv.strip() or None,
        max_rate=args.max_rate,
        full=args.full,
    )
    reader = DebugRos2OdomReader(
        logger=logger,
        topic=args.topic,
        position_offset=args.position_offset,
        ideal_rpy=args.ideal_rpy,
    )
    reader.start()
    print(f"waiting for every odom frame on {args.topic} ...")

    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        logger.close()


if __name__ == "__main__":
    main()
