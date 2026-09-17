#!/usr/bin/env python3
"""Wait for a fresh, stationary FAST-LIO odometry window."""

import argparse
import math
import time
from collections import deque

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node


def normalized_quaternion(msg):
    quat = np.asarray(
        [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(quat))
    if not math.isfinite(norm) or norm < 1.0e-12:
        return None
    return quat / norm


def quaternion_span(quaternions):
    max_angle = 0.0
    for i in range(len(quaternions)):
        dots = np.abs(quaternions[i + 1 :] @ quaternions[i])
        if dots.size:
            angle = float(
                np.max(2.0 * np.arccos(np.clip(dots, 0.0, 1.0)))
            )
            max_angle = max(max_angle, angle)
    return max_angle


class StableOdomWaiter(Node):
    def __init__(self, args):
        super().__init__("stable_odom_waiter")
        self.args = args
        self.samples = deque(maxlen=args.samples)
        self.last_source_stamp = None
        self.last_receive_time = None
        self.stable = False
        self.last_report_time = 0.0
        self.create_subscription(Odometry, args.topic, self.callback, 10)

    def callback(self, msg):
        receive_time = time.time()
        source_stamp = (
            float(msg.header.stamp.sec)
            + float(msg.header.stamp.nanosec) * 1.0e-9
        )
        source_age = receive_time - source_stamp
        position = np.asarray(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ],
            dtype=np.float64,
        )
        quaternion = normalized_quaternion(msg)
        rejection = None
        if (
            not np.isfinite(position).all()
            or quaternion is None
            or source_stamp <= 0.0
        ):
            rejection = "invalid pose or timestamp"
        elif source_age < -1.0 or source_age > self.args.max_source_age:
            rejection = (
                f"source age {source_age * 1000.0:.1f} ms exceeds "
                f"{self.args.max_source_age * 1000.0:.1f} ms"
            )
        elif (
            self.last_source_stamp is not None
            and source_stamp <= self.last_source_stamp
        ):
            rejection = "source timestamp did not advance"
        elif (
            self.last_receive_time is not None
            and receive_time - self.last_receive_time
            > self.args.max_receive_gap
        ):
            rejection = (
                f"receive gap {(receive_time - self.last_receive_time) * 1000.0:.1f} ms "
                f"exceeds {self.args.max_receive_gap * 1000.0:.1f} ms"
            )

        self.last_source_stamp = source_stamp
        self.last_receive_time = receive_time
        if rejection is not None:
            self.samples.clear()
            self.report(rejection)
            return

        self.samples.append(
            (position, quaternion, source_age, receive_time)
        )
        if len(self.samples) < self.args.samples:
            self.report(
                f"collecting fresh odometry "
                f"{len(self.samples)}/{self.args.samples}"
            )
            return

        positions = np.stack([sample[0] for sample in self.samples])
        quaternions = np.stack([sample[1] for sample in self.samples])
        position_span = float(
            np.max(
                np.linalg.norm(
                    positions[:, None, :] - positions[None, :, :], axis=-1
                )
            )
        )
        angle_span = quaternion_span(quaternions)
        max_age = max(sample[2] for sample in self.samples)
        window_duration = (
            self.samples[-1][3] - self.samples[0][3]
        )
        if (
            position_span <= self.args.max_position_span
            and angle_span <= self.args.max_angle_span
            and window_duration >= self.args.min_window_duration
        ):
            print(
                "FAST-LIO stable: "
                f"samples={len(self.samples)} "
                f"window={window_duration:.2f}s "
                f"position_span={position_span:.4f}m "
                f"angle_span={angle_span:.4f}rad "
                f"max_source_age={max_age * 1000.0:.1f}ms",
                flush=True,
            )
            self.stable = True
            return
        self.report(
            "unstable window: "
            f"duration={window_duration:.2f}s "
            f"position_span={position_span:.4f}m "
            f"angle_span={angle_span:.4f}rad "
            f"max_source_age={max_age * 1000.0:.1f}ms"
        )

    def report(self, message):
        now = time.monotonic()
        if now - self.last_report_time >= 1.0:
            print(f"FAST-LIO waiting: {message}", flush=True)
            self.last_report_time = now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/Odometry")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--max-position-span", type=float, default=0.05)
    parser.add_argument("--max-angle-span", type=float, default=0.05)
    parser.add_argument("--max-source-age", type=float, default=0.25)
    parser.add_argument("--max-receive-gap", type=float, default=0.30)
    parser.add_argument("--min-window-duration", type=float, default=2.0)
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("--samples must be at least 2")

    rclpy.init()
    node = StableOdomWaiter(args)
    deadline = time.monotonic() + args.timeout
    try:
        while rclpy.ok() and not node.stable:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"FAST-LIO did not become stable within {args.timeout:.1f}s"
                )
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
