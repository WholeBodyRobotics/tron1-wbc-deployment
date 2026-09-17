#!/usr/bin/env python3
"""Read-only live monitor for the physical J6/link6 height.

The monitor attaches to the ARX ZMQ state server and, when available, the
existing FAST-LIO /Odometry and latched /ground_height ROS 2 topics.  It does
not publish commands to the arm, legs, gripper, or policy.
"""

import argparse
import math
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from deploy_sf_tron1_arm_mujoco import (
    ARM_NAMES,
    TRAINING_URDF,
    ArmForwardKinematics,
    sdk_eef_to_link6,
)
from read_lidar_odom import Ros2OdomReader
from read_tron_arx_state import ArxStateReader


def format_value(value, digits=4):
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Continuously print the FK-computed physical J6/link6 height. "
            "This program is read-only and sends no robot commands."
        )
    )
    parser.add_argument("--rate", type=float, default=10.0, help="display rate in Hz")
    parser.add_argument("--arx-ip", default="127.0.0.1")
    parser.add_argument("--arx-port", type=int, default=8765)
    parser.add_argument("--arx-timeout-ms", type=int, default=300)
    parser.add_argument("--odom-topic", default="/Odometry")
    parser.add_argument("--ground-height-topic", default="/ground_height")
    parser.add_argument(
        "--ground-reference-topic",
        default="/ground_height_reference",
    )
    parser.add_argument(
        "--lidar-to-arm-base-xyz",
        nargs=3,
        type=float,
        default=[-0.14, 0.0, 0.0677],
        metavar=("X", "Y", "Z"),
        help="rigid MID360-origin to ARX base_link relative translation",
    )
    parser.add_argument(
        "--arx-ee-pose-is-link6",
        action="store_true",
        help="treat the ARX-reported ee_pose as link6 instead of SDK eef_link",
    )
    parser.add_argument(
        "--scroll",
        action="store_true",
        help="print a new line per sample instead of refreshing one terminal line",
    )
    args = parser.parse_args()

    if not np.isfinite(args.rate) or args.rate <= 0.0:
        parser.error("--rate must be positive")
    lidar_to_arm_base_xyz = np.asarray(
        args.lidar_to_arm_base_xyz, dtype=np.float64
    ).reshape(3)
    if not np.isfinite(lidar_to_arm_base_xyz).all():
        parser.error("--lidar-to-arm-base-xyz must be finite")

    arm_fk = ArmForwardKinematics(
        TRAINING_URDF,
        base_link="base_link",
        tip_link="link6",
    )
    arx = ArxStateReader(
        ip=args.arx_ip,
        port=args.arx_port,
        timeout_ms=args.arx_timeout_ms,
    )
    odom = Ros2OdomReader(
        topic=args.odom_topic,
        ground_height_topic=args.ground_height_topic,
        ground_reference_topic=args.ground_reference_topic,
        lidar_to_base_xyz=lidar_to_arm_base_xyz,
    )
    odom.start()

    print("J6 height monitor: READ ONLY; no robot commands are sent.")
    print(
        "Fields: world_z=fused ground-relative height, "
        "lidar_z=MID360-origin height, "
        "lidar_to_base_world_z=rotated relative-offset z, "
        "local_z=J6 relative to ARX base_link, "
        "base_delta_z=base motion since the paired height reference."
    )

    period = 1.0 / args.rate
    next_tick = time.monotonic()
    last_width = 0
    sample_count = 0
    world_min = math.inf
    world_max = -math.inf
    local_min = math.inf
    local_max = -math.inf

    try:
        while True:
            try:
                arm = arx.read()
            except Exception as exc:
                line = f"waiting for ARX ZMQ state: {exc}"
                if args.scroll or not sys.stdout.isatty():
                    print(line, flush=True)
                else:
                    print("\r" + line.ljust(last_width), end="", flush=True)
                    last_width = max(last_width, len(line))
                next_tick += period
                time.sleep(max(0.0, next_tick - time.monotonic()))
                continue

            q = np.asarray(arm["q"], dtype=np.float64).reshape(-1)[:6]
            fk_pos, fk_rot = arm_fk.pose(zip(ARM_NAMES, q))
            local_z = float(fk_pos[2])
            local_min = min(local_min, local_z)
            local_max = max(local_max, local_z)

            sdk_pose = np.asarray(arm["ee_pose"], dtype=np.float64).reshape(-1)[:6]
            sdk_rot = R.from_euler("xyz", sdk_pose[3:6]).as_matrix()
            if args.arx_ee_pose_is_link6:
                sdk_link6_pos = sdk_pose[:3]
                sdk_link6_rot = sdk_rot
            else:
                sdk_link6_pos, sdk_link6_rot = sdk_eef_to_link6(
                    sdk_pose[:3], sdk_rot
                )
            fk_sdk_mm = 1000.0 * float(np.linalg.norm(fk_pos - sdk_link6_pos))
            fk_sdk_deg = math.degrees(
                float(R.from_matrix(fk_rot.T @ sdk_link6_rot).magnitude())
            )

            odom_state = odom.read()
            ground_height = odom_state.get("ground_height")
            initial_arm_base_height = odom_state.get(
                "initial_arm_base_height"
            )
            level_z = (
                float(initial_arm_base_height) + local_z
                if initial_arm_base_height is not None
                else None
            )
            arm_base_z = None
            lidar_z = None
            lidar_to_base_world_z = None
            base_delta_z = None
            world_z = None
            odom_age_ms = None
            if odom_state.get("ok", False):
                world_lidar = np.asarray(
                    odom_state["world_lidar"], dtype=np.float64
                ).reshape(-1)
                lidar_z = float(world_lidar[2])
                rotated_lidar_to_base = np.asarray(
                    odom_state["rotated_lidar_to_base_xyz"],
                    dtype=np.float64,
                ).reshape(3)
                lidar_to_base_world_z = float(rotated_lidar_to_base[2])
                tf_world_arm_base = np.asarray(
                    odom_state["tf_world_arm_base"], dtype=np.float64
                ).reshape(4, 4)
                world_pos = (
                    tf_world_arm_base[:3, 3]
                    + tf_world_arm_base[:3, :3] @ fk_pos
                )
                arm_base_z = float(tf_world_arm_base[2, 3])
                if initial_arm_base_height is not None:
                    base_delta_z = (
                        arm_base_z - float(initial_arm_base_height)
                    )
                world_z = float(world_pos[2])
                source_stamp = float(odom_state.get("source_stamp", 0.0))
                if source_stamp > 0.0:
                    odom_age_ms = 1000.0 * (time.time() - source_stamp)
                world_min = min(world_min, world_z)
                world_max = max(world_max, world_z)

            sample_count += 1
            world_range = (
                world_max - world_min
                if np.isfinite(world_min) and np.isfinite(world_max)
                else None
            )
            line = (
                f"n={sample_count:06d} "
                f"world_z={format_value(world_z)} m "
                f"lidar_z={format_value(lidar_z)} m "
                f"lidar_to_base_world_z="
                f"{format_value(lidar_to_base_world_z)} m "
                f"arm_base_z={format_value(arm_base_z)} m "
                f"base_delta_z={format_value(base_delta_z)} m "
                f"local_z={local_z:.4f} m "
                f"level_z={format_value(level_z)} m "
                f"world_range={format_value(world_range)} m "
                f"local_range={local_max - local_min:.4f} m "
                f"fk_sdk={fk_sdk_mm:.2f} mm/{fk_sdk_deg:.3f} deg "
                f"odom_age={format_value(odom_age_ms, 1)} ms "
                f"q={np.array2string(q, precision=3, suppress_small=True)}"
            )
            if args.scroll or not sys.stdout.isatty():
                print(line, flush=True)
            else:
                print("\r" + line.ljust(last_width), end="", flush=True)
                last_width = max(last_width, len(line))

            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        if not args.scroll and sys.stdout.isatty():
            print()
        odom.stop()
        if arx.socket is not None:
            arx.socket.close(linger=0)
        arx.context.term()


if __name__ == "__main__":
    main()
