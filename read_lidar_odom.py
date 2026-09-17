#!/usr/bin/env python3
import argparse
import json
import math
import os
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as R


POSITION_OFFSET = np.asarray([-0.14, 0.0, 0.8277], dtype=np.float64)
LIDAR_TO_ARM_BASE_XYZ = np.asarray([-0.14, 0.0, 0.0677], dtype=np.float64)
# FAST-LIO's mapping.extrinsic_T: LiDAR origin expressed in the IMU/body
# frame.  /Odometry publishes state_point.pos (the IMU/body origin), not the
# LiDAR origin.
FASTLIO_LIDAR_TO_IMU_XYZ = np.asarray(
    [-0.011, -0.02329, 0.04412], dtype=np.float64
)

# 位置坐标重映射：
# new_x = old_y
# new_y = old_x
# new_z = -old_z
POS_REMAP = np.asarray(
    [
        [0.0, 1.0,  0.0],
        [1.0, 0.0,  0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)


def normalize_quat_xyzw(q):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)

    if n < 1.0e-12:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    q = q / n

    # q 和 -q 表示同一个姿态，这里统一 w 为正，避免跳变
    if q[3] < 0.0:
        q = -q

    return q


def wrap_angle(angle):
    """
    把角度限制到 [-pi, pi]
    """
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def wrap_rpy(rpy):
    rpy = np.asarray(rpy, dtype=np.float64).reshape(3)
    return np.asarray([wrap_angle(v) for v in rpy], dtype=np.float64)


def rpy_new_from_old_delta(delta_old_rpy):
    """
    按你最新描述的关系转换：

    原始 old_rpy:
        old_rpy[0] = old_roll   = 绕 old_x 的转动
        old_rpy[1] = old_pitch  = 绕 old_y 的转动
        old_rpy[2] = old_yaw    = 绕 old_z 的转动

    新 RPY:
        new_roll  = old_pitch
        new_pitch = old_roll
        new_yaw   = -old_yaw
    """

    old_roll = delta_old_rpy[0]
    old_pitch = delta_old_rpy[1]
    old_yaw = delta_old_rpy[2]

    new_roll = old_pitch
    new_pitch = old_roll
    new_yaw = -old_yaw

    return np.asarray([new_roll, new_pitch, new_yaw], dtype=np.float64)


def mapped_body_rotation(old_rot, ideal_rpy=(0.0, 0.0, 0.0)):
    """Map FAST-LIO body orientation into the robot/world axis convention.

    POS_REMAP is a proper rotation (det=+1), so use matrix conjugation instead
    of swapping Euler angles.  This remains exact for combined roll/pitch/yaw.
    """
    old_rot = np.asarray(old_rot, dtype=np.float64).reshape(3, 3)
    ideal_rot = R.from_euler(
        "xyz", np.asarray(ideal_rpy, dtype=np.float64).reshape(3)
    ).as_matrix()
    delta_old_rot = ideal_rot.T @ old_rot
    return POS_REMAP @ delta_old_rot @ POS_REMAP.T


def mapped_lidar_position(raw_imu_position, old_rot, lidar_to_imu_xyz):
    """Return the LiDAR origin in remapped FAST-LIO world coordinates."""
    raw_imu_position = np.asarray(raw_imu_position, dtype=np.float64).reshape(3)
    old_rot = np.asarray(old_rot, dtype=np.float64).reshape(3, 3)
    lidar_to_imu_xyz = np.asarray(lidar_to_imu_xyz, dtype=np.float64).reshape(3)
    raw_lidar_position = raw_imu_position + old_rot @ lidar_to_imu_xyz
    return POS_REMAP @ raw_lidar_position


def transform_from_pose6d(pose):
    pose = np.asarray(pose, dtype=np.float64).reshape(-1)[:6]
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = R.from_euler("xyz", pose[3:6]).as_matrix()
    tf[:3, 3] = pose[:3]
    return tf


def pose6d_from_transform(tf):
    tf = np.asarray(tf, dtype=np.float64).reshape(4, 4)
    return np.concatenate([tf[:3, 3], R.from_matrix(tf[:3, :3]).as_euler("xyz")])


class Ros2OdomReader:
    def __init__(
        self,
        topic="/Odometry",
        position_offset=POSITION_OFFSET,
        ground_height_topic=None,
        ground_reference_topic=None,
        ground_freeze_topic=None,
        lidar_to_base_z=0.0677,
        lidar_to_base_xyz=None,
        fastlio_lidar_to_imu_xyz=FASTLIO_LIDAR_TO_IMU_XYZ,
        ideal_rpy=(0.0, 0.0, 0.0),
        output_path=None,
        max_source_age=0.25,
        max_receive_gap=0.30,
        max_linear_speed=1.0,
        max_angular_speed=3.0,
        position_jump_margin=0.03,
        angle_jump_margin=0.05,
    ):
        import rclpy
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import Bool, Float64, Float64MultiArray

        self.rclpy = rclpy
        self.Odometry = Odometry
        self.Node = Node
        self.Float64 = Float64
        self.Float64MultiArray = Float64MultiArray
        self.Bool = Bool
        self.height_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.odom_qos = QoSProfile(depth=1)

        self.topic = topic
        self.position_offset = np.asarray(position_offset, dtype=np.float64).copy()
        self.ground_height_topic = ground_height_topic
        self.ground_reference_topic = ground_reference_topic
        self.ground_freeze_topic = ground_freeze_topic
        self.lidar_to_base_z = float(lidar_to_base_z)
        if lidar_to_base_xyz is None:
            lidar_to_base_xyz = [
                float(self.position_offset[0]),
                float(self.position_offset[1]),
                self.lidar_to_base_z,
            ]
        self.lidar_to_base_xyz = np.asarray(
            lidar_to_base_xyz, dtype=np.float64
        ).reshape(3)
        self.fastlio_lidar_to_imu_xyz = np.asarray(
            fastlio_lidar_to_imu_xyz, dtype=np.float64
        ).reshape(3)
        if not np.isfinite(self.lidar_to_base_xyz).all():
            raise ValueError("lidar_to_base_xyz must be finite")
        if not np.isfinite(self.fastlio_lidar_to_imu_xyz).all():
            raise ValueError("fastlio_lidar_to_imu_xyz must be finite")
        self.ground_height = None
        self.latest_ground_height = None
        self.ground_reference_received = False
        self.ground_reference_raw_pos = None
        self.ground_reference_raw_quat = None
        self.ground_reference_sequence = None
        self.ground_reference_generation = 0
        self.initial_arm_base_height = None

        # 理想姿态在“原始 odom 坐标系”下的 RPY
        # 默认 q=[0,0,0,1]，所以 ideal_rpy=[0,0,0]
        self.ideal_rpy = np.asarray(ideal_rpy, dtype=np.float64).reshape(3)

        self.output_path = output_path
        self.max_source_age = float(max_source_age)
        self.max_receive_gap = float(max_receive_gap)
        self.max_linear_speed = float(max_linear_speed)
        self.max_angular_speed = float(max_angular_speed)
        self.position_jump_margin = float(position_jump_margin)
        self.angle_jump_margin = float(angle_jump_margin)
        if self.output_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)

        # 位置用第一帧作为原点
        # 姿态不使用第一帧清零，而是减 ideal_rpy
        self.origin_pos = None
        self.world_lidar_anchor = None
        self.latest_mapped_pos = None
        self.latest_mapped_lidar_pos = None
        self.last_accepted_source_stamp = None
        self.last_accepted_receive_stamp = None
        self.last_accepted_mapped_pos = None
        self.last_accepted_mapped_lidar_pos = None
        self.last_accepted_old_rot = None
        self.rejected_odom_count = 0

        self.latest = None
        self.lock = threading.Lock()

        self.running = False
        self.node = None
        self.thread = None
        self.ground_freeze_publisher = None

    def start(self):
        if self.running:
            return

        self.rclpy.init(args=None)

        self.node = self.Node("direct_odom_reader")
        if self.ground_height_topic:
            self.node.create_subscription(
                self.Float64,
                self.ground_height_topic,
                self._ground_height_cb,
                self.height_qos,
            )
        if self.ground_reference_topic:
            self.node.create_subscription(
                self.Float64MultiArray,
                self.ground_reference_topic,
                self._ground_reference_cb,
                self.height_qos,
            )
        if self.ground_freeze_topic:
            self.ground_freeze_publisher = self.node.create_publisher(
                self.Bool,
                self.ground_freeze_topic,
                self.height_qos,
            )
        self.node.create_subscription(
            self.Odometry,
            self.topic,
            self._odom_cb,
            self.odom_qos,
        )

        self.running = True

        self.thread = threading.Thread(
            target=self._spin,
            daemon=True,
        )
        self.thread.start()

    def _spin(self):
        from rclpy.executors import ExternalShutdownException

        try:
            self.rclpy.spin(self.node)
        except ExternalShutdownException:
            # Normal when Ctrl+C or stop() shuts down the ROS context while
            # the executor thread is waiting for callbacks.
            pass

    def _ground_height_cb(self, msg):
        height = float(msg.data)
        if not np.isfinite(height) or not 0.50 <= height <= 1.30:
            return
        with self.lock:
            self.latest_ground_height = height
            # The scalar topic remains for compatibility.  When a paired
            # reference topic is configured, wait for it before anchoring:
            # its height and raw FAST-LIO position describe the same instant.
            if self.ground_height is None:
                self.ground_height = height
                if self.ground_reference_topic:
                    print(
                        f"Received radar ground height {height:.4f} m; "
                        "waiting for paired FAST-LIO pose reference"
                    )
                    return
                print(
                    f"Locked radar ground height {height:.4f} m; "
                    "waiting for the first FAST-LIO pose before computing "
                    "the rotated LiDAR-to-base transform"
                )

    def _ground_reference_cb(self, msg):
        values = np.asarray(msg.data, dtype=np.float64).reshape(-1)
        if values.size < 4:
            return
        height = float(values[0])
        raw_pos = values[1:4].copy()
        sequence = float(values[4]) if values.size >= 5 else None
        raw_quat = (
            normalize_quat_xyzw(values[5:9])
            if values.size >= 9
            else np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        )
        if (
            not np.isfinite(height)
            or not 0.50 <= height <= 1.30
            or not np.isfinite(raw_pos).all()
            or not np.isfinite(raw_quat).all()
        ):
            return

        reference_old_rot = R.from_quat(raw_quat).as_matrix()
        mapped_reference = mapped_lidar_position(
            raw_pos,
            reference_old_rot,
            self.fastlio_lidar_to_imu_xyz,
        )
        reference_base_rot = mapped_body_rotation(
            reference_old_rot, self.ideal_rpy
        )
        world_lidar_anchor = np.asarray(
            [0.0, 0.0, height], dtype=np.float64
        )
        reference_base_position = (
            world_lidar_anchor
            + reference_base_rot @ self.lidar_to_base_xyz
        )
        with self.lock:
            same_sequence = (
                sequence is not None
                and self.ground_reference_sequence is not None
                and sequence == self.ground_reference_sequence
            )
            same_legacy_reference = (
                sequence is None
                and self.ground_reference_sequence is None
                and self.ground_reference_raw_pos is not None
                and abs(height - self.ground_height) < 1.0e-12
                and np.allclose(
                    raw_pos,
                    self.ground_reference_raw_pos,
                    rtol=0.0,
                    atol=1.0e-12,
                )
            )
            if same_sequence or same_legacy_reference:
                return
            self.ground_height = height
            self.latest_ground_height = height
            self.position_offset = reference_base_position.copy()
            self.initial_arm_base_height = float(reference_base_position[2])
            self.origin_pos = mapped_reference.copy()
            self.world_lidar_anchor = world_lidar_anchor.copy()
            self.ground_reference_raw_pos = raw_pos.copy()
            self.ground_reference_raw_quat = raw_quat.copy()
            self.ground_reference_sequence = sequence
            self.ground_reference_received = True
            self.ground_reference_generation += 1
        print(
            f"Locked paired radar/FAST-LIO reference: ground={height:.4f} m "
            f"arm_base_z={reference_base_position[2]:.4f} m "
            f"rotated_lidar_to_base="
            f"{np.array2string(reference_base_rot @ self.lidar_to_base_xyz, precision=5)} "
            f"raw_imu={np.array2string(raw_pos, precision=5)}"
        )

    def freeze_ground_height_reference(self, timeout_s=3.0):
        if self.ground_freeze_publisher is None:
            return False
        with self.lock:
            previous_generation = self.ground_reference_generation
        msg = self.Bool()
        msg.data = True
        self.ground_freeze_publisher.publish(msg)

        deadline = time.monotonic() + max(0.0, timeout_s)
        while time.monotonic() < deadline:
            with self.lock:
                if self.ground_reference_generation > previous_generation:
                    return True
            time.sleep(0.01)
        return False

    def stop(self):
        if not self.running:
            return

        self.running = False

        if self.node is not None:
            try:
                self.node.destroy_node()
            except Exception:
                if self.rclpy.ok():
                    raise

        if self.rclpy.ok():
            self.rclpy.shutdown()
        if (
            self.thread is not None
            and self.thread.is_alive()
            and self.thread is not threading.current_thread()
        ):
            self.thread.join(timeout=1.0)

    def _odom_cb(self, msg):
        receive_stamp = time.time()
        source_stamp = (
            float(msg.header.stamp.sec)
            + float(msg.header.stamp.nanosec) * 1.0e-9
        )
        source_age = receive_stamp - source_stamp
        if (
            source_stamp <= 0.0
            or source_age < -1.0
            or source_age > self.max_source_age
        ):
            self._reject_odom(
                f"source age {source_age * 1000.0:.1f} ms exceeds "
                f"{self.max_source_age * 1000.0:.1f} ms"
            )
            return
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        old_pos = np.asarray(
            [
                float(p.x),
                float(p.y),
                float(p.z),
            ],
            dtype=np.float64,
        )
        old_quat_xyzw = normalize_quat_xyzw(
            [float(q.x), float(q.y), float(q.z), float(q.w)]
        )
        old_rot = R.from_quat(old_quat_xyzw).as_matrix()

        # /Odometry is FAST-LIO's IMU/body origin.  Recover the LiDAR origin
        # using mapping.extrinsic_T before applying the robot-axis remap.
        mapped_pos = POS_REMAP @ old_pos
        mapped_lidar_pos = mapped_lidar_position(
            old_pos,
            old_rot,
            self.fastlio_lidar_to_imu_xyz,
        )
        if self.last_accepted_source_stamp is not None:
            source_dt = source_stamp - self.last_accepted_source_stamp
            receive_dt = receive_stamp - self.last_accepted_receive_stamp
            if source_dt <= 0.0:
                self._reject_odom("source timestamp did not advance")
                return
            if receive_dt > self.max_receive_gap:
                # A fresh frame after a subscriber/CPU scheduling pause is
                # usable.  Rejecting it without advancing the baseline makes
                # every later frame exceed the same gap forever.  Keep the
                # source-age and pose-jump checks below, but allow this frame
                # to re-establish the receive-time baseline.
                if self.node is not None:
                    self.node.get_logger().warning(
                        f"FAST-LIO receive gap "
                        f"{receive_dt * 1000.0:.1f} ms; accepting the fresh "
                        "frame and resynchronizing",
                        throttle_duration_sec=1.0,
                    )
            position_step = float(
                np.linalg.norm(mapped_pos - self.last_accepted_mapped_pos)
            )
            bounded_dt = min(source_dt, self.max_receive_gap)
            position_limit = (
                self.position_jump_margin
                + self.max_linear_speed * bounded_dt
            )
            delta_rot = self.last_accepted_old_rot.T @ old_rot
            rotation_cosine = np.clip(
                (np.trace(delta_rot) - 1.0) * 0.5, -1.0, 1.0
            )
            angle_step = float(math.acos(float(rotation_cosine)))
            angle_limit = (
                self.angle_jump_margin
                + self.max_angular_speed * bounded_dt
            )
            if position_step > position_limit:
                self._reject_odom(
                    f"position jump {position_step:.4f} m exceeds "
                    f"{position_limit:.4f} m for dt={source_dt:.4f}s"
                )
                return
            if angle_step > angle_limit:
                self._reject_odom(
                    f"angle jump {angle_step:.4f} rad exceeds "
                    f"{angle_limit:.4f} rad for dt={source_dt:.4f}s"
                )
                return

        old_rpy = R.from_matrix(old_rot).as_euler("xyz")
        delta_old_rpy = wrap_rpy(old_rpy - self.ideal_rpy)
        new_rot = mapped_body_rotation(old_rot, self.ideal_rpy)
        new_rpy = R.from_matrix(new_rot).as_euler("xyz")
        orientation_xyzw = normalize_quat_xyzw(
            R.from_matrix(new_rot).as_quat()
        )

        with self.lock:
            self.latest_mapped_pos = mapped_pos.copy()
            self.latest_mapped_lidar_pos = mapped_lidar_pos.copy()
            ground_height = self.ground_height
            if self.ground_reference_topic:
                reference_ready = self.ground_reference_received
            else:
                reference_ready = ground_height is not None
                if reference_ready and self.origin_pos is None:
                    self.origin_pos = mapped_lidar_pos.copy()
                    self.world_lidar_anchor = np.asarray(
                        [0.0, 0.0, ground_height], dtype=np.float64
                    )
                    reference_base_position = (
                        self.world_lidar_anchor
                        + new_rot @ self.lidar_to_base_xyz
                    )
                    self.position_offset = reference_base_position.copy()
                    self.initial_arm_base_height = float(
                        reference_base_position[2]
                    )
                    print(
                        "Anchored first synchronized LiDAR pose with rotated "
                        "LiDAR-to-base translation"
                    )
            if reference_ready:
                origin_pos = self.origin_pos.copy()
                world_lidar_anchor = self.world_lidar_anchor.copy()
                initial_arm_base_height = self.initial_arm_base_height
            else:
                origin_pos = None
                world_lidar_anchor = None
                initial_arm_base_height = None

        if self.ground_height_topic and not reference_ready:
            return
        if origin_pos is None:
            # No ground source requested: keep the legacy arbitrary world
            # anchor, but still rotate the relative LiDAR-to-base vector.
            with self.lock:
                if self.origin_pos is None:
                    self.origin_pos = mapped_lidar_pos.copy()
                    initial_lidar_position = (
                        self.position_offset - new_rot @ self.lidar_to_base_xyz
                    )
                    self.world_lidar_anchor = initial_lidar_position.copy()
                    print("Anchored first LiDAR position frame")
                origin_pos = self.origin_pos.copy()
                world_lidar_anchor = self.world_lidar_anchor.copy()

        world_lidar_pos = (
            mapped_lidar_pos - origin_pos + world_lidar_anchor
        )
        # [-0.14, 0, 0.0677] is a rigid relative translation, not a world
        # offset.  Its x component contributes to world z under pitch.
        rotated_lidar_to_base = new_rot @ self.lidar_to_base_xyz
        world_pos = world_lidar_pos + rotated_lidar_to_base

        world_lidar_pose6d = np.concatenate([world_lidar_pos, new_rpy])
        world_pose6d = np.concatenate([world_pos, new_rpy])
        tf_world_arm_base = np.eye(4, dtype=np.float64)
        tf_world_arm_base[:3, :3] = new_rot
        tf_world_arm_base[:3, 3] = world_pos

        data = {
            "ok": True,
            # Keep stamp as the callback receipt time for compatibility with
            # existing freshness checks.  source_stamp is the producer's ROS
            # timestamp and exposes the true upstream/transport age.
            "stamp": receive_stamp,
            "receive_stamp": receive_stamp,
            "source_stamp": source_stamp,

            # [x, y, z]
            "position": world_pos,
            "ground_height": ground_height,
            "initial_base_height": (
                None
                if initial_arm_base_height is None
                else float(initial_arm_base_height)
            ),
            "initial_arm_base_height": (
                None
                if initial_arm_base_height is None
                else float(initial_arm_base_height)
            ),
            "ground_reference_received": self.ground_reference_received,
            "ground_reference_sequence": self.ground_reference_sequence,
            "ground_reference_raw_position": (
                None
                if self.ground_reference_raw_pos is None
                else self.ground_reference_raw_pos.copy()
            ),
            "ground_reference_raw_orientation_xyzw": (
                None
                if self.ground_reference_raw_quat is None
                else self.ground_reference_raw_quat.copy()
            ),
            "raw_position": old_pos,
            "mapped_position": mapped_pos,
            "mapped_lidar_position": mapped_lidar_pos,
            "world_lidar_position": world_lidar_pos,
            "lidar_to_base_xyz": self.lidar_to_base_xyz.copy(),
            "rotated_lidar_to_base_xyz": rotated_lidar_to_base,
            "fastlio_lidar_to_imu_xyz": (
                self.fastlio_lidar_to_imu_xyz.copy()
            ),

            # [qx, qy, qz, qw]
            "orientation_xyzw": orientation_xyzw,

            # [x, y, z, roll, pitch, yaw]
            "world_lidar": world_lidar_pose6d,
            "world_arm_base": world_pose6d,
            "tf_world_arm_base": tf_world_arm_base,
            # Backward-compatible aliases.  In this deployment "base" means
            # the ARX arm base_link, not the TRON base_Link.
            "world_base": world_pose6d,
            "tf_world_base": tf_world_arm_base,

            # 调试用
            "raw_odom_quat_xyzw": old_quat_xyzw,
            "old_rpy": old_rpy,
            "ideal_rpy": self.ideal_rpy,
            "delta_old_rpy": delta_old_rpy,
            "new_rpy": new_rpy,
        }

        with self.lock:
            self.latest = data
            self.last_accepted_source_stamp = source_stamp
            self.last_accepted_receive_stamp = receive_stamp
            self.last_accepted_mapped_pos = mapped_pos.copy()
            self.last_accepted_mapped_lidar_pos = mapped_lidar_pos.copy()
            self.last_accepted_old_rot = old_rot.copy()

        if self.output_path is not None:
            json_data = {
                "stamp": float(data["stamp"]),
                "position": data["position"].tolist(),
                "orientation_xyzw": data["orientation_xyzw"].tolist(),
                "world_arm_base": data["world_arm_base"].tolist(),
                # Backward-compatible alias for existing consumers.
                "world_base": data["world_base"].tolist(),

                # 调试用，可以后面删
                "old_rpy": data["old_rpy"].tolist(),
                "ideal_rpy": data["ideal_rpy"].tolist(),
                "delta_old_rpy": data["delta_old_rpy"].tolist(),
                "new_rpy": data["new_rpy"].tolist(),
            }

            tmp_path = self.output_path + ".tmp"

            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, separators=(",", ":"))

            os.replace(tmp_path, self.output_path)

    def _reject_odom(self, reason):
        self.rejected_odom_count += 1
        if self.node is not None:
            self.node.get_logger().warning(
                f"Rejected FAST-LIO odometry: {reason}",
                throttle_duration_sec=1.0,
            )

    def read(self):
        with self.lock:
            if self.latest is None:
                return {
                    "ok": False,
                    "stamp": 0.0,
                }

            result = dict(self.latest)
        source_age = time.time() - float(result.get("source_stamp", 0.0))
        if source_age > self.max_source_age:
            result["ok"] = False
            result["error"] = (
                f"FAST-LIO odometry stale: source age "
                f"{source_age * 1000.0:.1f} ms"
            )
        result["rejected_odom_count"] = self.rejected_odom_count
        return result

    def reanchor(self):
        with self.lock:
            anchor_mapped_lidar = self.last_accepted_mapped_lidar_pos
            if anchor_mapped_lidar is None:
                anchor_mapped_lidar = self.latest_mapped_lidar_pos
            if (
                anchor_mapped_lidar is None
                or self.origin_pos is None
                or self.world_lidar_anchor is None
            ):
                return False
            current_world_lidar = (
                anchor_mapped_lidar
                - self.origin_pos
                + self.world_lidar_anchor
            )
            self.world_lidar_anchor = current_world_lidar.copy()
            self.origin_pos = anchor_mapped_lidar.copy()
        print(
            "Reanchored FAST-LIO LiDAR origin with world-position continuity: "
            f"{np.array2string(current_world_lidar, precision=5)}"
        )
        return True

    def wait(self, timeout_s=3.0):
        deadline = time.monotonic() + max(0.0, timeout_s)

        while time.monotonic() < deadline:
            if self.read().get("ok", False):
                return True

            time.sleep(0.02)

        return self.read().get("ok", False)


def parse_vec3(text):
    values = [float(x) for x in str(text).replace(",", " ").split()]

    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected 3 values")

    return values


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--topic",
        default="/Odometry",
    )

    parser.add_argument(
        "--position-offset",
        type=parse_vec3,
        default=POSITION_OFFSET.tolist(),
        help="position offset: x y z, default='-0.14 0.0 0.8277'",
    )

    parser.add_argument(
        "--ideal-rpy",
        type=parse_vec3,
        default=[0.0, 0.0, 0.0],
        help="ideal rpy in original odom frame, unit: rad. default='0 0 0'",
    )

    parser.add_argument(
        "--rate",
        type=float,
        default=5.0,
        help="print rate in Hz",
    )

    parser.add_argument(
        "--out",
        default="",
        help="optional json output path",
    )

    args = parser.parse_args()

    output_path = args.out if args.out.strip() else None

    reader = Ros2OdomReader(
        topic=args.topic,
        position_offset=args.position_offset,
        ideal_rpy=args.ideal_rpy,
        output_path=output_path,
    )

    reader.start()

    try:
        while True:
            data = reader.read()

            if data.get("ok", False):
                print(
                    "raw_position="
                    + np.array2string(
                        data["raw_position"],
                        precision=4,
                        suppress_small=True,
                    )
                )

                print(
                    "mapped_position="
                    + np.array2string(
                        data["mapped_position"],
                        precision=4,
                        suppress_small=True,
                    )
                )

                print(
                    "position="
                    + np.array2string(
                        data["position"],
                        precision=4,
                        suppress_small=True,
                    )
                )

                print(
                    "orientation_xyzw="
                    + np.array2string(
                        data["orientation_xyzw"],
                        precision=4,
                        suppress_small=True,
                    )
                )

                print(
                    "world_lidar="
                    + np.array2string(
                        data["world_lidar"],
                        precision=4,
                        suppress_small=True,
                    )
                )

                print(
                    "old_rpy="
                    + np.array2string(
                        data["old_rpy"],
                        precision=4,
                        suppress_small=True,
                    )
                )

                print(
                    "delta_old_rpy="
                    + np.array2string(
                        data["delta_old_rpy"],
                        precision=4,
                        suppress_small=True,
                    )
                )

                print(
                    "new_rpy="
                    + np.array2string(
                        data["new_rpy"],
                        precision=4,
                        suppress_small=True,
                    )
                )

            else:
                print("waiting for odom...")

            time.sleep(1.0 / max(args.rate, 0.1))

    finally:
        reader.stop()


if __name__ == "__main__":
    main()
