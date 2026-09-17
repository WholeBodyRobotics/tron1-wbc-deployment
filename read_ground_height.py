#!/usr/bin/env python3
"""Estimate MID360 height above a locally planar ground surface."""

import argparse
import math
from collections import deque

import numpy as np
import rclpy
from livox_ros_driver2.msg import CustomMsg
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Imu
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Float64, Float64MultiArray, Header


GROUND_HEIGHT_TOPIC = "/ground_height"
GROUND_REFERENCE_TOPIC = "/ground_height_reference"
GROUND_FREEZE_TOPIC = "/ground_height_freeze"


def transient_local_qos():
    return QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def stamp_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def make_ground_reference(
    height, raw_odom_position, sequence=1, raw_odom_quaternion=None
):
    """Pack one atomic height/FAST-LIO reference sample.

    Layout: [height, raw IMU xyz, sequence, raw IMU quaternion xyzw].
    The orientation is required because both the FAST-LIO IMU-to-LiDAR
    extrinsic and the LiDAR-to-arm-base translation rotate with the body.
    """
    msg = Float64MultiArray()
    raw = np.asarray(raw_odom_position, dtype=np.float64).reshape(3)
    quat = (
        np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        if raw_odom_quaternion is None
        else np.asarray(raw_odom_quaternion, dtype=np.float64).reshape(4)
    )
    msg.data = [
        float(height),
        float(raw[0]),
        float(raw[1]),
        float(raw[2]),
        float(sequence),
        float(quat[0]),
        float(quat[1]),
        float(quat[2]),
        float(quat[3]),
    ]
    return msg


class FixedGroundHeightPublisher(Node):
    def __init__(self, height):
        super().__init__("fixed_ground_height_publisher")
        self.height = float(height)
        self.reference_odom_position = None
        self.reference_odom_quaternion = None
        self.latest_odom_position = None
        self.latest_odom_quaternion = None
        self.reference_sequence = 0
        self.frozen = False
        height_qos = transient_local_qos()
        self.height_publisher = self.create_publisher(
            Float64, GROUND_HEIGHT_TOPIC, height_qos
        )
        self.reference_publisher = self.create_publisher(
            Float64MultiArray, GROUND_REFERENCE_TOPIC, height_qos
        )
        self.create_subscription(
            Odometry,
            "/Odometry",
            self.odom_callback,
            1,
        )
        self.create_subscription(
            Bool,
            GROUND_FREEZE_TOPIC,
            self.freeze_callback,
            1,
        )
        self.create_timer(0.2, self.publish_height)
        self.publish_height()
        self.get_logger().warning(
            f"Using fixed ground height: {self.height:.4f} m"
        )

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.latest_odom_position = np.asarray(
            [float(p.x), float(p.y), float(p.z)], dtype=np.float64
        )
        self.latest_odom_quaternion = np.asarray(
            [float(q.x), float(q.y), float(q.z), float(q.w)], dtype=np.float64
        )
        if self.reference_odom_position is None:
            self.reference_odom_position = self.latest_odom_position.copy()
            self.reference_odom_quaternion = self.latest_odom_quaternion.copy()
            self.reference_sequence = 1
            self.get_logger().info(
                "Fixed ground height paired with FAST-LIO raw position "
                f"{np.array2string(self.reference_odom_position, precision=6)}"
            )
            self.publish_height()

    def freeze_callback(self, msg):
        if not bool(msg.data) or self.frozen or self.latest_odom_position is None:
            return
        self.reference_odom_position = self.latest_odom_position.copy()
        self.reference_odom_quaternion = self.latest_odom_quaternion.copy()
        self.reference_sequence += 1
        self.frozen = True
        self.get_logger().info(
            "Final fixed-height reference frozen at FAST-LIO raw position "
            f"{np.array2string(self.reference_odom_position, precision=6)}"
        )
        self.publish_height()

    def publish_height(self):
        msg = Float64()
        msg.data = self.height
        self.height_publisher.publish(msg)
        if self.reference_odom_position is not None:
            self.reference_publisher.publish(
                make_ground_reference(
                    self.height,
                    self.reference_odom_position,
                    self.reference_sequence,
                    self.reference_odom_quaternion,
                )
            )


class GroundHeightReader(Node):
    def __init__(self, stability_samples=20, max_stability_range=0.005):
        super().__init__("ground_height_reader")
        self.height_history = []
        self.stability_samples = int(stability_samples)
        self.max_stability_range = float(max_stability_range)
        self.startup_height_samples = deque(maxlen=self.stability_samples)
        self.initial_height = None
        self.current_height = None
        self.reference_odom_position = None
        self.reference_odom_quaternion = None
        self.current_reference_odom_position = None
        self.current_reference_odom_quaternion = None
        self.reference_sequence = 0
        self.frozen = False
        self.latest_odom_position = None
        self.latest_odom_quaternion = None
        self.latest_odom_stamp = None
        self.last_live_fit_time = 0.0
        # The +x ground sector is sparse (roughly 9 returns per scan), so use
        # 25 stationary startup scans before judging the plane.
        self.candidate_frames = deque(maxlen=25)
        self.last_cloud_stats_time = 0.0
        self.up_axis = None
        self.imu_sample_count = 0
        self.create_subscription(
            CustomMsg,
            "/livox/lidar",
            self.lidar_callback,
            qos_profile_sensor_data,
        )
        self.raw_cloud_publisher = self.create_publisher(
            PointCloud2, "/lidar_points_debug", 10
        )
        self.candidate_publisher = self.create_publisher(
            PointCloud2, "/ground_candidates", 10
        )
        self.inlier_publisher = self.create_publisher(
            PointCloud2, "/ground_inliers", 10
        )
        height_qos = transient_local_qos()
        self.height_publisher = self.create_publisher(
            Float64, GROUND_HEIGHT_TOPIC, height_qos
        )
        self.reference_publisher = self.create_publisher(
            Float64MultiArray, GROUND_REFERENCE_TOPIC, height_qos
        )
        # Keep the node and transient-local publishers alive for deployment
        # processes that subscribe later.  The reference message atomically
        # pairs ground height with the raw FAST-LIO position at lock time.
        self.create_timer(0.2, self.publish_initial_height)
        self.create_subscription(
            Odometry,
            "/Odometry",
            self.odom_callback,
            1,
        )
        self.create_subscription(
            Bool,
            GROUND_FREEZE_TOPIC,
            self.freeze_callback,
            1,
        )
        self.create_subscription(
            Imu,
            "/livox/imu",
            self.imu_callback,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            "Listening on /livox/lidar and /livox/imu; "
            "searching 360 deg for a gravity-aligned ground plane"
        )

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.latest_odom_position = np.asarray(
            [float(p.x), float(p.y), float(p.z)], dtype=np.float64
        )
        self.latest_odom_quaternion = np.asarray(
            [float(q.x), float(q.y), float(q.z), float(q.w)], dtype=np.float64
        )
        self.latest_odom_stamp = stamp_seconds(msg.header.stamp)

    def freeze_callback(self, msg):
        if not bool(msg.data) or self.frozen:
            return
        if (
            self.current_height is None
            or self.current_reference_odom_position is None
            or self.current_reference_odom_quaternion is None
        ):
            self.get_logger().warning(
                "Cannot freeze ground height before a live paired estimate exists"
            )
            return
        previous_height = self.initial_height
        self.initial_height = float(self.current_height)
        self.reference_odom_position = (
            self.current_reference_odom_position.copy()
        )
        self.reference_odom_quaternion = (
            self.current_reference_odom_quaternion.copy()
        )
        self.reference_sequence += 1
        self.frozen = True
        self.get_logger().info(
            "Final pre-output ground reference frozen: "
            f"height={self.initial_height:.4f} m "
            f"change_since_first_lock="
            f"{self.initial_height - previous_height:+.4f} m "
            f"raw_odom="
            f"{np.array2string(self.reference_odom_position, precision=6)}"
        )
        self.publish_initial_height()

    def imu_callback(self, msg):
        if self.frozen:
            return
        acceleration = np.asarray(
            [
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            ],
            dtype=np.float64,
        )
        norm = np.linalg.norm(acceleration)
        # Some Livox driver versions report acceleration in g, others in
        # m/s^2. Only the direction is needed, so do not assume either unit.
        if not np.isfinite(norm) or norm < 0.1:
            return
        # The MID360 is installed upside down.  Its stationary accelerometer
        # direction points toward physical up in the lidar frame (observed
        # near lidar -z), so keep that direction instead of negating it.
        measured_up = acceleration / norm
        # Keep the gravity-derived direction in the lidar frame.  Do not force
        # its z component positive: an upside-down or tilted installation can
        # have physical up pointing toward lidar -z.
        if self.up_axis is None:
            self.up_axis = measured_up
        else:
            self.up_axis = 0.98 * self.up_axis + 0.02 * measured_up
            self.up_axis /= np.linalg.norm(self.up_axis)
        self.imu_sample_count += 1

    def lidar_callback(self, msg):
        if self.frozen:
            return
        now = self.get_clock().now().nanoseconds * 1.0e-9
        if (
            self.initial_height is not None
            and now - self.last_live_fit_time < 0.20
        ):
            return
        if self.initial_height is not None:
            self.last_live_fit_time = now
        if len(msg.points) < 100:
            return

        # Keep every point here: a single MID360 packet can contain relatively
        # few floor returns. Computation is bounded later after ROI filtering.
        points = np.asarray(
            [(point.x, point.y, point.z) for point in msg.points],
            dtype=np.float64,
        )
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) < 100:
            return
        frame_id = msg.header.frame_id or "livox_frame"
        self.publish_cloud(
            self.raw_cloud_publisher, points[::2], frame_id, msg.header.stamp
        )
        if self.up_axis is None or self.imu_sample_count < 20:
            self.get_logger().warning(
                f"Waiting for stable IMU gravity direction: "
                f"{self.imu_sample_count}/20 samples",
                throttle_duration_sec=2.0,
            )
            return

        if now - self.last_cloud_stats_time >= 2.0:
            self.last_cloud_stats_time = now
            percentiles = np.percentile(points, [1, 10, 50, 90, 99], axis=0)
            print(
                "cloud percentiles [1,10,50,90,99]% (m):\n"
                f"  x={np.array2string(percentiles[:, 0], precision=3)}\n"
                f"  y={np.array2string(percentiles[:, 1], precision=3)}\n"
                f"  z={np.array2string(percentiles[:, 2], precision=3)}",
                flush=True,
            )

        # Gravity from the IMU defines up/down, so this remains valid when the
        # lidar is tilted or installed upside down.
        distance = np.linalg.norm(points, axis=1)
        down_distance = -(points @ self.up_axis)
        height_band = (
            (distance < 6.0)
            & (down_distance >= 0.30)
            & (down_distance < 1.30)
        )
        if now - self.last_cloud_stats_time < 0.05:
            height_band_angles = np.degrees(
                np.arctan2(points[height_band, 1], points[height_band, 0])
            )
            angle_bins = np.arange(-180.0, 181.0, 30.0)
            angle_counts, _ = np.histogram(height_band_angles, bins=angle_bins)
            print(
                "height-band directional counts: "
                f"+x={np.count_nonzero(height_band & (points[:, 0] > 0.3))}  "
                f"-x={np.count_nonzero(height_band & (points[:, 0] < -0.3))}  "
                f"+y={np.count_nonzero(height_band & (points[:, 1] > 0.3))}  "
                f"-y={np.count_nonzero(height_band & (points[:, 1] < -0.3))}  "
                f"up={np.array2string(self.up_axis, precision=3)}",
                flush=True,
            )
            print(
                "height-band azimuth histogram (30 deg bins):\n  "
                + "  ".join(
                    f"[{int(angle_bins[i])},{int(angle_bins[i + 1])}):"
                    f"{int(angle_counts[i])}"
                    for i in range(len(angle_counts))
                ),
                flush=True,
            )
        # Search all azimuths.  The previous fixed -45 degree sector could
        # point at a wall corner after the robot or lidar mounting changed.
        mask = (
            (distance > 0.15)
            & (distance < 3.0)
            & (down_distance >= 0.30)
            & (down_distance < 1.30)
        )
        frame_candidates = points[mask]
        self.publish_cloud(
            self.candidate_publisher,
            frame_candidates,
            frame_id,
            msg.header.stamp,
        )
        if len(frame_candidates) > 1500:
            step = max(1, len(frame_candidates) // 1500)
            frame_candidates = frame_candidates[::step]
        if self.initial_height is None:
            self.candidate_frames.append(frame_candidates)
            candidates = np.concatenate(self.candidate_frames, axis=0)
        else:
            # Once the first absolute reference is known, use only the current
            # scan so rope motion is not averaged away by a multi-second cloud
            # window.  This live estimate is used only until output starts.
            candidates = frame_candidates
        if len(candidates) < 100:
            self.get_logger().warning(
                f"Collecting ground candidates: {len(candidates)}/100 "
                f"from {len(self.candidate_frames)} frames",
                throttle_duration_sec=2.0,
            )
            return

        result = self.fit_ground_plane(candidates, self.up_axis)
        if result is None:
            self.get_logger().warning(
                "No reliable horizontal ground plane",
                throttle_duration_sec=2.0,
            )
            return

        height, normal, residual, inliers, fit_metrics = result
        inlier_count = len(inliers)
        self.publish_cloud(
            self.inlier_publisher, inliers, frame_id, msg.header.stamp
        )
        sample_index = len(self.startup_height_samples) + 1
        if self.initial_height is None:
            self.get_logger().info(
                "Ground fit sample "
                f"{sample_index}: height={height:.6f} m, "
                f"residual={residual:.6f} m, "
                f"inliers={inlier_count}/{fit_metrics['candidate_count']} "
                f"({fit_metrics['inlier_ratio']:.3f}), "
                f"aggregated_candidates={fit_metrics['input_candidate_count']}, "
                f"tilt={fit_metrics['tilt_deg']:.3f} deg, "
                f"normal={np.array2string(normal, precision=5)}, "
                f"center={np.array2string(fit_metrics['plane_center'], precision=5)}, "
                f"spans={np.array2string(fit_metrics['tangent_spans'], precision=5)}, "
                f"cells={fit_metrics['occupied_cells']}"
            )
        if self.initial_height is None:
            self.startup_height_samples.append(height)
            if len(self.startup_height_samples) < self.stability_samples:
                self.get_logger().info(
                    f"Collecting startup ground heights: "
                    f"{len(self.startup_height_samples)}/"
                    f"{self.stability_samples}",
                    throttle_duration_sec=1.0,
                )
                return

            startup_heights = np.asarray(
                self.startup_height_samples, dtype=np.float64
            )
            startup_spread = float(np.ptp(startup_heights))
            if startup_spread > self.max_stability_range:
                self.get_logger().warning(
                    "Waiting for suspension/ground height to settle: "
                    f"{self.stability_samples}-sample range="
                    f"{startup_spread:.4f} m > "
                    f"{self.max_stability_range:.4f} m",
                    throttle_duration_sec=1.0,
                )
                return

            cloud_stamp = stamp_seconds(msg.header.stamp)
            if (
                self.latest_odom_position is None
                or self.latest_odom_quaternion is None
                or self.latest_odom_stamp is None
                or abs(self.latest_odom_stamp - cloud_stamp) > 0.50
            ):
                self.get_logger().warning(
                    "Ground height is stable but no time-matched FAST-LIO "
                    "reference is available yet",
                    throttle_duration_sec=1.0,
                )
                return

            self.initial_height = float(np.median(startup_heights))
            self.current_height = self.initial_height
            self.reference_odom_position = self.latest_odom_position.copy()
            self.reference_odom_quaternion = (
                self.latest_odom_quaternion.copy()
            )
            self.current_reference_odom_position = (
                self.reference_odom_position.copy()
            )
            self.current_reference_odom_quaternion = (
                self.reference_odom_quaternion.copy()
            )
            self.reference_sequence = 1
            self.height_history = startup_heights.tolist()
            self.get_logger().info(
                "Startup ground-height self-check passed: "
                f"height={self.initial_height:.4f} m, "
                f"{self.stability_samples}-sample range="
                f"{startup_spread:.4f} m, "
                f"samples={np.array2string(startup_heights, precision=6)}"
            )
            self.get_logger().info(
                "Ground height paired with FAST-LIO raw position "
                f"{np.array2string(self.reference_odom_position, precision=6)}"
            )
            self.publish_initial_height()
            self.candidate_frames.clear()
            self.startup_height_samples.clear()
            self.get_logger().info(
                "Initial ground height and FAST-LIO reference locked; "
                "continuing live ground fits until output-start freeze"
            )
            return

        cloud_stamp = stamp_seconds(msg.header.stamp)
        if (
            self.latest_odom_position is None
            or self.latest_odom_quaternion is None
            or self.latest_odom_stamp is None
            or abs(self.latest_odom_stamp - cloud_stamp) > 0.50
        ):
            self.get_logger().warning(
                "Skipping live ground height without a time-matched FAST-LIO pose",
                throttle_duration_sec=1.0,
            )
            return
        self.current_height = float(height)
        self.current_reference_odom_position = self.latest_odom_position.copy()
        self.current_reference_odom_quaternion = (
            self.latest_odom_quaternion.copy()
        )
        self.get_logger().info(
            "Live pre-output ground height: "
            f"height={self.current_height:.4f} m "
            f"change_since_first_lock="
            f"{self.current_height - self.initial_height:+.4f} m",
            throttle_duration_sec=1.0,
        )
        self.publish_initial_height()

    def publish_initial_height(self):
        if self.initial_height is None:
            return
        height_msg = Float64()
        height_msg.data = (
            self.initial_height
            if self.current_height is None
            else self.current_height
        )
        self.height_publisher.publish(height_msg)
        if self.reference_odom_position is not None:
            self.reference_publisher.publish(
                make_ground_reference(
                    self.initial_height,
                    self.reference_odom_position,
                    self.reference_sequence,
                    self.reference_odom_quaternion,
                )
            )

    @staticmethod
    def publish_cloud(publisher, points, frame_id, stamp):
        header = Header()
        header.frame_id = frame_id
        header.stamp = stamp
        xyz = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        publisher.publish(point_cloud2.create_cloud_xyz32(header, xyz.tolist()))

    @staticmethod
    def fit_ground_plane(points, up_axis):
        if len(points) < 100:
            return None

        input_count = len(points)
        if len(points) > 4000:
            step = max(1, len(points) // 4000)
            points = points[::step]

        rng = np.random.default_rng(7)
        best_indices = None
        best_score = -1.0
        max_tilt_cos = math.cos(math.radians(20.0))

        for _ in range(400):
            sample_indices = rng.choice(len(points), size=3, replace=False)
            p0, p1, p2 = points[sample_indices]
            normal = np.cross(p1 - p0, p2 - p0)
            normal_norm = np.linalg.norm(normal)
            if normal_norm < 1.0e-6:
                continue
            normal /= normal_norm
            if np.dot(normal, up_axis) < 0.0:
                normal = -normal

            # Only accept planes close to perpendicular to gravity.
            if np.dot(normal, up_axis) < max_tilt_cos:
                continue

            plane_d = -float(np.dot(normal, p0))
            alignment = float(np.dot(normal, up_axis))
            height = abs(plane_d / alignment)
            if not 0.30 <= height <= 1.30:
                continue

            distances = np.abs(points @ normal + plane_d)
            indices = np.flatnonzero(distances < 0.030)
            if len(indices) < 80:
                continue

            residual = float(np.median(distances[indices]))
            score = len(indices) - 20.0 * residual
            if score > best_score:
                best_score = score
                best_indices = indices

        if best_indices is None:
            return None

        inliers = points[best_indices]
        if len(inliers) / len(points) < 0.12:
            return None

        # Refit the winning RANSAC plane using all of its inliers.
        center = np.mean(inliers, axis=0)
        _, _, vh = np.linalg.svd(inliers - center, full_matrices=False)
        normal = vh[-1]
        normal /= np.linalg.norm(normal)
        if np.dot(normal, up_axis) < 0.0:
            normal = -normal
        if np.dot(normal, up_axis) < max_tilt_cos:
            return None

        # Plane: normal.p + d = 0. Intersect it with the gravity ray from the
        # lidar origin so the result is vertical height, not lidar-z distance.
        plane_d = -float(np.dot(normal, center))
        alignment = float(np.dot(normal, up_axis))
        height = abs(plane_d / alignment)
        residuals = np.abs((inliers - center) @ normal)
        residual = float(np.median(residuals))

        if not 0.30 <= height <= 1.30 or residual > 0.025:
            return None

        # Reject narrow wall/floor edges and small robot surfaces.  Measure
        # support in the fitted plane itself, which also works for a tilted
        # lidar.  A real floor patch must have meaningful extent along both
        # tangent directions and occupy multiple 15 cm grid cells.
        tangent_points = inliers - center
        _, singular_values, vh = np.linalg.svd(tangent_points, full_matrices=False)
        plane_xy = tangent_points @ vh[:2].T
        spans = np.ptp(plane_xy, axis=0)
        if np.min(spans) < 0.35:
            return None
        if len(singular_values) < 2 or singular_values[1] / math.sqrt(len(inliers)) < 0.10:
            return None
        grid = np.floor(plane_xy / 0.15).astype(np.int64)
        occupied_cells = len(np.unique(grid, axis=0))
        if occupied_cells < 12:
            return None

        alignment = float(np.clip(np.dot(normal, up_axis), -1.0, 1.0))
        fit_metrics = {
            "candidate_count": int(len(points)),
            "input_candidate_count": int(input_count),
            "inlier_ratio": float(len(inliers) / len(points)),
            "tilt_deg": float(math.degrees(math.acos(alignment))),
            "plane_center": center.copy(),
            "tangent_spans": spans.copy(),
            "occupied_cells": int(occupied_cells),
        }
        return height, normal, residual, inliers, fit_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixed-height",
        type=float,
        default=None,
        help="Publish this fixed ground height instead of fitting a plane.",
    )
    parser.add_argument(
        "--stability-samples",
        type=int,
        default=20,
        help="number of consecutive valid estimates required before locking",
    )
    parser.add_argument(
        "--max-stability-range",
        type=float,
        default=0.005,
        help="maximum range of the locking window in metres",
    )
    args = parser.parse_args()
    if args.fixed_height is not None and not 0.50 <= args.fixed_height <= 1.30:
        parser.error("--fixed-height must be in [0.50, 1.30] meters")
    if args.stability_samples < 3:
        parser.error("--stability-samples must be at least 3")
    if not 0.0 < args.max_stability_range <= 0.10:
        parser.error("--max-stability-range must be in (0, 0.10] meters")

    rclpy.init()
    node = (
        FixedGroundHeightPublisher(args.fixed_height)
        if args.fixed_height is not None
        else GroundHeightReader(
            stability_samples=args.stability_samples,
            max_stability_range=args.max_stability_range,
        )
    )
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        # ROS 2 Humble can raise its private RCLError while another process is
        # shutting down the shared context.  Ignore only shutdown-time errors;
        # real runtime failures must still surface.
        if rclpy.ok():
            raise
    finally:
        try:
            node.destroy_node()
        except Exception:
            if rclpy.ok():
                raise
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
