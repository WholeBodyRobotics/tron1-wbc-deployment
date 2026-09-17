#!/usr/bin/env python3
"""Regression tests for suspension-height/FAST-LIO reference continuity."""

import time
import unittest
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation as R

from read_lidar_odom import POS_REMAP, Ros2OdomReader


def scalar_msg(value):
    return SimpleNamespace(data=float(value))


def reference_msg(
    height, raw_xyz, sequence=1, raw_quat=(0.0, 0.0, 0.0, 1.0)
):
    values = [float(height), *map(float, raw_xyz)]
    if sequence is not None:
        values.append(float(sequence))
        values.extend(map(float, raw_quat))
    return SimpleNamespace(data=values)


def odom_msg(raw_xyz, stamp, raw_quat=(0.0, 0.0, 0.0, 1.0)):
    sec = int(stamp)
    nanosec = int(round((stamp - sec) * 1.0e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=sec, nanosec=nanosec)
        ),
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(
                    x=float(raw_xyz[0]),
                    y=float(raw_xyz[1]),
                    z=float(raw_xyz[2]),
                ),
                orientation=SimpleNamespace(
                    x=float(raw_quat[0]),
                    y=float(raw_quat[1]),
                    z=float(raw_quat[2]),
                    w=float(raw_quat[3]),
                ),
            )
        ),
    )


class GroundHeightReferenceTest(unittest.TestCase):
    def make_reader(self):
        return Ros2OdomReader(
            ground_height_topic="/ground_height",
            ground_reference_topic="/ground_height_reference",
            lidar_to_base_xyz=[-0.14, 0.0, 0.0677],
        )

    def test_late_reader_recovers_drop_since_height_lock(self):
        reader = self.make_reader()
        reader._ground_height_cb(scalar_msg(0.85))
        reader._ground_reference_cb(reference_msg(0.85, [0.0, 0.0, 0.0]))

        # In this deployment raw FAST-LIO +z maps to physical/world -z.
        # The reader starts after a 2 cm suspension drop but must still
        # reconstruct that drop from the paired raw-odom reference.
        reader._odom_cb(
            odom_msg([0.0, 0.0, 0.02], time.time() - 0.02)
        )
        state = reader.read()
        self.assertTrue(state["ok"])
        self.assertAlmostEqual(state["initial_arm_base_height"], 0.9177, 7)
        self.assertAlmostEqual(state["position"][2], 0.8977, 7)

    def test_reanchor_preserves_accumulated_drop(self):
        reader = self.make_reader()
        reader._ground_reference_cb(reference_msg(0.85, [0.0, 0.0, 0.0]))
        base_stamp = time.time() - 0.10
        reader._odom_cb(odom_msg([0.0, 0.0, 0.02], base_stamp))
        before = reader.read()["position"].copy()

        self.assertTrue(reader.reanchor())
        reader._odom_cb(odom_msg([0.0, 0.0, 0.03], base_stamp + 0.02))
        after = reader.read()["position"].copy()

        np.testing.assert_allclose(before, [-0.14, 0.0, 0.8977], atol=1e-9)
        self.assertAlmostEqual(after[2], 0.8877, 7)
        self.assertAlmostEqual(after[2] - before[2], -0.01, 7)

    def test_final_pre_output_reference_replaces_first_lock(self):
        reader = self.make_reader()
        reader._ground_reference_cb(
            reference_msg(0.85, [0.0, 0.0, 0.0], sequence=1)
        )

        # Rope is lowered by 5 cm during startup.  The live plane fit at the
        # output boundary publishes a new absolute height paired with the
        # corresponding raw FAST-LIO position.
        reader._ground_reference_cb(
            reference_msg(0.80, [0.0, 0.0, 0.05], sequence=2)
        )
        reader._odom_cb(
            odom_msg([0.0, 0.0, 0.05], time.time() - 0.02)
        )
        state = reader.read()
        self.assertAlmostEqual(state["position"][2], 0.8677, 7)
        self.assertAlmostEqual(state["initial_arm_base_height"], 0.8677, 7)
        self.assertEqual(state["ground_reference_sequence"], 2.0)

    def test_output_freeze_waits_for_new_reference_sequence(self):
        reader = self.make_reader()
        reader._ground_reference_cb(
            reference_msg(0.85, [0.0, 0.0, 0.0], sequence=1)
        )

        class FreezePublisher:
            def publish(self, _msg):
                reader._ground_reference_cb(
                    reference_msg(0.80, [0.0, 0.0, 0.05], sequence=2)
                )

        reader.ground_freeze_publisher = FreezePublisher()
        self.assertTrue(reader.freeze_ground_height_reference(timeout_s=0.1))
        self.assertEqual(reader.ground_reference_sequence, 2.0)
        self.assertAlmostEqual(reader.initial_arm_base_height, 0.8677, 7)

    def test_relative_lidar_to_base_translation_rotates_under_pitch(self):
        reader = self.make_reader()
        reader._ground_reference_cb(
            reference_msg(0.85, [0.0, 0.0, 0.0], sequence=1)
        )

        # Construct a raw FAST-LIO rotation whose remapped body rotation is a
        # +20 degree robot pitch. Keep the physical LiDAR origin fixed by
        # compensating the IMU origin for the rotating IMU-to-LiDAR lever arm.
        pitch = np.deg2rad(20.0)
        expected_base_rot = R.from_euler("y", pitch).as_matrix()
        raw_rot = POS_REMAP.T @ expected_base_rot @ POS_REMAP
        raw_quat = R.from_matrix(raw_rot).as_quat()
        lidar_in_imu = reader.fastlio_lidar_to_imu_xyz
        raw_imu_position = lidar_in_imu - raw_rot @ lidar_in_imu

        reader._odom_cb(
            odom_msg(
                raw_imu_position,
                time.time() - 0.02,
                raw_quat=raw_quat,
            )
        )
        state = reader.read()
        expected_lidar_world = np.asarray([0.0, 0.0, 0.85])
        expected_base_world = (
            expected_lidar_world
            + expected_base_rot @ np.asarray([-0.14, 0.0, 0.0677])
        )

        np.testing.assert_allclose(
            state["world_lidar_position"], expected_lidar_world, atol=1e-9
        )
        np.testing.assert_allclose(
            state["position"], expected_base_world, atol=1e-9
        )
        # The former world-constant implementation returned 0.9177 m here.
        self.assertGreater(abs(state["position"][2] - 0.9177), 0.04)


if __name__ == "__main__":
    unittest.main()
