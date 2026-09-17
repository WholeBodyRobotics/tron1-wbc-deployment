import argparse
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from read_lidar_odom import Ros2OdomReader, pose6d_from_transform, transform_from_pose6d
from read_tron_arx_state import ArxStateReader


def fmt(values):
    return np.array2string(np.asarray(values), precision=4, suppress_small=True)


def quat_xyzw_from_pose6d(pose):
    pose = np.asarray(pose, dtype=np.float64).reshape(-1)[:6]
    return R.from_euler("xyz", pose[3:6]).as_quat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arx-ip", default=os.getenv("ARX5_ZMQ_IP", "127.0.0.1"))
    parser.add_argument("--arx-port", type=int, default=int(os.getenv("ARX5_ZMQ_PORT", "8765")))
    parser.add_argument("--arx-timeout-ms", type=int, default=200)
    parser.add_argument("--odom-topic", default="/Odometry")
    parser.add_argument("--rate", type=float, default=5.0)
    parser.add_argument("--wait-s", type=float, default=3.0)
    args = parser.parse_args()

    arx_reader = ArxStateReader(args.arx_ip, args.arx_port, args.arx_timeout_ms)
    odom_reader = Ros2OdomReader(topic=args.odom_topic)
    odom_reader.start()

    try:
        if not odom_reader.wait(args.wait_s):
            print(f"WARNING: no odom after {args.wait_s:.1f}s")

        dt = 1.0 / max(args.rate, 0.1)
        while True:
            odom = odom_reader.read()
            try:
                arm = arx_reader.read()
            except Exception as exc:
                arm = {"ok": False, "error": str(exc)}

            print("=" * 80)
            if not isinstance(arm, dict) or not arm.get("ok", False):
                error = arm.get("error", "unavailable") if isinstance(arm, dict) else "unavailable"
                print(f"arx_error={error}")
                time.sleep(dt)
                continue
            if not odom.get("ok", False):
                print("odom_error=unavailable")
                time.sleep(dt)
                continue

            base_ee = np.asarray(arm["ee_pose"], dtype=np.float64).reshape(-1)[:6]
            world_base = np.asarray(odom["world_base"], dtype=np.float64).reshape(-1)[:6]
            tf_world_ee = odom["tf_world_base"] @ transform_from_pose6d(base_ee)
            world_ee = pose6d_from_transform(tf_world_ee)

            print(f"world_base_pos={fmt(world_base[:3])}")
            print(f"world_base_rpy={fmt(world_base[3:6])}")
            print(f"world_base_quat_xyzw={fmt(quat_xyzw_from_pose6d(world_base))}")
            print(f"base_ee_pos={fmt(base_ee[:3])}")
            print(f"base_ee_rpy={fmt(base_ee[3:6])}")
            print(f"base_ee_quat_xyzw={fmt(quat_xyzw_from_pose6d(base_ee))}")
            print(f"world_ee_pos={fmt(world_ee[:3])}")
            print(f"world_ee_rpy={fmt(world_ee[3:6])}")
            print(f"world_ee_quat_xyzw={fmt(quat_xyzw_from_pose6d(world_ee))}")
            time.sleep(dt)
    finally:
        odom_reader.stop()


if __name__ == "__main__":
    main()
