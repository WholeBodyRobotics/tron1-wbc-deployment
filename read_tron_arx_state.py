import argparse
import os
import time
from functools import partial

import numpy as np

os.environ.setdefault("ROBOT_TYPE", "SF_TRON1A_ARX5ARM")

import limxsdk.datatypes as datatypes
from limxsdk.robot import Robot, RobotType
import limxsdk.robot.Rate as Rate


LEG_JOINT_NUM = 8
ARM_JOINT_NUM = 6


class TronStateReader:
    def __init__(self, robot_ip):
        self.robot = Robot(RobotType.PointFoot)
        if not self.robot.init(robot_ip):
            raise RuntimeError(f"Failed to init LimX robot at {robot_ip}")

        self.motor_number = self.robot.getMotorNumber()
        self.robot_state = datatypes.RobotState()
        self.robot_state.q = [0.0] * self.motor_number
        self.robot_state.dq = [0.0] * self.motor_number
        self.robot_state.tau = [0.0] * self.motor_number

        self.imu_data = datatypes.ImuData()
        # LimX ImuData stores quaternions as (w, x, y, z).
        self.imu_data.quat[0] = 1.0
        self.imu_data.quat[1] = 0.0
        self.imu_data.quat[2] = 0.0
        self.imu_data.quat[3] = 0.0

        self.diagnostics = {}
        self.robot_state_count = 0
        self.imu_count = 0
        self.diagnostic_count = 0

        self._robot_state_cb = partial(self._on_robot_state)
        self._imu_cb = partial(self._on_imu)
        self._diagnostic_cb = partial(self._on_diagnostic)

        self.robot.subscribeRobotState(self._robot_state_cb)
        self.robot.subscribeImuData(self._imu_cb)
        self.robot.subscribeDiagnosticValue(self._diagnostic_cb)

    def _on_robot_state(self, robot_state):
        self.robot_state = robot_state
        self.robot_state_count += 1

    def _on_imu(self, imu_data):
        self.imu_data = imu_data
        self.imu_count += 1

    def _on_diagnostic(self, diagnostic):
        self.diagnostics[diagnostic.name] = diagnostic
        self.diagnostic_count += 1

    def wait(self, timeout_s=3.0):
        deadline = time.monotonic() + max(0.0, timeout_s)
        while time.monotonic() < deadline:
            if self.robot_state_count > 0 or self.imu_count > 0:
                return True
            time.sleep(0.02)
        return self.robot_state_count > 0 or self.imu_count > 0

    def read(self):
        q = _fit(self.robot_state.q, LEG_JOINT_NUM)
        dq = _fit(self.robot_state.dq, LEG_JOINT_NUM)
        tau = _fit(self.robot_state.tau, LEG_JOINT_NUM)
        return {
            "stamp": getattr(self.robot_state, "stamp", 0),
            "q": q,
            "dq": dq,
            "tau": tau,
            "imu": {
                "stamp": getattr(self.imu_data, "stamp", 0),
                "acc": np.asarray(self.imu_data.acc, dtype=np.float64).reshape(-1),
                "gyro": np.asarray(self.imu_data.gyro, dtype=np.float64).reshape(-1),
                "quat": np.asarray(self.imu_data.quat, dtype=np.float64).reshape(-1),
            },
            "counts": {
                "state": self.robot_state_count,
                "imu": self.imu_count,
                "diagnostic": self.diagnostic_count,
            },
            "diagnostics": self.diagnostics,
        }


class ArxStateReader:
    def __init__(self, ip="127.0.0.1", port=8765, timeout_ms=200):
        import zmq

        self.zmq = zmq
        self.ip = ip
        self.port = int(port)
        self.timeout_ms = int(timeout_ms)
        self.context = zmq.Context()
        self.socket = None
        self._connect()

    def _connect(self):
        if self.socket is not None:
            self.socket.close(linger=0)
        self.socket = self.context.socket(self.zmq.REQ)
        self.socket.setsockopt(self.zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(self.zmq.SNDTIMEO, self.timeout_ms)
        self.socket.setsockopt(self.zmq.LINGER, 0)
        self.socket.connect(f"tcp://{self.ip}:{self.port}")

    def request(self, cmd, data=None, timeout_ms=None):
        old_rcvtimeo = self.socket.getsockopt(self.zmq.RCVTIMEO)
        old_sndtimeo = self.socket.getsockopt(self.zmq.SNDTIMEO)
        if timeout_ms is not None:
            self.socket.setsockopt(self.zmq.RCVTIMEO, int(timeout_ms))
            self.socket.setsockopt(self.zmq.SNDTIMEO, int(timeout_ms))
        try:
            self.socket.send_pyobj({"cmd": cmd, "data": data})
            reply = self.socket.recv_pyobj()
            if not isinstance(reply, dict) or reply.get("cmd") != cmd:
                raise RuntimeError(f"Unexpected ARX reply: {reply}")
            if isinstance(reply.get("data"), str) and reply["data"].lower().startswith("error"):
                raise RuntimeError(reply["data"])
            return reply["data"]
        except Exception:
            self._connect()
            raise
        finally:
            if self.socket is not None and timeout_ms is not None:
                self.socket.setsockopt(self.zmq.RCVTIMEO, old_rcvtimeo)
                self.socket.setsockopt(self.zmq.SNDTIMEO, old_sndtimeo)

    def read(self):
        data = self.request("GET_STATE", None)
        return {
            "stamp": data.get("timestamp", 0),
            "ok": True,
            "q": _fit(data.get("joint_pos", []), ARM_JOINT_NUM),
            "dq": _fit(data.get("joint_vel", []), ARM_JOINT_NUM),
            "tau": _fit(data.get("joint_torque", []), ARM_JOINT_NUM),
            "ee_pose": _fit(data.get("ee_pose", []), 6),
            "gripper_pos": data.get("gripper_pos", None),
        }


class StateReader:
    def __init__(self, robot_ip, arx_ip="127.0.0.1", arx_port=8765, arx_timeout_ms=200, enable_arx=True):
        self.tron = TronStateReader(robot_ip)
        self.arx = None
        if enable_arx:
            self.arx = ArxStateReader(arx_ip, arx_port, arx_timeout_ms)

    def wait(self, timeout_s=3.0):
        return self.tron.wait(timeout_s)

    def read(self):
        arm = None
        if self.arx is not None:
            try:
                arm = self.arx.read()
            except Exception as exc:
                arm = {
                    "stamp": 0,
                    "ok": False,
                    "error": str(exc),
                    "q": np.zeros((ARM_JOINT_NUM,), dtype=np.float64),
                    "dq": np.zeros((ARM_JOINT_NUM,), dtype=np.float64),
                    "tau": np.zeros((ARM_JOINT_NUM,), dtype=np.float64),
                    "ee_pose": np.zeros((6,), dtype=np.float64),
                    "gripper_pos": None,
                }
        return {
            "time": time.time(),
            "tron": self.tron.read(),
            "arx": arm,
        }


def _fit(values, size):
    src = np.asarray(values, dtype=np.float64).reshape(-1)
    out = np.zeros((size,), dtype=np.float64)
    out[: min(size, src.size)] = src[:size]
    return out


def _fmt(values):
    return np.array2string(np.asarray(values), precision=4, suppress_small=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-ip", default=os.getenv("TRON1_IP", "10.192.1.2"))
    parser.add_argument("--arx-ip", default=os.getenv("ARX5_ZMQ_IP", "127.0.0.1"))
    parser.add_argument("--arx-port", type=int, default=int(os.getenv("ARX5_ZMQ_PORT", "8765")))
    parser.add_argument("--no-arx", action="store_true")
    parser.add_argument("--rate", type=float, default=2.0)
    parser.add_argument("--timeout-ms", type=int, default=200)
    parser.add_argument("--wait-s", type=float, default=3.0)
    args = parser.parse_args()

    reader = StateReader(
        robot_ip=args.robot_ip,
        arx_ip=args.arx_ip,
        arx_port=args.arx_port,
        arx_timeout_ms=args.timeout_ms,
        enable_arx=not args.no_arx,
    )
    print(f"TRON connected robot_ip={args.robot_ip} motor_number={reader.tron.motor_number}")
    if not reader.wait(args.wait_s):
        print(f"WARNING: no TRON state/imu callbacks after {args.wait_s:.1f}s")

    rate = Rate(max(args.rate, 0.1))
    while True:
        state = reader.read()
        tron = state["tron"]
        arm = state["arx"]

        print("=" * 80)
        print(f"counts={tron['counts']} stamp={tron['stamp']}")
        print(f"tron_q={_fmt(tron['q'])}")
        print(f"tron_dq={_fmt(tron['dq'])}")
        print(f"imu_quat={_fmt(tron['imu']['quat'])}")
        if isinstance(arm, dict) and not arm.get("ok", True):
            print(f"arx_error={arm['error']}")
        elif arm is not None:
            print(f"arx_q={_fmt(arm['q'])}")
            print(f"arx_ee_pose={_fmt(arm['ee_pose'])}")
        rate.sleep()


if __name__ == "__main__":
    main()
