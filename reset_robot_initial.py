#!/usr/bin/env python3
import argparse
import os
import time

import numpy as np

from deploy_sf_tron1_arm_mujoco import (
    ARM_IDS,
    DEFAULT_Q,
    LEG_IDS,
    RealOutput,
    fmt,
)
from read_tron_arx_state import StateReader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Return the TRON legs and ARX arm to the policy initial state."
    )
    parser.add_argument(
        "--robot-ip",
        default=os.getenv("TRON1_IP", "10.192.1.2"),
    )
    parser.add_argument(
        "--arx-ip",
        default=os.getenv("ARX5_ZMQ_IP", "127.0.0.1"),
    )
    parser.add_argument(
        "--arx-port",
        type=int,
        default=int(os.getenv("ARX5_ZMQ_PORT", "8765")),
    )
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--arm-duration", type=float, default=2.0)
    parser.add_argument("--leg-duration", type=float, default=8.0)
    parser.add_argument("--hold-duration", type=float, default=3.0)
    parser.add_argument("--leg-kp-scale", type=float, default=1.0)
    parser.add_argument("--arx-max-delta", type=float, default=0.2)
    parser.add_argument("--max-arm-error", type=float, default=0.1)
    parser.add_argument("--max-leg-error", type=float, default=0.25)
    return parser.parse_args()


def main():
    args = parse_args()
    rate_hz = max(1.0, float(args.rate))
    dt = 1.0 / rate_hz

    reader = StateReader(
        robot_ip=args.robot_ip,
        arx_ip=args.arx_ip,
        arx_port=args.arx_port,
        arx_timeout_ms=200,
        enable_arx=True,
    )
    if not reader.wait(3.0):
        raise RuntimeError("No TRON state/IMU callbacks after 3.0s")

    deadline = time.monotonic() + 3.0
    state = None
    while time.monotonic() < deadline:
        state = reader.read()
        if isinstance(state.get("arx"), dict) and state["arx"].get("ok", False):
            break
        time.sleep(0.05)
    else:
        error = state.get("arx", {}).get("error", "unavailable")
        raise RuntimeError(f"No valid ARX state after 3.0s: {error}")

    output = RealOutput(
        reader,
        enable_legs=True,
        enable_arm=True,
        arm_max_step=0.01,
        max_leg_step=0.002,
        leg_kp_scale=args.leg_kp_scale,
        arx_max_delta=args.arx_max_delta,
    )

    print(f"arm_initial_target={fmt(DEFAULT_Q[ARM_IDS])}")
    print(f"leg_initial_target={fmt(DEFAULT_Q[LEG_IDS])}")
    print(f"arm_before={fmt(state['arx']['q'])}")
    print(f"leg_before={fmt(state['tron']['q'])}")

    try:
        arm_q = output.reset_arm(
            rate_hz=rate_hz,
            duration=max(0.1, args.arm_duration),
            hold=0.5,
        )
        arm_error = DEFAULT_Q[ARM_IDS] - np.asarray(arm_q, dtype=np.float64)
        print(f"arm_after_reset={fmt(arm_q)}")
        print(f"arm_reset_error={fmt(arm_error)}")

        leg_q = output.reset_legs(
            rate_hz=rate_hz,
            duration=max(0.1, args.leg_duration),
            hold=0.5,
        )
        leg_error = DEFAULT_Q[LEG_IDS] - np.asarray(leg_q, dtype=np.float64)
        print(f"leg_after_reset={fmt(leg_q)}")
        print(f"leg_reset_error={fmt(leg_error)}")

        hold_steps = max(1, int(max(0.0, args.hold_duration) * rate_hz))
        for _ in range(hold_steps):
            output.publish(reader.read(), DEFAULT_Q)
            time.sleep(dt)

        final_state = reader.read()
        final_arm = np.asarray(final_state["arx"]["q"], dtype=np.float64)[:6]
        final_leg = np.asarray(final_state["tron"]["q"], dtype=np.float64)[:8]
        final_arm_error = DEFAULT_Q[ARM_IDS] - final_arm
        final_leg_error = DEFAULT_Q[LEG_IDS] - final_leg
        print(f"arm_final={fmt(final_arm)}")
        print(f"arm_final_error={fmt(final_arm_error)}")
        print(f"leg_final={fmt(final_leg)}")
        print(f"leg_final_error={fmt(final_leg_error)}")

        arm_max_error = float(np.max(np.abs(final_arm_error)))
        leg_max_error = float(np.max(np.abs(final_leg_error)))
        print(
            f"reset_complete arm_max_error={arm_max_error:.4f} "
            f"leg_max_error={leg_max_error:.4f}"
        )
        if arm_max_error > args.max_arm_error:
            raise RuntimeError(
                f"Arm initial-state error {arm_max_error:.4f} > "
                f"{args.max_arm_error:.4f}"
            )
        if leg_max_error > args.max_leg_error:
            raise RuntimeError(
                f"Leg initial-state error {leg_max_error:.4f} > "
                f"{args.max_leg_error:.4f}"
            )
    finally:
        output.stop_legs()


if __name__ == "__main__":
    main()
