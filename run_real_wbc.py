import argparse
import os
import re
import time
from argparse import BooleanOptionalAction

import numpy as np
import yaml

import limxsdk.datatypes as datatypes

from run_wbc_policy import LEG_NAMES, WbcObservationBuilder, WbcOnnxPolicy, fmt
from read_lidar_odom import Ros2OdomReader
from read_tron_arx_state import StateReader


ARM_ACTION_INDICES = [0, 3, 6, 9, 12, 13]
LEG_ACTION_INDICES = [1, 4, 7, 10, 2, 5, 8, 11]
ARM_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6"]


def parse_last_arx_command(error):
    match = re.search(r"last command:\s*\[([^\]]+)\]", str(error), flags=re.MULTILINE)
    if match is None:
        return None
    values = np.fromstring(match.group(1).replace("\n", " "), sep=" ")
    if values.size != 6:
        return None
    return values.astype(np.float64)


def fmt_named(names, values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return " ".join(f"{name}={values[i]: .4f}" for i, name in enumerate(names[: values.size]))


def deployment_state_safe(builder, use_ros2_odom, max_base_norm=5.0, max_ee_error_norm=5.0, max_se3=20.0):
    use_world_target = bool(builder.cfg.get("ee_target", {}).get("use_world_frame", False))
    if use_ros2_odom and use_world_target and not getattr(builder, "ee_target_from_world", False):
        return False, "ros2 odom unavailable; world target cannot be converted to base frame"
    base_world = getattr(builder, "base_world", None)
    if use_ros2_odom and base_world is not None:
        base_norm = float(np.linalg.norm(np.asarray(base_world, dtype=np.float64)[:3]))
        if not np.isfinite(base_norm) or base_norm > max_base_norm:
            return False, f"base world position norm abnormal: {base_norm:.3f} > {max_base_norm:.3f}"
    ee_error = np.asarray(getattr(builder, "ee_error", np.zeros(6)), dtype=np.float64)
    ee_error_norm = float(np.linalg.norm(ee_error[:3]))
    if not np.all(np.isfinite(ee_error)) or ee_error_norm > max_ee_error_norm:
        return False, f"EE position error abnormal: {ee_error_norm:.3f} > {max_ee_error_norm:.3f}"
    se3 = float(getattr(builder, "ee_se3", 0.0))
    if not np.isfinite(se3) or se3 > max_se3:
        return False, f"EE se3 abnormal: {se3:.3f} > {max_se3:.3f}"
    return True, ""


def debug_target_sweep(builder, policy, state, base_target, z_offsets):
    saved_target = builder.ee_target.copy()
    saved_source = builder.ee_target_source
    saved_stamp = builder.ee_target_stamp
    saved_last_actions = builder.last_actions.copy()
    saved_ee_se3 = builder.ee_se3
    saved_ee_se3_actual = getattr(builder, "ee_se3_actual", builder.ee_se3)
    saved_ee_se3_last_update = getattr(builder, "ee_se3_last_update", None)
    saved_ee_last_target_for_se3 = (
        None
        if getattr(builder, "ee_last_target_for_se3", None) is None
        else builder.ee_last_target_for_se3.copy()
    )
    saved_history = policy.history.copy()
    saved_hidden = policy.hidden.copy()
    saved_history_ready = policy.history_ready

    rows = []
    try:
        for z_offset in z_offsets:
            builder.ee_target = np.asarray(base_target, dtype=np.float64).copy()
            builder.ee_target[2] += float(z_offset)
            builder.ee_target_source = f"debug_sweep dz={z_offset:+.2f}"
            policy.history[:] = saved_history
            policy.hidden[:] = saved_hidden
            policy.history_ready = saved_history_ready
            builder.last_actions[:] = saved_last_actions
            obs, contact_obs, _, _, _ = builder.build(state)
            action = policy.infer(obs, contact_obs)
            arm_action = np.asarray(action, dtype=np.float64).reshape(-1)[ARM_ACTION_INDICES]
            rows.append((z_offset, builder.ee_target_obs.copy(), builder.ee_current.copy(), builder.ee_error.copy(), arm_action))
    finally:
        builder.ee_target = saved_target
        builder.ee_target_source = saved_source
        builder.ee_target_stamp = saved_stamp
        builder.last_actions[:] = saved_last_actions
        builder.ee_se3 = saved_ee_se3
        builder.ee_se3_actual = saved_ee_se3_actual
        builder.ee_se3_last_update = saved_ee_se3_last_update
        builder.ee_last_target_for_se3 = saved_ee_last_target_for_se3
        policy.history[:] = saved_history
        policy.hidden[:] = saved_hidden
        policy.history_ready = saved_history_ready
        builder.build(state)

    return rows


class WbcRealOutput:
    def __init__(
        self,
        robot,
        cfg,
        enable_legs=True,
        enable_arm=True,
        arm_action_scale=1.0,
        arm_max_joint_step=0.02,
        arx_max_joint_delta=0.2,
    ):
        self.robot = robot
        self.cfg = cfg
        self.control = cfg["control"]
        self.default_angles = cfg["init_state"]["default_joint_angle"]
        self.enable_legs = enable_legs
        self.enable_arm = enable_arm
        self.cmd_mode = int(os.getenv("LIMX_CMD_MODE", "0"))
        self.robot_cmd = datatypes.RobotCmd()
        self.robot_cmd.mode = [self.cmd_mode] * len(LEG_NAMES)
        self.robot_cmd.q = [0.0] * len(LEG_NAMES)
        self.robot_cmd.dq = [0.0] * len(LEG_NAMES)
        self.robot_cmd.tau = [0.0] * len(LEG_NAMES)
        self.robot_cmd.Kp = [0.0] * len(LEG_NAMES)
        self.robot_cmd.Kd = [1.0] * len(LEG_NAMES)
        self.arm_joint_target = None
        self.arm_action_scale = float(arm_action_scale)
        self.arm_max_joint_step = float(arm_max_joint_step)
        self.arx_max_joint_delta = float(arx_max_joint_delta)

    def reset_arm_home(self, reader, timeout_ms):
        if reader.arx is None:
            return None
        reader.arx.request("RESET_TO_HOME", None, timeout_ms=timeout_ms)
        state = reader.arx.read()
        if isinstance(state, dict) and state.get("ok", False):
            self.arm_joint_target = np.asarray(state["q"], dtype=np.float64).reshape(-1)[:6]
        return self.arm_joint_target

    def reset_arm_policy_init(self, reader, duration_s=2.0, rate_hz=50.0, hold_s=1.0):
        if reader.arx is None:
            return None
        state = reader.arx.read()
        if not isinstance(state, dict) or not state.get("ok", False):
            return None
        current_q = np.asarray(state["q"], dtype=np.float64).reshape(-1)[:6]
        init_angles = self.cfg["init_state"].get("init_stand_joint_angle", self.default_angles)
        target_q = np.asarray(
            [float(init_angles.get(name, self.default_angles.get(name, 0.0))) for name in ARM_NAMES],
            dtype=np.float64,
        )
        steps = max(1, int(max(duration_s, 0.0) * max(rate_hz, 1.0)))
        dt = 1.0 / max(rate_hz, 1.0)

        for step in range(1, steps + 1):
            alpha = step / steps
            q_cmd = (1.0 - alpha) * current_q + alpha * target_q
            reader.arx.request(
                "SET_JOINT_POS",
                {
                    "joint_pos": q_cmd.astype(np.float64),
                    "gripper_pos": None,
                    "max_delta": self.arx_max_joint_delta,
                },
            )
            time.sleep(dt)
        hold_steps = max(1, int(max(hold_s, 0.0) * max(rate_hz, 1.0)))
        for _ in range(hold_steps):
            reader.arx.request(
                "SET_JOINT_POS",
                {
                    "joint_pos": target_q.astype(np.float64),
                    "gripper_pos": None,
                    "max_delta": self.arx_max_joint_delta,
                },
            )
            time.sleep(dt)
        final_state = reader.arx.read()
        final_q = np.asarray(final_state["q"], dtype=np.float64).reshape(-1)[:6]
        error_q = final_q - target_q
        try:
            sync_reply = reader.arx.request("SYNC_LAST_COMMAND", None)
            if isinstance(sync_reply, dict) and "joint_pos" in sync_reply:
                self.arm_joint_target = np.asarray(sync_reply["joint_pos"], dtype=np.float64).reshape(-1)[:6]
            else:
                self.arm_joint_target = final_q.copy()
        except Exception:
            self.arm_joint_target = final_q.copy()
        return {"target": target_q, "actual": final_q, "error": error_q}

    def reset_legs_home(self, reader, duration_s=2.0, rate_hz=50.0):
        state = reader.read()
        current_q = np.asarray(state["tron"]["q"], dtype=np.float64).reshape(-1)[: len(LEG_NAMES)]
        init_angles = self.cfg["init_state"].get("init_stand_joint_angle", self.default_angles)
        target_q = np.asarray([float(init_angles.get(name, self.default_angles.get(name, 0.0))) for name in LEG_NAMES])
        steps = max(1, int(max(duration_s, 0.0) * max(rate_hz, 1.0)))
        dt = 1.0 / max(rate_hz, 1.0)

        for step in range(1, steps + 1):
            alpha = step / steps
            q_cmd = (1.0 - alpha) * current_q + alpha * target_q
            for i, name in enumerate(LEG_NAMES):
                kp, kd, _ = self.joint_gain(name)
                self.robot_cmd.q[i] = float(q_cmd[i])
                self.robot_cmd.dq[i] = 0.0
                self.robot_cmd.tau[i] = 0.0
                self.robot_cmd.Kp[i] = kp
                self.robot_cmd.Kd[i] = kd
            self.robot_cmd.stamp = time.time_ns()
            self.robot.publishRobotCmd(self.robot_cmd)
            time.sleep(dt)
        return target_q

    def joint_gain(self, name):
        if "ankle" in name:
            return (
                float(self.control["ankle_joint_stiffness"]),
                float(self.control["ankle_joint_damping"]),
                float(self.control["ankle_joint_torque_limit"]),
            )
        return (
            float(self.control["leg_joint_stiffness"]),
            float(self.control["leg_joint_damping"]),
            float(self.control["leg_joint_torque_limit"]),
        )

    def apply_leg_actions(self, state, actions, max_step):
        leg_q = np.asarray(state["tron"]["q"], dtype=np.float64).reshape(-1)[: len(LEG_NAMES)]
        leg_dq = np.asarray(state["tron"]["dq"], dtype=np.float64).reshape(-1)[: len(LEG_NAMES)]
        scale = float(self.control["action_scale_pos"])
        leg_actions = np.asarray(actions, dtype=np.float64).reshape(-1)[LEG_ACTION_INDICES]

        for i, name in enumerate(LEG_NAMES):
            kp, kd, torque_limit = self.joint_gain(name)
            init = float(self.default_angles.get(name, 0.0))
            action_min = leg_q[i] - init + (kd * leg_dq[i] - torque_limit) / kp
            action_max = leg_q[i] - init + (kd * leg_dq[i] + torque_limit) / kp
            action = np.clip(float(leg_actions[i]), action_min / scale, action_max / scale)
            target = init + action * scale
            if max_step > 0.0:
                target = np.clip(target, leg_q[i] - max_step, leg_q[i] + max_step)

            self.robot_cmd.q[i] = float(target)
            self.robot_cmd.dq[i] = 0.0
            self.robot_cmd.tau[i] = 0.0
            self.robot_cmd.Kp[i] = kp
            self.robot_cmd.Kd[i] = kd

    def apply_arm_actions(self, reader, state, actions):
        if reader.arx is None:
            return None
        arm = state.get("arx")
        if not isinstance(arm, dict) or not arm.get("ok", False):
            return None

        arm_q = np.asarray(arm["q"], dtype=np.float64).reshape(-1)[:6].copy()
        if self.arm_joint_target is None:
            self.arm_joint_target = arm_q.copy()

        arm_default = np.asarray(
            [float(self.default_angles.get(name, 0.0)) for name in ARM_NAMES],
            dtype=np.float64,
        )
        arm_actions = np.asarray(actions, dtype=np.float64).reshape(-1)[ARM_ACTION_INDICES]
        desired = arm_default + arm_actions * self.arm_action_scale
        target = desired.copy()
        if self.arm_max_joint_step > 0.0:
            target = np.clip(target, arm_q - self.arm_max_joint_step, arm_q + self.arm_max_joint_step)

        try:
            data = reader.arx.request(
                "SET_JOINT_POS",
                {
                    "joint_pos": target.astype(np.float64),
                    "gripper_pos": None,
                    "max_delta": self.arx_max_joint_delta,
                },
            )
            if isinstance(data, dict) and "joint_pos" in data:
                self.arm_joint_target = np.asarray(data["joint_pos"], dtype=np.float64).reshape(-1)[:6]
            else:
                self.arm_joint_target = target.copy()
            return {
                "ok": True,
                "action": arm_actions.copy(),
                "desired": desired.copy(),
                "cmd": target.copy(),
                "target": self.arm_joint_target.copy(),
            }
        except Exception as exc:
            return {
                "ok": False,
                "error": str(exc),
                "action": arm_actions.copy(),
                "desired": desired.copy(),
                "cmd": target.copy(),
                "target": self.arm_joint_target.copy(),
            }

    def publish(self, reader, state, actions, max_leg_step):
        if self.enable_legs:
            self.apply_leg_actions(state, actions, max_leg_step)
            self.robot_cmd.stamp = time.time_ns()
            self.robot.publishRobotCmd(self.robot_cmd)
        arm_target = None
        if self.enable_arm:
            arm_target = self.apply_arm_actions(reader, state, actions)
        return arm_target

    def stop_legs(self):
        self.robot_cmd.Kp = [0.0] * len(LEG_NAMES)
        self.robot_cmd.Kd = [1.0] * len(LEG_NAMES)
        self.robot_cmd.tau = [0.0] * len(LEG_NAMES)
        self.robot_cmd.stamp = time.time_ns()
        self.robot.publishRobotCmd(self.robot_cmd)


def load_policy(model_dir, odom_reader=None, ee_command_file=None):
    config_file = os.path.join(model_dir, "params.yaml")
    policy_dir = os.path.join(model_dir, "policy")
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)["PointfootCfg"]
    builder = WbcObservationBuilder(config_file, odom_reader=odom_reader, ee_command_file=ee_command_file)
    policy = WbcOnnxPolicy(
        policy_dir=policy_dir,
        obs_size=cfg["size"]["observations_size"],
        contact_obs_size=55,
        history_len=cfg["size"]["obs_history_length"],
        actions_size=cfg["size"]["actions_size"],
        sample_latent=cfg.get("rfm", {}).get("sample_next_obs_latent", False),
    )
    return cfg, builder, policy


def main():
    parser = argparse.ArgumentParser()
    root = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(root, "controllers", "model", "SF_TRON1A_ARX5ARM")
    parser.add_argument("--robot-ip", default=os.getenv("TRON1_IP", "10.192.1.2"))
    parser.add_argument("--arx-ip", default=os.getenv("ARX5_ZMQ_IP", "127.0.0.1"))
    parser.add_argument("--arx-port", type=int, default=int(os.getenv("ARX5_ZMQ_PORT", "8765")))
    parser.add_argument("--model-dir", default=model_dir)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--timeout-ms", type=int, default=50)
    parser.add_argument("--arm-reset-timeout-ms", type=int, default=10000)
    parser.add_argument("--wait-s", type=float, default=3.0)
    parser.add_argument("--print-period", type=float, default=0.5)
    parser.add_argument("--max-leg-step", type=float, default=None)
    parser.add_argument("--arm-action-scale", type=float, default=None)
    parser.add_argument("--arm-max-pos-step", type=float, default=None)
    parser.add_argument("--arm-max-joint-step", type=float, default=None)
    parser.add_argument("--arm-max-rpy-step", type=float, default=None)
    parser.add_argument("--arm-max-delta-norm", type=float, default=None)
    parser.add_argument("--real-action-clip", type=float, default=None)
    parser.add_argument("--max-safe-base-norm", type=float, default=None)
    parser.add_argument("--max-safe-ee-error", type=float, default=None)
    parser.add_argument("--max-safe-ee-se3", type=float, default=None)
    parser.add_argument("--use-ros2-odom", action=BooleanOptionalAction, default=None)
    parser.add_argument("--odom-topic", default=None)
    parser.add_argument("--ee-command-file", default=None)
    parser.add_argument("--ignore-ee-command-file", action="store_true")
    parser.add_argument("--debug-target-sweep", action="store_true")
    parser.add_argument("--enable-output", action="store_true")
    parser.add_argument("--no-legs", action="store_true")
    parser.add_argument("--no-arm", action="store_true")
    args = parser.parse_args()

    config_file = os.path.join(args.model_dir, "params.yaml")
    with open(config_file, "r") as f:
        cfg_preview = yaml.safe_load(f)["PointfootCfg"]
    deploy_cfg = cfg_preview.get("deploy", {})
    use_ros2_odom = bool(
        args.use_ros2_odom if args.use_ros2_odom is not None else deploy_cfg.get("use_ros2_odom", False)
    )
    odom_topic = args.odom_topic if args.odom_topic is not None else deploy_cfg.get("odom_topic", "/Odometry")
    odom_reader = None
    if use_ros2_odom:
        odom_reader = Ros2OdomReader(
            topic=odom_topic,
            position_offset=deploy_cfg.get("odom_position_offset", [-0.14, 0.0, 0.8277]),
        )
        odom_reader.start()
        odom_reader.wait(args.wait_s)

    ee_command_file = "" if args.ignore_ee_command_file else args.ee_command_file
    cfg, builder, policy = load_policy(args.model_dir, odom_reader=odom_reader, ee_command_file=ee_command_file)
    deploy_cfg = cfg.get("deploy", {})
    max_leg_step = float(args.max_leg_step if args.max_leg_step is not None else deploy_cfg.get("max_leg_step", 0.02))
    arm_action_scale = float(
        args.arm_action_scale
        if args.arm_action_scale is not None
        else os.getenv("ARX5_ARM_ACTION_SCALE", deploy_cfg.get("arm_action_scale", 0.005))
    )
    arm_max_joint_step = float(
        args.arm_max_joint_step
        if args.arm_max_joint_step is not None
        else os.getenv("ARX5_ARM_MAX_JOINT_STEP", deploy_cfg.get("arm_max_joint_step", 0.02))
    )
    arx_max_joint_delta = float(os.getenv("ARX5_MAX_JOINT_DELTA", deploy_cfg.get("arx_max_joint_delta", 0.2)))
    real_action_clip = float(args.real_action_clip if args.real_action_clip is not None else deploy_cfg.get("real_action_clip", 3.0))
    max_safe_base_norm = float(args.max_safe_base_norm if args.max_safe_base_norm is not None else deploy_cfg.get("max_safe_base_norm", 5.0))
    max_safe_ee_error = float(args.max_safe_ee_error if args.max_safe_ee_error is not None else deploy_cfg.get("max_safe_ee_error", 5.0))
    max_safe_ee_se3 = float(args.max_safe_ee_se3 if args.max_safe_ee_se3 is not None else deploy_cfg.get("max_safe_ee_se3", 20.0))
    leg_reset_duration = float(deploy_cfg.get("leg_reset_duration", 2.0))
    arm_reset_duration = float(deploy_cfg.get("arm_reset_duration", leg_reset_duration))
    arm_init_hold = float(deploy_cfg.get("arm_init_hold", 1.0))
    reader = StateReader(
        robot_ip=args.robot_ip,
        arx_ip=args.arx_ip,
        arx_port=args.arx_port,
        arx_timeout_ms=args.timeout_ms,
        enable_arx=not args.no_arm,
    )
    if not reader.wait(args.wait_s):
        raise RuntimeError(f"No TRON state/imu callbacks after {args.wait_s:.1f}s")

    output = WbcRealOutput(
        robot=reader.tron.robot,
        cfg=cfg,
        enable_legs=not args.no_legs,
        enable_arm=not args.no_arm,
        arm_action_scale=arm_action_scale,
        arm_max_joint_step=arm_max_joint_step,
        arx_max_joint_delta=arx_max_joint_delta,
    )

    print(f"real_wbc ready output={args.enable_output} rate={args.rate}Hz duration={args.duration}s")
    print(f"action_order={builder.joint_names}")
    print(f"use_ros2_odom={use_ros2_odom} odom_topic={odom_topic}")
    print(f"legs={not args.no_legs} arm={not args.no_arm} max_leg_step={max_leg_step}")
    print(f"arm_action_indices={dict(zip(ARM_NAMES, ARM_ACTION_INDICES))}")
    print(
        f"arm_action_scale={arm_action_scale} "
        f"arm_max_joint_step={arm_max_joint_step} "
        f"arx_max_joint_delta={arx_max_joint_delta} "
        f"arm_reset_duration={arm_reset_duration} arm_init_hold={arm_init_hold} "
        f"leg_reset_duration={leg_reset_duration}"
    )
    print(
        f"real_action_clip={real_action_clip} "
        f"max_safe_base_norm={max_safe_base_norm} "
        f"max_safe_ee_error={max_safe_ee_error} "
        f"max_safe_ee_se3={max_safe_ee_se3}"
    )
    if max_leg_step > 0.03:
        print(f"WARNING: max_leg_step={max_leg_step} is large for real hardware; 0.01-0.02 is recommended.")
    if args.arm_max_pos_step is not None or args.arm_max_rpy_step is not None or args.arm_max_delta_norm is not None:
        print("WARNING: --arm-max-pos-step/--arm-max-rpy-step/--arm-max-delta-norm are ignored in joint-action mode.")
    if args.enable_output and not args.no_arm:
        arm_home = output.reset_arm_home(reader, args.arm_reset_timeout_ms)
        print(f"arm_reset_home={fmt(arm_home) if arm_home is not None else None}")
        arm_policy_init = output.reset_arm_policy_init(
            reader,
            duration_s=arm_reset_duration,
            rate_hz=args.rate,
            hold_s=arm_init_hold,
        )
        if isinstance(arm_policy_init, dict):
            print(f"arm_policy_init_target={fmt_named(ARM_NAMES, arm_policy_init['target'])}")
            print(f"arm_policy_init_actual={fmt_named(ARM_NAMES, arm_policy_init['actual'])}")
            print(f"arm_policy_init_error={fmt_named(ARM_NAMES, arm_policy_init['error'])}")
        else:
            print(f"arm_policy_init={arm_policy_init}")
    if args.enable_output and not args.no_legs:
        leg_home = output.reset_legs_home(reader, duration_s=leg_reset_duration, rate_hz=args.rate)
        print(f"leg_reset_home={fmt(leg_home)}")
    if odom_reader is not None:
        if odom_reader.reanchor():
            print("Re-anchored odom frame after reset, before WBC loop")
        else:
            print("WARNING: failed to re-anchor odom frame before WBC loop")
    dt = 1.0 / max(args.rate, 1.0)
    start = time.monotonic()
    next_print = 0.0

    try:
        while True:
            state = reader.read()
            obs, contact_obs, q, dq, tau = builder.build(state)
            actions = policy.infer(obs, contact_obs)
            clip = float(cfg["normalization"]["clip_scales"]["clip_actions"])
            clip = min(clip, real_action_clip)
            actions = np.clip(actions, -clip, clip)
            builder.update_last_actions(actions)

            arm_target = None
            safe_to_output, unsafe_reason = deployment_state_safe(
                builder,
                use_ros2_odom=use_ros2_odom,
                max_base_norm=max_safe_base_norm,
                max_ee_error_norm=max_safe_ee_error,
                max_se3=max_safe_ee_se3,
            )
            if args.enable_output:
                if safe_to_output:
                    arm_target = output.publish(reader, state, actions, max_leg_step)
                else:
                    output.stop_legs()

            now = time.monotonic()
            if now >= next_print:
                next_print = now + max(args.print_period, dt)
                arx_ok = isinstance(state.get("arx"), dict) and state["arx"].get("ok", False)
                print("=" * 80)
                print(f"elapsed={now - start:.2f}s output={args.enable_output} arx_ok={arx_ok}")
                if not safe_to_output:
                    print(f"OUTPUT BLOCKED: {unsafe_reason}")
                print("EE task:")
                print(f"  target(raw):       {fmt(builder.ee_target)}")
                print(f"  source:     {builder.ee_target_source}")
                print(f"  target_obs(base):  {fmt(builder.ee_target_obs)}")
                print(f"  ee_pose_obs(base): {fmt(builder.ee_current)}")
                if getattr(builder, "ee_current_world_pose", None) is not None:
                    print(f"  ee_pose_fk(world debug): {fmt(builder.ee_current_world_pose)}")
                if getattr(builder, "base_world", None) is not None:
                    print(f"  base(world): {fmt(builder.base_world)}")
                    print(f"  ee-base dW:  {fmt(builder.ee_world_delta)}")
                print(f"  error:      {fmt(builder.ee_error)}")
                print(f"  se3_ref:    {builder.ee_se3:.4f}")
                print(f"  se3_actual: {getattr(builder, 'ee_se3_actual', builder.ee_se3):.4f}")
                if args.debug_target_sweep:
                    print("EE target sweep debug:")
                    for dz, target_obs, current_obs, error_obs, arm_action in debug_target_sweep(
                        builder,
                        policy,
                        state,
                        builder.ee_target,
                        [-0.3, 0.0, 0.3],
                    ):
                        print(
                            f"  dz={dz:+.2f} "
                            f"target_z={target_obs[2]: .4f} current_z={current_obs[2]: .4f} "
                            f"err_z={error_obs[2]: .4f} "
                            f"arm_action={fmt_named(ARM_NAMES, arm_action)}"
                        )
                arm_state = state.get("arx")
                if isinstance(arm_state, dict) and arm_state.get("ok", False):
                    print("ARM state:")
                    print(f"  q_actual:   {fmt_named(ARM_NAMES, arm_state['q'])}")
                    print(f"  ee_pose_fk(base): {fmt(arm_state['ee_pose'])}")
                if isinstance(arm_target, dict):
                    print("ARM action:")
                    print(f"  policy:     {fmt_named(ARM_NAMES, arm_target['action'])}")
                    print(f"  q_des:      {fmt_named(ARM_NAMES, arm_target['desired'])}")
                    print(f"  q_cmd:      {fmt_named(ARM_NAMES, arm_target['cmd'])}")
                    if arm_target.get("ok", False):
                        print(f"  q_return:   {fmt_named(ARM_NAMES, arm_target['target'])}")
                    else:
                        print(f"  error:      {arm_target['error']}")
                        print(f"  q_hold:     {fmt_named(ARM_NAMES, arm_target['target'])}")
                print("TRON legs:")
                print(f"  q:          {fmt_named(LEG_NAMES, state['tron']['q'])}")
                print(f"  action:     {fmt_named(LEG_NAMES, np.asarray(actions)[LEG_ACTION_INDICES])}")

            if args.duration > 0.0 and now - start >= args.duration:
                break
            time.sleep(dt)
    finally:
        if args.enable_output and not args.no_legs:
            output.stop_legs()
        if odom_reader is not None:
            odom_reader.stop()


if __name__ == "__main__":
    main()
