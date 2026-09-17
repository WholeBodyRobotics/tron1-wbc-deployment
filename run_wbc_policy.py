import argparse
import json
import os
import time

import numpy as np
import onnxruntime as ort
import yaml
from scipy.spatial.transform import Rotation as R

from read_tron_arx_state import StateReader
from read_lidar_odom import Ros2OdomReader, transform_from_pose6d


LEG_NAMES = [
    "abad_L_Joint",
    "hip_L_Joint",
    "knee_L_Joint",
    "ankle_L_Joint",
    "abad_R_Joint",
    "hip_R_Joint",
    "knee_R_Joint",
    "ankle_R_Joint",
]

ARM_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6"]


def last_positive_dim(shape):
    for dim in reversed(shape):
        if isinstance(dim, int) and dim > 0:
            return dim
    return 0


def rot6d_from_rpy(rpy):
    rot = R.from_euler("xyz", rpy).as_matrix()
    return np.array(
        [rot[0, 0], rot[1, 0], rot[2, 0], rot[0, 1], rot[1, 1], rot[2, 1]],
        dtype=np.float32,
    )


def rot6d_from_matrix(rot):
    rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    return np.array(
        [rot[0, 0], rot[1, 0], rot[2, 0], rot[0, 1], rot[1, 1], rot[2, 1]],
        dtype=np.float32,
    )


def se3_distance(command_pos, command_rot, ee_pos, ee_rot):
    pos_err = float(np.linalg.norm(np.asarray(command_pos, dtype=np.float64) - np.asarray(ee_pos, dtype=np.float64)))
    rel_rot = np.asarray(command_rot, dtype=np.float64).reshape(3, 3) @ np.asarray(ee_rot, dtype=np.float64).reshape(3, 3).T
    cos_angle = float(np.clip((np.trace(rel_rot) - 1.0) * 0.5, -1.0, 1.0))
    return 2.0 * pos_err + float(np.arccos(cos_angle))


def limx_quat_to_xyzw(quat):
    quat = np.asarray(quat, dtype=np.float64).reshape(-1)
    if quat.size < 4:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    xyzw = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)
    norm = np.linalg.norm(xyzw)
    if norm < 1.0e-6:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return xyzw / norm


class WbcOnnxPolicy:
    def __init__(self, policy_dir, obs_size, contact_obs_size, history_len, actions_size, sample_latent=False):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]

        self.actor = ort.InferenceSession(os.path.join(policy_dir, "actor.onnx"), sess_options=opts, providers=providers)
        self.contact = ort.InferenceSession(os.path.join(policy_dir, "contactNet.onnx"), sess_options=opts, providers=providers)
        self.gru = ort.InferenceSession(os.path.join(policy_dir, "gru.onnx"), sess_options=opts, providers=providers)

        self.actor_inputs = [x.name for x in self.actor.get_inputs()]
        self.actor_outputs = [x.name for x in self.actor.get_outputs()]
        self.contact_inputs = [x.name for x in self.contact.get_inputs()]
        self.contact_outputs = [x.name for x in self.contact.get_outputs()]
        self.gru_inputs = [x.name for x in self.gru.get_inputs()]
        self.gru_outputs = [x.name for x in self.gru.get_outputs()]

        self.obs_size = last_positive_dim(self.actor.get_inputs()[0].shape) or obs_size
        self.contact_obs_size = last_positive_dim(self.contact.get_inputs()[0].shape) or contact_obs_size
        self.history_len = history_len
        self.actions_size = actions_size
        self.sample_latent = sample_latent
        self.gru_latent_size = max(last_positive_dim(x.shape) for x in self.gru.get_outputs()) or 128
        self.next_obs_latent_size = (self.gru_latent_size - 3) // 2
        self.actor_latent_size = 3 + self.next_obs_latent_size
        self.history = np.zeros((history_len, self.contact_obs_size), dtype=np.float32)
        self.hidden = np.zeros((1, 1, self.gru_latent_size), dtype=np.float32)
        self.history_ready = False

    @staticmethod
    def fit(x, dim):
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == dim:
            return x
        out = np.zeros((dim,), dtype=np.float32)
        out[: min(x.size, dim)] = x[:dim]
        return out

    def infer(self, obs, contact_obs):
        obs = self.fit(obs, self.obs_size)
        contact_obs = self.fit(contact_obs, self.contact_obs_size)
        if not self.history_ready:
            self.history[:] = contact_obs[None, :]
            self.hidden[:] = 0.0
            self.history_ready = True
        else:
            self.history[:-1] = self.history[1:]
            self.history[-1] = contact_obs

        contact_latent = self.contact.run(
            self.contact_outputs,
            {self.contact_inputs[0]: self.history.reshape(1, self.history_len, self.contact_obs_size)},
        )[0]
        contact_latent = np.asarray(contact_latent, dtype=np.float32).reshape(-1)[-self.gru_latent_size :]

        gru_values = self.gru.run(
            self.gru_outputs,
            {self.gru_inputs[0]: contact_latent.reshape(1, -1), self.gru_inputs[1]: self.hidden},
        )
        out0 = np.asarray(gru_values[0], dtype=np.float32)
        out1 = np.asarray(gru_values[1], dtype=np.float32)
        if out0.ndim == 2 and out1.ndim == 3:
            gru_latent = out0.reshape(-1)
            self.hidden = out1.reshape(1, 1, self.gru_latent_size)
        else:
            gru_latent = out1.reshape(-1)
            self.hidden = out0.reshape(1, 1, self.gru_latent_size)

        actor_latent = np.zeros((self.actor_latent_size,), dtype=np.float32)
        actor_latent[:3] = gru_latent[:3]
        mu = gru_latent[3 : 3 + self.next_obs_latent_size]
        logvar = gru_latent[3 + self.next_obs_latent_size : 3 + 2 * self.next_obs_latent_size]
        if self.sample_latent:
            actor_latent[3:] = mu + np.sqrt(np.exp(logvar) + 1.0e-4) * np.random.standard_normal(mu.shape).astype(np.float32)
        else:
            actor_latent[3:] = mu

        action = self.actor.run(
            self.actor_outputs,
            {self.actor_inputs[0]: obs.reshape(1, -1), self.actor_inputs[1]: actor_latent.reshape(1, -1)},
        )[0]
        return np.asarray(action, dtype=np.float32).reshape(-1)[: self.actions_size]


class WbcObservationBuilder:
    def __init__(self, config_file, odom_reader=None, ee_command_file=None):
        with open(config_file, "r") as f:
            cfg = yaml.safe_load(f)["PointfootCfg"]
        self.cfg = cfg
        self.odom_reader = odom_reader
        self.ee_command_file = (
            cfg.get("ee_target", {}).get("command_file", "/tmp/ee_command.json")
            if ee_command_file is None
            else ee_command_file
        )
        self.joint_names = cfg["init_state"]["joint_names"]
        self.default_angles = cfg["init_state"]["default_joint_angle"]
        self.init_joint_angles = np.array([self.default_angles.get(name, 0.0) for name in self.joint_names], dtype=np.float64)
        self.imu_orientation_offset = np.array(list(cfg["imu_orientation_offset"].values()), dtype=np.float64)
        self.ee_target = np.asarray(
            cfg.get("ee_target", {}).get("manual_set_pos", [0.145308, 0.0, 0.140205])
            + cfg.get("ee_target", {}).get("manual_set_rpy", [0.0, 0.5, 0.0]),
            dtype=np.float64,
        )
        self.ee_target_stamp = 0.0
        self.ee_target_source = "params"
        self.ee_se3 = 0.0
        self.ee_se3_actual = 0.0
        self.ee_se3_decrease_vel = float(cfg.get("ee_target", {}).get("se3_decrease_vel", 0.95))
        self.ee_se3_last_update = time.monotonic()
        self.ee_last_target_for_se3 = None
        self.ee_current = np.zeros((6,), dtype=np.float64)
        self.ee_target_obs = self.ee_target.copy()
        self.ee_error = np.zeros((6,), dtype=np.float64)
        self.ee_current_world = False
        self.base_world = None
        self.ee_world_delta = None
        self.ee_current_world_pose = None
        self.ee_target_from_world = False
        self.last_actions = np.zeros((cfg["size"]["actions_size"],), dtype=np.float32)

    def update_ee_target_from_file(self):
        if not self.ee_command_file or not os.path.exists(self.ee_command_file):
            return
        try:
            stamp = os.path.getmtime(self.ee_command_file)
            if stamp <= self.ee_target_stamp:
                return
            with open(self.ee_command_file, "r") as f:
                data = json.load(f)
            if "pose" in data:
                target = np.asarray(data["pose"], dtype=np.float64).reshape(-1)[:6]
            else:
                target = np.asarray(data.get("position", []) + data.get("rpy", []), dtype=np.float64).reshape(-1)[:6]
            if target.size == 6 and np.all(np.isfinite(target)):
                self.ee_target = target
                self.ee_target_stamp = stamp
                self.ee_target_source = self.ee_command_file
        except Exception:
            return

    def world_debug_from_base_ee(self, ee_pose_base):
        self.base_world = None
        self.ee_world_delta = None
        self.ee_current_world_pose = None
        if self.odom_reader is None:
            return
        odom = self.odom_reader.read()
        if not odom.get("ok", False):
            return
        tf_world_base = odom["tf_world_base"]
        tf_base_ee = transform_from_pose6d(ee_pose_base)
        tf_world_ee = tf_world_base @ tf_base_ee
        self.base_world = np.concatenate([tf_world_base[:3, 3], R.from_matrix(tf_world_base[:3, :3]).as_euler("xyz")])
        self.ee_world_delta = tf_world_ee[:3, 3] - tf_world_base[:3, 3]
        self.ee_current_world_pose = np.concatenate([tf_world_ee[:3, 3], R.from_matrix(tf_world_ee[:3, :3]).as_euler("xyz")])

    def target_pose_for_obs(self):
        use_world = bool(self.cfg.get("ee_target", {}).get("use_world_frame", False))
        self.ee_target_from_world = False
        target_rot = R.from_euler("xyz", self.ee_target[3:6]).as_matrix()
        if not use_world:
            return self.ee_target[:3], target_rot, False

        if self.odom_reader is None:
            return self.ee_target[:3], target_rot, False
        odom = self.odom_reader.read()
        if not odom.get("ok", False):
            return self.ee_target[:3], target_rot, False

        tf_world_base = odom["tf_world_base"]
        tf_world_target = transform_from_pose6d(self.ee_target)
        tf_base_target = np.linalg.inv(tf_world_base) @ tf_world_target
        self.ee_target_from_world = True
        return tf_base_target[:3, 3], tf_base_target[:3, :3], True

    def joint_state(self, state):
        q_map = {name: 0.0 for name in self.joint_names}
        dq_map = {name: 0.0 for name in self.joint_names}
        tau_map = {name: 0.0 for name in self.joint_names}

        tron = state["tron"]
        for i, name in enumerate(LEG_NAMES):
            q_map[name] = float(tron["q"][i])
            dq_map[name] = float(tron["dq"][i])
            tau_map[name] = float(tron["tau"][i])

        arm = state.get("arx")
        if isinstance(arm, dict) and arm.get("ok", False):
            for i, name in enumerate(ARM_NAMES):
                q_map[name] = float(arm["q"][i])
                dq_map[name] = float(arm["dq"][i])
                tau_map[name] = float(arm["tau"][i])
        else:
            for name in ARM_NAMES:
                q_map[name] = float(self.default_angles.get(name, 0.0))

        q = np.array([q_map[name] for name in self.joint_names], dtype=np.float64)
        dq = np.array([dq_map[name] for name in self.joint_names], dtype=np.float64)
        tau = np.array([tau_map[name] for name in self.joint_names], dtype=np.float64)
        return q, dq, tau

    def build(self, state):
        self.update_ee_target_from_file()
        tron = state["tron"]
        imu_quat = limx_quat_to_xyzw(tron["imu"]["quat"])
        rot_wb = R.from_quat(imu_quat).as_matrix()
        projected_gravity = rot_wb.T @ np.array([0.0, 0.0, -1.0])
        base_ang_vel = np.asarray(tron["imu"]["gyro"], dtype=np.float64).reshape(-1)[:3]
        if base_ang_vel.size < 3:
            base_ang_vel = np.pad(base_ang_vel, (0, 3 - base_ang_vel.size))

        offset_rot = R.from_euler("zyx", self.imu_orientation_offset).as_matrix()
        base_ang_vel = offset_rot @ base_ang_vel
        projected_gravity = offset_rot @ projected_gravity

        q, dq, tau = self.joint_state(state)
        pos_rel_no_ankle = []
        for i, name in enumerate(self.joint_names):
            if "ankle" not in name:
                pos_rel_no_ankle.append(q[i] - self.init_joint_angles[i])

        arm = state.get("arx")
        if isinstance(arm, dict) and arm.get("ok", False):
            ee_pose = np.asarray(arm["ee_pose"], dtype=np.float64).reshape(-1)[:6]
        else:
            ee_pose = np.asarray([0.145308, 0.0, 0.140205, 0.0, 0.5, 0.0], dtype=np.float64)

        current_ee_pos = ee_pose[:3].copy()
        current_ee_rot = R.from_euler("xyz", ee_pose[3:6]).as_matrix()
        self.world_debug_from_base_ee(ee_pose)
        current_ee_pos_obs = current_ee_pos
        self.ee_current = np.concatenate([current_ee_pos_obs[:3], R.from_matrix(current_ee_rot).as_euler("xyz")])
        target_pos_obs, target_rot_obs, target_is_from_world = self.target_pose_for_obs()
        self.ee_target_obs = np.concatenate([target_pos_obs[:3], R.from_matrix(target_rot_obs).as_euler("xyz")])
        self.ee_target_from_world = bool(target_is_from_world)
        self.ee_current_world = False
        ee_pose_b = np.concatenate([current_ee_pos_obs[:3], rot6d_from_matrix(current_ee_rot)])
        rel_rot = target_rot_obs @ current_ee_rot.T
        self.ee_error = np.concatenate(
            [
                target_pos_obs[:3] - current_ee_pos_obs[:3],
                R.from_matrix(rel_rot).as_rotvec(),
            ]
        )
        ee_target_b = np.concatenate([target_pos_obs[:3], rot6d_from_matrix(target_rot_obs)])
        self.ee_se3_actual = se3_distance(target_pos_obs[:3], target_rot_obs, current_ee_pos_obs[:3], current_ee_rot)
        target_key = np.concatenate([target_pos_obs[:3], rot6d_from_matrix(target_rot_obs)])
        now = time.monotonic()
        if self.ee_last_target_for_se3 is None or np.max(np.abs(target_key - self.ee_last_target_for_se3)) > 0.1:
            self.ee_se3 = self.ee_se3_actual
            self.ee_last_target_for_se3 = target_key.copy()
        else:
            dt = max(0.0, now - self.ee_se3_last_update)
            self.ee_se3 = max(0.0, float(self.ee_se3) - self.ee_se3_decrease_vel * dt)
        self.ee_se3_last_update = now

        policy_obs = np.concatenate(
            [
                base_ang_vel[:3],
                projected_gravity[:3],
                ee_target_b,
                np.asarray(pos_rel_no_ankle),
                dq,
                self.last_actions,
                ee_pose_b,
                np.array([self.ee_se3]),
            ]
        ).astype(np.float32)
        contact_obs = np.concatenate(
            [
                base_ang_vel[:3],
                projected_gravity[:3],
                np.asarray(pos_rel_no_ankle),
                dq,
                tau,
                ee_pose_b,
            ]
        ).astype(np.float32)
        return policy_obs, contact_obs, q, dq, tau

    def update_last_actions(self, actions):
        self.last_actions[:] = np.asarray(actions, dtype=np.float32).reshape(-1)[: self.last_actions.size]


def fmt(x):
    return np.array2string(np.asarray(x), precision=4, suppress_small=True)


def main():
    parser = argparse.ArgumentParser()
    root = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(root, "controllers", "model", "SF_TRON1A_ARX5ARM")
    parser.add_argument("--robot-ip", default=os.getenv("TRON1_IP", "10.192.1.2"))
    parser.add_argument("--arx-ip", default=os.getenv("ARX5_ZMQ_IP", "127.0.0.1"))
    parser.add_argument("--arx-port", type=int, default=int(os.getenv("ARX5_ZMQ_PORT", "8765")))
    parser.add_argument("--model-dir", default=model_dir)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--timeout-ms", type=int, default=50)
    parser.add_argument("--wait-s", type=float, default=3.0)
    parser.add_argument("--use-ros2-odom", action="store_true")
    parser.add_argument("--odom-topic", default="/Odometry")
    parser.add_argument("--ee-command-file", default=None)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--print-period", type=float, default=0.5)
    args = parser.parse_args()

    config_file = os.path.join(args.model_dir, "params.yaml")
    policy_dir = os.path.join(args.model_dir, "policy")
    with open(config_file, "r") as f:
        cfg_preview = yaml.safe_load(f)["PointfootCfg"]
    deploy_cfg = cfg_preview.get("deploy", {})
    odom_reader = None
    if args.use_ros2_odom:
        odom_reader = Ros2OdomReader(
            topic=args.odom_topic,
        )
        odom_reader.start()
        odom_reader.wait(args.wait_s)

    builder = WbcObservationBuilder(config_file, odom_reader=odom_reader, ee_command_file=args.ee_command_file)
    cfg = builder.cfg
    policy = WbcOnnxPolicy(
        policy_dir=policy_dir,
        obs_size=cfg["size"]["observations_size"],
        contact_obs_size=55,
        history_len=cfg["size"]["obs_history_length"],
        actions_size=cfg["size"]["actions_size"],
        sample_latent=cfg.get("rfm", {}).get("sample_next_obs_latent", False),
    )
    reader = StateReader(
        robot_ip=args.robot_ip,
        arx_ip=args.arx_ip,
        arx_port=args.arx_port,
        arx_timeout_ms=args.timeout_ms,
        enable_arx=True,
    )
    if not reader.wait(args.wait_s):
        print(f"WARNING: no TRON state/imu callbacks after {args.wait_s:.1f}s")

    print(f"WBC policy ready obs={policy.obs_size} contact_obs={policy.contact_obs_size} actions={policy.actions_size}")
    print(f"action_order={builder.joint_names}")
    dt = 1.0 / max(args.rate, 1.0)
    next_print = 0.0

    while True:
        state = reader.read()
        obs, contact_obs, q, dq, tau = builder.build(state)
        actions = policy.infer(obs, contact_obs)
        builder.update_last_actions(actions)

        now = time.monotonic()
        if now >= next_print or args.once:
            next_print = now + max(args.print_period, dt)
            arx_ok = isinstance(state.get("arx"), dict) and state["arx"].get("ok", False)
            print("=" * 80)
            print(f"tron_counts={state['tron']['counts']} arx_ok={arx_ok}")
            print(f"obs_shape={obs.shape} contact_obs_shape={contact_obs.shape}")
            print(f"ee_target={fmt(builder.ee_target)} ee_se3={builder.ee_se3:.4f}")
            print(f"q14={fmt(q)}")
            print(f"action14={fmt(actions)}")

        if args.once:
            break
        time.sleep(dt)

    if odom_reader is not None:
        odom_reader.stop()


if __name__ == "__main__":
    main()
