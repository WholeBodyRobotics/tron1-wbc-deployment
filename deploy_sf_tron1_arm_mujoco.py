#!/usr/bin/env python3
"""Minimal real-robot deployment matching run_sf_tron1_arm_mujoco.py."""

import argparse
import copy
import importlib
import json
import math
import os
import select
import struct
import sys
import termios
import threading
import time
import tty
import types
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R

import limxsdk.datatypes as datatypes

from read_lidar_odom import Ros2OdomReader, transform_from_pose6d
from read_tron_arx_state import ArxStateReader, StateReader


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = ROOT / "policy/deploy3/exported"
TRAINING_URDF = (
    ROOT / "controllers/model/SF_TRON1A_ARXR5ARM/assembly.urdf"
)

JOINT_NAMES = (
    "abad_L_Joint",
    "abad_R_Joint",
    "hip_L_Joint",
    "hip_R_Joint",
    "knee_L_Joint",
    "knee_R_Joint",
    "J1",
    "ankle_L_Joint",
    "ankle_R_Joint",
    "J2",
    "J3",
    "J4",
    "J5",
    "J6",
)
LEG_NAMES = (
    "abad_L_Joint",
    "hip_L_Joint",
    "knee_L_Joint",
    "ankle_L_Joint",
    "abad_R_Joint",
    "hip_R_Joint",
    "knee_R_Joint",
    "ankle_R_Joint",
)
ARM_NAMES = ("J1", "J2", "J3", "J4", "J5", "J6")
LEG_NAME_TO_I = {name: i for i, name in enumerate(LEG_NAMES)}
ARM_NAME_TO_I = {name: i for i, name in enumerate(ARM_NAMES)}
ARM_IDS = np.array([JOINT_NAMES.index(name) for name in ARM_NAMES], dtype=int)
LEG_IDS = np.array([JOINT_NAMES.index(name) for name in LEG_NAMES], dtype=int)
NO_ANKLE = np.array(["ankle" not in name for name in JOINT_NAMES], dtype=bool)
ACTION_SCALE = np.array(
    [
        0.6
        if name in LEG_NAMES
        else 0.3
        if name in ("J1", "J2", "J3")
        else 0.2
        for name in JOINT_NAMES
    ],
    dtype=np.float64,
)

DEFAULT_Q = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float64,
)
KP = np.array([40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 18.0, 45.0, 45.0, 18.0, 18.0, 4.0, 4.0, 4.0])
KD = np.array([1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.0, 0.8, 0.8, 1.0, 1.0, 0.5, 0.5, 0.5])
TORQUE_LIMIT = np.array([80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 18.0, 40.0, 40.0, 18.0, 18.0, 3.0, 3.0, 3.0])
# WBC EE is the physical J6/link6 origin.  Keep the legacy eef_link name
# coincident with link6; no +X tool offset is applied.
TRAINING_EEF_OFFSET_POS = np.zeros(3, dtype=np.float64)
# L5_umi.urdf fixes SDK eef_link at the DAS_Controller_V3_with_flange
# gripper base_link.  This gripper base_link is also the policy optical-center
# frame.  Both SDK feedback and Diffusion commands therefore use this one
# physical frame; WBC itself commands the arm's J6/link6 origin.
GRIPPER_BASE_LINK_OFFSET_POS = np.array(
    [0.1039414, 0.0000388, 0.0767217], dtype=np.float64
)
GRIPPER_BASE_LINK_OFFSET_RPY = np.array([0.0, 0.2618, 0.0], dtype=np.float64)
SDK_EEF_OFFSET_POS = GRIPPER_BASE_LINK_OFFSET_POS
SDK_EEF_OFFSET_RPY = GRIPPER_BASE_LINK_OFFSET_RPY

POLICY_DT = 0.02
HISTORY = 10
OBS_DIM = 65
CONTACT_DIM = 55
ACTION_DIM = 14


def fmt(x):
    return np.array2string(np.asarray(x), precision=4, suppress_small=True)


def fmt_named(names, values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return " ".join(f"{name}={values[i]: .4f}" for i, name in enumerate(names[: values.size]))


class KeyboardCommandInput:
    """Read XYZ command increments from a terminal without requiring Enter."""

    KEY_DELTAS = {
        "w": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "s": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
        "a": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "d": np.array([0.0, -1.0, 0.0], dtype=np.float64),
        "r": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        "f": np.array([0.0, 0.0, -1.0], dtype=np.float64),
    }

    def __init__(self, step, gripper_enabled=False):
        self.step = float(step)
        self.gripper_enabled = bool(gripper_enabled)
        if not np.isfinite(self.step) or self.step <= 0.0:
            raise ValueError("--keyboard-step must be a finite positive number")
        if not sys.stdin.isatty():
            raise RuntimeError(
                "--keyboard-command requires deployment stdin to be an interactive terminal"
            )
        self.fd = sys.stdin.fileno()
        self.previous_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        print(
            "keyboard_command=enabled "
            f"step={self.step:.3f}m "
            "keys=[W:+X S:-X A:+Y D:-Y R:+Z F:-Z] "
            + (
                "gripper=[T:+open G:-close] "
                if self.gripper_enabled
                else ""
            )
            + "(no Enter required)"
        )

    def poll_delta(self):
        delta = np.zeros(3, dtype=np.float64)
        pressed = []
        while select.select([self.fd], [], [], 0.0)[0]:
            data = os.read(self.fd, 128).decode("utf-8", errors="ignore")
            # Ignore terminal escape sequences (for example arrow keys) so
            # their trailing A/B/C/D bytes cannot become motion commands.
            index = 0
            while index < len(data):
                if data[index] == "\x1b":
                    index += 1
                    if index < len(data) and data[index] == "[":
                        index += 1
                    if index < len(data):
                        index += 1
                    continue
                key = data[index].lower()
                index += 1
                if key in self.KEY_DELTAS:
                    delta += self.KEY_DELTAS[key] * self.step
                    pressed.append(key.upper())
                elif self.gripper_enabled and key in ("t", "g"):
                    pressed.append(key.upper())
        return delta, pressed

    def close(self):
        if self.previous_settings is not None:
            termios.tcsetattr(
                self.fd, termios.TCSADRAIN, self.previous_settings
            )
            self.previous_settings = None


class GenGripperControl:
    """Own the GenRobot DataBus serial connection for keyboard control."""

    def __init__(
        self,
        sdk_root,
        serial_port,
        encoder_frequency,
        initial_width,
        feedback_timeout,
    ):
        self.serial_port = str(serial_port)
        self.encoder_value = None
        self.last_invalid_encoder_value = None
        self.invalid_encoder_count = 0
        self.encoder_lock = threading.Lock()
        self.encoder_ready = threading.Event()
        self.bus = None

        for name, value in (
            ("initial width", initial_width),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 0.103:
                raise ValueError(f"Gen gripper {name} must be in [0, 0.103] m")
        if encoder_frequency <= 0.0:
            raise ValueError("--gripper-encoder-frequency must be positive")

        scripts_dir = Path(sdk_root).expanduser().resolve() / "scripts"
        if not (scripts_dir / "databus.py").is_file():
            raise FileNotFoundError(
                f"Gen gripper SDK databus.py not found in {scripts_dir}"
            )

        # Loading scripts/__init__.py also imports the optional camera stack
        # and cv2.  Keyboard control only needs DataBus, so expose the scripts
        # directory as a private namespace package and load databus directly.
        package_name = "_wbc_gen_gripper_scripts"
        package = sys.modules.get(package_name)
        if package is None:
            package = types.ModuleType(package_name)
            package.__path__ = [str(scripts_dir)]
            package.__package__ = package_name
            sys.modules[package_name] = package
        DataBus = importlib.import_module(
            f"{package_name}.databus"
        ).DataBus

        def encoder_callback(data):
            try:
                value = float(struct.unpack(">f", bytes(data[:4]))[0])
            except Exception as exc:
                print(f"gen_gripper_encoder_error={exc}")
                return
            if not np.isfinite(value):
                return
            if not 0.0 <= value <= 0.103:
                with self.encoder_lock:
                    self.last_invalid_encoder_value = value
                    self.invalid_encoder_count += 1
                    invalid_count = self.invalid_encoder_count
                if invalid_count == 1:
                    print(
                        "gen_gripper_encoder_invalid="
                        f"{value:.4f}m expected_range=[0.0000,0.1030]m"
                    )
                return
            with self.encoder_lock:
                self.encoder_value = value
            self.encoder_ready.set()

        self.bus = DataBus(
            tty_port=self.serial_port,
            encoder_freq=float(encoder_frequency),
            encoder_callback=encoder_callback,
            initial_target_distance=float(initial_width),
        )
        if not self.encoder_ready.wait(float(feedback_timeout)):
            with self.encoder_lock:
                last_invalid = self.last_invalid_encoder_value
            self.close()
            invalid_detail = (
                ""
                if last_invalid is None
                else f"; last invalid raw value was {last_invalid:.4f}m"
            )
            raise RuntimeError(
                "No valid Gen gripper encoder feedback in [0, 0.103] m within "
                f"{feedback_timeout:g}s on {self.serial_port}{invalid_detail}"
            )
        print(
            f"gen_gripper=ready port={self.serial_port} "
            f"encoder={self.current_width():.4f}m "
            f"initial_target={float(initial_width):.4f}m"
        )

    def current_width(self):
        with self.encoder_lock:
            return self.encoder_value

    def set_width(self, width):
        width = float(width)
        if not np.isfinite(width) or not 0.0 <= width <= 0.103:
            raise ValueError(f"Gen gripper width must be in [0, 0.103] m: {width}")
        self.bus.set_target_distance(width)
        current = self.current_width()
        current_text = "unavailable" if current is None else f"{current:.4f}m"
        print(
            f"gen_gripper_target={width:.4f}m "
            f"encoder={current_text}"
        )

    def increment_width(self, delta):
        previous = float(self.bus.get_target_distance())
        target = float(np.clip(previous + float(delta), 0.0, 0.103))
        self.set_width(target)
        print(
            f"gen_gripper_delta={target - previous:+.4f}m "
            f"previous_target={previous:.4f}m"
        )

    def close(self):
        if self.bus is not None:
            self.bus.stop()
            self.bus = None


def gripper_base_link_pose_to_wbc_world(
    gripper_base_link_pose_arm_base,
    tf_world_arm_base,
):
    """Convert an arm-base DAS gripper ``base_link`` pose to world J6/link6."""
    pose = np.asarray(
        gripper_base_link_pose_arm_base, dtype=np.float64
    ).reshape(6)
    tf_world_arm_base = np.asarray(
        tf_world_arm_base, dtype=np.float64
    ).reshape(4, 4)

    tf_arm_base_gripper_base_link = np.eye(4, dtype=np.float64)
    tf_arm_base_gripper_base_link[:3, :3] = R.from_rotvec(
        pose[3:6]
    ).as_matrix()
    tf_arm_base_gripper_base_link[:3, 3] = pose[:3]

    tf_link6_gripper_base_link = transform_from_pose6d(
        np.concatenate(
            (
                GRIPPER_BASE_LINK_OFFSET_POS,
                GRIPPER_BASE_LINK_OFFSET_RPY,
            )
        )
    )
    tf_gripper_base_link_link6 = np.linalg.inv(
        tf_link6_gripper_base_link
    )
    tf_world_link6 = (
        tf_world_arm_base
        @ tf_arm_base_gripper_base_link
        @ tf_gripper_base_link_link6
    )
    return np.concatenate(
        (
            tf_world_link6[:3, 3],
            rpy_from_rot(tf_world_link6[:3, :3]),
        )
    )


def rot_from_rpy(rpy):
    return R.from_euler("xyz", np.asarray(rpy, dtype=np.float64).reshape(3)).as_matrix()


def rpy_from_rot(rot):
    return R.from_matrix(np.asarray(rot, dtype=np.float64).reshape(3, 3)).as_euler("xyz")


def sdk_eef_to_training_eef(ee_pos, ee_rot):
    """Convert the SDK eef frame to the policy's assembly.urdf eef frame."""
    link6_pos, link6_rot = sdk_eef_to_link6(ee_pos, ee_rot)
    training_pos = link6_pos + link6_rot @ TRAINING_EEF_OFFSET_POS
    return training_pos, link6_rot


def sdk_eef_to_link6(ee_pos, ee_rot):
    """Convert the SDK eef frame to the physical J6/link6 frame."""
    sdk_offset_rot = rot_from_rpy(SDK_EEF_OFFSET_RPY)
    link6_rot = ee_rot @ sdk_offset_rot.T
    link6_pos = ee_pos - link6_rot @ SDK_EEF_OFFSET_POS
    return link6_pos, link6_rot


def ee_frame_transform(source_frame, target_frame):
    """Return the fixed transform from source EE frame to target EE frame."""
    source_frame = str(source_frame)
    target_frame = str(target_frame)
    valid_frames = ("eef_link", "j6")
    if source_frame not in valid_frames or target_frame not in valid_frames:
        raise ValueError(
            f"Unsupported EE frame conversion: {source_frame} -> {target_frame}"
        )
    if source_frame == target_frame:
        return np.eye(4, dtype=np.float64)
    tf_j6_eef = transform_from_pose6d(
        np.concatenate(
            (TRAINING_EEF_OFFSET_POS, np.zeros(3, dtype=np.float64))
        )
    )
    if source_frame == "j6":
        return tf_j6_eef
    return np.linalg.inv(tf_j6_eef)


def command_pose_to_policy_ee(command_pose, command_ee_frame, policy_ee_frame):
    """Convert an external J6/eef_link command to the policy's trained EE."""
    tf_command = transform_from_pose6d(command_pose)
    return tf_command @ ee_frame_transform(command_ee_frame, policy_ee_frame)


def pose6d(position, rotation):
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    return np.concatenate((np.asarray(position, dtype=np.float64).reshape(3), rotation[:, 0], rotation[:, 1]))


def rotation_angle(rotation):
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(math.acos(float(cosine)))


def limx_quat_to_xyzw(quat):
    quat = np.asarray(quat, dtype=np.float64).reshape(-1)
    if quat.size < 4:
        return np.array([0.0, 0.0, 0.0, 1.0])
    xyzw = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)
    norm = np.linalg.norm(xyzw)
    if norm < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return xyzw / norm


def idealize_observation_state(state):
    state = copy.deepcopy(state)
    tron = state["tron"]
    tron["q"] = np.zeros(8, dtype=np.float64)
    tron["dq"] = np.zeros(8, dtype=np.float64)
    tron["tau"] = np.zeros(8, dtype=np.float64)
    tron["imu"]["gyro"] = np.zeros(3, dtype=np.float64)
    tron["imu"]["quat"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    arm = state.get("arx")
    if isinstance(arm, dict):
        arm["ok"] = True
        arm["q"] = DEFAULT_Q[ARM_IDS].copy()
        arm["dq"] = np.zeros(6, dtype=np.float64)
        arm["tau"] = np.zeros(6, dtype=np.float64)
    return state


class ArmForwardKinematics:
    def __init__(self, urdf_path, base_link="base_link", tip_link="link6"):
        root = ET.parse(Path(urdf_path)).getroot()
        by_child = {}
        for element in root.findall("joint"):
            parent = element.find("parent")
            child = element.find("child")
            if parent is None or child is None:
                continue
            origin = element.find("origin")
            axis = element.find("axis")
            xyz = self._vec(origin.get("xyz") if origin is not None else None, [0.0, 0.0, 0.0])
            rpy = self._vec(origin.get("rpy") if origin is not None else None, [0.0, 0.0, 0.0])
            joint_axis = self._vec(axis.get("xyz") if axis is not None else None, [1.0, 0.0, 0.0])
            by_child[child.get("link")] = {
                "name": element.get("name"),
                "type": element.get("type", "fixed"),
                "parent": parent.get("link"),
                "xyz": xyz,
                "rpy": rpy,
                "axis": joint_axis,
            }

        chain = []
        link = tip_link
        while link != base_link:
            if link not in by_child:
                raise RuntimeError(f"No URDF FK chain from {base_link} to {tip_link}: stopped at {link}")
            joint = by_child[link]
            chain.append(joint)
            link = joint["parent"]
        self.chain = list(reversed(chain))

    @staticmethod
    def _vec(text, default):
        if text is None:
            return np.asarray(default, dtype=np.float64)
        return np.asarray([float(value) for value in text.split()], dtype=np.float64)

    def pose(self, joint_positions):
        joint_positions = dict(joint_positions)
        tf = np.eye(4, dtype=np.float64)
        for joint in self.chain:
            tf = tf @ transform_from_pose6d(np.concatenate((joint["xyz"], joint["rpy"])))
            if joint["type"] in ("revolute", "continuous"):
                angle = float(joint_positions.get(joint["name"], 0.0))
                motion = np.eye(4, dtype=np.float64)
                motion[:3, :3] = R.from_rotvec(joint["axis"] * angle).as_matrix()
                tf = tf @ motion
        return tf[:3, 3].copy(), tf[:3, :3].copy()


class ThreeOnnxPolicy:
    def __init__(self, model_dir, sample_latent=False, seed=0):
        model_dir = Path(model_dir).expanduser().resolve()
        missing = [name for name in ("actor.onnx", "contactNet.onnx", "gru.onnx") if not (model_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {missing} in {model_dir}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        self.actor = ort.InferenceSession(str(model_dir / "actor.onnx"), opts, providers=providers)
        self.contact = ort.InferenceSession(str(model_dir / "contactNet.onnx"), opts, providers=providers)
        self.gru = ort.InferenceSession(str(model_dir / "gru.onnx"), opts, providers=providers)
        self.sample_latent = bool(sample_latent)
        self.rng = np.random.default_rng(seed)
        self.hidden = np.zeros((1, 1, 131), dtype=np.float32)

        actor_inputs = self.actor.get_inputs()
        if actor_inputs[0].shape[-1] != OBS_DIM or actor_inputs[1].shape[-1] != 67:
            raise ValueError(f"Unexpected actor inputs: {[x.shape for x in actor_inputs]}")
        if self.contact.get_inputs()[0].shape[-1] != CONTACT_DIM:
            raise ValueError(f"Unexpected contactNet input: {self.contact.get_inputs()[0].shape}")
        if self.gru.get_inputs()[0].shape[-1] != 131:
            raise ValueError(f"Unexpected GRU input: {[x.shape for x in self.gru.get_inputs()]}")

    def reset(self):
        self.hidden.fill(0.0)

    def __call__(self, obs, history):
        contact_name = self.contact.get_inputs()[0].name
        contact_out = self.contact.run(None, {contact_name: history[np.newaxis, :, :].astype(np.float32)})[0]
        contact_out = np.asarray(contact_out[-1:], dtype=np.float32)

        gru_inputs = self.gru.get_inputs()
        gru_out, hidden = self.gru.run(None, {gru_inputs[0].name: contact_out, gru_inputs[1].name: self.hidden})
        self.hidden = np.asarray(hidden, dtype=np.float32)
        gru_out = np.asarray(gru_out, dtype=np.float32)

        mean = gru_out[:, 3:67]
        if self.sample_latent:
            logvar = gru_out[:, 67:131]
            latent = mean + np.sqrt(np.exp(logvar) + 1e-4) * self.rng.standard_normal(mean.shape).astype(np.float32)
        else:
            latent = mean
        actor_latent = np.concatenate((gru_out[:, :3], latent), axis=1).astype(np.float32)

        actor_inputs = self.actor.get_inputs()
        action = self.actor.run(
            None,
            {
                actor_inputs[0].name: obs[np.newaxis, :].astype(np.float32),
                actor_inputs[1].name: actor_latent,
            },
        )[0]
        action = np.asarray(action[0], dtype=np.float64)
        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise RuntimeError(f"Invalid actor output: {action}")
        return np.clip(action, -100.0, 100.0)


class RealMujocoStyleDeploy:
    def __init__(
        self,
        reader,
        policy,
        command,
        command_frame,
        command_ee_frame="j6",
        policy_ee_frame="j6",
        odom_reader=None,
        arx_ee_pose_is_link6=False,
        ee_pose_source="training_fk",
        action_smoothing=0.0,
        se3_decay_rate=1.0,
    ):
        self.reader = reader
        self.policy = policy
        self.command = np.asarray(command, dtype=np.float64).reshape(6)
        self.command_frame = command_frame
        self.command_ee_frame = str(command_ee_frame)
        if self.command_ee_frame not in ("eef_link", "j6"):
            raise ValueError(
                f"Unsupported command EE frame: {self.command_ee_frame}"
            )
        self.policy_ee_frame = str(policy_ee_frame)
        if self.policy_ee_frame not in ("eef_link", "j6"):
            raise ValueError(
                f"Unsupported policy EE frame: {self.policy_ee_frame}"
            )
        self.odom_reader = odom_reader
        self.cycle_odom = (
            odom_reader.read() if odom_reader is not None else {"ok": False}
        )
        self.arx_ee_pose_is_link6 = bool(arx_ee_pose_is_link6)
        self.ee_pose_source = ee_pose_source
        policy_tip_link = (
            "link6" if self.policy_ee_frame == "j6" else "eef_link"
        )
        self.arm_fk = ArmForwardKinematics(
            TRAINING_URDF,
            base_link="base_link",
            tip_link=policy_tip_link,
        )
        mount_fk = ArmForwardKinematics(
            TRAINING_URDF,
            base_link="base_Link",
            tip_link="base_link",
        )
        mount_pos, mount_rot = mount_fk.pose({})
        self.tf_policy_base_arm_base = np.eye(4, dtype=np.float64)
        self.tf_policy_base_arm_base[:3, :3] = mount_rot
        self.tf_policy_base_arm_base[:3, 3] = mount_pos
        self.tf_arm_base_policy_base = np.linalg.inv(
            self.tf_policy_base_arm_base
        )
        self.action_smoothing = float(np.clip(action_smoothing, 0.0, 0.99))
        self.se3_decay_rate = max(0.0, float(se3_decay_rate))
        self.last_action = np.zeros(ACTION_DIM, dtype=np.float64)
        self.last_action_input = np.zeros(ACTION_DIM, dtype=np.float64)
        self.last_torque = np.zeros(ACTION_DIM, dtype=np.float64)
        self.actor_action = np.zeros(ACTION_DIM, dtype=np.float64)
        self.raw_action = np.zeros(ACTION_DIM, dtype=np.float64)
        self.effective_action = np.zeros(ACTION_DIM, dtype=np.float64)
        self.desired_q = DEFAULT_Q.copy()
        self.history = deque(maxlen=HISTORY)
        self.se3_ref = 0.0
        self.se3_actual = 0.0
        self.se3_reset_pending = False
        self.se3_reset_reason = ""
        self.se3_reset_applied = False
        self.se3_reset_value = None
        self._init_history()

    def refresh_odom_snapshot(self):
        """Latch one odom frame for all transforms in the current policy cycle."""
        self.cycle_odom = (
            self.odom_reader.read()
            if self.odom_reader is not None
            else {"ok": False}
        )
        return self.cycle_odom

    def odom_snapshot(self):
        return self.cycle_odom

    def _init_history(self):
        state = self.reader.read()
        # Real deployment starts after the reset controller is already holding
        # DEFAULT_Q, so its initial commanded PD torque estimate is zero.
        self.last_torque.fill(0.0)
        contact = self.contact_observation(state)
        self.history.clear()
        for _ in range(HISTORY):
            self.history.append(contact.copy())
        self.se3_ref = self.initial_se3_distance(state)
        self.policy.reset()

    def request_se3_reset(self, reason):
        """Reset trajectory progress on the next state-backed inference."""
        self.se3_reset_pending = True
        self.se3_reset_reason = str(reason)

    def joint_state(self, state):
        tron = state["tron"]
        arm = state.get("arx")
        q = np.zeros(ACTION_DIM, dtype=np.float64)
        dq = np.zeros(ACTION_DIM, dtype=np.float64)
        tau = np.zeros(ACTION_DIM, dtype=np.float64)
        q[LEG_IDS] = np.asarray(tron["q"], dtype=np.float64).reshape(-1)[:8]
        dq[LEG_IDS] = np.asarray(tron["dq"], dtype=np.float64).reshape(-1)[:8]
        tau[LEG_IDS] = np.asarray(tron["tau"], dtype=np.float64).reshape(-1)[:8]
        if isinstance(arm, dict) and arm.get("ok", False):
            q[ARM_IDS] = np.asarray(arm["q"], dtype=np.float64).reshape(-1)[:6]
            dq[ARM_IDS] = np.asarray(arm["dq"], dtype=np.float64).reshape(-1)[:6]
            tau[ARM_IDS] = np.asarray(arm["tau"], dtype=np.float64).reshape(-1)[:6]
        else:
            q[ARM_IDS] = DEFAULT_Q[ARM_IDS]
        return q, dq, tau

    def base_rotation(self, state):
        quat_xyzw = limx_quat_to_xyzw(state["tron"]["imu"]["quat"])
        return R.from_quat(quat_xyzw).as_matrix()

    def base_angular_velocity(self, state):
        gyro = np.asarray(state["tron"]["imu"]["gyro"], dtype=np.float64).reshape(-1)
        if gyro.size < 3:
            gyro = np.pad(gyro, (0, 3 - gyro.size))
        return gyro[:3]

    def projected_gravity(self, state):
        return self.base_rotation(state).T @ np.array([0.0, 0.0, -1.0])

    def ee_pose_arm_base(self, state):
        """Return the configured EE pose relative to the ARX arm base_link."""
        arm = state.get("arx")
        if not isinstance(arm, dict) or not arm.get("ok", False):
            return self.arm_fk.pose(
                zip(ARM_NAMES, DEFAULT_Q[ARM_IDS])
            )
        if self.ee_pose_source == "training_fk":
            q = np.asarray(arm["q"], dtype=np.float64).reshape(-1)[:6]
            return self.arm_fk.pose(zip(ARM_NAMES, q))
        ee = np.asarray(arm["ee_pose"], dtype=np.float64).reshape(-1)[:6]
        return self._ee_pose_from_raw(ee)

    def ee_pose_base(self, state):
        """Return EE in deploy2's training base: the TRON root base_Link."""
        arm_pos, arm_rot = self.ee_pose_arm_base(state)
        tf_arm_ee = np.eye(4, dtype=np.float64)
        tf_arm_ee[:3, :3] = arm_rot
        tf_arm_ee[:3, 3] = arm_pos
        tf_policy_ee = self.tf_policy_base_arm_base @ tf_arm_ee
        return (
            tf_policy_ee[:3, 3].copy(),
            tf_policy_ee[:3, :3].copy(),
        )

    def world_arm_base_transform(self):
        """Return world -> ARX arm base_link from MID360/FAST-LIO."""
        if self.odom_reader is None:
            return None
        odom = self.odom_snapshot()
        if not odom.get("ok", False):
            return None
        return np.asarray(
            odom.get("tf_world_arm_base", odom["tf_world_base"]),
            dtype=np.float64,
        ).reshape(4, 4)

    def world_policy_base_transform(self):
        """Return world -> TRON root base_Link for policy-frame transforms."""
        tf_world_arm_base = self.world_arm_base_transform()
        if tf_world_arm_base is None:
            return None
        return tf_world_arm_base @ self.tf_arm_base_policy_base

    def _ee_pose_from_raw(self, ee):
        ee_pos = ee[:3].copy()
        ee_rot = rot_from_rpy(ee[3:6])
        if self.arx_ee_pose_is_link6:
            link6_pos, link6_rot = ee_pos, ee_rot
        else:
            link6_pos, link6_rot = sdk_eef_to_link6(ee_pos, ee_rot)
        if self.policy_ee_frame == "j6":
            return link6_pos, link6_rot
        return (
            link6_pos + link6_rot @ TRAINING_EEF_OFFSET_POS,
            link6_rot,
        )

    def raw_and_obs_ee_pose_base(self, state):
        arm = state.get("arx")
        if not isinstance(arm, dict) or not arm.get("ok", False):
            obs_pos, obs_rot = self.ee_pose_base(state)
            return obs_pos, obs_rot, obs_pos, obs_rot
        ee = np.asarray(arm["ee_pose"], dtype=np.float64).reshape(-1)[:6]
        raw_pos = ee[:3].copy()
        raw_rot = rot_from_rpy(ee[3:6])
        obs_pos, obs_rot = self.ee_pose_base(state)
        return raw_pos, raw_rot, obs_pos, obs_rot

    def target_pose_base(self):
        tf_command_target = command_pose_to_policy_ee(
            self.command, self.command_ee_frame, self.policy_ee_frame
        )
        if self.command_frame == "base":
            return (
                tf_command_target[:3, 3].copy(),
                tf_command_target[:3, :3].copy(),
            )
        if self.odom_reader is None:
            raise RuntimeError("world command requires --use-ros2-odom")
        tf_world_policy_base = self.world_policy_base_transform()
        if tf_world_policy_base is None:
            raise RuntimeError("ros2 odom unavailable")
        tf_base_target = (
            np.linalg.inv(tf_world_policy_base) @ tf_command_target
        )
        return tf_base_target[:3, 3].copy(), tf_base_target[:3, :3].copy()

    def policy_ee_to_command_ee_base(self, ee_pos_base, ee_rot_base):
        """Express the policy's trained EE pose as the command EE frame."""
        tf_base_policy = np.eye(4, dtype=np.float64)
        tf_base_policy[:3, :3] = np.asarray(
            ee_rot_base, dtype=np.float64
        ).reshape(3, 3)
        tf_base_policy[:3, 3] = np.asarray(
            ee_pos_base, dtype=np.float64
        ).reshape(3)
        tf_base_command = tf_base_policy @ ee_frame_transform(
            self.policy_ee_frame, self.command_ee_frame
        )
        return (
            tf_base_command[:3, 3].copy(),
            tf_base_command[:3, :3].copy(),
        )

    def ee_pose_world(self, ee_pos_base, ee_rot_base):
        tf_world_policy_base = self.world_policy_base_transform()
        if tf_world_policy_base is None:
            return None
        base_rot = tf_world_policy_base[:3, :3]
        base_pos = tf_world_policy_base[:3, 3]
        ee_pos_world = base_pos + base_rot @ ee_pos_base
        ee_rot_world = base_rot @ ee_rot_base
        return ee_pos_world, ee_rot_world

    def contact_observation(self, state):
        q, dq, tau = self.joint_state(state)
        ee_pos, ee_rot = self.ee_pose_base(state)
        obs = np.concatenate(
            (
                self.base_angular_velocity(state),
                self.projected_gravity(state),
                (q - DEFAULT_Q)[NO_ANKLE],
                dq,
                self.last_torque,
                pose6d(ee_pos, ee_rot),
            )
        )
        if obs.shape != (CONTACT_DIM,):
            raise RuntimeError(f"contact obs shape={obs.shape}")
        return obs.astype(np.float32)

    def policy_observation(self, state):
        q, dq, _ = self.joint_state(state)
        ee_pos, ee_rot = self.ee_pose_base(state)
        target_pos, target_rot = self.target_pose_base()
        obs = np.concatenate(
            (
                self.base_angular_velocity(state),
                self.projected_gravity(state),
                pose6d(target_pos, target_rot),
                (q - DEFAULT_Q)[NO_ANKLE],
                dq,
                self.last_action,
                pose6d(ee_pos, ee_rot),
                np.array([self.se3_ref]),
            )
        )
        if obs.shape != (OBS_DIM,):
            raise RuntimeError(f"policy obs shape={obs.shape}")
        return np.clip(obs, -100.0, 100.0).astype(np.float32)

    def initial_se3_distance(self, state):
        ee_pos, ee_rot = self.ee_pose_base(state)
        target_pos, target_rot = self.target_pose_base()
        return float(2.0 * np.linalg.norm(target_pos - ee_pos) + rotation_angle(target_rot @ ee_rot.T))

    def infer(self, state):
        contact = self.contact_observation(state)
        self.history.append(contact)
        history = np.stack(self.history, axis=0)
        self.last_action_input = self.last_action.copy()
        # A teleop target change starts a new trajectory.  Apply the reset
        # before building obs_t so Actor sees the current SE(3) distance in
        # the same cycle as the new target, rather than the previous
        # trajectory's already-decayed zero.
        self.se3_actual = self.initial_se3_distance(state)
        self.se3_reset_applied = False
        self.se3_reset_value = None
        if self.se3_reset_pending:
            self.se3_ref = self.se3_actual
            self.se3_reset_pending = False
            self.se3_reset_applied = True
            self.se3_reset_value = float(self.se3_ref)
            print(
                f"se3_ref_reset reason={self.se3_reset_reason} "
                f"value={self.se3_ref:.4f}"
            )
        actor_action = self.policy(self.policy_observation(state), history)
        self.actor_action = actor_action.copy()
        if self.action_smoothing > 0.0:
            self.raw_action = self.action_smoothing * self.raw_action + (1.0 - self.action_smoothing) * actor_action
        else:
            self.raw_action = actor_action
        self.se3_ref = max(0.0, self.se3_ref - self.se3_decay_rate * POLICY_DT)
        # Match IsaacLab/Sim2Sim mdp.last_action: the next observation receives
        # this cycle's direct actor output, before torque/step limiting or
        # hardware command tracking.  This update happens after obs_t was
        # constructed, so obs_(t+1) receives action_t.
        self.last_action = self.actor_action.copy()
        return self.raw_action

    def compute_pd_targets(self, state):
        q, dq, _ = self.joint_state(state)
        action_min = q - DEFAULT_Q + (KD * dq - TORQUE_LIMIT) / KP
        action_max = q - DEFAULT_Q + (KD * dq + TORQUE_LIMIT) / KP
        scaled_action = self.raw_action * ACTION_SCALE
        self.effective_action = np.clip(scaled_action, action_min, action_max)
        self.desired_q = DEFAULT_Q + self.effective_action
        return self.desired_q

    def record_applied_targets(self, state, output_reply):
        q, dq, _ = self.joint_state(state)
        applied_q = q.copy()
        if isinstance(output_reply, dict):
            arm_reply = output_reply.get("arm")
            if (
                isinstance(arm_reply, dict)
                and "cmd" in arm_reply
                and "error" not in arm_reply
            ):
                applied_q[ARM_IDS] = np.asarray(
                    arm_reply["cmd"], dtype=np.float64
                ).reshape(-1)[:6]
            leg_reply = output_reply.get("legs")
            if isinstance(leg_reply, dict) and "cmd" in leg_reply:
                applied_q[LEG_IDS] = np.asarray(
                    leg_reply["cmd"], dtype=np.float64
                ).reshape(-1)[:8]
        self.last_torque = np.clip(
            KP * (applied_q - q) - KD * dq,
            -TORQUE_LIMIT,
            TORQUE_LIMIT,
        )


class RealOutput:
    def __init__(
        self,
        reader,
        enable_legs=True,
        enable_arm=True,
        arm_max_step=0.0,
        max_leg_step=0.0,
        leg_kp_scale=1.0,
        arx_max_delta=0.2,
    ):
        self.reader = reader
        self.enable_legs = enable_legs
        self.enable_arm = enable_arm
        self.arm_max_step = float(arm_max_step)
        self.max_leg_step = float(max_leg_step)
        self.leg_kp_scale = float(leg_kp_scale)
        self.arx_max_delta = float(arx_max_delta)
        self.cmd = datatypes.RobotCmd()
        self.cmd.mode = [int(os.getenv("LIMX_CMD_MODE", "0"))] * len(LEG_NAMES)
        self.cmd.q = [0.0] * len(LEG_NAMES)
        self.cmd.dq = [0.0] * len(LEG_NAMES)
        self.cmd.tau = [0.0] * len(LEG_NAMES)
        self.cmd.Kp = [0.0] * len(LEG_NAMES)
        self.cmd.Kd = [1.0] * len(LEG_NAMES)
        self.last_arm_cmd = None
        self.last_leg_cmd = None
        self._leg_cmd_lock = threading.Lock()
        self._leg_publish_stop = threading.Event()
        self._leg_publish_thread = None
        self._leg_publish_rate = 0.0
        self._leg_publish_error = None

    def _publish_leg_cmd_locked(self):
        self.cmd.stamp = time.time_ns()
        self.reader.tron.robot.publishRobotCmd(self.cmd)

    def _leg_publish_loop(self):
        period = 1.0 / max(self._leg_publish_rate, 1.0)
        next_tick = time.monotonic()
        try:
            while not self._leg_publish_stop.is_set():
                with self._leg_cmd_lock:
                    self._publish_leg_cmd_locked()
                next_tick += period
                delay = next_tick - time.monotonic()
                if delay > 0.0:
                    self._leg_publish_stop.wait(delay)
                else:
                    next_tick = time.monotonic()
        except Exception as exc:
            self._leg_publish_error = exc
            self._leg_publish_stop.set()

    def start_leg_publisher(self, rate_hz=500.0):
        if not self.enable_legs:
            return
        if self._leg_publish_thread is not None and self._leg_publish_thread.is_alive():
            return
        self._leg_publish_rate = float(rate_hz)
        if self._leg_publish_rate <= 0.0:
            raise ValueError("leg publish rate must be positive")
        with self._leg_cmd_lock:
            if self.last_leg_cmd is not None:
                for i in range(len(LEG_NAMES)):
                    idx = LEG_IDS[i]
                    self.cmd.q[i] = float(self.last_leg_cmd[i])
                    self.cmd.dq[i] = 0.0
                    self.cmd.tau[i] = 0.0
                    self.cmd.Kp[i] = float(KP[idx] * self.leg_kp_scale)
                    self.cmd.Kd[i] = float(KD[idx])
        self._leg_publish_error = None
        self._leg_publish_stop.clear()
        self._leg_publish_thread = threading.Thread(
            target=self._leg_publish_loop,
            name="limx_robot_cmd_publisher",
            daemon=True,
        )
        self._leg_publish_thread.start()

    def stop_leg_publisher(self, damping=True):
        thread = self._leg_publish_thread
        if thread is not None:
            self._leg_publish_stop.set()
            thread.join(timeout=1.0)
            self._leg_publish_thread = None
        if damping and self.enable_legs:
            self.stop_legs()

    def set_sim2sim_arm_gains(self):
        """Apply and verify the arm PD gains used by the Sim2Sim policy."""
        if not self.enable_arm or self.reader.arx is None:
            return None
        previous = self.reader.arx.request("GET_GAIN", None, timeout_ms=1000)
        target_kp = KP[ARM_IDS].copy()
        target_kd = KD[ARM_IDS].copy()
        self.reader.arx.request(
            "SET_GAIN",
            {
                "kp": target_kp,
                "kd": target_kd,
                "gripper_kp": float(previous["gripper_kp"]),
                "gripper_kd": float(previous["gripper_kd"]),
            },
            timeout_ms=1000,
        )
        applied = self.reader.arx.request("GET_GAIN", None, timeout_ms=1000)
        applied_kp = np.asarray(applied["kp"], dtype=np.float64).reshape(-1)[:6]
        applied_kd = np.asarray(applied["kd"], dtype=np.float64).reshape(-1)[:6]
        if not (
            np.allclose(applied_kp, target_kp, rtol=0.0, atol=1e-6)
            and np.allclose(applied_kd, target_kd, rtol=0.0, atol=1e-6)
        ):
            raise RuntimeError(
                "ARX gain verification failed: "
                f"expected kp={target_kp}, kd={target_kd}; "
                f"received kp={applied_kp}, kd={applied_kd}"
            )
        return applied

    def reset_arm(self, rate_hz=50.0, duration=2.0, hold=0.5):
        if not self.enable_arm or self.reader.arx is None:
            return None
        self.reader.arx.request("RESET_TO_HOME", None, timeout_ms=10000)
        state = self.reader.arx.read()
        current = np.asarray(state["q"], dtype=np.float64).reshape(-1)[:6]
        if not np.isfinite(current).all():
            raise RuntimeError(f"Invalid ARX joint state after reset: {current}")
        # Replace a stale/corrupted server-side last command with the measured
        # current pose before applying the normal max-delta safety check.
        self.reader.arx.request(
            "SET_JOINT_POS",
            {"joint_pos": current, "gripper_pos": None, "max_delta": 0.0},
        )
        target = DEFAULT_Q[ARM_IDS]
        steps = max(1, int(duration * rate_hz))
        dt = 1.0 / rate_hz
        for i in range(1, steps + 1):
            q = (1.0 - i / steps) * current + (i / steps) * target
            self.reader.arx.request("SET_JOINT_POS", {"joint_pos": q, "gripper_pos": None, "max_delta": self.arx_max_delta})
            time.sleep(dt)
        for _ in range(max(1, int(hold * rate_hz))):
            self.reader.arx.request("SET_JOINT_POS", {"joint_pos": target, "gripper_pos": None, "max_delta": self.arx_max_delta})
            time.sleep(dt)
        try:
            self.reader.arx.request("SYNC_LAST_COMMAND", None)
        except Exception:
            pass
        actual = np.asarray(self.reader.arx.read()["q"], dtype=np.float64).reshape(-1)[:6]
        self.last_arm_cmd = actual.copy()
        self.set_sim2sim_arm_gains()
        return actual

    def reset_legs(self, rate_hz=50.0, duration=2.0, hold=0.3):
        if not self.enable_legs:
            return None
        state = self.reader.read()
        current = np.asarray(state["tron"]["q"], dtype=np.float64).reshape(-1)[:8]
        target = DEFAULT_Q[LEG_IDS]
        steps = max(1, int(duration * rate_hz))
        dt = 1.0 / rate_hz
        for i in range(1, steps + 1):
            q = (1.0 - i / steps) * current + (i / steps) * target
            for j, name in enumerate(LEG_NAMES):
                idx = LEG_IDS[j]
                self.cmd.q[j] = float(q[j])
                self.cmd.dq[j] = 0.0
                self.cmd.tau[j] = 0.0
                self.cmd.Kp[j] = float(KP[idx] * self.leg_kp_scale)
                self.cmd.Kd[j] = float(KD[idx])
            with self._leg_cmd_lock:
                self._publish_leg_cmd_locked()
            time.sleep(dt)
        for _ in range(max(1, int(hold * rate_hz))):
            for j, name in enumerate(LEG_NAMES):
                idx = LEG_IDS[j]
                self.cmd.q[j] = float(target[j])
                self.cmd.dq[j] = 0.0
                self.cmd.tau[j] = 0.0
                self.cmd.Kp[j] = float(KP[idx] * self.leg_kp_scale)
                self.cmd.Kd[j] = float(KD[idx])
            with self._leg_cmd_lock:
                self._publish_leg_cmd_locked()
            time.sleep(dt)
        self.last_leg_cmd = target.copy()
        return self.reader.read()["tron"]["q"]

    def hold_arm(self, q, rate_hz=50.0, duration=1.0):
        if not self.enable_arm or self.reader.arx is None:
            return None
        q = np.asarray(q, dtype=np.float64).reshape(6)
        steps = max(1, int(duration * rate_hz))
        dt = 1.0 / rate_hz
        for _ in range(steps):
            self.reader.arx.request(
                "SET_JOINT_POS",
                {"joint_pos": q, "gripper_pos": None, "max_delta": self.arx_max_delta},
            )
            time.sleep(dt)
        self.last_arm_cmd = q.copy()
        return np.asarray(self.reader.arx.read()["q"], dtype=np.float64).reshape(-1)[:6]

    def test_one_arm_joint(self, joint_name, delta, rate_hz=50.0, duration=1.0, hold=1.0):
        if joint_name not in ARM_NAME_TO_I:
            raise ValueError(f"unknown arm joint {joint_name}, choose one of {list(ARM_NAMES)}")
        start = self.reset_arm(rate_hz=rate_hz)
        target = start.copy()
        target[ARM_NAME_TO_I[joint_name]] += float(delta)
        steps = max(1, int(duration * rate_hz))
        dt = 1.0 / rate_hz
        for i in range(1, steps + 1):
            q = (1.0 - i / steps) * start + (i / steps) * target
            self.reader.arx.request(
                "SET_JOINT_POS",
                {"joint_pos": q, "gripper_pos": None, "max_delta": self.arx_max_delta},
            )
            time.sleep(dt)
        actual = self.hold_arm(target, rate_hz=rate_hz, duration=hold)
        # Return through the same interpolated trajectory.  Sending `start`
        # directly can violate the server's max_delta guard when the requested
        # diagnostic displacement is larger than arx_max_delta.
        for i in range(1, steps + 1):
            q = (1.0 - i / steps) * target + (i / steps) * start
            self.reader.arx.request(
                "SET_JOINT_POS",
                {"joint_pos": q, "gripper_pos": None, "max_delta": self.arx_max_delta},
            )
            time.sleep(dt)
        self.hold_arm(start, rate_hz=rate_hz, duration=hold)
        return start, target, actual

    def hold_legs(self, q, rate_hz=50.0, duration=1.0):
        if not self.enable_legs:
            return None
        q = np.asarray(q, dtype=np.float64).reshape(8)
        steps = max(1, int(duration * rate_hz))
        dt = 1.0 / rate_hz
        for _ in range(steps):
            for j, name in enumerate(LEG_NAMES):
                idx = LEG_IDS[j]
                self.cmd.q[j] = float(q[j])
                self.cmd.dq[j] = 0.0
                self.cmd.tau[j] = 0.0
                self.cmd.Kp[j] = float(KP[idx] * self.leg_kp_scale)
                self.cmd.Kd[j] = float(KD[idx])
            with self._leg_cmd_lock:
                self._publish_leg_cmd_locked()
            time.sleep(dt)
        self.last_leg_cmd = q.copy()
        return self.reader.read()["tron"]["q"]

    def test_one_leg_joint(self, joint_name, delta, rate_hz=50.0, duration=1.0, hold=1.0):
        if joint_name not in LEG_NAME_TO_I:
            raise ValueError(f"unknown leg joint {joint_name}, choose one of {list(LEG_NAMES)}")
        home = DEFAULT_Q[LEG_IDS].copy()
        target = home.copy()
        target[LEG_NAME_TO_I[joint_name]] += float(delta)
        self.reset_legs(rate_hz=rate_hz, duration=duration, hold=0.2)
        self.hold_legs(target, rate_hz=rate_hz, duration=hold)
        actual = self.reader.read()["tron"]["q"]
        self.hold_legs(home, rate_hz=rate_hz, duration=hold)
        return actual

    def publish(self, state, desired_q):
        if self._leg_publish_error is not None:
            raise RuntimeError(
                f"LimX {self._leg_publish_rate:.1f}Hz publisher failed: "
                f"{self._leg_publish_error}"
            )
        leg_reply = None
        if self.enable_legs:
            current_leg_q = np.asarray(state["tron"]["q"], dtype=np.float64).reshape(-1)[:8]
            desired_leg_q = desired_q[LEG_IDS].copy()
            if self.last_leg_cmd is None:
                self.last_leg_cmd = current_leg_q.copy()
            prev_leg_cmd = self.last_leg_cmd.copy()
            leg_q = desired_leg_q.copy()
            if self.max_leg_step > 0.0:
                leg_q = np.clip(leg_q, prev_leg_cmd - self.max_leg_step, prev_leg_cmd + self.max_leg_step)
            with self._leg_cmd_lock:
                for i, name in enumerate(LEG_NAMES):
                    idx = LEG_IDS[i]
                    self.cmd.q[i] = float(leg_q[i])
                    self.cmd.dq[i] = 0.0
                    self.cmd.tau[i] = 0.0
                    self.cmd.Kp[i] = float(KP[idx] * self.leg_kp_scale)
                    self.cmd.Kd[i] = float(KD[idx])
                if self._leg_publish_thread is None:
                    self._publish_leg_cmd_locked()
            self.last_leg_cmd = leg_q.copy()
            leg_reply = {
                "current": current_leg_q,
                "desired": desired_leg_q,
                "prev_cmd": prev_leg_cmd,
                "cmd": leg_q.copy(),
            }

        arm_reply = None
        if self.enable_arm and self.reader.arx is not None and isinstance(state.get("arx"), dict) and state["arx"].get("ok", False):
            current = np.asarray(state["arx"]["q"], dtype=np.float64).reshape(-1)[:6]
            arm_desired_raw = desired_q[ARM_IDS].copy()
            if self.last_arm_cmd is None:
                self.last_arm_cmd = current.copy()
            prev_cmd = self.last_arm_cmd.copy()
            arm_cmd = arm_desired_raw.copy()
            if self.arm_max_step > 0.0:
                arm_cmd = np.clip(arm_cmd, prev_cmd - self.arm_max_step, prev_cmd + self.arm_max_step)
            arm_reply = {
                "current": current.copy(),
                "desired": arm_desired_raw.copy(),
                "prev_cmd": prev_cmd.copy(),
                "cmd": arm_cmd.copy(),
                "reply_q": None,
            }
            try:
                reply = self.reader.arx.request(
                    "SET_JOINT_POS",
                    {"joint_pos": arm_cmd, "gripper_pos": None, "max_delta": self.arx_max_delta},
                )
                arm_reply["raw_reply"] = reply
                if isinstance(reply, dict) and "joint_pos" in reply:
                    arm_reply["reply_q"] = np.asarray(reply["joint_pos"], dtype=np.float64).reshape(-1)[:6]
                self.last_arm_cmd = arm_cmd.copy()
            except Exception as exc:
                arm_reply["error"] = str(exc)
        return {"arm": arm_reply, "legs": leg_reply}

    def stop_legs(self):
        with self._leg_cmd_lock:
            self.cmd.Kp = [0.0] * len(LEG_NAMES)
            self.cmd.Kd = [1.0] * len(LEG_NAMES)
            self.cmd.tau = [0.0] * len(LEG_NAMES)
            self._publish_leg_cmd_locked()

    def emergency_hold(self, state):
        errors = []
        if self.enable_arm and self.reader.arx is not None:
            arm = state.get("arx")
            if isinstance(arm, dict) and arm.get("ok", False):
                current = np.asarray(arm["q"], dtype=np.float64).reshape(-1)[:6]
                try:
                    self.reader.arx.request(
                        "SET_JOINT_POS",
                        {"joint_pos": current, "gripper_pos": None, "max_delta": 0.0},
                        timeout_ms=1000,
                    )
                    self.last_arm_cmd = current.copy()
                except Exception as exc:
                    errors.append(f"arm={exc}")
        if self.enable_legs:
            current = np.asarray(state["tron"]["q"], dtype=np.float64).reshape(-1)[:8]
            with self._leg_cmd_lock:
                for i in range(len(LEG_NAMES)):
                    idx = LEG_IDS[i]
                    self.cmd.q[i] = float(current[i])
                    self.cmd.dq[i] = 0.0
                    self.cmd.tau[i] = 0.0
                    self.cmd.Kp[i] = float(KP[idx] * self.leg_kp_scale)
                    self.cmd.Kd[i] = float(KD[idx])
                self._publish_leg_cmd_locked()
            self.last_leg_cmd = current.copy()
        return errors


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-ip", default=os.getenv("TRON1_IP", "10.192.1.2"))
    parser.add_argument("--arx-ip", default=os.getenv("ARX5_ZMQ_IP", "127.0.0.1"))
    parser.add_argument("--arx-port", type=int, default=int(os.getenv("ARX5_ZMQ_PORT", "8765")))
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--command", nargs=6, type=float, default=[0.15, 0.0, 1.0, 0.0, 0.0, 0.0])
    parser.add_argument("--command-frame", choices=("world", "base"), default="world")
    parser.add_argument(
        "--keyboard-command",
        action="store_true",
        help=(
            "Change command XYZ from the terminal without Enter: "
            "W/S=+/-X, A/D=+/-Y, R/F=+/-Z."
        ),
    )
    parser.add_argument(
        "--keyboard-step",
        type=float,
        default=0.1,
        help="XYZ increment in metres for each keyboard key press (default: 0.1).",
    )
    parser.add_argument(
        "--gen-gripper",
        action="store_true",
        help="Enable GenRobot serial gripper control; keyboard T=open and G=close.",
    )
    parser.add_argument("--gripper-port", default="/dev/ttyUSB0")
    parser.add_argument(
        "--gripper-sdk-root",
        default="/home/phi/python__runner/umi-deploy/gen_con_sdk_python_release",
    )
    parser.add_argument("--gripper-encoder-frequency", type=float, default=30.0)
    parser.add_argument("--gripper-feedback-timeout", type=float, default=3.0)
    parser.add_argument("--gripper-initial-width", type=float, default=0.08)
    parser.add_argument(
        "--gripper-step",
        type=float,
        default=0.01,
        help="Gripper opening increment per T/G key press in metres (default: 0.01).",
    )
    parser.add_argument(
        "--command-ee-frame",
        choices=("eef_link", "j6"),
        default="j6",
        help=(
            "Frame represented by --command (default: j6). 'j6' means the "
            "physical J6/link6 origin."
        ),
    )
    parser.add_argument(
        "--policy-ee-frame",
        choices=("eef_link", "j6"),
        default="j6",
        help=(
            "EE frame used during policy training (default: j6). deploy2 "
            "was trained with J6/link6."
        ),
    )
    parser.add_argument(
        "--hold-current-ee",
        action="store_true",
        help="Use the post-reset current EE world pose as the policy target.",
    )
    parser.add_argument(
        "--pre-diffusion-hold-position",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help=(
            "Before the first fresh Diffusion chunk, actively command this "
            "J6 world position while retaining the startup EE orientation."
        ),
    )
    parser.add_argument(
        "--pre-diffusion-hold-pose",
        nargs=6,
        type=float,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        default=None,
        help=(
            "Before the first fresh Diffusion chunk, actively command this "
            "complete J6 world pose."
        ),
    )
    parser.add_argument(
        "--hold-current-ee-base",
        action="store_true",
        help=(
            "Use the current configured command EE (eef_link or J6) "
            "base-frame pose as target without ROS2 odom."
        ),
    )
    parser.add_argument("--use-ros2-odom", action="store_true")
    parser.add_argument("--odom-topic", default="/Odometry")
    parser.add_argument("--ground-height-topic", default="/ground_height")
    parser.add_argument(
        "--ground-reference-topic",
        default="/ground_height_reference",
        help=(
            "Atomic [height, raw odom xyz] reference used to preserve motion "
            "after the ground-height measurement."
        ),
    )
    parser.add_argument(
        "--ground-freeze-topic",
        default="/ground_height_freeze",
        help="Freeze the final live ground/odom reference immediately before output.",
    )
    parser.add_argument(
        "--lidar-to-base-xyz",
        nargs=3,
        type=float,
        default=[-0.14, 0.0, 0.0677],
        metavar=("X", "Y", "Z"),
        help=(
            "Rigid MID360-origin to ARX base_link translation; it is rotated "
            "by the current body orientation before entering world coordinates."
        ),
    )
    parser.add_argument(
        "--fastlio-lidar-to-imu-xyz",
        nargs=3,
        type=float,
        default=[-0.011, -0.02329, 0.04412],
        metavar=("X", "Y", "Z"),
        help="FAST-LIO mapping.extrinsic_T (LiDAR origin in IMU/body frame).",
    )
    parser.add_argument(
        "--lidar-to-base-z",
        type=float,
        default=None,
        help="Deprecated compatibility override for only the Z component.",
    )
    parser.add_argument(
        "--ee-command-file",
        default="",
        help="JSON file providing runtime XYZ/RPY target and a latched estop request.",
    )
    parser.add_argument(
        "--bridge-command-timeout",
        type=float,
        default=2.0,
        help="Reject Diffusion bridge files older than this many seconds.",
    )
    parser.add_argument(
        "--require-diffusion-command",
        action="store_true",
        help=(
            "Block real policy output until a fresh Diffusion chunk arrives, "
            "and hold current joints if the chunk stream times out."
        ),
    )
    parser.add_argument(
        "--freeze-world-base",
        action="store_true",
        help=(
            "Freeze world-to-robot-base at startup. Use only when the robot "
            "base is physically stationary, such as an arm-only bench test."
        ),
    )
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument(
        "--leg-publish-rate",
        type=float,
        default=500.0,
        help="LimX RobotCmd publication rate; policy inference remains at --rate.",
    )
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--enable-output", action="store_true")
    parser.add_argument(
        "--reset-before-dry-run",
        action="store_true",
        help="Reset enabled arm/legs, then run inference without publishing policy commands.",
    )
    parser.add_argument(
        "--debug-ideal-observation",
        action="store_true",
        help="Dry-run ONNX with exact training-home state instead of measured state.",
    )
    parser.add_argument("--no-legs", action="store_true")
    parser.add_argument("--no-arm", action="store_true")
    parser.add_argument("--arm-max-step", type=float, default=0.0)
    parser.add_argument("--max-arm-home-error", type=float, default=0.1)
    parser.add_argument("--arm-home-only", action="store_true")
    parser.add_argument("--arm-joint-test", choices=ARM_NAMES)
    parser.add_argument("--arm-test-delta", type=float, default=0.02)
    parser.add_argument("--arm-test-hold", type=float, default=1.0)
    parser.add_argument("--max-leg-step", type=float, default=0.0)
    parser.add_argument("--leg-kp-scale", type=float, default=0.5)
    parser.add_argument("--leg-reset-duration", type=float, default=2.0)
    parser.add_argument("--max-leg-home-error", type=float, default=0.1)
    parser.add_argument(
        "--skip-leg-reset",
        action="store_true",
        help="Keep the measured leg pose at startup and skip the leg home trajectory/check.",
    )
    parser.add_argument("--leg-home-only", action="store_true")
    parser.add_argument(
        "--leg-hold-current-only",
        action="store_true",
        help="Hold the measured leg pose with PD only; no reset, arm, or policy.",
    )
    parser.add_argument("--leg-joint-test", choices=LEG_NAMES)
    parser.add_argument("--leg-test-delta", type=float, default=0.02)
    parser.add_argument("--leg-test-hold", type=float, default=1.0)
    parser.add_argument("--arx-max-delta", type=float, default=0.2)
    parser.add_argument("--ee-pose-source", choices=("training_fk", "sdk"), default="training_fk")
    parser.add_argument(
        "--arx-ee-pose-is-link6",
        action="store_true",
        help="Treat the SDK EE pose as link6; default converts L5_umi eef_link to link6.",
    )
    parser.add_argument(
        "--convert-arx-eef-to-link6",
        action="store_false",
        dest="arx_ee_pose_is_link6",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--action-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--se3-decay-rate",
        type=float,
        default=0.5,
        help="Reference SE3 decrease per second; training range is 0.5 to 1.4.",
    )
    parser.add_argument("--max-ee-position-regression", type=float, default=0.05)
    parser.add_argument("--max-se3-regression", type=float, default=0.10)
    parser.add_argument("--max-arm-track-error", type=float, default=0.10)
    parser.add_argument("--max-leg-track-error", type=float, default=0.25)
    parser.add_argument("--sample-latent", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-period", type=float, default=0.5)
    parser.add_argument(
        "--diagnostic-log",
        default="",
        help="Optional 50 Hz JSONL log for timing, state, action and command analysis.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.debug_ideal_observation and args.enable_output:
        raise RuntimeError("--debug-ideal-observation cannot be used with --enable-output")
    if args.keyboard_command and args.require_diffusion_command:
        raise RuntimeError(
            "--keyboard-command cannot be combined with --require-diffusion-command"
        )
    if args.gen_gripper and not args.keyboard_command:
        raise RuntimeError("--gen-gripper currently requires --keyboard-command")
    for gripper_arg_name in ("gripper_initial_width",):
        gripper_value = float(getattr(args, gripper_arg_name))
        if (
            not np.isfinite(gripper_value)
            or not 0.0 <= gripper_value <= 0.103
        ):
            raise RuntimeError(
                f"--{gripper_arg_name.replace('_', '-')} must be in [0, 0.103] m"
            )
    if (
        not np.isfinite(args.gripper_step)
        or not 0.0 < args.gripper_step <= 0.103
    ):
        raise RuntimeError("--gripper-step must be in (0, 0.103] m")
    policy_archive_name = Path(args.model_dir).expanduser().resolve().parent.name
    if policy_archive_name == "deploy2" and args.policy_ee_frame != "j6":
        raise RuntimeError(
            "deploy2 was trained on J6/link6; add --policy-ee-frame j6"
        )
    if args.require_diffusion_command and args.command_ee_frame != "j6":
        raise RuntimeError(
            "Diffusion bridge commands are already converted to J6/link6; "
            "use --command-ee-frame j6"
        )

    if args.arm_home_only or args.arm_joint_test is not None:
        if args.no_arm:
            raise RuntimeError("arm debug mode requires ARX to be enabled")
        arx_only_reader = type("ArxOnlyReader", (), {})()
        arx_only_reader.arx = ArxStateReader(args.arx_ip, args.arx_port, timeout_ms=200)
        output = RealOutput(
            arx_only_reader,
            enable_legs=False,
            enable_arm=True,
            arm_max_step=args.arm_max_step,
            max_leg_step=args.max_leg_step,
            leg_kp_scale=args.leg_kp_scale,
            arx_max_delta=args.arx_max_delta,
        )
        try:
            print(
                "arm_controller_config="
                f"{arx_only_reader.arx.request('GET_CONTROLLER_CONFIG', None, timeout_ms=1000)}"
            )
        except Exception as exc:
            print(f"arm_controller_config_error={exc}")
        try:
            print(f"arm_gain_before_reset={arx_only_reader.arx.request('GET_GAIN', None, timeout_ms=1000)}")
        except Exception as exc:
            print(f"arm_gain_before_reset_error={exc}")
        if args.arm_joint_test is not None:
            start_q, target_q, actual_q = output.test_one_arm_joint(
                args.arm_joint_test,
                args.arm_test_delta,
                rate_hz=args.rate,
                duration=1.0,
                hold=args.arm_test_hold,
            )
            print(f"arm_joint_test={args.arm_joint_test} delta={args.arm_test_delta}")
            print(f"arm_joint_start={fmt(start_q)}")
            print(f"arm_joint_target={fmt(target_q)}")
            print(f"arm_joint_actual={fmt(actual_q)}")
            print(f"arm_joint_response={fmt(actual_q - start_q)}")
            try:
                print(f"arm_gain_after_reset={arx_only_reader.arx.request('GET_GAIN', None, timeout_ms=1000)}")
            except Exception as exc:
                print(f"arm_gain_after_reset_error={exc}")
            return
        arm_q = output.reset_arm(rate_hz=args.rate)
        arm_error = DEFAULT_Q[ARM_IDS] - arm_q
        try:
            print(f"arm_gain_after_reset={arx_only_reader.arx.request('GET_GAIN', None, timeout_ms=1000)}")
        except Exception as exc:
            print(f"arm_gain_after_reset_error={exc}")
        print(f"arm_reset_actual={fmt(arm_q)}")
        print(f"arm_reset_error={fmt(arm_error)}")
        return

    odom_reader = None
    if args.use_ros2_odom:
        lidar_to_base_xyz = np.asarray(
            args.lidar_to_base_xyz, dtype=np.float64
        ).reshape(3)
        if args.lidar_to_base_z is not None:
            lidar_to_base_xyz[2] = float(args.lidar_to_base_z)
        odom_reader = Ros2OdomReader(
            topic=args.odom_topic,
            ground_height_topic=args.ground_height_topic,
            ground_reference_topic=args.ground_reference_topic,
            ground_freeze_topic=args.ground_freeze_topic,
            lidar_to_base_xyz=lidar_to_base_xyz,
            fastlio_lidar_to_imu_xyz=args.fastlio_lidar_to_imu_xyz,
        )
        odom_reader.start()
        if not odom_reader.wait(10.0):
            raise RuntimeError(
                f"No synchronized ROS2 odom ({args.odom_topic}) and ground height "
                f"reference ({args.ground_reference_topic}) received after 10.0s"
            )
        initial_odom = odom_reader.read()
        print(
            f"startup_ground_height={initial_odom['ground_height']:.4f} "
            f"startup_arm_base_height="
            f"{initial_odom['initial_arm_base_height']:.4f}"
        )
        print(
            "lidar_to_arm_base_relative="
            f"{fmt(lidar_to_base_xyz)} fastlio_lidar_to_imu="
            f"{fmt(np.asarray(args.fastlio_lidar_to_imu_xyz))}"
        )

    reader = StateReader(
        robot_ip=args.robot_ip,
        arx_ip=args.arx_ip,
        arx_port=args.arx_port,
        arx_timeout_ms=50,
        enable_arx=not args.no_arm,
    )
    if not reader.wait(3.0):
        raise RuntimeError("No TRON state/imu callbacks")

    if args.leg_home_only or args.leg_hold_current_only or args.leg_joint_test is not None:
        output = RealOutput(
            reader,
            enable_legs=True,
            enable_arm=False,
            arm_max_step=args.arm_max_step,
            max_leg_step=args.max_leg_step,
            leg_kp_scale=args.leg_kp_scale,
            arx_max_delta=args.arx_max_delta,
        )
        print(
            f"leg debug mode publish_rate={args.leg_publish_rate}Hz "
            f"leg_kp_scale={args.leg_kp_scale}"
        )
        if args.leg_hold_current_only:
            if not args.enable_output:
                raise RuntimeError(
                    "--leg-hold-current-only physically enables leg PD; "
                    "add --enable-output to confirm"
                )
            hold_q = np.asarray(
                reader.read()["tron"]["q"], dtype=np.float64
            ).reshape(-1)[:8]
            output.last_leg_cmd = hold_q.copy()
            output.start_leg_publisher(args.leg_publish_rate)
            leg_kp = KP[LEG_IDS] * args.leg_kp_scale
            leg_kd = KD[LEG_IDS]
            print(f"leg_pd_hold_target={fmt(hold_q)}")
            print(f"leg_pd_hold_kp={fmt(leg_kp)}")
            print(f"leg_pd_hold_kd={fmt(leg_kd)}")
            print(f"leg_pd_hold_cmd_mode={output.cmd.mode}")
            print("Leg PD hold active; press Ctrl+C to switch to damping and exit.")
            start_hold = time.monotonic()
            next_hold_print = 0.0
            try:
                while args.duration <= 0.0 or time.monotonic() - start_hold < args.duration:
                    if output._leg_publish_error is not None:
                        raise RuntimeError(
                            f"leg publisher failed: {output._leg_publish_error}"
                        )
                    now = time.monotonic()
                    if now >= next_hold_print:
                        next_hold_print = now + 1.0
                        actual_q = np.asarray(
                            reader.read()["tron"]["q"], dtype=np.float64
                        ).reshape(-1)[:8]
                        error_q = hold_q - actual_q
                        print(f"leg_q={fmt(actual_q)}")
                        print(f"leg_hold_error={fmt(error_q)} max={np.max(np.abs(error_q)):.4f} rad")
                    time.sleep(0.02)
            finally:
                output.stop_leg_publisher(damping=True)
                print("Leg PD hold stopped; damping command sent.")
            return
        leg_q = output.reset_legs(
            rate_hz=args.leg_publish_rate, duration=args.leg_reset_duration
        )
        print(f"leg_reset_actual={fmt(leg_q)}")
        print(f"leg_reset_error={fmt(DEFAULT_Q[LEG_IDS] - leg_q)}")
        if args.leg_home_only:
            print("leg_home_only complete; holding home and exiting.")
            output.hold_legs(
                DEFAULT_Q[LEG_IDS],
                rate_hz=args.leg_publish_rate,
                duration=args.leg_test_hold,
            )
            output.stop_legs()
            return
        actual = output.test_one_leg_joint(
            args.leg_joint_test,
            args.leg_test_delta,
            rate_hz=args.leg_publish_rate,
            duration=args.leg_reset_duration,
            hold=args.leg_test_hold,
        )
        print(f"leg_joint_test={args.leg_joint_test} delta={args.leg_test_delta}")
        print(f"leg_joint_test_actual={fmt(actual)}")
        output.stop_legs()
        return

    if args.hold_current_ee and args.hold_current_ee_base:
        raise RuntimeError("Choose only one current-EE hold mode")
    if (
        args.pre_diffusion_hold_position is not None
        and args.pre_diffusion_hold_pose is not None
    ):
        raise RuntimeError(
            "Choose only one of --pre-diffusion-hold-position and "
            "--pre-diffusion-hold-pose"
        )
    if (
        args.pre_diffusion_hold_position is not None
        or args.pre_diffusion_hold_pose is not None
    ) and not (
        args.hold_current_ee and args.require_diffusion_command
    ):
        raise RuntimeError(
            "A pre-Diffusion hold target requires both "
            "--hold-current-ee and --require-diffusion-command"
        )
    if args.command_frame == "world" and odom_reader is None and not args.hold_current_ee_base:
        raise RuntimeError("--command-frame world requires --use-ros2-odom")

    policy = ThreeOnnxPolicy(args.model_dir, sample_latent=args.sample_latent, seed=args.seed)
    output = RealOutput(
        reader,
        enable_legs=not args.no_legs,
        enable_arm=not args.no_arm,
        arm_max_step=args.arm_max_step,
        max_leg_step=args.max_leg_step,
        leg_kp_scale=args.leg_kp_scale,
        arx_max_delta=args.arx_max_delta,
    )

    print(
        f"mujoco-style real deploy output={args.enable_output} "
        f"policy_rate={args.rate}Hz leg_publish_rate={args.leg_publish_rate}Hz "
        f"duration={args.duration}s"
    )
    print(f"ONNX={Path(args.model_dir).expanduser().resolve()}")
    print(
        f"requested_command({args.command_ee_frame},{args.command_frame})="
        f"{fmt(args.command)}"
    )
    policy_ee_detail = (
        "J6/link6 origin"
        if args.policy_ee_frame == "j6"
        else f"J6/link6 +X {TRAINING_EEF_OFFSET_POS[0]:.3f} m"
    )
    print(
        f"policy_internal_ee={args.policy_ee_frame} ({policy_ee_detail})"
    )
    print(
        "physical_arm_fk=base_link->link6; "
        "policy_pose_base=TRON base_Link"
    )
    print(f"arm_max_step={args.arm_max_step} max_leg_step={args.max_leg_step} leg_kp_scale={args.leg_kp_scale} arx_max_delta={args.arx_max_delta}")
    print(f"action_scale={fmt_named(JOINT_NAMES, ACTION_SCALE)}")
    print(f"skip_leg_reset={args.skip_leg_reset}")
    print(f"ee_pose_source={args.ee_pose_source}")
    if args.ee_pose_source == "sdk":
        print(f"arx_ee_pose={'sdk_link6' if args.arx_ee_pose_is_link6 else 'L5_umi eef_link->link6'}")
    print(f"action_smoothing={args.action_smoothing}")
    print(f"se3_decay_rate={args.se3_decay_rate}")
    print(
        "safety="
        f"position_regression:{args.max_ee_position_regression} "
        f"se3_regression:{args.max_se3_regression} "
        f"arm_track:{args.max_arm_track_error} "
        f"leg_track:{args.max_leg_track_error}"
    )
    if not args.no_arm:
        try:
            arm_controller_config = reader.arx.request(
                "GET_CONTROLLER_CONFIG", None, timeout_ms=1000
            )
            print(f"arm_controller_config={arm_controller_config}")
        except Exception as exc:
            print(f"arm_controller_config_error={exc}")

    reset_before_inference = args.enable_output or args.reset_before_dry_run
    if (
        args.enable_output
        and not args.no_legs
        and args.skip_leg_reset
    ):
        # Skipping homing must not mean disabling leg control.  Hold the
        # measured pose immediately while the arm, odometry and policy finish
        # initializing; the policy loop will replace this target later.
        leg_q = np.asarray(reader.read()["tron"]["q"], dtype=np.float64).reshape(-1)[:8]
        output.last_leg_cmd = leg_q.copy()
        output.start_leg_publisher(args.leg_publish_rate)
        print(f"leg_reset_skipped_pd_hold={fmt(leg_q)}")

    if reset_before_inference and not args.no_arm:
        arm_q = output.reset_arm(rate_hz=args.rate)
        print(f"arm_reset_actual={fmt(arm_q)}")
        arm_home_error = DEFAULT_Q[ARM_IDS] - np.asarray(arm_q, dtype=np.float64)
        print(f"arm_reset_error={fmt(arm_home_error)}")
        arm_gain = reader.arx.request("GET_GAIN", None, timeout_ms=1000)
        print(f"arm_gain_after_reset={arm_gain}")
        arm_kp = np.asarray(arm_gain["kp"], dtype=np.float64).reshape(-1)[:6]
        if not np.isfinite(arm_kp).all() or np.any(arm_kp <= 0.0):
            raise RuntimeError(f"Invalid arm Kp after reset: {arm_kp}; policy output blocked")
        if np.max(np.abs(arm_home_error)) > args.max_arm_home_error:
            if not args.no_legs:
                output.stop_legs()
            raise RuntimeError(
                f"Arm did not reach home: max error {np.max(np.abs(arm_home_error)):.4f} "
                f"> {args.max_arm_home_error:.4f}; policy output blocked"
            )
    if reset_before_inference and not args.no_legs and not args.skip_leg_reset:
        leg_q = output.reset_legs(
            rate_hz=args.leg_publish_rate, duration=args.leg_reset_duration
        )
        print(f"leg_reset_actual={fmt(leg_q)}")
        leg_home_error = DEFAULT_Q[LEG_IDS] - np.asarray(leg_q, dtype=np.float64)
        print(f"leg_reset_error={fmt(leg_home_error)}")
        if np.max(np.abs(leg_home_error)) > args.max_leg_home_error:
            output.stop_legs()
            raise RuntimeError(
                f"Legs did not reach home: max error {np.max(np.abs(leg_home_error)):.4f} "
                f"> {args.max_leg_home_error:.4f}; policy output blocked"
            )
    elif (
        reset_before_inference
        and not args.no_legs
        and not args.enable_output
    ):
        leg_q = np.asarray(reader.read()["tron"]["q"], dtype=np.float64).reshape(-1)[:8]
        output.last_leg_cmd = leg_q.copy()
        print(f"leg_reset_skipped_current={fmt(leg_q)}")
    frozen_world_base = None
    if odom_reader is not None and hasattr(
        odom_reader, "freeze_ground_height_reference"
    ):
        old_stamp = odom_reader.read().get("stamp", 0.0)
        if not odom_reader.freeze_ground_height_reference(timeout_s=3.0):
            raise RuntimeError(
                "Failed to freeze the final pre-output ground/FAST-LIO reference"
            )
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            odom = odom_reader.read()
            if odom.get("ok", False) and odom.get("stamp", 0.0) != old_stamp:
                break
            time.sleep(0.01)
        else:
            raise RuntimeError("No fresh ROS2 odom frame after ground-reference freeze")
        print(
            f"output_start_ground_height={odom['ground_height']:.4f} "
            f"output_start_arm_base_height="
            f"{odom['world_arm_base'][2]:.4f}"
        )
    if args.freeze_world_base:
        if odom_reader is None:
            raise RuntimeError("--freeze-world-base requires --use-ros2-odom")
        frozen_odom = odom_reader.read()
        if not frozen_odom.get("ok", False):
            raise RuntimeError("--freeze-world-base has no valid startup odom")
        frozen_world_base = np.asarray(
            frozen_odom["tf_world_base"], dtype=np.float64
        ).reshape(4, 4).copy()
        print(
            "frozen_world_base="
            f"{fmt(np.concatenate((frozen_world_base[:3, 3], rpy_from_rot(frozen_world_base[:3, :3]))))}"
        )

    mount_fk = ArmForwardKinematics(
        TRAINING_URDF,
        base_link="base_Link",
        tip_link="base_link",
    )
    mount_pos, mount_rot = mount_fk.pose({})
    tf_policy_base_arm_base = np.eye(4, dtype=np.float64)
    tf_policy_base_arm_base[:3, :3] = mount_rot
    tf_policy_base_arm_base[:3, 3] = mount_pos

    command = np.asarray(args.command, dtype=np.float64)
    command_frame = args.command_frame
    if args.hold_current_ee or args.hold_current_ee_base:
        deadline = time.monotonic() + 3.0
        arm = None
        while time.monotonic() < deadline:
            state = reader.read()
            arm = state.get("arx")
            if isinstance(arm, dict) and arm.get("ok", False):
                break
            time.sleep(0.05)
        else:
            error = arm.get("error", "unavailable") if isinstance(arm, dict) else "unavailable"
            raise RuntimeError(f"current-EE hold has no valid ARX state after 3.0s: {error}")
        ee = np.asarray(arm["ee_pose"], dtype=np.float64).reshape(-1)[:6]
        if args.ee_pose_source == "training_fk":
            hold_tip_link = (
                "link6" if args.command_ee_frame == "j6" else "eef_link"
            )
            arm_fk = ArmForwardKinematics(
                TRAINING_URDF,
                base_link="base_link",
                tip_link=hold_tip_link,
            )
            if args.debug_ideal_observation:
                arm_q = DEFAULT_Q[ARM_IDS]
            else:
                arm_q = np.asarray(arm["q"], dtype=np.float64).reshape(-1)[:6]
            ee_pos, ee_rot = arm_fk.pose(zip(ARM_NAMES, arm_q))
        else:
            ee_pos = ee[:3].copy()
            ee_rot = rot_from_rpy(ee[3:6])
            if args.arx_ee_pose_is_link6:
                if args.command_ee_frame == "eef_link":
                    ee_pos = ee_pos + ee_rot @ TRAINING_EEF_OFFSET_POS
            else:
                if args.command_ee_frame == "j6":
                    ee_pos, ee_rot = sdk_eef_to_link6(ee_pos, ee_rot)
                else:
                    ee_pos, ee_rot = sdk_eef_to_training_eef(
                        ee_pos, ee_rot
                    )

    if args.hold_current_ee_base:
        tf_arm_ee = np.eye(4, dtype=np.float64)
        tf_arm_ee[:3, :3] = ee_rot
        tf_arm_ee[:3, 3] = ee_pos
        tf_policy_ee = tf_policy_base_arm_base @ tf_arm_ee
        command = np.concatenate(
            (
                tf_policy_ee[:3, 3],
                rpy_from_rot(tf_policy_ee[:3, :3]),
            )
        )
        command_frame = "base"
        print(f"hold_current_ee_target(TRON_base_Link)={fmt(command)}")
    elif args.hold_current_ee:
        if odom_reader is None:
            raise RuntimeError("--hold-current-ee requires --use-ros2-odom")
        odom = odom_reader.read()
        if not odom.get("ok", False):
            raise RuntimeError("--hold-current-ee requires a valid ROS2 odom frame")
        tf_world_arm_base = np.asarray(
            odom.get("tf_world_arm_base", odom["tf_world_base"]),
            dtype=np.float64,
        ).reshape(4, 4)
        ee_world_pos = (
            tf_world_arm_base[:3, 3]
            + tf_world_arm_base[:3, :3] @ ee_pos
        )
        ee_world_rot = tf_world_arm_base[:3, :3] @ ee_rot
        command = np.concatenate((ee_world_pos, rpy_from_rot(ee_world_rot)))
        command_frame = "world"
        print(f"hold_current_ee_target(world)={fmt(command)}")
        if args.pre_diffusion_hold_position is not None:
            command[:3] = np.asarray(
                args.pre_diffusion_hold_position, dtype=np.float64
            )
            print(
                "pre_diffusion_hold_target(j6,world)="
                f"{fmt(command)}"
            )
        elif args.pre_diffusion_hold_pose is not None:
            command = np.asarray(
                args.pre_diffusion_hold_pose, dtype=np.float64
            )
            print(
                "pre_diffusion_hold_target(j6,world)="
                f"{fmt(command)}"
            )

    deploy_reader = reader
    if args.debug_ideal_observation:
        deploy_reader = type(
            "IdealObservationReader",
            (),
            {"read": lambda self: idealize_observation_state(reader.read())},
        )()
        print("debug_ideal_observation=True")

    # Initialize history only after the real robot has reached its reset pose
    # and the odometry origin has produced a fresh frame.
    deploy = RealMujocoStyleDeploy(
        deploy_reader,
        policy,
        command,
        command_frame,
        command_ee_frame=args.command_ee_frame,
        policy_ee_frame=args.policy_ee_frame,
        odom_reader=odom_reader,
        arx_ee_pose_is_link6=args.arx_ee_pose_is_link6,
        ee_pose_source=args.ee_pose_source,
        action_smoothing=args.action_smoothing,
        se3_decay_rate=args.se3_decay_rate,
    )

    dt = 1.0 / max(args.rate, 1.0)
    start = time.monotonic()
    next_policy_tick = time.perf_counter()
    next_print = 0.0
    previous_loop_start = None
    best_position_error = math.inf
    best_se3 = math.inf
    safety_stopped = False
    bridge_command_id = None
    bridge_rejected_command_id = None
    bridge_timestamps = np.empty((0,), dtype=np.float64)
    bridge_world_poses = np.empty((0, 6), dtype=np.float64)
    bridge_last_command_time = None
    bridge_output_gate_state = None
    manual_command_token = None
    diagnostic_file = None
    keyboard_input = None
    gen_gripper = None
    if args.diagnostic_log:
        diagnostic_path = Path(args.diagnostic_log).expanduser().resolve()
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostic_file = diagnostic_path.open("w", encoding="utf-8", buffering=1)
        print(f"diagnostic_log={diagnostic_path}")
    if args.enable_output and args.require_diffusion_command:
        # A missing Diffusion command means "keep the startup posture", not
        # "publish an uninitialised/damping-only leg command".  Seed both
        # output paths from the measured joints before the 500 Hz publisher
        # starts.  A later chunk timeout uses the same measured-pose hold.
        initial_hold_state = reader.read()
        initial_hold_errors = output.emergency_hold(initial_hold_state)
        if initial_hold_errors:
            raise RuntimeError(
                "Failed to establish initial Diffusion hold: "
                f"{initial_hold_errors}"
            )
        print(
            "diffusion_initial_hold=measured_joint_posture "
            f"arm={('disabled' if output.last_arm_cmd is None else fmt(output.last_arm_cmd))} "
            f"legs={('disabled' if output.last_leg_cmd is None else fmt(output.last_leg_cmd))}"
        )
    if args.enable_output and not args.no_legs:
        output.start_leg_publisher(args.leg_publish_rate)
    try:
        if args.gen_gripper:
            gen_gripper = GenGripperControl(
                sdk_root=args.gripper_sdk_root,
                serial_port=args.gripper_port,
                encoder_frequency=args.gripper_encoder_frequency,
                initial_width=args.gripper_initial_width,
                feedback_timeout=args.gripper_feedback_timeout,
            )
        if args.keyboard_command:
            keyboard_input = KeyboardCommandInput(
                args.keyboard_step,
                gripper_enabled=gen_gripper is not None,
            )
        while True:
            loop_start = time.perf_counter()
            deploy.refresh_odom_snapshot()
            if frozen_world_base is not None:
                deploy.cycle_odom = copy.deepcopy(deploy.cycle_odom)
                deploy.cycle_odom["tf_world_base"] = frozen_world_base.copy()
                deploy.cycle_odom["tf_world_arm_base"] = frozen_world_base.copy()
                frozen_arm_base_pose = np.concatenate(
                    (
                        frozen_world_base[:3, 3],
                        rpy_from_rot(frozen_world_base[:3, :3]),
                    )
                )
                deploy.cycle_odom["world_base"] = frozen_arm_base_pose
                deploy.cycle_odom["world_arm_base"] = frozen_arm_base_pose
            loop_dt_ms = (
                None
                if previous_loop_start is None
                else (loop_start - previous_loop_start) * 1000.0
            )
            previous_loop_start = loop_start
            if args.ee_command_file and os.path.exists(args.ee_command_file):
                try:
                    command_file_mtime_ns = os.stat(
                        args.ee_command_file
                    ).st_mtime_ns
                    with open(args.ee_command_file, "r", encoding="utf-8") as f:
                        live_command = json.load(f)
                    if bool(live_command.get("estop", False)):
                        if args.enable_output:
                            output.stop_leg_publisher(damping=True)
                            if not args.no_arm and reader.arx is not None:
                                reader.arx.request(
                                    "SET_TO_DAMPING", None, timeout_ms=1000
                                )
                        safety_stopped = True
                        raise RuntimeError(
                            "SOFTWARE E-STOP requested by the XYZ/RPY control panel"
                        )
                    live_pose = np.asarray(
                        live_command.get("pose", []), dtype=np.float64
                    ).reshape(-1)
                    live_frame = str(live_command.get("frame", "")).strip()
                    if live_frame == "arm_base_gripper_base_link":
                        candidate_id = live_command.get("sequence")
                        source_stamp = float(live_command.get("stamp", 0.0))
                        command_age = time.time() - source_stamp
                        if (
                            source_stamp <= 0.0
                            or command_age > args.bridge_command_timeout
                            or command_age < -1.0
                        ):
                            if candidate_id != bridge_rejected_command_id:
                                print(
                                    "diffusion_wbc_chunk_rejected="
                                    f"stale age={command_age:.3f}s "
                                    f"id={candidate_id}"
                                )
                            bridge_rejected_command_id = candidate_id
                            live_frame = "stale_arm_base_gripper_base_link"
                    if live_frame == "arm_base_gripper_base_link":
                        if odom_reader is None:
                            raise RuntimeError(
                                "Diffusion/WBC bridge requires --use-ros2-odom"
                            )
                        rotation_representation = str(
                            live_command.get("rotation_representation", "")
                        )
                        if rotation_representation != "rotvec":
                            raise ValueError(
                                "Diffusion/WBC bridge requires "
                                "rotation_representation=rotvec"
                            )
                        # ``sequence`` identifies target content. ``stamp`` is
                        # only a freshness heartbeat. This lets a manual gate
                        # refresh the watchdog without reprojecting the same
                        # arm-base target through changing odometry or resetting
                        # the WBC SE(3) task every inference cycle.
                        command_id = live_command.get("sequence")
                        if command_id != bridge_command_id:
                            timestamps = np.asarray(
                                live_command.get("timestamps", []),
                                dtype=np.float64,
                            ).reshape(-1)
                            poses = np.asarray(
                                live_command.get("poses", []),
                                dtype=np.float64,
                            )
                            if poses.ndim != 2 or poses.shape[1] != 6:
                                raise ValueError(
                                    "Diffusion/WBC bridge poses must have "
                                    f"shape (N, 6), got {poses.shape}"
                                )
                            if len(timestamps) != len(poses) or len(poses) == 0:
                                raise ValueError(
                                    "Diffusion/WBC bridge requires equal, "
                                    "non-empty timestamps and poses"
                                )
                            if not np.all(np.isfinite(timestamps)) or not np.all(
                                np.isfinite(poses)
                            ):
                                raise ValueError(
                                    "Diffusion/WBC bridge received non-finite values"
                                )
                            if np.any(np.diff(timestamps) < 0.0):
                                raise ValueError(
                                    "Diffusion/WBC bridge timestamps are not ordered"
                                )
                            odom = deploy.odom_snapshot()
                            if not odom.get("ok", False):
                                raise RuntimeError(
                                    "Diffusion/WBC bridge has no valid world/base pose"
                                )
                            bridge_world_poses = np.stack(
                                [
                                    gripper_base_link_pose_to_wbc_world(
                                        pose,
                                        odom.get(
                                            "tf_world_arm_base",
                                            odom["tf_world_base"],
                                        ),
                                    )
                                    for pose in poses
                                ],
                                axis=0,
                            )
                            bridge_timestamps = timestamps
                            bridge_command_id = command_id
                            bridge_last_command_time = time.time()
                            deploy.command_frame = "world"
                            print(
                                "diffusion_wbc_chunk="
                                f"{len(bridge_world_poses)} "
                                f"id={bridge_command_id} "
                                f"first={fmt(bridge_world_poses[0])} "
                                f"last={fmt(bridge_world_poses[-1])}"
                            )
                        else:
                            bridge_last_command_time = time.time()
                    elif live_pose.size == 6 and np.all(np.isfinite(live_pose)):
                        # Apply a manual file only when the producer publishes
                        # a new version.  This lets terminal keyboard increments
                        # and the GUI coexist: an unchanged GUI target must not
                        # overwrite a later keyboard increment every 20 ms.
                        manual_token = (
                            live_command.get("stamp", command_file_mtime_ns),
                            tuple(float(value) for value in live_pose),
                        )
                        if manual_token != manual_command_token:
                            manual_command_token = manual_token
                            if not np.array_equal(live_pose, deploy.command):
                                deploy.command = live_pose.copy()
                                deploy.request_se3_reset("command_file")
                                print(
                                    f"runtime_target({deploy.command_frame})="
                                    f"{fmt(deploy.command)}"
                                )
                except RuntimeError:
                    raise
                except Exception as exc:
                    print(f"runtime_command_file_error={exc}")
            if bridge_world_poses.size:
                bridge_index = int(
                    np.searchsorted(
                        bridge_timestamps, time.time(), side="right"
                    )
                    - 1
                )
                if bridge_index >= 0:
                    bridge_index = min(
                        bridge_index, len(bridge_world_poses) - 1
                    )
                    bridge_pose = bridge_world_poses[bridge_index]
                    if not np.array_equal(bridge_pose, deploy.command):
                        deploy.command = bridge_pose.copy()
                        # Each timestamped Diffusion pose is a new WBC task
                        # target.  Keep the policy's scheduled SE(3) reference
                        # consistent with that target instead of continuing the
                        # decay that belonged to the previous chunk step.
                        deploy.request_se3_reset(
                            f"diffusion_step_{bridge_index}"
                        )
            if keyboard_input is not None:
                keyboard_delta, keyboard_pressed = (
                    keyboard_input.poll_delta()
                )
                if keyboard_pressed:
                    motion_keys = [
                        key
                        for key in keyboard_pressed
                        if key in ("W", "S", "A", "D", "R", "F")
                    ]
                    if motion_keys:
                        deploy.command[:3] += keyboard_delta
                        deploy.request_se3_reset("keyboard")
                        print(
                            f"keyboard_keys={''.join(motion_keys)} "
                            f"delta_xyz={fmt(keyboard_delta)} "
                            f"runtime_target({deploy.command_frame})="
                            f"{fmt(deploy.command)}"
                        )
                    if gen_gripper is not None:
                        for key in keyboard_pressed:
                            if key == "T":
                                gen_gripper.increment_width(
                                    args.gripper_step
                                )
                            elif key == "G":
                                gen_gripper.increment_width(
                                    -args.gripper_step
                                )
            state_read_start = time.perf_counter()
            state = deploy_reader.read()
            state_read_ms = (time.perf_counter() - state_read_start) * 1000.0
            inference_start = time.perf_counter()
            deploy.infer(state)
            inference_ms = (time.perf_counter() - inference_start) * 1000.0
            desired_q = deploy.compute_pd_targets(state)
            ee_pos_now, _ = deploy.ee_pose_base(state)
            target_pos_now, _ = deploy.target_pose_base()
            position_error = float(np.linalg.norm(target_pos_now - ee_pos_now))
            safety_reason = None
            if position_error > best_position_error + args.max_ee_position_regression:
                safety_reason = (
                    f"EE position regression {position_error:.4f} > "
                    f"best {best_position_error:.4f} + {args.max_ee_position_regression:.4f}"
                )
            elif deploy.se3_actual > best_se3 + args.max_se3_regression:
                safety_reason = (
                    f"SE3 regression {deploy.se3_actual:.4f} > "
                    f"best {best_se3:.4f} + {args.max_se3_regression:.4f}"
                )
            best_position_error = min(best_position_error, position_error)
            best_se3 = min(best_se3, deploy.se3_actual)
            if safety_reason is not None and args.enable_output:
                hold_errors = output.emergency_hold(state)
                safety_stopped = True
                detail = f"; hold errors: {hold_errors}" if hold_errors else ""
                raise RuntimeError(f"SAFETY STOP: {safety_reason}{detail}")

            bridge_gate_reason = None
            if args.require_diffusion_command:
                if bridge_last_command_time is None:
                    if (
                        args.pre_diffusion_hold_position is None
                        and args.pre_diffusion_hold_pose is None
                    ):
                        bridge_gate_reason = "waiting_for_first_fresh_chunk"
                else:
                    bridge_command_age = time.time() - bridge_last_command_time
                    if bridge_command_age > args.bridge_command_timeout:
                        bridge_gate_reason = (
                            f"chunk_timeout age={bridge_command_age:.3f}s"
                        )
            if args.require_diffusion_command:
                bridge_gate_state = (
                    "enabled"
                    if bridge_gate_reason is None
                    else f"blocked:{bridge_gate_reason}"
                )
                if bridge_gate_state != bridge_output_gate_state:
                    print(f"diffusion_output_gate={bridge_gate_state}")
                    if (
                        args.enable_output
                        and bridge_output_gate_state == "enabled"
                        and bridge_gate_reason is not None
                    ):
                        hold_errors = output.emergency_hold(state)
                        if hold_errors:
                            print(
                                "diffusion_output_gate_hold_errors="
                                f"{hold_errors}"
                            )
                    bridge_output_gate_state = bridge_gate_state

            output_reply = {}
            publish_ms = 0.0
            if args.enable_output and bridge_gate_reason is None:
                publish_start = time.perf_counter()
                output_reply = output.publish(state, desired_q)
                publish_ms = (time.perf_counter() - publish_start) * 1000.0
                deploy.record_applied_targets(state, output_reply)
                arm_reply = output_reply.get("arm")
                if isinstance(arm_reply, dict) and "cmd" in arm_reply:
                    arm_track_errors = np.asarray(
                        arm_reply["cmd"] - arm_reply["current"], dtype=np.float64
                    )
                    arm_track_index = int(np.argmax(np.abs(arm_track_errors)))
                    arm_track_error = float(abs(arm_track_errors[arm_track_index]))
                    if arm_track_error > args.max_arm_track_error:
                        safety_reason = (
                            f"arm tracking error {arm_track_error:.4f} > "
                            f"{args.max_arm_track_error:.4f}; "
                            f"joint={ARM_NAMES[arm_track_index]} "
                            f"cmd={arm_reply['cmd'][arm_track_index]:.4f} "
                            f"current={arm_reply['current'][arm_track_index]:.4f}"
                        )
                leg_reply = output_reply.get("legs")
                if safety_reason is None and isinstance(leg_reply, dict):
                    leg_track_errors = np.asarray(
                        leg_reply["cmd"] - leg_reply["current"], dtype=np.float64
                    )
                    leg_track_index = int(np.argmax(np.abs(leg_track_errors)))
                    leg_track_error = float(abs(leg_track_errors[leg_track_index]))
                    if leg_track_error > args.max_leg_track_error:
                        safety_reason = (
                            f"leg tracking error {leg_track_error:.4f} > "
                            f"{args.max_leg_track_error:.4f}; "
                            f"joint={LEG_NAMES[leg_track_index]} "
                            f"cmd={leg_reply['cmd'][leg_track_index]:.4f} "
                            f"current={leg_reply['current'][leg_track_index]:.4f}"
                        )
                if safety_reason is not None:
                    hold_errors = output.emergency_hold(state)
                    safety_stopped = True
                    detail = f"; hold errors: {hold_errors}" if hold_errors else ""
                    raise RuntimeError(f"SAFETY STOP: {safety_reason}{detail}")

            if diagnostic_file is not None:
                q_log, dq_log, tau_log = deploy.joint_state(state)
                ee_arm_pos_log, ee_arm_rot_log = (
                    deploy.ee_pose_arm_base(state)
                )
                arm_state_log = state.get("arx")
                sdk_eef_log = []
                sdk_link6_log = []
                fk_link6_log = []
                fk_sdk_position_error = None
                fk_sdk_orientation_error = None
                if (
                    isinstance(arm_state_log, dict)
                    and arm_state_log.get("ok", False)
                ):
                    arm_q_log = np.asarray(
                        arm_state_log["q"], dtype=np.float64
                    ).reshape(-1)[:6]
                    fk_link6_pos_log, fk_link6_rot_log = (
                        deploy.arm_fk.pose(zip(ARM_NAMES, arm_q_log))
                    )
                    fk_link6_log = np.concatenate(
                        [
                            fk_link6_pos_log,
                            rpy_from_rot(fk_link6_rot_log),
                        ]
                    ).tolist()
                    sdk_eef_pose_log = np.asarray(
                        arm_state_log["ee_pose"], dtype=np.float64
                    ).reshape(-1)[:6]
                    sdk_eef_rot_log = rot_from_rpy(sdk_eef_pose_log[3:6])
                    sdk_eef_log = sdk_eef_pose_log.tolist()
                    if args.arx_ee_pose_is_link6:
                        sdk_link6_pos_log = sdk_eef_pose_log[:3].copy()
                        sdk_link6_rot_log = sdk_eef_rot_log
                    else:
                        sdk_link6_pos_log, sdk_link6_rot_log = (
                            sdk_eef_to_link6(
                                sdk_eef_pose_log[:3], sdk_eef_rot_log
                            )
                        )
                    sdk_link6_log = np.concatenate(
                        [
                            sdk_link6_pos_log,
                            rpy_from_rot(sdk_link6_rot_log),
                        ]
                    ).tolist()
                    fk_sdk_position_error = float(
                        np.linalg.norm(
                            fk_link6_pos_log - sdk_link6_pos_log
                        )
                    )
                    fk_sdk_orientation_error = float(
                        R.from_matrix(
                            fk_link6_rot_log.T @ sdk_link6_rot_log
                        ).magnitude()
                    )
                ee_pos_log, ee_rot_log = deploy.ee_pose_base(state)
                command_ee_pos_log, command_ee_rot_log = (
                    deploy.policy_ee_to_command_ee_base(
                        ee_pos_log, ee_rot_log
                    )
                )
                target_pos_log, target_rot_log = deploy.target_pose_base()
                arm_reply_log = (
                    output_reply.get("arm")
                    if isinstance(output_reply, dict)
                    else None
                )
                leg_reply_log = (
                    output_reply.get("legs")
                    if isinstance(output_reply, dict)
                    else None
                )
                odom_log = deploy.odom_snapshot()
                record_wall_time = time.time()
                odom_stamp = float(odom_log.get("receive_stamp", odom_log.get("stamp", 0.0)))
                odom_source_stamp = float(odom_log.get("source_stamp", 0.0))
                odom_age_ms = (
                    (record_wall_time - odom_stamp) * 1000.0
                    if odom_log.get("ok", False) and odom_stamp > 0.0
                    else None
                )
                odom_source_age_ms = (
                    (record_wall_time - odom_source_stamp) * 1000.0
                    if odom_log.get("ok", False) and odom_source_stamp > 0.0
                    else None
                )
                ee_world_log = []
                command_ee_world_log = []
                target_world_log = []
                if odom_log.get("ok", False):
                    ee_world_pose = deploy.ee_pose_world(
                        ee_pos_log, ee_rot_log
                    )
                    command_ee_world_pose = deploy.ee_pose_world(
                        command_ee_pos_log, command_ee_rot_log
                    )
                    if ee_world_pose is not None:
                        ee_world_pos_log, ee_world_rot_log = ee_world_pose
                        ee_world_log = np.concatenate(
                            [
                                ee_world_pos_log,
                                rpy_from_rot(ee_world_rot_log),
                            ]
                        ).tolist()
                    if command_ee_world_pose is not None:
                        (
                            command_ee_world_pos_log,
                            command_ee_world_rot_log,
                        ) = command_ee_world_pose
                        command_ee_world_log = np.concatenate(
                            [
                                command_ee_world_pos_log,
                                rpy_from_rot(command_ee_world_rot_log),
                            ]
                        ).tolist()
                    if deploy.command_frame == "world":
                        target_world_log = deploy.command.tolist()
                    else:
                        target_world_pose = deploy.ee_pose_world(
                            target_pos_log, target_rot_log
                        )
                        if target_world_pose is not None:
                            target_world_pos_log, target_world_rot_log = (
                                target_world_pose
                            )
                            target_world_log = np.concatenate(
                                [
                                    target_world_pos_log,
                                    rpy_from_rot(target_world_rot_log),
                                ]
                            ).tolist()
                record = {
                    "wall_time": record_wall_time,
                    "elapsed": time.monotonic() - start,
                    "loop_dt_ms": loop_dt_ms,
                    "state_read_ms": state_read_ms,
                    "inference_ms": inference_ms,
                    "publish_ms": publish_ms,
                    "loop_work_ms": (time.perf_counter() - loop_start) * 1000.0,
                    "tron_stamp": state["tron"].get("stamp", 0),
                    "tron_counts": state["tron"].get("counts", {}),
                    "arx_stamp": (
                        state.get("arx", {}).get("stamp", 0)
                        if isinstance(state.get("arx"), dict)
                        else 0
                    ),
                    "odom_ok": bool(odom_log.get("ok", False)),
                    "odom_stamp": odom_stamp,
                    "odom_age_ms": odom_age_ms,
                    "odom_receive_stamp": odom_stamp,
                    "odom_source_stamp": odom_source_stamp,
                    "odom_source_age_ms": odom_source_age_ms,
                    # Raw FAST-LIO IMU/body pose in /Odometry coordinates.
                    "lidar_odom_raw": (
                        np.concatenate(
                            [
                                np.asarray(odom_log["raw_position"]),
                                np.asarray(odom_log["old_rpy"]),
                            ]
                        ).tolist()
                        if odom_log.get("ok", False)
                        else []
                    ),
                    # Remapped FAST-LIO translation before first-frame anchoring.
                    "lidar_odom_mapped_xyz": (
                        np.asarray(odom_log["mapped_position"]).tolist()
                        if odom_log.get("ok", False)
                        else []
                    ),
                    "lidar_origin_mapped_xyz": (
                        np.asarray(
                            odom_log["mapped_lidar_position"]
                        ).tolist()
                        if odom_log.get("ok", False)
                        else []
                    ),
                    "lidar_world": (
                        np.asarray(odom_log["world_lidar"]).tolist()
                        if odom_log.get("ok", False)
                        else []
                    ),
                    "lidar_to_base_relative_xyz": (
                        np.asarray(odom_log["lidar_to_base_xyz"]).tolist()
                        if odom_log.get("ok", False)
                        else []
                    ),
                    "lidar_to_base_rotated_world_xyz": (
                        np.asarray(
                            odom_log["rotated_lidar_to_base_xyz"]
                        ).tolist()
                        if odom_log.get("ok", False)
                        else []
                    ),
                    # ARX arm base_link after IMU->LiDAR and the rotated rigid
                    # LiDAR->arm-base transform.
                    "base_world": (
                        np.asarray(odom_log["world_base"]).tolist()
                        if odom_log.get("ok", False)
                        else []
                    ),
                    "arm_base_world": (
                        np.asarray(odom_log["world_arm_base"]).tolist()
                        if odom_log.get("ok", False)
                        else []
                    ),
                    "q": q_log.tolist(),
                    "dq": dq_log.tolist(),
                    "tau": tau_log.tolist(),
                    "ee_base": np.concatenate(
                        [ee_pos_log, rpy_from_rot(ee_rot_log)]
                    ).tolist(),
                    "ee_policy_base": np.concatenate(
                        [ee_pos_log, rpy_from_rot(ee_rot_log)]
                    ).tolist(),
                    "ee_arm_base": np.concatenate(
                        [
                            ee_arm_pos_log,
                            rpy_from_rot(ee_arm_rot_log),
                        ]
                    ).tolist(),
                    # Independent FK cross-check.  The ARX SDK reports its
                    # gripper-base eef_link, so convert that pose back to the
                    # physical J6/link6 before comparing it with training FK.
                    "ee_sdk_eef_arm_base": sdk_eef_log,
                    "ee_sdk_link6_arm_base": sdk_link6_log,
                    "ee_fk_link6_arm_base": fk_link6_log,
                    "ee_fk_sdk_position_error": fk_sdk_position_error,
                    "ee_fk_sdk_orientation_error": (
                        fk_sdk_orientation_error
                    ),
                    "ee_world": ee_world_log,
                    "command_ee_frame": deploy.command_ee_frame,
                    "policy_ee_frame": deploy.policy_ee_frame,
                    "command_ee_base": np.concatenate(
                        [
                            command_ee_pos_log,
                            rpy_from_rot(command_ee_rot_log),
                        ]
                    ).tolist(),
                    "command_ee_world": command_ee_world_log,
                    "requested_command": deploy.command.tolist(),
                    "target_base": np.concatenate(
                        [target_pos_log, rpy_from_rot(target_rot_log)]
                    ).tolist(),
                    "target_world": target_world_log,
                    "pos_error": float(
                        np.linalg.norm(target_pos_log - ee_pos_log)
                    ),
                    "orientation_error": float(
                        R.from_matrix(target_rot_log.T @ ee_rot_log).magnitude()
                    ),
                    "se3_ref": float(deploy.se3_ref),
                    "se3_actual": float(deploy.se3_actual),
                    "se3_reset_applied": bool(
                        deploy.se3_reset_applied
                    ),
                    "se3_reset_value": deploy.se3_reset_value,
                    "se3_reset_reason": (
                        deploy.se3_reset_reason
                        if deploy.se3_reset_applied
                        else ""
                    ),
                    "actor_action": deploy.actor_action.tolist(),
                    "effective_action": deploy.effective_action.tolist(),
                    "desired_q": desired_q.tolist(),
                    "last_action_obs": deploy.last_action_input.tolist(),
                    "last_action_next_obs": deploy.last_action.tolist(),
                    "last_torque_obs": deploy.last_torque.tolist(),
                    "arm_current": (
                        np.asarray(arm_reply_log.get("current", [])).tolist()
                        if isinstance(arm_reply_log, dict)
                        else []
                    ),
                    "arm_cmd": (
                        np.asarray(arm_reply_log.get("cmd", [])).tolist()
                        if isinstance(arm_reply_log, dict)
                        else []
                    ),
                    "leg_current": (
                        np.asarray(leg_reply_log.get("current", [])).tolist()
                        if isinstance(leg_reply_log, dict)
                        else []
                    ),
                    "leg_desired": (
                        np.asarray(leg_reply_log.get("desired", [])).tolist()
                        if isinstance(leg_reply_log, dict)
                        else []
                    ),
                    "leg_cmd": (
                        np.asarray(leg_reply_log.get("cmd", [])).tolist()
                        if isinstance(leg_reply_log, dict)
                        else []
                    ),
                    "leg_publisher_error": (
                        None
                        if output._leg_publish_error is None
                        else str(output._leg_publish_error)
                    ),
                }
                diagnostic_file.write(
                    json.dumps(
                        record,
                        separators=(",", ":"),
                        default=lambda value: (
                            value.item()
                            if isinstance(value, np.generic)
                            else str(value)
                        ),
                    )
                    + "\n"
                )

            now = time.monotonic()
            if now >= next_print:
                next_print = now + max(args.print_period, dt)
                _, _, ee_pos, ee_rot = deploy.raw_and_obs_ee_pose_base(state)
                target_pos, target_rot = deploy.target_pose_base()
                current_command_pos, current_command_rot = (
                    deploy.policy_ee_to_command_ee_base(ee_pos, ee_rot)
                )
                frame = "base"
                current_pos = current_command_pos
                current_rot = current_command_rot
                display_target_pos = deploy.command[:3].copy()
                display_target_rot = rot_from_rpy(deploy.command[3:6])
                if deploy.command_frame == "world":
                    ee_world = deploy.ee_pose_world(
                        current_command_pos, current_command_rot
                    )
                    if ee_world is not None:
                        frame = "world"
                        current_pos, current_rot = ee_world

                current_pose = np.concatenate(
                    [current_pos, rpy_from_rot(current_rot)]
                )
                target_pose = np.concatenate(
                    [display_target_pos, rpy_from_rot(display_target_rot)]
                )
                pos_error = np.linalg.norm(display_target_pos - current_pos)
                orientation_error = R.from_matrix(
                    display_target_rot.T @ current_rot
                ).magnitude()

                odom_print = deploy.odom_snapshot()
                if odom_print.get("ok", False):
                    print_wall_time = time.time()
                    receive_stamp = float(
                        odom_print.get("receive_stamp", odom_print.get("stamp", 0.0))
                    )
                    source_stamp = float(odom_print.get("source_stamp", 0.0))
                    callback_age_ms = (print_wall_time - receive_stamp) * 1000.0
                    source_age_text = (
                        f"{(print_wall_time - source_stamp) * 1000.0:.1f} ms"
                        if source_stamp > 0.0
                        else "unavailable"
                    )
                    print(
                        f"lidar_odom_raw="
                        f"{fmt(np.concatenate([odom_print['raw_position'], odom_print['old_rpy']]))}"
                    )
                    print(
                        f"lidar_world={fmt(odom_print['world_lidar'])} "
                        f"lidar_to_base_world="
                        f"{fmt(odom_print['rotated_lidar_to_base_xyz'])}"
                    )
                    print(
                        f"arm_base_world={fmt(odom_print['world_arm_base'])} "
                        f"callback_age={callback_age_ms:.1f} ms "
                        f"source_age={source_age_text}"
                    )
                else:
                    print("lidar_odom=unavailable")

                print(
                    f"current_{deploy.command_ee_frame}({frame})="
                    f"{fmt(current_pose)}"
                )
                print(
                    f"target_{deploy.command_ee_frame}({frame})="
                    f"{fmt(target_pose)}"
                )
                print(f"pos_error={pos_error:.4f} m")
                print(f"orientation_error={orientation_error:.4f} rad")
                print(f"action={fmt(deploy.effective_action)}")

            if args.duration > 0 and now - start >= args.duration:
                break
            # Absolute-deadline scheduling: computation time is part of the
            # 20 ms policy period instead of being added on top of it.
            next_policy_tick += dt
            sleep_s = next_policy_tick - time.perf_counter()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                # Do not accumulate lag after an overrun; resume from the
                # current time and expose the missed deadline in loop_dt_ms.
                next_policy_tick = time.perf_counter()
    finally:
        if keyboard_input is not None:
            keyboard_input.close()
        if gen_gripper is not None:
            gen_gripper.close()
        if args.enable_output and not args.no_legs:
            output.stop_leg_publisher(damping=not safety_stopped)
        elif reset_before_inference and not args.no_legs and not safety_stopped:
            output.stop_legs()
        if odom_reader is not None:
            odom_reader.stop()
        if diagnostic_file is not None:
            diagnostic_file.close()


if __name__ == "__main__":
    main()
