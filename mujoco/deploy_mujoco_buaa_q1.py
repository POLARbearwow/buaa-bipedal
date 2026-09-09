"""Deploy the Isaac Lab BUAA Q1 flat policy in MuJoCo.

The policy interface matches ``BuaaQ1FlatEnvCfg`` exactly.  No torque logging
or terrain transfer is performed; MuJoCo supplies a flat plane and the Xbox
joystick supplies ``(vx, vy, wz)`` commands.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml

HERE = Path(__file__).resolve().parent
# The repository directory is also named ``mujoco``. Remove that local path
# before importing the installed MuJoCo package, otherwise Python can resolve
# this directory as a namespace package when launched from the repo root.
sys.path = [p for p in sys.path if Path(p or ".").resolve() not in {HERE, HERE.parent}]
import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402

sys.path.insert(0, str(HERE.parent / "M1-HIMLoco" / "mujoco"))
from joystick_interface import JoystickInterface  # noqa: E402


DEFAULT_CONFIG = HERE / "configs" / "buaa_q1_flat.yaml"


def _quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qw, qv = q[0], q[1:4]
    return v * (2.0 * qw * qw - 1.0) - np.cross(qv, v) * qw * 2.0 + qv * np.dot(qv, v) * 2.0


def _resolve(path: str, config_path: Path) -> str:
    path = path.replace("{PROJECT_ROOT}", str(HERE.parent))
    path = path.replace("{CONFIG_DIR}", str(config_path.parent))
    return str(Path(path).expanduser().resolve())


class DeployBuaaQ1:
    # Isaac Lab prints this articulation order (preserve_order=False).
    ARTICULATION_JOINTS = [
        "hip_yaw_l", "hip_yaw_r", "hip_roll_l", "hip_roll_r", "hip_pitch_l",
        "hip_pitch_r", "knee_pitch_l", "knee_pitch_r", "ankle_pitch_l", "ankle_pitch_r",
    ]
    # JointPositionActionCfg order (preserve_order=True).
    CONFIGURED_JOINTS = [
        "hip_yaw_l", "hip_roll_l", "hip_pitch_l", "knee_pitch_l", "ankle_pitch_l",
        "hip_yaw_r", "hip_roll_r", "hip_pitch_r", "knee_pitch_r", "ankle_pitch_r",
    ]

    def __init__(self, cfg: dict, onnx_path: str | None = None):
        self.cfg = cfg
        config_path = Path(cfg["config_path"])
        self.xml_path = _resolve(cfg["xml_path"], config_path)
        self.policy_path = str(Path(onnx_path or _resolve(cfg["policy_path"], config_path)).resolve())
        if not os.path.isfile(self.xml_path):
            raise FileNotFoundError(f"MJCF scene not found: {self.xml_path}")
        if not os.path.isfile(self.policy_path):
            raise FileNotFoundError(f"ONNX policy not found: {self.policy_path}")

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = float(cfg["simulation_dt"])
        self.session = ort.InferenceSession(self.policy_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_dim = self._last_dim(self.session.get_inputs()[0].shape)
        self.output_dim = self._last_dim(self.session.get_outputs()[0].shape)
        if self.input_dim <= 0:
            self.input_dim = 39

        self.obs_order = str(cfg.get("obs_order", "articulation")).lower()
        self.action_order = str(cfg.get("action_order", "configured")).lower()
        orders = {"articulation": self.ARTICULATION_JOINTS, "configured": self.CONFIGURED_JOINTS}
        if self.obs_order not in orders or self.action_order not in orders:
            raise ValueError("obs_order/action_order must be 'articulation' or 'configured'")
        self.OBS_JOINTS = list(orders[self.obs_order])
        self.ACTION_JOINTS = list(orders[self.action_order])

        # YAML vectors are stored in the documented configured/articulation
        # orders; remap them whenever a different runtime order is selected.
        def remap(values, source_names, target_names):
            by_name = dict(zip(source_names, np.asarray(values, dtype=np.float32)))
            return np.asarray([by_name[name] for name in target_names], dtype=np.float32)

        self.default_angles = remap(cfg["default_angles"], self.CONFIGURED_JOINTS, self.ACTION_JOINTS)
        self.kp = remap(cfg["kp"], self.CONFIGURED_JOINTS, self.ACTION_JOINTS)
        self.kd = remap(cfg["kd"], self.CONFIGURED_JOINTS, self.ACTION_JOINTS)
        self.torque_limits = remap(cfg["torque_limits"], self.CONFIGURED_JOINTS, self.ACTION_JOINTS)
        self.obs_default_angles = remap(cfg["obs_default_angles"], self.ARTICULATION_JOINTS, self.OBS_JOINTS)
        # JointPositionToLimitsActionCfg semantics: normalized actions are
        # mapped to the articulation's soft joint limits, not added to the
        # default pose with a fixed radian scale.
        self.soft_joint_pos_limit_factor = float(cfg.get("soft_joint_pos_limit_factor", 0.90))
        self.joint_vel_scale = float(cfg["joint_vel_scale"])
        self.ang_vel_scale = float(cfg["ang_vel_scale"])
        self.cmd = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(10, dtype=np.float32)
        self.obs = np.zeros(57, dtype=np.float32)
        self.qpos = np.zeros(10, dtype=np.float32)
        self.qvel = np.zeros(10, dtype=np.float32)
        self.obs_qpos = np.zeros(10, dtype=np.float32)
        self.obs_qvel = np.zeros(10, dtype=np.float32)
        self.quat = np.array([1, 0, 0, 0], dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.gravity = np.array([0, 0, -1], dtype=np.float32)
        self.action_qpos_ids = [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self.ACTION_JOINTS]
        self.action_qvel_ids = [self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self.ACTION_JOINTS]
        self.obs_qpos_ids = [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self.OBS_JOINTS]
        self.obs_qvel_ids = [self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self.OBS_JOINTS]
        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{n}_motor") for n in self.ACTION_JOINTS]
        if any(x < 0 for x in self.action_qpos_ids + self.action_qvel_ids + self.obs_qpos_ids + self.obs_qvel_ids + self.actuator_ids):
            raise RuntimeError("MJCF is missing one or more BUAA Q1 joints/actuators")
        action_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.ACTION_JOINTS
        ]
        hard_limits = np.asarray(self.model.jnt_range[action_joint_ids], dtype=np.float32)
        limit_center = hard_limits.mean(axis=1)
        limit_half_range = 0.5 * (hard_limits[:, 1] - hard_limits[:, 0])
        self.soft_joint_limits = np.column_stack(
            (limit_center - self.soft_joint_pos_limit_factor * limit_half_range,
             limit_center + self.soft_joint_pos_limit_factor * limit_half_range)
        )
        self.joystick = JoystickInterface(cfg["joystick_device"], float(cfg["joystick_max_v_x"]), float(cfg["joystick_max_v_y"]), float(cfg["joystick_max_omega"]))
        self.control_decimation = int(cfg["control_decimation"])
        self.debug_log = bool(cfg.get("debug_log", False))
        self.debug_interval = max(1, int(cfg.get("debug_interval", 1)))
        self.step_count = 0
        self.control_count = 0
        self.last_target = self.default_angles.copy()
        self.last_computed_torque = np.zeros(10, dtype=np.float32)
        self.reset()
        # Match Isaac Lab play timing: infer the first action immediately
        # after reset, before the first physics/control interval.
        self._read_command()
        self._read_state()
        self._make_observation()
        self._infer_action()
        print(f"[BUAA Q1] MJCF: {self.xml_path}")
        print(f"[BUAA Q1] ONNX: {self.policy_path} (input={self.input_dim}, output={self.output_dim})")
        print(f"[BUAA Q1] observation order ({self.obs_order}): {self.OBS_JOINTS}")
        print(f"[BUAA Q1] action order ({self.action_order}): {self.ACTION_JOINTS}")
        for name, actuator_id in zip(self.ACTION_JOINTS, self.actuator_ids):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            print(
                f"[BUAA Q1] actuator {name}: axis={self.model.jnt_axis[joint_id].tolist()} "
                f"gear={self.model.actuator_gear[actuator_id, 0]:.1f} "
                f"ctrl={self.model.actuator_ctrlrange[actuator_id].tolist()}"
            )

    @staticmethod
    def _last_dim(shape) -> int:
        return int(shape[-1]) if isinstance(shape[-1], (int, np.integer)) else -1

    def reset(self) -> None:
        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "default_pos")
        if key >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key)
        else:
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[2] = 0.35
        mujoco.mj_forward(self.model, self.data)
        self.cmd.fill(0)
        self.last_action.fill(0)

    def _read_state(self) -> None:
        self.qpos[:] = self.data.qpos[self.action_qpos_ids]
        self.qvel[:] = self.data.qvel[self.action_qvel_ids]
        self.obs_qpos[:] = self.data.qpos[self.obs_qpos_ids]
        self.obs_qvel[:] = self.data.qvel[self.obs_qvel_ids]
        self.quat = self.data.qpos[3:7].astype(np.float32)
        try:
            self.ang_vel = self.data.sensor("body_gyro").data.astype(np.float32).copy()
        except KeyError:
            self.ang_vel = self.data.qvel[3:6].astype(np.float32)
        self.gravity = _quat_rotate_inverse(self.quat, np.array([0, 0, -1], dtype=np.float32))

    def _read_command(self) -> None:
        if self.joystick.available:
            self.cmd[:] = self.joystick.get_command()

    def _make_observation(self) -> None:
        self.obs[0:3] = self.ang_vel * self.ang_vel_scale
        self.obs[3:6] = self.gravity
        self.obs[6:9] = self.cmd
        self.obs[9:19] = self.obs_qpos - self.obs_default_angles
        self.obs[19:29] = self.obs_qvel * self.joint_vel_scale
        self.obs[29:39] = self.last_action
        # Isaac Lab's flat task has 39 policy values when base_lin_vel and scan
        # are disabled. Keep a fixed 57-vector for exported checkpoints that
        # retain the base-linear-velocity slot; zero padding is explicit.
        if self.input_dim == 39:
            self.policy_input = self.obs[:39]
        else:
            self.policy_input = np.zeros(self.input_dim, dtype=np.float32)
            self.policy_input[:39] = self.obs[:39]

    def _infer_action(self) -> None:
        output = self.session.run([self.output_name], {self.input_name: self.policy_input[None, : self.input_dim]})[0]
        action = np.asarray(output, dtype=np.float32).reshape(-1)
        self.last_action.fill(0)
        self.last_action[: min(10, action.size)] = action[:10]

    def _apply_pd(self) -> None:
        normalized_action = np.clip(self.last_action, -1.0, 1.0)
        target = self.soft_joint_limits[:, 0] + 0.5 * (normalized_action + 1.0) * (
            self.soft_joint_limits[:, 1] - self.soft_joint_limits[:, 0]
        )
        torque = self.kp * (target - self.qpos) - self.kd * self.qvel
        self.last_target[:] = target
        self.last_computed_torque[:] = torque
        torque = np.clip(torque, -self.torque_limits, self.torque_limits)
        self.data.ctrl[self.actuator_ids] = torque

    def _print_debug(self) -> None:
        """Print one control-step snapshot; angles are rad and torques N*m."""
        if not self.debug_log or self.control_count % self.debug_interval != 0:
            return
        applied = np.asarray(self.data.actuator_force[self.actuator_ids], dtype=np.float32)
        ctrl = np.asarray(self.data.ctrl[self.actuator_ids], dtype=np.float32)
        print(f"\n[DEBUG] control_step={self.control_count} command={self.cmd.tolist()}")
        print("joint                    action(raw)  action(clamped)  target(rad)  qpos(rad)  computed(Nm)  ctrl(Nm)  applied(Nm)")
        for i, name in enumerate(self.ACTION_JOINTS):
            print(
                f"{name:24s} {self.last_action[i]:11.5f} {np.clip(self.last_action[i], -1, 1):15.5f} "
                f"{self.last_target[i]:11.5f} {self.qpos[i]:10.5f} {self.last_computed_torque[i]:13.5f} "
                f"{ctrl[i]:9.5f} {applied[i]:11.5f}"
            )
        print(
            f"[DEBUG] max_abs_action={np.max(np.abs(self.last_action)):.5f} "
            f"max_abs_computed={np.max(np.abs(self.last_computed_torque)):.5f} "
            f"max_abs_applied={np.max(np.abs(applied)):.5f}"
        )

    def run(self) -> None:
        duration = float(self.cfg.get("simulation_duration", 1e12))
        start = time.time()
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running() and time.time() - start < duration:
                tick = time.time()
                self._read_command()
                self._read_state()
                self._apply_pd()
                mujoco.mj_step(self.model, self.data)
                self.step_count += 1
                if self.step_count % self.control_decimation == 0:
                    self._read_state()
                    # At this point qpos and actuator_force correspond to the
                    # action applied during the preceding control interval.
                    self._print_debug()
                    self._make_observation()
                    self._infer_action()
                    self.control_count += 1
                viewer.sync()
                delay = self.model.opt.timestep - (time.time() - tick)
                if delay > 0:
                    time.sleep(delay)

    def close(self) -> None:
        self.joystick.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--onnx", type=Path, default=None)
    parser.add_argument("--obs-order", choices=("articulation", "configured"), default=None,
                        help="joint order for observation position/velocity values")
    parser.add_argument("--action-order", choices=("articulation", "configured"), default=None,
                        help="joint order for policy actions and PD/actuator mapping")
    parser.add_argument("--debug", action="store_true", help="print action, joint state, and torque diagnostics")
    parser.add_argument("--debug-interval", type=int, default=None,
                        help="print diagnostics every N control steps (default: config or 1)")
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if args.obs_order is not None:
        cfg["obs_order"] = args.obs_order
    if args.action_order is not None:
        cfg["action_order"] = args.action_order
    if args.debug:
        cfg["debug_log"] = True
    if args.debug_interval is not None:
        cfg["debug_interval"] = args.debug_interval
    cfg["config_path"] = str(args.config.resolve())
    deploy = DeployBuaaQ1(cfg, str(args.onnx) if args.onnx else None)
    try:
        deploy.run()
    finally:
        deploy.close()


if __name__ == "__main__":
    main()
