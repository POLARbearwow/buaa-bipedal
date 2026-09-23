#!/usr/bin/env python3
"""Convert Q1 free-joint qpos motion data into the 26-D AMP JSON format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


JOINT_NAMES = [
    "hip_yaw_l", "hip_roll_l", "hip_pitch_l", "knee_pitch_l", "ankle_pitch_l",
    "hip_yaw_r", "hip_roll_r", "hip_pitch_r", "knee_pitch_r", "ankle_pitch_r",
]
FOOT_NAMES = ["ankle_pitch_l", "ankle_pitch_r"]
DEFAULT_JOINT_POS = np.array([-0.36, 0.18, 2.35, -0.33, -0.77, 0.36, -0.18, -2.35, 0.33, 0.77])


def convert(input_path: Path, output_path: Path, xml_path: Path | None) -> None:
    source = np.load(input_path, allow_pickle=True)
    xml = xml_path or Path(str(source["robot_xml"].item()))
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    qpos = np.asarray(source["qpos"], dtype=np.float64)
    names = [str(name) for name in source["robot_joint_names"].tolist()]
    if qpos.ndim != 2 or qpos.shape[1] != 17 or names != JOINT_NAMES:
        raise ValueError(f"expected qpos [N,17] and the standard Q1 joint order, got {qpos.shape}, {names}")

    joint_qpos_adr = [model.jnt_qposadr[model.joint(name).id] for name in JOINT_NAMES]
    foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in FOOT_NAMES]
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    # The environment exposes joint position relative to its default pose.
    # Keep the expert convention identical to Q1AmpVecEnvWrapper.
    joint_pos = qpos[:, 7:] - DEFAULT_JOINT_POS
    joint_vel = np.gradient(np.unwrap(joint_pos, axis=0), axis=0) * float(source["fps"].item())
    frames = []
    for i, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        root_pos = data.xpos[root_id]
        root_rot = data.xmat[root_id].reshape(3, 3)
        feet_base = np.concatenate([root_rot.T @ (data.xpos[foot_id] - root_pos) for foot_id in foot_ids])
        frame = np.concatenate([joint_pos[i], joint_vel[i], feet_base])
        frames.append(frame.tolist())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "LoopMode": "Wrap",
        "FrameDuration": 1.0 / float(source["fps"].item()),
        "MotionWeight": 1.0,
        "Frames": frames,
    }
    output_path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {output_path}: {len(frames)} frames x {len(frames[0])} values")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--xml", type=Path, default=None)
    args = parser.parse_args()
    convert(args.input, args.output, args.xml)
