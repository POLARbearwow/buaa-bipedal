"""Check whether the BUAA Q1 target pose is a stable physics equilibrium.

The test holds the configured default joint pose with the articulation's
position controller, starts from noisy initial states, and reports the actual
settled trunk height and tilt.  A low trunk height alone is never considered
success: a fallen robot has a large tilt and/or root velocity.

Examples:
    python scripts/settle_test_buaa_q1.py --headless
    python scripts/settle_test_buaa_q1.py --headless --num_episodes 20 --settle_seconds 3
    python scripts/settle_test_buaa_q1.py --num_episodes 8 --joint_noise 0.15 --tilt_noise_deg 8
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

import local_assets  # isort: skip


parser = argparse.ArgumentParser(description="Run noisy-init settle tests for BUAA Q1.")
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--settle_seconds", type=float, default=3.0)
parser.add_argument("--joint_noise", type=float, default=0.10, help="Uniform joint-position noise in radians.")
parser.add_argument("--tilt_noise_deg", type=float, default=5.0, help="Uniform roll/pitch noise in degrees.")
parser.add_argument("--root_z_noise", type=float, default=0.03, help="Uniform initial root-z noise in meters.")
parser.add_argument("--tail_fraction", type=float, default=0.20, help="Fraction of samples used for settle statistics.")
parser.add_argument("--max_tilt_deg", type=float, default=15.0, help="Pass threshold for tail 95th-percentile tilt.")
parser.add_argument("--max_root_speed", type=float, default=0.25, help="Pass threshold for tail 95th-percentile root speed.")
parser.add_argument("--num_steps", type=int, default=0, help="Optional hard cap; 0 derives it from settle_seconds.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.kit_args = local_assets.use_local_assets(args_cli.kit_args)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot_lab.assets.robots.buaa_q1 import BUAA_Q1_CFG


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    x = torch.tensor(values, dtype=torch.float64)
    return float(torch.quantile(x, q))


def tilt_degrees(root_quat_w: torch.Tensor) -> float:
    """Angle between the robot's local +Z axis and world +Z, in degrees."""
    local_z = quat_apply(root_quat_w.unsqueeze(0), torch.tensor([[0.0, 0.0, 1.0]], device=root_quat_w.device))[0]
    cosine = torch.clamp(local_z[2], -1.0, 1.0)
    return float(torch.rad2deg(torch.acos(cosine)))


def main() -> None:
    if args_cli.num_episodes < 1 or args_cli.settle_seconds <= 0:
        raise ValueError("num_episodes must be >= 1 and settle_seconds must be > 0")
    if not 0.0 < args_cli.tail_fraction <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1]")

    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.001, device=args_cli.device))
    sim.set_camera_view([2.2, -2.2, 1.5], [0.0, 0.0, 0.45])
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robot = Articulation(BUAA_Q1_CFG.replace(prim_path="/World/Q1"))
    sim.reset()
    dt = sim.get_physics_dt()
    steps = args_cli.num_steps or max(1, math.ceil(args_cli.settle_seconds / dt))
    tail_start = max(0, int(steps * (1.0 - args_cli.tail_fraction)))

    target_joint_pos = robot.data.default_joint_pos.clone()
    target_root = robot.data.default_root_state.clone()
    print("[INFO] Joint order:", list(robot.data.joint_names))
    print("[INFO] Target joint pose (rad):", target_joint_pos[0].detach().cpu().tolist())
    print(f"[INFO] dt={dt:.6f}, steps={steps}, tail={steps - tail_start} samples")

    passed = 0
    all_heights: list[float] = []
    for episode in range(args_cli.num_episodes):
        # Reinitialize from the same target with controlled, reproducible noise.
        root_pose = target_root[:, :7].clone()
        state_device = target_joint_pos.device
        roll_pitch = (2.0 * torch.rand(2, device=state_device) - 1.0) * math.radians(args_cli.tilt_noise_deg)
        noise_quat = quat_from_euler_xyz(roll_pitch[0:1], roll_pitch[1:2], torch.zeros(1, device=state_device))[0]
        root_pose[:, 3:7] = quat_mul(noise_quat.unsqueeze(0), root_pose[:, 3:7])
        root_pose[:, 2] += (2.0 * torch.rand(1, device=state_device) - 1.0) * args_cli.root_z_noise
        joint_pos = target_joint_pos + (2.0 * torch.rand_like(target_joint_pos) - 1.0) * args_cli.joint_noise
        zero_root_vel = torch.zeros_like(target_root[:, 7:])
        zero_joint_vel = torch.zeros_like(robot.data.default_joint_vel)
        robot.write_root_pose_to_sim(root_pose)
        robot.write_root_velocity_to_sim(zero_root_vel)
        robot.write_joint_state_to_sim(joint_pos, zero_joint_vel)
        robot.set_joint_position_target(target_joint_pos)
        robot.write_data_to_sim()
        robot.reset()

        heights: list[float] = []
        tilts: list[float] = []
        root_speeds: list[float] = []
        for step in range(steps):
            robot.set_joint_position_target(target_joint_pos)
            robot.write_data_to_sim()
            sim.step()
            robot.update(dt)
            if step >= tail_start:
                heights.append(float(robot.data.root_pos_w[0, 2]))
                tilts.append(tilt_degrees(robot.data.root_quat_w[0]))
                root_speeds.append(float(torch.linalg.vector_norm(robot.data.root_lin_vel_w[0])))

        h_med = percentile(heights, 0.50)
        h_min, h_max = min(heights), max(heights)
        tilt_p95 = percentile(tilts, 0.95)
        speed_p95 = percentile(root_speeds, 0.95)
        ok = tilt_p95 <= args_cli.max_tilt_deg and speed_p95 <= args_cli.max_root_speed
        passed += int(ok)
        all_heights.extend(heights)
        print(
            f"[EP {episode + 1:02d}/{args_cli.num_episodes}] {'PASS' if ok else 'FAIL'} "
            f"z_med={h_med:.4f}m z_range=[{h_min:.4f}, {h_max:.4f}] "
            f"tilt_p95={tilt_p95:.2f}deg speed_p95={speed_p95:.3f}m/s"
        )

    print("[SUMMARY]")
    print(f"  pass_rate={passed}/{args_cli.num_episodes} ({passed / args_cli.num_episodes:.1%})")
    print(f"  measured_tail_root_z_median={percentile(all_heights, 0.50):.4f} m")
    print("  Note: use this measured z as the revision-specific standing height; do not hard-code it across URDF revisions.")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
