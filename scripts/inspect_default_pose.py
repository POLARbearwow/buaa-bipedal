"""Inspect the Q1 default pose in Isaac Sim without creating an RL environment.

Usage:
    python scripts/inspect_default_pose.py
    python scripts/inspect_default_pose.py --headless --num_steps 1000
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Load Q1 at its default pose in Isaac Sim.")
parser.add_argument(
    "--num_steps",
    type=int,
    default=0,
    help="Number of physics steps to run; 0 keeps the GUI open until it is closed.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

# Allow the script to run before or after an editable package installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "source" / "robot_lab"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from robot_lab.assets.robots.q1 import Q1_CFG


def main() -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.001, device=args_cli.device))
    sim.set_camera_view([2.2, -2.2, 1.4], [0.0, 0.0, 0.4])

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robot = Articulation(Q1_CFG.replace(prim_path="/World/Q1"))
    sim.reset()

    root_state = robot.data.default_root_state.clone()
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(robot.data.default_joint_vel)
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)
    robot.write_data_to_sim()
    robot.reset()

    print("[INFO] Q1 loaded at its configured default pose.")
    print("[INFO] Joint order:", robot.data.joint_names)
    print("[INFO] Joint positions (rad):", joint_pos[0].tolist())

    step_count = 0
    while simulation_app.is_running() and (args_cli.num_steps == 0 or step_count < args_cli.num_steps):
        robot.set_joint_position_target(joint_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
        step_count += 1


if __name__ == "__main__":
    main()
    simulation_app.close()
