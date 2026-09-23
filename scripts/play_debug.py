"""Run the normal play script and periodically dump Q1 action/torque state.

This wrapper intentionally leaves ``scripts/play.py`` unchanged.  It hooks the
environment returned by ``gym.make`` and prints values after every environment
step for environment 0.  Angles are radians and torques are N*m.

Example:
    python scripts/play_debug.py --task buaa-q1-flat --num_envs 1 \
        --checkpoint <checkpoint> --debug-interval 10
"""

from __future__ import annotations

import argparse
import runpy
import types

import gymnasium as gym
import torch


def _install_debug_hook(interval: int, action_threshold: float) -> None:
    original_make = gym.make

    def debug_make(*args, **kwargs):
        env = original_make(*args, **kwargs)
        original_step = env.step
        state = {"step": 0}

        def debug_step(self, actions):
            result = original_step(actions)
            state["step"] += 1
            step = state["step"]

            if step % interval != 0:
                return result

            try:
                robot = self.unwrapped.scene["robot"]
                action_term = self.unwrapped.action_manager.get_term("joint_pos")
                raw = action_term.raw_actions[0].detach().cpu()
                processed = action_term.processed_actions[0].detach().cpu()
                # Match the action term's configured safety clip.
                if action_term.cfg.clip is not None:
                    display_action = raw.clamp(*action_term.cfg.clip)
                else:
                    display_action = raw
                target = robot.data.joint_pos_target[0].detach().cpu()
                qpos = robot.data.joint_pos[0].detach().cpu()
                computed = robot.data.computed_torque[0].detach().cpu()
                applied = robot.data.applied_torque[0].detach().cpu()
                names = list(robot.data.joint_names)

                # State/reward diagnostics needed to distinguish a bad stand
                # gate from a policy that is actively tracking a walking prior.
                command = self.unwrapped.command_manager.get_command("base_velocity")[0].detach().cpu()
                root_lin = robot.data.root_lin_vel_b[0].detach().cpu()
                root_ang = robot.data.root_ang_vel_b[0].detach().cpu()
                gravity = robot.data.projected_gravity_b[0].detach().cpu()
                print(
                    "[DEBUG] command(vx,vy,wz)="
                    f"{command.tolist()} norm={float(command.norm()):.4f} "
                    f"base_lin={root_lin.tolist()} base_ang={root_ang.tolist()} "
                    f"gravity={gravity.tolist()}"
                )
                try:
                    sensor = self.unwrapped.scene.sensors["contact_forces"]
                    foot_ids = self.unwrapped.scene["robot"].find_bodies(
                        [".*ankle_pitch_[lr]"], preserve_order=True
                    )[0]
                    contact_force = sensor.data.net_forces_w[0, foot_ids].detach().cpu()
                    foot_vel = robot.data.body_lin_vel_w[0, foot_ids, :2].detach().cpu()
                    print(
                        "[DEBUG] feet contact_fz="
                        f"{contact_force[:, 2].tolist()} foot_vel_xy="
                        f"{foot_vel.tolist()}"
                    )
                except (AttributeError, KeyError, IndexError, TypeError, RuntimeError) as exc:
                    print(f"[DEBUG] Could not read contact diagnostics: {exc}")
                reward_manager = getattr(self.unwrapped, "reward_manager", None)
                step_reward = getattr(reward_manager, "_step_reward", None)
                if step_reward is not None:
                    names_rm = getattr(reward_manager, "_term_names", [])
                    values = step_reward[0].detach().cpu().tolist()
                    print("[DEBUG] reward terms (raw manager values):")
                    print("  " + " ".join(f"{n}={v:.4g}" for n, v in zip(names_rm, values)))

                print(f"\n[DEBUG] step={step}")
                print("joint                    action(clamped)  target(rad)  qpos(rad)  computed(Nm)  applied(Nm)")
                for i, name in enumerate(names):
                    # The action term may address only a subset/order of joints.
                    action_i = float("nan")
                    processed_i = float("nan")
                    ids = getattr(action_term, "_joint_ids", None)
                    if isinstance(ids, slice):
                        action_i = float(display_action[i])
                    elif ids is not None:
                        matches = [j for j, joint_id in enumerate(ids) if int(joint_id) == i]
                        if matches:
                            action_i = float(display_action[matches[0]])
                    print(
                        f"{name:28s} {action_i:10.4f} {float(target[i]):12.5f} "
                        f"{float(qpos[i]):10.5f} {float(computed[i]):13.5f} {float(applied[i]):11.5f}"
                    )
                max_action = float(torch.nan_to_num(raw, nan=0.0).abs().max())
                if max_action >= action_threshold:
                    print(f"[DEBUG] WARNING: max raw action={max_action:.5f}")
            except (AttributeError, KeyError, IndexError, TypeError) as exc:
                print(f"[DEBUG] Could not read articulation debug state: {exc}")
            return result

        env.step = types.MethodType(debug_step, env)
        return env

    gym.make = debug_make


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--debug-interval", type=int, default=10)
    parser.add_argument("--action-threshold", type=float, default=100.0)
    debug_args, remaining = parser.parse_known_args()
    if debug_args.debug_interval < 1:
        parser.error("--debug-interval must be >= 1")

    _install_debug_hook(debug_args.debug_interval, debug_args.action_threshold)

    # play.py owns all Isaac Lab/RSL-RL argument parsing and simulator startup.
    import sys

    sys.argv = [sys.argv[0], *remaining]
    runpy.run_path("scripts/play.py", run_name="__main__")


if __name__ == "__main__":
    main()
