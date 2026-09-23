"""Isaac Lab adapter exposing the AMP transition interface required by RSL-RL."""

from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply_inverse
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class Q1AmpVecEnvWrapper(RslRlVecEnvWrapper):
    """RSL-RL wrapper with a 26-D Q1 AMP observation.

    The layout is joint position (10), joint velocity (10), and the two ankle
    positions (3 + 3), all expressed using the Q1 action joint order and the
    base-link frame.

    The wrapper also gates the AMP style reward: as soon as the commanded
    velocity is (almost) zero the environment is considered standing and the
    AMP reward is disabled (see :meth:`get_amp_reward_gate`).
    """

    amp_obs_dim = 26

    def __init__(self, env, clip_actions=None):
        super().__init__(env, clip_actions=clip_actions)
        self._amp_env = env.unwrapped
        self._amp_reset_env_ids = torch.empty(0, dtype=torch.long, device=self.device)
        self._amp_robot = self._amp_env.scene["robot"]

        joint_names = list(getattr(self._amp_env.cfg, "joint_names", []))
        if len(joint_names) != 10:
            raise ValueError(f"Q1 AMP expects 10 configured joints, got {len(joint_names)}")
        self._amp_joint_ids = self._amp_robot.find_joints(joint_names, preserve_order=True)[0]
        foot_names = ["ankle_pitch_l", "ankle_pitch_r"]
        self._amp_foot_ids = self._amp_robot.find_bodies(foot_names, preserve_order=True)[0]
        if len(self._amp_foot_ids) != 2:
            raise ValueError(f"Q1 AMP expects both ankle bodies, got ids {self._amp_foot_ids}")

        cfg = self._amp_env.cfg
        self._amp_reward_command_name = str(getattr(cfg, "amp_reward_command_name", "base_velocity"))
        self._amp_reward_command_threshold = float(getattr(cfg, "stand_command_threshold", 0.05))

    @property
    def step_dt(self):
        return self._amp_env.step_dt

    @property
    def reset_env_ids(self):
        return self._amp_reset_env_ids

    def get_amp_obs_for_expert_trans(self) -> torch.Tensor:
        robot = self._amp_robot
        joint_pos = robot.data.joint_pos[:, self._amp_joint_ids]
        joint_vel = robot.data.joint_vel[:, self._amp_joint_ids]
        foot_pos_w = robot.data.body_pos_w[:, self._amp_foot_ids, :]
        foot_pos_base = quat_apply_inverse(
            robot.data.root_quat_w[:, None, :].expand(-1, foot_pos_w.shape[1], -1),
            foot_pos_w - robot.data.root_pos_w[:, None, :],
        )
        amp_obs = torch.cat(
            (
                joint_pos - robot.data.default_joint_pos[:, self._amp_joint_ids],
                joint_vel,
                foot_pos_base.flatten(start_dim=1),
            ),
            dim=-1,
        )
        if amp_obs.shape[-1] != self.amp_obs_dim:
            raise RuntimeError(f"Q1 AMP observation has shape {amp_obs.shape}, expected last dim 26")
        return amp_obs

    def get_amp_reward_gate(self) -> torch.Tensor:
        """Per-environment multiplier that disables the AMP reward while standing.

        The AMP prior is a walking motion, so an environment that is commanded to
        stand still must not be pulled towards that prior.  Environments whose command
        norm is below ``stand_command_threshold`` get a gate of ``0.0``; all others get
        ``1.0``.  The norm covers the linear and the yaw command, matching the command
        threshold used by the reward terms.  Only the AMP part of the reward is scaled by
        the gate, the task reward is left untouched.
        """
        command = self._amp_env.command_manager.get_command(self._amp_reward_command_name)
        is_standing = torch.norm(command, dim=-1) < self._amp_reward_command_threshold
        return (~is_standing).to(dtype=command.dtype)

    def step(self, actions):
        obs, rewards, dones, infos = super().step(actions)
        self._amp_reset_env_ids = torch.nonzero(dones, as_tuple=False).flatten()
        return obs, rewards, dones, infos
