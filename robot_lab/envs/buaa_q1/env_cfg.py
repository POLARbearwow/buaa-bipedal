# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Flat-ground velocity task for the BUAA Q1-v1 model."""

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg

import robot_lab.mdp as mdp
from robot_lab.assets.robots.buaa_q1 import BUAA_Q1_CFG
from robot_lab.envs.velocity_env_cfg import LocomotionVelocityRoughEnvCfg


@configclass
class BuaaQ1FlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for the BUAA Q1-v1 robot on flat terrain.

    This is intentionally a complete task definition rather than a subclass
    of the original Q1 task. Keeping the settings here makes this task
    independently editable while retaining the current Q1 training setup.
    The robot asset itself starts with all ten actuated joints at zero.
    """

    base_link_name = "base_link"
    foot_link_name = ".*ankle_pitch_[lr]"
    joint_names = [
        "hip_yaw_l",
        "hip_roll_l",
        "hip_pitch_l",
        "knee_pitch_l",
        "ankle_pitch_l",
        "hip_yaw_r",
        "hip_roll_r",
        "hip_pitch_r",
        "knee_pitch_r",
        "ankle_pitch_r",
    ]
    # Resolve action joints in articulation order, matching the robot asset.
    action_joint_names = [
        "hip_yaw_l",
        "hip_yaw_r",
        "hip_roll_l",
        "hip_roll_r",
        "hip_pitch_l",
        "hip_pitch_r",
        "knee_pitch_l",
        "knee_pitch_r",
        "ankle_pitch_l",
        "ankle_pitch_r",
    ]
    
    # The "standing" command threshold, shared by the command generator deadzone
    # (robot_lab/mdp/commands.py), the AMP reward gate and every reward term that gates on the
    # command (the loop at the end of ``__post_init__``).  A command is therefore either treated
    # as "no command" or as a real walking command, never something in between.
    stand_command_threshold = 0.05
    amp_reward_command_name = "base_velocity"

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = BUAA_Q1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.num_envs = 4096
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None
        # Match the 2026-09-18 16:26 run's serialized phase-observation config.
        # The function defaults retain the same runtime values.
        for obs_cfg in (self.observations.policy, self.observations.critic):
            obs_cfg.phase.params.pop("command_threshold", None)
            obs_cfg.phase.params.pop("phase_offset_l", None)
            obs_cfg.phase.params.pop("phase_offset_r", None)
        # The RSL-RL environment wrapper clips raw actions to +/-100 before
        # this term scales them and adds the default joint pose.
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=self.action_joint_names,
            scale=0.25,
            use_default_offset=True,
            clip=None,
            preserve_order=True,
        )

        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = ["^(?!base_link$).*"]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # Start with nominal dynamics to match the RoboTamer training setup.
        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_rigid_body_mass_others = None
        self.events.randomize_com_positions = None
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_push_robot = None

        # Match RoboTamer's reset perturbations before introducing stronger randomization.
        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (-0.2, 0.2),
        }
        # self.events.randomize_reset_base.params["velocity_range"] = {
        #     "x": (-0.5, 0.5),
        #     "y": (-0.5, 0.5),
        #     "z": (-0.2, 0.2),
        #     "roll": (-0.2, 0.2),
        #     "pitch": (-0.2, 0.2),
        #     "yaw": (-0.2, 0.2),
        # }
        self.events.randomize_reset_base.params["velocity_range"] = {
            "x": (-0.0, 0.0),
            "y": (-0.0, 0.0),
            "z": (-0.0, 0.0),
            "roll": (-0.0, 0.0),
            "pitch": (-0.0, 0.0),
            "yaw": (-0.0, 0.0),
        }

        self.rewards.is_terminated.weight = -200.0
        
        self.rewards.stand_still.weight = -2.0
        # Penalize vertical base motion (body-frame z velocity) to discourage
        # bouncing while preserving the commanded horizontal velocity task.
        self.rewards.lin_vel_z_l2.weight = -1.5
        self.rewards.ang_vel_xy_l2.weight = -0.3
        self.rewards.flat_orientation_l2.weight = -0.5
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.base_height_tracking.weight = 1.0
        self.rewards.base_height_tracking.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.base_orientation_tracking.weight = 0.5
        self.rewards.base_orientation_tracking.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.joint_torques_l2.weight = -3e-4 #-3e-4
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.joint_torques_above_threshold.weight = -0.0025
        self.rewards.joint_torques_above_threshold.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.joint_torques_above_threshold.params["threshold_ratio"] = 0.7
        # Joint-velocity penalty, split by command magnitude. Environments whose
        # command norm is <= ``stand_command_threshold`` are treated as "standing"
        # and get their own weight, separate from the moving environments.
        # self.rewards.joint_vel_l2.weight = -0.0
        self.rewards.joint_vel_l2_stand.weight = -0.05
        self.rewards.joint_vel_l2_stand.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.joint_vel_l2_moving.weight = -1e-4
        self.rewards.joint_vel_l2_moving.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.joint_acc_l2.weight = -1.25e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.joint_pos_limits.weight = -0.005
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.joint_pos_penalty.weight = -0.0
        self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.joint_pos_tracking_l2.weight = -0.0
        self.rewards.joint_pos_tracking_l2.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.action_rate_l2.weight = -0.001
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_yaw_frame_exp
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        # self.rewards.track_ang_vel_z_exp.weight = 0.5

        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_z_world_exp
        # Gait rewards (tune this block together).
        # self.rewards.feet_air_time.weight = 1.0  # legacy reward disabled
        self.rewards.phase_conditioned_contact.weight = 1.0
        self.rewards.phase_conditioned_contact.params["cycle_time"] = 0.667
        self.rewards.phase_conditioned_contact.params.pop("air_ratio", None)
        self.rewards.phase_conditioned_contact.params.pop("phase_offset_l", None)
        self.rewards.phase_conditioned_contact.params.pop("phase_offset_r", None)
        self.rewards.phase_conditioned_contact.params.pop("transition_width", None)
        # Keep the phase reward's left/right ordering explicit.
        self.rewards.phase_conditioned_contact.params["sensor_cfg"].body_names = [
            ".*ankle_pitch_[lr]",
        ]
        # This term was not part of the 2026-09-18 16:26 baseline.
        self.rewards.fly.weight = -0.5
        self.rewards.fly.params["threshold"] = 1.0
        self.rewards.fly.params["sensor_cfg"].body_names = [
            "ankle_pitch_l",
            "ankle_pitch_r",
        ]
        self.rewards.feet_height.weight = -1.0
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.params["target_height"] = 0.7
        self.rewards.feet_slide.weight = -0.2
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        
        self.rewards.feet_y_distance.weight = -4.0
        self.rewards.feet_y_distance.params["asset_cfg"].body_names = ["ankle_pitch_l", "ankle_pitch_r"]
        self.rewards.upward.weight = 0.2
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (-0.4, 0.7)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)

        # Keep every command gated reward term (stand_still, joint_vel_l2_stand/moving,
        # joint_pos_penalty, phase_conditioned_contact, ...) on the same threshold as the
        # command generator deadzone, so that no term can disagree about what "standing"
        # means.  Terms without a ``command_threshold`` parameter are skipped.
        for term_name in dir(self.rewards):
            term_params = getattr(getattr(self.rewards, term_name, None), "params", None)
            if isinstance(term_params, dict) and "command_threshold" in term_params:
                term_params["command_threshold"] = self.stand_command_threshold

        if self.__class__.__name__ == "BuaaQ1FlatEnvCfg":
            self.disable_zero_weight_rewards()
