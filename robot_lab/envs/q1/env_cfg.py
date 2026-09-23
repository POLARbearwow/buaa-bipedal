# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

import robot_lab.mdp as mdp
from robot_lab.assets.robots.q1 import Q1_CFG
from robot_lab.envs.velocity_env_cfg import LocomotionVelocityRoughEnvCfg


@configclass
class Q1FlatEnvCfg(LocomotionVelocityRoughEnvCfg):
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

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = Q1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
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
        self.actions.joint_pos.joint_names = self.joint_names
        self.actions.joint_pos.scale = 0.5
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}

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
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.flat_orientation_l2.weight = -0.2
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.base_height_tracking.weight = 1.0
        self.rewards.base_height_tracking.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.base_orientation_tracking.weight = 1.5
        self.rewards.base_orientation_tracking.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.joint_torques_l2.weight = -1.5e-7
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.joint_torques_above_threshold.weight = -0.1
        self.rewards.joint_torques_above_threshold.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.joint_torques_above_threshold.params["threshold_ratio"] = 0.7
        self.rewards.joint_acc_l2.weight = -1.25e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.joint_pos_limits.weight = -0.5
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.joint_pos_penalty.weight = -1.0
        self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.joint_pos_tracking_l2.weight = -0.2
        self.rewards.joint_pos_tracking_l2.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_yaw_frame_exp
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_z_world_exp
        # Gait rewards (tune this block together).
        # self.rewards.feet_air_time.weight = 1.0  # legacy reward disabled
        self.rewards.phase_conditioned_contact.weight = 1.0
        self.rewards.phase_conditioned_contact.params["cycle_time"] = 0.667
        self.rewards.phase_conditioned_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = -1.0
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.params["target_height"] = 0.10
        self.rewards.feet_slide.weight = -0.2
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.upward.weight = 2.0
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 0.7)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        if self.__class__.__name__ == "Q1FlatEnvCfg":
            self.disable_zero_weight_rewards()
