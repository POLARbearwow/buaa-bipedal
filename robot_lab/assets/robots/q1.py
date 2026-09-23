# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR


Q1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/q1/urdf/q1.urdf",
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=True,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            "hip_yaw_l": 0.4,
            "hip_roll_l": -0.1,
            "hip_pitch_l": -1.5,
            "knee_pitch_l": 1.0,
            "ankle_pitch_l": -1.3,
            "hip_yaw_r": -0.4,
            "hip_roll_r": 0.1,
            "hip_pitch_r": 1.5,
            "knee_pitch_r": -1.0,
            "ankle_pitch_r": 1.3,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "hip_yaw": ImplicitActuatorCfg(
            joint_names_expr=["hip_yaw_.*"], stiffness=55.0, damping=0.3, effort_limit_sim=20.0, velocity_limit_sim=30.0
        ),
        "hip_roll": ImplicitActuatorCfg(
            joint_names_expr=["hip_roll_.*"], stiffness=105.0, damping=2.5, effort_limit_sim=60.0, velocity_limit_sim=10.0
        ),
        "hip_pitch": ImplicitActuatorCfg(
            joint_names_expr=["hip_pitch_.*"], stiffness=75.0, damping=0.3, effort_limit_sim=20.0, velocity_limit_sim=30.0
        ),
        "knee": ImplicitActuatorCfg(
            joint_names_expr=["knee_pitch_.*"], stiffness=45.0, damping=0.5, effort_limit_sim=20.0, velocity_limit_sim=30.0
        ),
        "ankle": ImplicitActuatorCfg(
            joint_names_expr=["ankle_pitch_.*"], stiffness=30.0, damping=0.25, effort_limit_sim=20.0, velocity_limit_sim=30.0
        ),
    },
)
