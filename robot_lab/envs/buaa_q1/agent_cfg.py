# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from robot_lab.envs.q1.agent_cfg import Q1FlatAMPRunnerCfg, Q1FlatPPORunnerCfg


@configclass
class BuaaQ1FlatPPORunnerCfg(Q1FlatPPORunnerCfg):
    """Same PPO settings as Q1, with an independent experiment directory."""

    experiment_name = "buaa_q1_flat"
    clip_actions = 100.0


@configclass
class BuaaQ1FlatAMPRunnerCfg(Q1FlatAMPRunnerCfg):
    """AMP configuration for the BUAA Q1 task."""

    experiment_name = "buaa_q1_flat_amp"
    max_iterations = 8000
    clip_actions = 100.0
    amp_motion_files = ["datasets/motion_amp_expert/0007_walking001_q1_amp.json"]
