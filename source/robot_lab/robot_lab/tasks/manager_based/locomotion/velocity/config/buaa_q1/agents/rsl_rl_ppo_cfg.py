# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity.config.q1.agents.rsl_rl_ppo_cfg import Q1FlatPPORunnerCfg


@configclass
class BuaaQ1FlatPPORunnerCfg(Q1FlatPPORunnerCfg):
    """Same PPO settings as Q1, with an independent experiment directory."""

    experiment_name = "buaa_q1_flat"
