# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents


_TASK_KWARGS = {
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:BuaaQ1FlatEnvCfg",
    "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BuaaQ1FlatPPORunnerCfg",
}

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Buaa-Q1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs=_TASK_KWARGS,
)

# Short alias matching the model/task name used in local training commands.
gym.register(
    id="buaa-q1-flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs=_TASK_KWARGS,
)
