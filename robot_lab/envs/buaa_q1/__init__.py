"""BUAA Q1 robot environment registration."""

import gymnasium as gym


_TASK_KWARGS = {
    "env_cfg_entry_point": "robot_lab.envs.buaa_q1.env_cfg:BuaaQ1FlatEnvCfg",
    "rsl_rl_cfg_entry_point": "robot_lab.envs.buaa_q1.agent_cfg:BuaaQ1FlatPPORunnerCfg",
    "rsl_rl_amp_cfg_entry_point": "robot_lab.envs.buaa_q1.agent_cfg:BuaaQ1FlatAMPRunnerCfg",
}

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Buaa-Q1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs=_TASK_KWARGS,
)

gym.register(
    id="buaa-q1-flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs=_TASK_KWARGS,
)
