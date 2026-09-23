"""Q1 robot environment registration."""

import gymnasium as gym


gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Q1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "robot_lab.envs.q1.env_cfg:Q1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": "robot_lab.envs.q1.agent_cfg:Q1FlatPPORunnerCfg",
        "rsl_rl_amp_cfg_entry_point": "robot_lab.envs.q1.agent_cfg:Q1FlatAMPRunnerCfg",
    },
)
