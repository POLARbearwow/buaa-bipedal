# Q1 的 AMP + RL 迁移指南

本文说明如何把 TienKung-Lab 的 AMP（Adversarial Motion Priors）+ PPO
训练流程迁移到 Q1 training。Q1 当前已经可以使用普通 PPO，但 AMP 需要额外的
RSL-RL 模块、环境接口和专家动作数据。

本仓库已经提供第一版迁移实现：本地 `rsl_rl/` 是 TienKung-Lab 的定制包，
`robot_lab/rl/amp_wrapper.py` 提供 Q1 的 26 维 AMP 状态，训练脚本按 agent
入口自动选择普通 PPO 或 AMP runner。

## 1. 当前两个项目的差异

Q1 training 使用 IsaacLab 的 Manager-Based 环境和外部安装的 RSL-RL：

```text
gym.make -> RslRlVecEnvWrapper -> rsl_rl.runners.OnPolicyRunner -> PPO
```

TienKung-Lab 使用仓库内定制的 RSL-RL：

```text
自定义 VecEnv -> AmpOnPolicyRunner -> AMPPPO -> PPO + Discriminator
```

Q1 的 `setup.py` 已固定依赖 `rsl-rl-lib==2.3.1`，便于验证与
TienKung-Lab 的本地 RSL-RL 是否兼容。注意，TienKung-Lab 的 2.3.1 是包含
AMP 扩展的定制版本，不等同于 PyPI 上的普通 PPO 版本。

## 2. 推荐的迁移顺序

按以下顺序实施，每一步都先验证再进入下一步：

1. 安装并确认 RSL-RL 2.3.1 的实际导入路径。
2. 复制 AMP 相关算法、Runner、判别器、loader 和 replay buffer。
3. 保持 Q1 的普通 PPO 环境不变，先增加 AMP 状态接口。
4. 生成 Q1 专家动作文件并检查维度、关节顺序和数值范围。
5. 将 AMPLoader 改为支持 Q1 的状态维度。
6. 将 Q1 runner 切换为 `AmpOnPolicyRunner`，运行小规模 AMP 训练。

## 3. 安装本地 RSL-RL

在用于 IsaacLab 的 Python 环境中执行：

```bash
cd /home/niu/Desktop/TienKung-Lab/rsl_rl
python -m pip install -e . --no-deps
```

确认当前加载的包不是其他位置的同名包：

```bash
python -c "import rsl_rl, importlib.metadata as m; print(rsl_rl.__file__); print(m.version('rsl-rl-lib'))"
```

输出版本应为 `2.3.1`，路径应指向 TienKung-Lab 的 `rsl_rl` 目录。

Q1 项目重新安装时使用：

```bash
cd /home/niu/Desktop/q1_training
python -m pip install -e . --no-deps
```

q1 的 AMP agent 入口是 `rsl_rl_amp_cfg_entry_point`；普通 PPO 仍使用
`rsl_rl_cfg_entry_point`。安装后确认本地包优先：

```bash
cd /home/niu/Desktop/q1_training
python -c "import rsl_rl; print(rsl_rl.__file__)"
```

## 4. 需要迁移的 RSL-RL 文件

从 TienKung-Lab 迁移以下文件到 Q1 使用的 RSL-RL 包中：

```text
rsl_rl/algorithms/amp_ppo.py
rsl_rl/runners/amp_on_policy_runner.py
rsl_rl/modules/discriminator.py
rsl_rl/storage/replay_buffer.py
rsl_rl/utils/motion_loader.py
rsl_rl/utils/motion_loader_for_display.py   # 仅需要动作播放时迁移
rsl_rl/utils/utils.py                       # Normalizer
```

并检查这些导出文件：

```text
rsl_rl/algorithms/__init__.py
rsl_rl/runners/__init__.py
rsl_rl/modules/__init__.py
rsl_rl/storage/__init__.py
rsl_rl/utils/__init__.py
```

`AMPPPO` 同时优化 PPO policy/value 网络和 discriminator；
`AmpOnPolicyRunner` 负责采集 AMP 状态、计算 AMP reward 和组织 expert batch；
`Discriminator` 区分专家状态转移与策略状态转移；`ReplayBuffer` 保存策略产生的
AMP transitions；`AMPLoader` 从专家文件随机采样相邻帧。

## 5. 为 Q1 增加 AMP 状态

在 Q1 环境或 Q1 专用 wrapper 中实现：

```python
def get_amp_obs_for_expert_trans(self):
    ...
```

推荐 Q1 使用 26 维 AMP 状态：

```text
10 维关节位置
10 维关节速度
3 维左脚相对 base_link 的位置
3 维右脚相对 base_link 的位置
```

脚端位置需要从世界坐标转换到 base 坐标系：

```python
foot_relative = foot_world - root_world
foot_base = quat_apply(quat_conjugate(root_quat), foot_relative)
```

关节位置必须明确采用绝对位置或相对于默认姿态的位置。策略 AMP 状态和专家
AMP 数据必须采用同一种定义。

Manager-Based 环境通过 `RslRlVecEnvWrapper` 使用时，还要确认 wrapper 能提供：

```text
get_amp_obs_for_expert_trans()
step_dt
num_envs
device
reset_env_ids
```

如果 wrapper 没有这些属性，增加 `Q1AmpVecEnvWrapper` 转发到
`env.unwrapped`，尤其是终止环境 ID，因为 Runner 会为终止 transition 处理
AMP next state。

## 6. 修改 AMPLoader 的维度

TienKung-Lab 当前 AMPLoader 把维度写死为：

```text
20 joint positions + 20 joint velocities + 12 end-effector values = 52
```

Q1 不能直接使用这个固定布局。应将 `JOINT_POS_SIZE`、`JOINT_VEL_SIZE` 和
`END_EFFECTOR_POS_SIZE` 改成构造参数，或创建 Q1 专用 loader：

```text
joint_pos_size = 10
joint_vel_size = 10
end_effector_pos_size = 6
observation_dim = 26
discriminator input_dim = 52
```

不要简单截断 TienKung 的 52 维文件；每一列的语义必须和
`get_amp_obs_for_expert_trans()` 完全一致。

## 7. 生成 Q1 专家数据

Q1 expert 文件每一帧应为：

```text
[q1_joint_pos(10), q1_joint_vel(10), left_foot_pos_base(3), right_foot_pos_base(3)]
```

文件可使用 TienKung-Lab 的 JSON 结构：

```json
{
  "LoopMode": "Wrap",
  "FrameDuration": 0.033,
  "MotionWeight": 1.0,
  "Frames": [[...], [...]]
}
```

数据处理要求：

- 关节顺序必须与 Q1 的 action joint order 一致；
- 角度单位使用弧度；
- 速度使用相邻帧差分除以 `dt`；
- 足端位置必须使用 Q1 的实际 URDF/body 定义计算；
- 删除或修复 NaN、Inf、关节限位越界和速度尖峰；
- `FrameDuration`、速度计算的 `dt` 和环境 AMP transition 时间应一致。

生成后先检查：

```python
assert len(frames[0]) == 26
```

并播放动作确认脚底没有穿地、漂浮或左右脚交换。

## 8. 修改 Q1 Agent 配置

在 `robot_lab/envs/q1/agent_cfg.py` 中增加 AMP 字段：

```python
algorithm = RslRlPpoAlgorithmCfg(
    class_name="AMPPPO",
    ...,
)

runner_class_name = "AmpOnPolicyRunner"
amp_reward_coef = 0.3
amp_motion_files = [
    "robot_lab/envs/q1/datasets/motion_amp_expert/walk.txt"
]
amp_num_preload_transitions = 200000
amp_task_reward_lerp = 0.7
amp_discr_hidden_dims = [512, 256, 128]
```

`amp_task_reward_lerp` 越大，任务 reward 占比越高；越小，动作风格约束越强。
初次验证建议使用较小的 `amp_num_preload_transitions` 和环境数量，确认内存占用。

当前配置已经定义为 `Q1FlatAMPRunnerCfg`。将 `amp_motion_files` 填成专家 JSON
文件路径，然后用 `--agent rsl_rl_amp_cfg_entry_point` 启动；没有专家文件时
训练脚本会直接报错。

## 9. 修改训练入口

Q1 当前训练入口使用：

```python
from rsl_rl.runners import OnPolicyRunner
runner = OnPolicyRunner(...)
```

需要改为按配置选择 Runner：

```python
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner

runner_class = eval(agent_cfg.runner_class_name)
runner = runner_class(
    env,
    agent_cfg.to_dict(),
    log_dir=log_dir,
    device=agent_cfg.device,
)
```

同时将 `RslRlVecEnvWrapper` 替换为或扩展为能转发 AMP 接口的 Q1 wrapper。

## 10. 验证清单

先用普通 PPO 验证 Q1，再启用 AMP。每个阶段检查：

```text
普通 PPO：环境 reset、动作维度、观测维度和 reward 正常
AMP 状态：shape 为 [num_envs, 26]，无 NaN/Inf
专家数据：每帧 26 维，关节顺序和脚端坐标正确
Loader：expert state 和 next state 均为 [batch, 26]
判别器：输入为 [batch, 52]，输出为 [batch, 1]
训练：expert_d 接近 +1，policy_d 接近 -1，AMP reward 非零
```

以当前 BUAA Q1 任务和已转换专家动作为例，第一次 AMP 运行：

```bash
python scripts/train.py --task buaa-q1-flat \
  --agent rsl_rl_amp_cfg_entry_point \
  --headless --num_envs 64 --max_iterations 10
```

确认流程稳定后，再逐步增大环境数量、预加载 transition 数量和训练迭代数。

在启动仿真前先验证专家文件：

```bash
python scripts/validate_amp_motion.py path/to/q1_walk.json
```

该检查要求每帧严格为 26 维，并检查 `FrameDuration`、`LoopMode` 和 NaN/Inf。

## 11. 常见不兼容原因

- Q1 使用的是外部官方 `OnPolicyRunner`，没有 `AMPPPO` 和 AMP loader；
- Q1 Manager-Based wrapper 没有 Runner 需要的 AMP 方法或 `reset_env_ids`；
- AMPLoader 仍按 TienKung 的 52 维布局读取 Q1 数据；
- 专家数据关节顺序与 action 顺序不一致；
- 专家数据使用绝对角度，而策略 AMP 状态使用相对默认姿态角度；
- `FrameDuration` 与环境步长不一致，导致速度和 transition 插值尺度错误；
- 实际导入了另一个位置的 `rsl_rl`，而不是 TienKung-Lab 的定制版本。

## 12. 当前交接状态（2026-09-15）

本节是当前实现的交接说明。后续 agent 应以本节为当前状态，不要只参考前面的
初始迁移方案。

### 12.1 已完成的代码

已经将 TienKung-Lab 的定制 RSL-RL 复制到 q1_training 的本地 `rsl_rl/` 包，
包括：

```text
rsl_rl/algorithms/amp_ppo.py
rsl_rl/runners/amp_on_policy_runner.py
rsl_rl/modules/discriminator.py
rsl_rl/storage/replay_buffer.py
rsl_rl/utils/motion_loader.py
rsl_rl/utils/utils.py
```

q1_training 已新增 Q1 AMP wrapper：

```text
robot_lab/rl/amp_wrapper.py
```

wrapper 输出 26 维 AMP observation，布局固定为：

```text
[10 relative joint positions,
 10 joint velocities,
 3 left ankle position in base_link frame,
 3 right ankle position in base_link frame]
```

关节角是相对于 IsaacLab 中 `robot.data.default_joint_pos` 的相对角度；脚端位置
是相对于 `base_link` 的 `(x, y, z)`，不是脚端到 base 的单一欧氏距离。

训练脚本 `scripts/train.py` 已按 agent 配置选择 runner，并将仓库根目录放到
`sys.path` 前面，避免误加载环境中安装的官方 RSL-RL：

```text
普通 PPO: OnPolicyRunner + RslRlVecEnvWrapper
AMP PPO:  AmpOnPolicyRunner + Q1AmpVecEnvWrapper
```

### 12.2 当前任务入口

当前主要任务是：

```text
buaa-q1-flat
```

该任务已经注册两个 agent 入口：

```text
rsl_rl_cfg_entry_point      -> BuaaQ1FlatPPORunnerCfg
rsl_rl_amp_cfg_entry_point  -> BuaaQ1FlatAMPRunnerCfg
```

AMP 配置文件：

```text
robot_lab/envs/buaa_q1/agent_cfg.py
```

当前 AMP 配置的关键值：

```text
algorithm.class_name       = AMPPPO
amp_reward_coef            = 0.3
amp_task_reward_lerp       = 0.7
amp_num_preload_transitions = 20000
amp_joint_pos_size         = 10
amp_joint_vel_size         = 10
amp_end_effector_pos_size  = 6
learning_rate              = 3e-4
schedule                   = fixed
```

普通 PPO 配置不应被 AMP 配置修改。普通 PPO 已经验证约 80 次迭代可以学会站立。

### 12.3 专家数据状态

原始专家文件：

```text
0007_Walking001_stageii.npz_smplx_buaa_q1_medium.npz
```

NPZ 字段中：

```text
qpos              shape = [3936, 17]
                  7 维 floating base + 10 维关节位置
fps               = 120
robot_joint_names = 10 个 Q1 关节，顺序与 Q1 AMP 定义一致
```

已经使用 `scripts/convert_q1_npz_to_amp.py` 转换为：

```text
datasets/motion_amp_expert/0007_walking001_q1_amp.json
```

转换结果：

```text
3936 frames
26 values per frame
FrameDuration = 0.008333 seconds
max joint velocity = 6.718 rad/s
```

转换过程使用 NPZ 中的 MuJoCo XML 做 forward kinematics，计算左右
`ankle_pitch_l`、`ankle_pitch_r` 相对于 `base_link` 的位置。关节角同时减去
当前 BUAA Q1 默认姿态，使其与 Q1 wrapper 的 AMP 状态定义一致。

当前 loader 会根据 IsaacLab 的环境步长采样 expert transition。BUAA Q1 的环境
步长是：

```text
physics dt = 0.001
decimation = 15
environment step dt = 0.015 seconds
```

专家帧间隔是 `0.008333` 秒，和环境步长不同是允许的，AMPLoader 会进行时间插值。
后续 agent 仍需确认 MuJoCo XML 和 IsaacLab URDF 的关节零位、脚端几何位置一致。

### 12.4 已验证内容

已验证：

```text
本地 rsl_rl 可以导入
AMPPPO / Discriminator / ReplayBuffer 核心接口可以运行
AMP 源码通过 compileall
专家 JSON 通过 26 维、NaN/Inf、FrameDuration 检查
buaa-q1-flat 可以解析 BuaaQ1FlatAMPRunnerCfg
AMP 训练可以进入 IsaacLab 环境创建阶段
```

AMP 最小启动命令：

```bash
conda activate zgl_isaaclab
cd /home/niu/Desktop/q1_training
python scripts/train.py \
  --task buaa-q1-flat \
  --agent rsl_rl_amp_cfg_entry_point \
  --headless \
  --num_envs 64 \
  --max_iterations 10
```

### 12.5 当前观察到的训练问题

AMP 训练已经能运行，但训练效果不如普通 PPO。最近一次 TensorBoard 日志显示：

```text
illegal_contact = 1.0
time_out = 0.0
mean episode length approximately 270 steps
mean amp policy prediction approximately -0.7
mean amp expert prediction approximately +0.7
```

这说明：

1. 判别器已经能够区分策略 transition 和 expert transition，AMP 判别器没有完全
   崩溃；
2. 策略大约在 `270 * 0.015 = 4.05` 秒后发生非法接触或摔倒；
3. 当前首要问题是 AMP policy 的早期稳定性、默认姿态和 reward 组合，而不是
   expert JSON 的基本维度错误；
4. 不应仅根据 `Mean reward` 判断问题，因为该 reward 已包含 AMP reward 和 task
   reward 的组合。

TensorBoard 中需要重点查看：

```text
Episode_Termination/illegal_contact
Episode_Termination/time_out
Train/mean_episode_length
Reward/mean_raw_amp_reward_step
Reward/mean_task_reward_step
Reward/amp_reward_gate_mean
Loss/amp_policy_pred
Loss/amp_expert_pred
Loss/learning_rate
```

### 12.6 已完成的 AMP 修正

已完成以下修正：

1. 修复 Q1 wrapper 中 root quaternion 对脚端 batch 的错误扩展方式。
2. 修复 AMP normalizer：使用原始 AMP state 更新均值方差，不再使用归一化后的
   state 更新统计量。
3. 修复 discriminator gradient penalty，使其使用和 discriminator forward 相同
   的归一化坐标系。
4. 将 AMP PPO 的初始学习率改为 `3e-4`，并使用 fixed schedule，避免 adaptive
   schedule 在早期频繁把学习率降到 `1e-5`。
5. 新增原始 AMP reward 和 task reward 的 TensorBoard 日志。
6. AMP reward 按 command 门控：`Q1AmpVecEnvWrapper.get_amp_reward_gate()` 返回
   每个环境一个 0/1 系数，被判为 standing 的环境 AMP reward 被置零，只保留
   task reward。这样原地站立的环境不会被 walking 专家动作先验拖着一直动。门控值
   本身记在 `Reward/amp_reward_gate_mean`（1 表示全部环境都在使用 AMP reward，
   0 表示全部被门控）。
7. command threshold 统一为三个分量（线速度 + yaw）的普通范数
   `torch.norm(command) <= command_threshold`，reward term 和 AMP 门控都用这一条，
   并且 buaa-q1 里所有这类 term 的阈值都被设成同一个值
   `stand_command_threshold = 0.05`（`robot_lab/envs/buaa_q1/env_cfg.py` 的
   `__post_init__` 末尾遍历 rewards，凡是有 `command_threshold` 参数的 term 都改成它），
   与命令生成器的死区阈值一致。base 配置
   （`robot_lab/envs/velocity_env_cfg.py`）现在把所有带该门控的 term 的
   `command_threshold` 都显式写成默认值 `0.1`，其它任务行为不变。
   命令生成器的死区（`UniformThresholdVelocityCommand._resample_command`，当前阈值
   `0.05`）只作用于线速度部分，`ang_vel_z` 不会被清零，所以如果以后把
   `commands.base_velocity.ranges.ang_vel_z` 恢复到 `(-0.5, 0.5)`，需要重新考虑
   “静止”的定义：一个范数要求 `|wz| < 0.05`，会让 yaw 命令主导静止判定
   （`rel_standing_envs` 之外的环境几乎都不会被判为静止）。届时的选择是收窄
   `ang_vel_z` 范围、把 yaw 也纳入死区，或让 yaw 使用单独的容忍度。

8. 新增 AMP 相似度诊断日志，用来判断 policy 的步态和专家动作到底有多像。除了
   已有的 `Loss/amp_policy_pred`、`Loss/amp_expert_pred` 和
   `Reward/mean_raw_amp_reward_step`，`AMPPPO.update()` 每个 iteration 还会统计
   policy 与专家数据集的 AMP 特征分布差异（`AMPPPO.amp_similarity_metrics()`），
   runner 把它写到 TensorBoard 的 `AMP/` 组，同时在终端每次迭代打印三行：

   ```text
   AMP obs mean dist: ... (0=expert) [joint_pos ..., joint_vel ..., end_effector_pos ...]
   AMP obs std ratio: ... [joint_pos ..., joint_vel ..., end_effector_pos ...]
   AMP discriminator acc: ... (0.5=indistinguishable)
   ```

   - `obs_mean_dist`：policy 特征均值与专家特征均值之差，用专家的标准差归一化后取平均。
     `0` 表示和专家统计完全一致，`1` 表示平均差了一个专家标准差。专家参考是整份
     `0007_walking001_q1_amp.json`（`amp_data.all_trajectories_full`），不是随机采样
     minibatch，所以不受 batch 噪声影响。
   - `obs_std_ratio`：policy 特征标准差与专家标准差的比值。`< 1` 说明 policy 的关节
     幅度比参考动作小（动作偏保守），`> 1` 说明幅度更大。
   - `disc_accuracy`：判别器把 policy 样本判成 policy、专家样本判成专家的比例。
     `0.5` 表示判别器完全分不出来（风格已经对齐），`1.0` 表示一眼可分。对抗训练下
     这个值长期贴近 1.0 属正常，重点看它是否随训练下降。
   - 注意：AMP replay buffer 里存的是所有环境的 transition（包括被门控关掉 AMP
     reward 的 standing 环境），所以这组诊断统计包含约 `rel_standing_envs` 比例的
     站立数据。

### 12.7 下一步建议

后续 agent 应按以下顺序继续，不要同时修改大量 reward 权重：

1. 使用当前 AMP 配置重新运行短实验，确认新增的 raw AMP reward 和 task reward
   日志是否符合预期。
2. 用 `num_envs=1` 或 `num_envs=16` 播放/检查初始姿态，确认默认姿态不会立即
   触发 base_link 接触。
3. 对比普通 PPO 和 AMP 的 `illegal_contact` 与 episode length，确认 AMP 是否
   在第一个 rollout 就破坏了原本可站立的策略学习过程。
4. 检查 MuJoCo XML 与 IsaacLab URDF 的以下一致性：
   - 10 个关节顺序；
   - 关节零位和正方向；
   - 左右脚 body 的名称和几何位置；
   - base_link 的高度和初始姿态。
5. 在环境稳定前，不要提高 `amp_reward_coef`。当前 `0.3` 已足够作为初始值。
6. 如果 AMP policy 仍然快速摔倒，先降低 AMP 影响进行 warm-up，例如临时使用
   更高的 `amp_task_reward_lerp`，或先用普通 PPO checkpoint 初始化 policy，再
   继续 AMP 训练。
7. 只有当 episode length 稳定增长后，再调 `amp_reward_coef`、判别器网络规模和
   replay buffer 大小。

### 12.8 交付验收标准

后续修改至少应满足：

```text
AMP observation shape = [num_envs, 26]
expert state shape     = [batch, 26]
expert next state      = [batch, 26]
discriminator input    = [batch, 52]
所有 AMP state 无 NaN/Inf
illegal_contact 不再每个 iteration 都为 1.0
episode length 明显超过当前约 270 steps
raw AMP reward 非零且没有 NaN
policy prediction 和 expert prediction 保持相反符号
普通 PPO 配置和结果不被 AMP 修改破坏
```

### 12.9 重要运行约束

必须使用 IsaacLab 对应的 conda 环境：

```bash
conda activate zgl_isaaclab
```

如果直接使用 `/home/niu/IsaacLab/isaaclab.sh`，需要确认该脚本实际指向的 Python
环境。此前该脚本曾指向 `umr` 环境，导致 `import isaaclab` 失败；而在当前环境中
还需要确认 NVIDIA driver 对 Isaac Sim 可见：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

`omni.physx` 的 stage detach、plugin unload 等 warning 通常是 Isaac Sim 退出时
的清理警告，不要把它们和真正的 Python traceback 或 CUDA 初始化错误混淆。
