# BUAA Q1 的 Noise Std、Entropy 与动作链路总结

## 现象

PPO/AMP-PPO 的策略用高斯分布产生动作。当前策略配置为 `noise_std_type=scalar`、`init_noise_std=1.0`，std 是可训练参数，并没有固定目标值或上限。动作维数为 10，Gaussian entropy 是 10 个动作维度 entropy 的和：

```text
H = sum_i 0.5 * log(2 * pi * e * std_i^2)
```

如果各维 std 相同，则 `H = 10 * (0.5 * log(2*pi*e) + log(std))`。所以 std 增大时，entropy 也会随之增大。PPO loss 中包含 `-entropy_coef * entropy`，当前 `entropy_coef=0.01`，会鼓励保留/增加策略随机性；它并不会将 entropy 或 std 固定在某个值。

## 日志证据

09-23 的 AMP 运行 `logs/rsl_rl/buaa_q1_flat_amp/2026-09-23_14-33-57` 使用旧的 `[-1,1]` 限位映射，且 `action_rate_l2=-0.001`。该进程启动时保存的 YAML 仍是 `JointPositionToLimitsAction`/`rescale_to_limits=true`、`clip_actions=null`，没有加载后来修改的动作配置。因此它不是新方案的验证结果。该旧运行在约第 1053 次迭代时的数据如下：

| 指标 | 第 0 次 | 第 602 次 | 第 1053 次 |
| --- | ---: | ---: | ---: |
| `Policy/mean_noise_std` | 1.01 | 3.94 | 6.06 |
| `Loss/entropy` | 14.23 | 27.85 | 32.17 |
| raw action 中 `|a|>1` 的比例 | 31.8% | 95.7% | 97.9% |
| raw `|a|` 的 P95 | 1.97 | 17.93 | 32.71 |
| 限位映射后的目标靠近软限位比例 | 32.3% | 95.7% | 97.9% |
| 平均归一化限位余量 | 0.184 | 0.0107 | 0.00535 |

这说明旧方案下，std/entropy 增长伴随着大量 raw action 超出 `[-1,1]`，映射后的目标集中在关节软限位边缘。许多不同的大 raw action 会得到相同或近似相同的边界目标，但 entropy 仍按映射前的 Gaussian raw action 计算。因此，entropy 上升不等同于实际关节动作仍在有效增加探索。

同日的另一次运行 `2026-09-23_13-32-33` 使用 `action_rate_l2=-0.1`，约第 606 次迭代 std 为 0.88；改为 `-0.001` 的运行 `2026-09-23_14-33-57` 约第 602 次迭代 std 为 3.94。两者迭代数接近，支持较弱的动作变化惩罚与更大的 std 有关。不过较强惩罚并不构成 std 上限，也不能单独证明因果机制。

以上动作饱和率和软限位比例来自旧的 `[-1,1]` 动作映射。旧进程写出的 `Action/raw_outside_fraction` 阈值是 `|a|>1`，不是新方案下 `±100` 的裁剪越界率。源码修改也不会热加载到已运行进程；新指标要等重新启动训练后才会按 `Action/raw_outside_clip_fraction` 等新 tag 写入。

## 可能的问题

旧的 Q1 动作项将动作裁剪到 `[-1,1]`，然后将这个归一化值映射到每个关节的软限位。策略本身的 Gaussian 分布仍然是无界的，而环境只看到经裁剪/映射的结果。于是可能出现：

1. entropy 项推动 raw action 分布继续变宽。
2. 越界 raw action 被压到相同的动作边界，无法带来对应的有效控制变化。
3. 目标频繁贴近软限位，策略噪声变大但有效探索未必变多。

`action_rate_l2` 会惩罚相邻动作变化，因此会影响策略行为和 std 的学习；减弱该项后 std 增长更快是符合现有对照日志的，但该项不直接约束 Gaussian std。

## 调整后的动作方案

当前 BUAA Q1 普通 PPO 和 AMP-PPO 均按 TienKung-Lab 的动作语义配置：

```text
clipped_action = clamp(raw_action, -100, 100)
joint_position_target = default_joint_position + 0.25 * clipped_action
```

这里的 `±100` 是 raw policy action 的大范围安全裁剪，不是关节限位。`0.25` 的单位是弧度/action unit。目标不再被映射或夹紧到软关节限位；越界目标由奖励、执行器和仿真动力学共同处理，因此必须持续监控软限位指标。

训练配置将 `clip_actions=100.0` 放在 RSL-RL 环境 wrapper。wrapper 在动作进入环境前裁剪 raw action；`JointPositionActionCfg` 使用 `scale=0.25`、`use_default_offset=True`、`clip=None`，避免 action term 再以不同顺序裁剪。RSL-RL 的 PPO rollout/log-prob 仍对应策略采样的 raw action，环境执行的是裁剪后的动作；这与 TienKung 在环境 step 中裁剪动作的设计一致。

## 从策略到仿真力矩

```text
policy observation
  -> Actor 输出 Gaussian mean
  -> Normal(mean, std) 采样 raw_action
  -> RSL-RL wrapper 裁剪到 [-100, 100]
  -> JointPositionActionCfg: target = default_joint_pos + 0.25 * clipped_action
  -> Isaac Lab 将关节位置 target 交给 implicit position actuator
  -> 执行器按 stiffness/damping 对位置误差和关节速度产生驱动力矩
  -> effort_limit_sim=20 N*m 限制可施加的关节力矩
  -> PhysX 推进仿真并返回下一步观测/奖励
```

控制器力矩可近似理解为 `tau = Kp * (target - q) - Kd * qdot`，实际求解由 Isaac Lab/PhysX 的 implicit actuator 完成；最终执行力矩受每组执行器 `effort_limit_sim=20 N*m` 限制。这个力矩限制不等于位置目标限制，目标仍可能在软限位之外。

MuJoCo 部署使用相同的动作转换：先把 ONNX raw action 裁剪到 `[-100,100]`，再乘 `0.25` 并加默认关节角，之后用 PD 计算力矩并按 `20 N*m` 裁剪。配置在 `mujoco/configs/buaa_q1_flat.yaml`。

## 训练期间观察

普通 PPO 和 AMP-PPO 每个 rollout 都记录以下 TensorBoard 指标：

- `Policy/mean_noise_std`、`Loss/entropy`：策略分布尺度及其熵。
- `Action/raw_abs_p95`、`Action/raw_abs_p99`：未裁剪 raw action 幅度分位数。
- `Action/raw_outside_clip_fraction`、`Action/raw_clip_excess_mean`：超过 `±100` 裁剪范围的比例和超出幅度。
- `Action/clipped_abs_mean`、`Action/clipped_boundary_fraction`：环境接收的裁剪动作幅度及碰到裁剪边界的比例。
- `Action/target_near_limit_fraction`、`Action/target_limit_margin_mean`：关节目标靠近/越过软限位的比例与有符号余量；负余量表示超出软限位。

判断 std/entropy 增长是否仍变成无效饱和时，重点对照 raw action 分位数、`raw_outside_clip_fraction`、`target_near_limit_fraction` 和限位余量。新动作语义下 `|raw_action|>1` 不是裁剪指标，文档和 TensorBoard 均不再把它作为饱和判断。

## 复现逐步动作调试

训练中的 raw policy action 统计在 TensorBoard 按 rollout 记录。若要检查单环境的关节目标、状态和力矩，可用 `scripts/play_debug.py` 加载一个按**当前动作语义训练的新 checkpoint**：

```bash
./isaaclab.sh -p scripts/play_debug.py \
  --task buaa-q1-flat \
  --agent rsl_rl_amp_cfg_entry_point \
  --checkpoint "logs/rsl_rl/buaa_q1_flat_amp/NEW_RUN/model_100.pt" \
  --num_envs 1 \
  --debug-interval 10 \
  --action-threshold 100
```

普通 PPO checkpoint 使用 `--agent rsl_rl_cfg_entry_point`。命令每 10 个控制步输出一次，包括速度命令、机身速度/姿态、脚端接触、各关节的 `action(clamped)`、`target(rad)`、`qpos(rad)`、`computed(Nm)`、`applied(Nm)`，以及 reward manager 的各 reward 值。

在新动作配置下，`action(clamped)` 是 RSL-RL wrapper 裁剪后进入 action term 的动作；目标角满足 `target = default_joint_pos + 0.25 * action(clamped)`。这个逐步输出看不到 wrapper 裁剪前的 actor raw sample；训练时的 pre-clip raw action 要看 TensorBoard 的 `Action/raw_abs_p95/p99` 和 `Action/raw_outside_clip_fraction`。`--action-threshold 100` 在收到的动作达到安全裁剪边界时发出警告。控制台 debug 默认不保存为文件，可用 shell `tee` 另存输出。

建议按以下顺序判断问题位置：

1. `Policy/mean_noise_std` 和 `Loss/entropy` 同步上升，说明策略分布变宽，但不代表环境实际目标等比例变大。
2. 对照 `Action/raw_abs_p95/p99`、`raw_outside_clip_fraction` 和 `clipped_boundary_fraction`，判断 raw 分布是否变宽以及是否撞到 `±100` 安全裁剪。
3. 看 `target_near_limit_fraction` 和 `target_limit_margin_mean`。margin 为负表示目标越过软限位；若目标贴边但没有触发 `±100` clip，仍可能有位置目标风险。
4. 用逐步 debug 比较目标角、实际关节角、计算力矩和施加力矩。若 computed torque 很大而 applied torque 长期卡在 `±20 N*m`，说明执行器力矩饱和；若目标越界但力矩没有饱和，问题在目标/策略行为而非 effort clip。
5. 最后结合 tracking reward、episode length 和 AMP 指标判断整体训练，不要只用 reward 或 entropy 单一指标判断收敛。

09-23 的逐关节控制台 debug 输出没有保存为日志文件；上面的定量历史值来自 TensorBoard event。新动作方案需要在新 checkpoint 上重新播放并保存 debug 输出，不能用旧 checkpoint 在新映射下作等价比较。

## 注意事项

- 动作解释改变后必须从头训练；旧 checkpoint 不兼容新的动作映射。
- 已经启动的训练不会热加载源代码或新配置，需要停止并重新启动才能使用新动作链路。
- 目前完成了 Python 静态语法检查，但尚未在本机 Isaac Lab 环境中启动仿真验证。
