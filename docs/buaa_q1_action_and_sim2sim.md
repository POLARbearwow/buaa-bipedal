# BUAA Q1 Action and MuJoCo Sim2Sim Notes

## Action semantics

### Previous configuration

The original `JointPositionActionCfg` used the default joint pose as an offset:

```text
target = default_joint_pos + policy_action * 0.5
```

The actor output was not bounded. The old action clip (`[-100, 100]`) was not a
joint-limit clip, so an output of `20` could produce a target about `10 rad`
away from the default pose. The PD controller then requested a large torque,
which was finally clipped by the actuator effort limit (`20 N*m`).

### Current configuration

`buaa-q1-flat` follows TienKung-Lab's action semantics:

```text
clipped_action = clamp(policy_action, -100, 100)
target = default_joint_pos + clipped_action * 0.25 rad
```

The policy output is not normalized to `[-1, 1]`, and the target is not
remapped or clamped to the soft joint limits. The RSL-RL environment wrapper
clips raw actions to `[-100, 100]` before the action term applies scale and
offset, matching TienKung-Lab; this is not the joint-position limit.

Joint-limit rewards and actuator dynamics must therefore discourage unsafe
targets. Check `Action/target_near_limit_fraction` and
`Action/target_limit_margin_mean` during training.

The new action order is the articulation order:

```text
hip_yaw_l, hip_yaw_r,
hip_roll_l, hip_roll_r,
hip_pitch_l, hip_pitch_r,
knee_pitch_l, knee_pitch_r,
ankle_pitch_l, ankle_pitch_r
```

The policy must be retrained after this semantic change. Old checkpoints are
not compatible with the new action mapping.

## MuJoCo mapping

`mujoco/deploy_mujoco_buaa_q1.py` applies the same clipped, scaled default-pose
offset as training, then computes PD torque:

```text
target = default_joint_pos + clip(policy_action, -100, 100) * 0.25
computed_torque = kp * (target - qpos) - kd * qvel
applied_torque = clip(computed_torque, -20, 20)
```

The matching MuJoCo configuration sets the same action scale and safety clip:

```yaml
obs_order: articulation
action_order: articulation
action_scale: 0.25
clip_actions: 100.0
```

If `/dev/input/js0` is absent, the command remains `(vx, vy, wz) = (0, 0, 0)`.
This is expected and does not prevent the simulation from starting.

## First-action timing

The first observation after reset has `last_action = 0`. The correct sequence is:

```text
reset
-> observe with last_action = 0
-> infer action_1
-> immediately apply action_1
-> observe again with last_action = action_1
-> infer and apply action_2
```

MuJoCo performs this initial inference before entering the physics loop, matching
Isaac Lab `play.py`. Previously it executed one control interval with zero
action before calculating `action_1`, causing an initial movement toward the
limit midpoint.

## Debug logging

Run:

```bash
python mujoco/deploy_mujoco_buaa_q1.py --debug --debug-interval 5
```

The log fields are:

```text
action(raw)       policy output before clipping
action(clamped)   output after clipping to [-1, 1]
target(rad)       mapped joint position target
qpos(rad)         actual joint position
computed(Nm)      PD torque before actuator clipping
ctrl(Nm)          command written to MuJoCo actuator
applied(Nm)       MuJoCo actuator_force actually applied
```

Sustained `applied(Nm) = +/-20` indicates torque saturation. Large raw action
values are not by themselves a problem under the new action term; the mapped
target and applied torque are the values to inspect.
