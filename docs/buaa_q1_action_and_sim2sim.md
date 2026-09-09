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

`buaa-q1-flat` uses `JointPositionToLimitsActionCfg` with
`rescale_to_limits=True`:

```text
normalized_action = clamp(policy_action, -1, 1)
target = lower_soft_limit
       + (normalized_action + 1) / 2
       * (upper_soft_limit - lower_soft_limit)
```

The policy output itself is still not guaranteed to be in `[-1, 1]`. A raw
output of `8` is displayed as `8`, but the action term uses `1` for target
generation. The final target is inside the per-joint soft limits.

With `soft_joint_pos_limit_factor = 0.90`:

```text
policy output = -1 -> lower soft limit
policy output =  0 -> soft-limit midpoint
policy output = +1 -> upper soft limit
policy output =  8 -> treated as +1
```

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

`mujoco/deploy_mujoco_buaa_q1.py` applies the same soft-limit mapping using
the MJCF hard ranges and factor `0.90`, then computes PD torque:

```text
computed_torque = kp * (target - qpos) - kd * qvel
applied_torque = clip(computed_torque, -20, 20)
```

The matching MuJoCo configuration is:

```yaml
obs_order: articulation
action_order: articulation
soft_joint_pos_limit_factor: 0.90
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
