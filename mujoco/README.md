# BUAA Q1 MuJoCo sim2sim

This deploys the policy trained by `buaa-q1-flat` on a flat MuJoCo plane.
The observation order is the Isaac Lab policy order after the task disables
`base_lin_vel` and `height_scan`:

`base_ang_vel(3) | projected_gravity(3) | velocity_command(3) | joint_pos_rel(10) | joint_vel(10) | last_action(10)`

The joint/action order is:

Observation order: `hip_yaw_l, hip_yaw_r, hip_roll_l, hip_roll_r, hip_pitch_l, hip_pitch_r, knee_pitch_l, knee_pitch_r, ankle_pitch_l, ankle_pitch_r`.

Action order follows the explicit Isaac action configuration: `hip_yaw_l, hip_roll_l, hip_pitch_l, knee_pitch_l, ankle_pitch_l, hip_yaw_r, hip_roll_r, hip_pitch_r, knee_pitch_r, ankle_pitch_r`.

The Xbox controller uses the same Linux joystick mapping as the M1 deploy:
left stick vertical = `vx`, left stick horizontal = `vy`, right stick horizontal = `wz`.

## Run

Install the runtime dependencies in the active Python environment:

```bash
pip install mujoco onnxruntime pyyaml numpy
```

Then run from the repository root:

```bash
python mujoco/deploy_mujoco_buaa_q1.py \
  --onnx logs/rsl_rl/buaa_q1_flat/2026-09-08_20-29-59/exported/policy.onnx
```

The default policy path is already set in `mujoco/configs/buaa_q1_flat.yaml`.
Override `--config` when using another checkpoint. Set `joystick_device` to
the controller's `/dev/input/jsX` device if it is not `/dev/input/js0`.

## Regenerate MJCF

The checked-in scene was generated from the BUAA Q1 URDF. Regenerate it after
changing that URDF with:

```bash
python mujoco/convert_urdf_to_mjcf.py \
  source/robot_lab/robot_lab/assets/data/buaa-q1-v1/q1/urdf/q1.urdf \
  source/robot_lab/robot_lab/assets/data/buaa-q1-v1/q1/mjcf/scene.xml
```
