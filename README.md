# Q1 Robot Lab

This repository is a self-contained Isaac Lab extension for the Q1 flat-ground
velocity-tracking task. It includes the Q1 URDF and meshes, task configuration,
PPO configuration, and RSL-RL training/play scripts. Isaac Lab and Isaac Sim
remain external prerequisites and must be installed according to the Isaac Lab
installation guide.

## Install

From an Isaac Lab Python environment:

```bash
cd q1_training
python -m pip install -e .
```

The package expects Isaac Lab's `isaaclab`, `isaaclab_tasks`, and `isaaclab_rl`
modules to be available in that environment.

## Project layout

The Python package is kept at the repository root, following the layout used by
TienKung-Lab:

```text
robot_lab/          # package, assets, environments and MDP terms
robot_lab/envs/q1/       # Q1 task configs and Gym registration
robot_lab/envs/buaa_q1/  # BUAA Q1 task configs and Gym registration
robot_lab/mdp/      # reusable commands, rewards and terminations
scripts/            # train, play and inspection entry points
mujoco/             # MuJoCo conversion and deployment helpers
logs/               # training outputs
```

After moving from an older checkout, reinstall the package so editable-install
metadata is refreshed:

```bash
python -m pip install -e . --no-deps
```

## Local asset mirror (avoids slow startup)

Isaac Sim resolves scene assets (the grid ground plane, the sky HDRI, the
velocity-arrow props) through NVIDIA's asset CDN by default. That CDN is slow or
unreachable from some networks, and Isaac Sim then blocks for minutes while
creating the scene, which looks like `play` hanging on startup.

Mirror the few assets this task needs once, and everything resolves locally:

```bash
scripts/fetch_local_assets.sh              # downloads into ~/isaac_assets/isaac_5.0
scripts/fetch_local_assets.sh /some/path   # or into an explicit directory
```

`scripts/play.py`, `scripts/play_joystick.py`, `scripts/train.py` and
`scripts/inspect_default_pose.py` pick the mirror up automatically through
`scripts/local_assets.py`, which passes
`--/persistent/isaac/asset_root/cloud=<mirror>` to Kit. Set
`Q1_ISAAC_ASSET_ROOT` to use a different directory. If the mirror is missing the
scripts warn and fall back to the remote asset root.

The velocity task additionally guards the sky HDRI: the dome light only
references the texture when the file exists on disk, so a missing mirror yields
a plain dome light instead of a multi-minute network stall.

## Train

```bash
python scripts/train.py --task buaa-q1-flat --headless
```

Useful overrides include `--num_envs`, `--max_iterations`, `--seed`, and
`--checkpoint` (for resuming an existing run).

## Play a checkpoint

```bash
python scripts/play.py --task buaa-q1-flat \
  --checkpoint logs/rsl_rl/q1_flat/<run>/model_*.pt
```

Replace the checkpoint path with the model produced by training. Logs are
written under `logs/rsl_rl/q1_flat`.

The BUAA Q1-v1 model uses the `buaa-q1-flat` task. Its initial joint pose is
currently all zeros:

```bash
python scripts/train.py --task buaa-q1-flat --headless
```

Logs are written under `logs/rsl_rl/buaa_q1_flat`.

## AMP + RL migration

The step-by-step guide for porting TienKung-Lab's AMP+PPO pipeline is in
[`docs/amp_rl_migration.md`](docs/amp_rl_migration.md).
