# Q1 standalone training

This directory is a self-contained Isaac Lab extension for the Q1 flat-ground
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
