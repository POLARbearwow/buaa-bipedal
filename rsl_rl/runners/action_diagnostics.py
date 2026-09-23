"""Action saturation diagnostics for clipped joint-position actions."""

from __future__ import annotations

import torch


def get_action_diagnostics(env, raw_actions: torch.Tensor) -> dict[str, torch.Tensor] | None:
    """Summarize raw actions and realized joint targets when the env exposes limits."""
    base_env = getattr(env, "_amp_env", None)
    if base_env is None:
        base_env = env.unwrapped

    cfg = base_env.cfg
    action_joint_names = list(getattr(cfg, "action_joint_names", []))
    if not action_joint_names:
        return None

    clip_limit = getattr(env, "clip_actions", None)
    if clip_limit is None:
        clip_range = getattr(cfg.actions.joint_pos, "clip", None)
        if clip_range is None:
            return None
        clip_limit = max(abs(float(clip_range[0])), abs(float(clip_range[1])))
    else:
        clip_limit = float(clip_limit)

    robot = base_env.scene["robot"]
    action_joint_ids = robot.find_joints(action_joint_names, preserve_order=True)[0]
    abs_actions = raw_actions.detach().abs().flatten()
    outside_clip = abs_actions > clip_limit
    clip_excess = (abs_actions - clip_limit).clamp_min(0.0)
    clipped_actions = raw_actions.detach().clamp(-clip_limit, clip_limit)

    target = robot.data.joint_pos_target[:, action_joint_ids].detach()
    limits = robot.data.soft_joint_pos_limits[:, action_joint_ids]
    lower, upper = limits[..., 0], limits[..., 1]
    span = (upper - lower).clamp_min(1.0e-6)
    normalized_target = 2.0 * (target - lower) / span - 1.0
    limit_margin = torch.minimum(target - lower, upper - target) / span

    return {
        "raw_outside_clip_fraction": outside_clip.float().mean(),
        "raw_clip_excess_mean": clip_excess.mean(),
        "raw_clip_excess_mean_outside": (
            clip_excess[outside_clip].mean() if outside_clip.any() else clip_excess.new_zeros(())
        ),
        "raw_abs_p95": torch.quantile(abs_actions, 0.95),
        "raw_abs_p99": torch.quantile(abs_actions, 0.99),
        "clipped_abs_mean": clipped_actions.abs().mean(),
        "clipped_boundary_fraction": (abs_actions >= clip_limit).float().mean(),
        "target_near_limit_fraction": (normalized_target.abs() >= 0.99).float().mean(),
        "target_limit_margin_mean": limit_margin.mean(),
    }
