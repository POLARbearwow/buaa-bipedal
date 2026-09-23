#!/usr/bin/env python3
"""Validate a Q1 AMP JSON file before starting Isaac Sim training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def validate(path: Path) -> None:
    data = json.loads(path.read_text())
    frames = data.get("Frames")
    if not isinstance(frames, list) or len(frames) < 2:
        raise ValueError("Frames must contain at least two samples")
    if data.get("LoopMode", "Wrap") != "Wrap":
        raise ValueError("Q1 AMP currently requires LoopMode=Wrap")
    duration = float(data["FrameDuration"])
    if duration <= 0:
        raise ValueError("FrameDuration must be positive")
    widths = {len(frame) for frame in frames}
    if widths != {26}:
        raise ValueError(f"every frame must have 26 values, got widths {sorted(widths)}")
    values = [value for frame in frames for value in frame]
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Frames contain NaN or Inf")
    velocities = [frame[26 - 6 - 10 : 26 - 6] for frame in frames]
    max_velocity = max(abs(float(value)) for frame in velocities for value in frame)
    if max_velocity > 100.0:
        raise ValueError(f"joint velocity spike is too large: {max_velocity:.3f} rad/s")
    print(f"valid: {path} ({len(frames)} frames, dt={duration:.6f}s, dim=26, max_joint_vel={max_velocity:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("motion_files", nargs="+", type=Path)
    args = parser.parse_args()
    for motion_file in args.motion_files:
        validate(motion_file)
