# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Point Isaac Sim at a local mirror of the NVIDIA asset CDN.

Isaac Sim resolves scene assets (the grid ground-plane USD, the sky HDRI, MDL
materials, ...) through the carb setting ``/persistent/isaac/asset_root/cloud``,
which defaults to NVIDIA's public asset CDN.  When that CDN is slow or blocked,
Isaac Sim blocks for minutes while creating the scene and ``play``/``train``
look like they hang on startup.

:func:`use_local_assets` rewrites ``kit_args`` so Kit boots with the asset root
pointing at a local copy of ``Assets/Isaac/5.0``.  Call it right before the
:class:`~isaaclab.app.AppLauncher` is created:

.. code-block:: python

    args_cli.kit_args = local_assets.use_local_assets(args_cli.kit_args)

The mirror location defaults to ``~/isaac_assets/isaac_5.0`` and can be changed
with the ``Q1_ISAAC_ASSET_ROOT`` environment variable.  If the mirror does not
exist the override is skipped and the remote asset root is used as before.
"""

from __future__ import annotations

import os

DEFAULT_ASSET_ROOT = os.path.expanduser("~/isaac_assets/isaac_5.0")
"""Default location of the mirrored ``Assets/Isaac/5.0`` tree."""

ASSET_ROOT_ENV_VAR = "Q1_ISAAC_ASSET_ROOT"
"""Environment variable that overrides :data:`DEFAULT_ASSET_ROOT`."""

ASSET_ROOT_SETTING = "/persistent/isaac/asset_root/cloud"
"""Carb setting holding the asset root used by Isaac Sim and Isaac Lab."""


def get_asset_root() -> str:
    """Return the configured local asset root as an absolute path."""
    return os.path.abspath(os.path.expanduser(os.environ.get(ASSET_ROOT_ENV_VAR, DEFAULT_ASSET_ROOT)))


def use_local_assets(kit_args: str = "") -> str:
    """Extend ``kit_args`` with a local asset-root override when it is available.

    Args:
        kit_args: Kit command line arguments collected by :class:`AppLauncher`.

    Returns:
        ``kit_args`` with ``--/persistent/isaac/asset_root/cloud=<mirror>``
        appended.  The input is returned unchanged when the mirror is missing,
        so the previous (remote) behaviour is preserved.
    """
    if ASSET_ROOT_SETTING in kit_args:
        return kit_args

    asset_root = get_asset_root()
    if not os.path.isdir(os.path.join(asset_root, "Isaac")):
        print(
            f"[WARN] Local Isaac asset mirror not found at '{asset_root}'. Falling back to the"
            f" NVIDIA asset CDN, which can stall startup. Set {ASSET_ROOT_ENV_VAR} to relocate the mirror."
        )
        return kit_args

    print(f"[INFO] Using local Isaac asset mirror: {asset_root}")
    return f"{kit_args} --{ASSET_ROOT_SETTING}={asset_root}".strip()
