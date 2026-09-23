#!/usr/bin/env bash
# Mirror the Isaac Sim assets used by these tasks into a local directory so that
# play/train no longer depend on NVIDIA's asset CDN while starting up.
#
# Usage:
#   scripts/fetch_local_assets.sh [target_dir]
#
# Default target: $Q1_ISAAC_ASSET_ROOT, or ~/isaac_assets/isaac_5.0.
# Downloads are resumable, so the script can be re-run after a failure.
set -euo pipefail

BASE_URL="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0"
ROOT="${1:-${Q1_ISAAC_ASSET_ROOT:-$HOME/isaac_assets/isaac_5.0}}"

files=(
  "Isaac/Environments/Grid/default_environment.usd"
  "Isaac/Environments/Grid/Materials/Textures/Wireframe_blue.png"
  "Isaac/Environments/Grid/Materials/Textures/WireframeBlur_basecolor.png"
  "Isaac/Environments/Grid/Materials/Textures/WireframeBlur_blue.png"
  "Isaac/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"
  "Isaac/Props/UIElements/arrow_x.usd"
  "Isaac/Props/UIElements/frame_prim.usd"
  "Isaac/IsaacLab/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl"
)

for rel in "${files[@]}"; do
  dest="$ROOT/$rel"
  mkdir -p "$(dirname "$dest")"
  echo "[fetch] $rel"
  curl -fL -C - --retry 40 --retry-all-errors --retry-delay 5 --connect-timeout 20 -o "$dest" "$BASE_URL/$rel"
done

echo "[done] assets mirrored to $ROOT"
