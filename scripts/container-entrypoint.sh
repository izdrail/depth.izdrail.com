#!/usr/bin/env bash
set -Eeuo pipefail

assets_dir="${DEPTH_ASSETS_DIR:-/opt/marigold-assets}"
mkdir -p "$assets_dir"

if python scripts/validate_inference_assets.py --assets-dir "$assets_dir"; then
  echo "[boot] Reusing cached inference assets from $assets_dir"
else
  echo "[boot] Preparing inference assets in $assets_dir"
  echo "[boot] The first start downloads roughly 42 GB; later starts reuse the persistent cache."
  python scripts/download_assets.py --assets-dir "$assets_dir" --inference-only
  python scripts/validate_inference_assets.py --assets-dir "$assets_dir"
fi

exec "$@"
