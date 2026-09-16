#!/usr/bin/env python3
"""Validate the minimal asset set required by the production depth service."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

PROMPT_PREFIX = "qwen_edit_2509_qwen_depth_realimg512"


def required_paths(assets_dir: Path) -> tuple[Path, ...]:
    qwen = assets_dir / "checkpoints" / "Qwen-Image-Edit-2509"
    marigold = assets_dir / "checkpoints" / "Marigold-V2"
    embeddings = marigold / "qwen_text_embeddings"
    return (
        qwen / "model_index.json",
        qwen / "transformer",
        qwen / "vae",
        marigold / "depth" / "Log-stage2" / "trainables.safetensors",
        embeddings / f"{PROMPT_PREFIX}_prompt_embeds.pt",
        embeddings / f"{PROMPT_PREFIX}_prompt_mask.pt",
    )


def missing_paths(assets_dir: Path) -> list[Path]:
    return [path for path in required_paths(assets_dir) if not path.exists()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path(os.getenv("DEPTH_ASSETS_DIR", "assets")),
    )
    args = parser.parse_args()
    assets_dir = args.assets_dir.expanduser().resolve()
    missing = missing_paths(assets_dir)
    if missing:
        print("Missing required inference assets:")
        for path in missing:
            print(f"- {path}")
        return 1
    print(f"Inference assets validated under {assets_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
