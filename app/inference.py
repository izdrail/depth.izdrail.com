"""In-process, single-model wrapper around Marigold V2's inference graph."""
from __future__ import annotations

import asyncio
import io
import os
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = Path(os.getenv("DEPTH_ASSETS_DIR", REPO_ROOT / "assets")).resolve()
DEFAULT_CHECKPOINT: Final = "depth/Log-stage2"
SUPPORTED_CHECKPOINTS: Final = (DEFAULT_CHECKPOINT,)


class InferenceConfigurationError(RuntimeError):
    """Required model assets or GPU runtime are unavailable."""


class InferenceOutOfMemoryError(RuntimeError):
    """The GPU could not satisfy an inference allocation."""


@dataclass(frozen=True)
class Prediction:
    raw: np.ndarray
    display: np.ndarray
    duration_ms: float


class MarigoldInference:
    """Own one GPU model and serialize calls to preserve predictable VRAM use."""

    def __init__(self, assets_dir: Path = ASSETS_DIR) -> None:
        self.assets_dir = Path(assets_dir)
        self._model = None
        self._initialise_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _paths(self, checkpoint: str) -> tuple[Path, Path, Path]:
        if checkpoint not in SUPPORTED_CHECKPOINTS:
            raise ValueError(f"Unsupported checkpoint: {checkpoint}")
        qwen = self.assets_dir / "checkpoints" / "Qwen-Image-Edit-2509"
        marigold = self.assets_dir / "checkpoints" / "Marigold-V2"
        return qwen, marigold / checkpoint, marigold / "qwen_text_embeddings"

    def load(self, checkpoint: str = DEFAULT_CHECKPOINT) -> None:
        """Build the same registered VAE/DiT/LoRA graph as the CLI, once."""
        if self.loaded:
            return
        with self._initialise_lock:
            if self.loaded:
                return
            qwen, checkpoint_dir, embed_dir = self._paths(checkpoint)
            required = (
                qwen,
                checkpoint_dir / "trainables.safetensors",
                embed_dir / "qwen_edit_2509_qwen_depth_realimg512_prompt_embeds.pt",
                embed_dir / "qwen_edit_2509_qwen_depth_realimg512_prompt_mask.pt",
            )
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                raise InferenceConfigurationError(
                    "Missing model assets; run scripts/download_assets.py --skip-datasets: "
                    + ", ".join(missing)
                )

            import torch
            from accelerate import Accelerator
            from omegaconf import OmegaConf
            from safetensors.torch import load_file

            if not torch.cuda.is_available():
                raise InferenceConfigurationError(
                    "CUDA is required. Run the container with NVIDIA Container Toolkit and --gpus all."
                )

            from marigoldv2.core.builder import build_transforms, register_experiment_modules
            from marigoldv2.core.registry import REGISTRY
            from marigoldv2.network.change_network_mode import set_to_eval
            from marigoldv2.script.train.util import make_load_trainables_hook
            from marigoldv2.util.config_resolvers import recursive_load_config

            cfg = recursive_load_config(str(REPO_ROOT / "evaluation/config/inference_depth.yaml"))
            cfg.paths = OmegaConf.create(
                {
                    "ckpt_qwen_image_edit": str(qwen),
                    "embed_dir": str(embed_dir),
                }
            )
            cfg.device = "cuda"
            cfg.random_seed = 2025
            cfg.network_graph.QwenImageEdit2509Step.kwargs.prefix = (
                "qwen_edit_2509_qwen_depth_realimg512"
            )
            cfg.network_graph["FolderDepthPrediction"] = {
                "kwargs": {
                    "input_key": "out/pixel_pred",
                    "output_key": "out/depth_pred",
                }
            }
            cfg.register_modules = list(cfg.register_modules) + [
                "marigoldv2.validation.folder_steps"
            ]

            REGISTRY["cfg"] = cfg
            REGISTRY["network_components"].clear()
            register_experiment_modules(cfg)
            build_transforms(cfg.network_components, "network_components")({})
            graph = build_transforms(cfg.network_graph, "network_graph")

            accelerator = Accelerator(mixed_precision="bf16")
            trainables = checkpoint_dir / "trainables.safetensors"
            has_vae = any(key.startswith("VAE.") for key in load_file(trainables, device="cpu"))
            loader = make_load_trainables_hook(
                REGISTRY,
                accelerator,
                exclude_components=[] if has_vae else ["VAE"],
            )
            loader([], str(checkpoint_dir))
            for component in REGISTRY["network_components"].values():
                if isinstance(component, torch.nn.Module):
                    component.requires_grad_(False)
            set_to_eval()
            self._model = graph

    @staticmethod
    def _seed(seed: int) -> None:
        import torch

        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def predict(
        self,
        image: Image.Image,
        resolution: int,
        checkpoint: str,
        seed: int,
    ) -> Prediction:
        import torch
        from marigoldv2.dataset.dataloading.util import _lanczos_resize_chw

        if resolution not in (256, 512):
            raise ValueError("resolution must be 256 or 512")
        self.load(checkpoint)
        started = time.perf_counter()
        with self._inference_lock:
            self._seed(seed)
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).transpose(2, 0, 1)
            tensor = torch.from_numpy(rgb).float()
            tensor = _lanczos_resize_chw(tensor, (resolution, resolution))
            batch = {"rgb_norm": (tensor / 255.0 * 2.0 - 1.0).unsqueeze(0), "out": {}}
            try:
                with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    self._model(batch)
                raw = batch["out"]["depth_pred"][0, 0].float().cpu().numpy()
            except torch.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                raise InferenceOutOfMemoryError(
                    f"CUDA ran out of memory at {resolution}x{resolution}; try resolution=256"
                ) from exc
            finally:
                batch.clear()

        raw = np.asarray(raw, dtype=np.float32)
        # Reuse the CLI visualizer so the HTTP PNG has the same Spectral mapping.
        from marigoldv2.validation.folder_steps import _depth_rgb

        display = np.asarray(
            Image.fromarray(_depth_rgb(raw)).resize(
                image.size, Image.Resampling.BILINEAR
            )
        )
        return Prediction(raw, display, (time.perf_counter() - started) * 1000)

    async def predict_async(self, *args, **kwargs) -> Prediction:
        return await asyncio.to_thread(self.predict, *args, **kwargs)


def encode_npy(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.save(output, np.asarray(array, dtype=np.float32), allow_pickle=False)
    return output.getvalue()
