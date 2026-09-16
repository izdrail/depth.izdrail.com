import subprocess
import sys


def test_inference_only_flag_is_documented():
    result = subprocess.run(
        [sys.executable, "scripts/download_assets.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--inference-only" in result.stdout
    assert "depth/Log-stage2" in result.stdout


def test_asset_validator_rejects_an_empty_cache(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_inference_assets.py",
            "--assets-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Missing required inference assets" in result.stdout


def test_asset_validator_accepts_complete_minimal_cache(tmp_path):
    qwen = tmp_path / "checkpoints" / "Qwen-Image-Edit-2509"
    marigold = tmp_path / "checkpoints" / "Marigold-V2"
    for directory in (
        qwen / "transformer",
        qwen / "vae",
        marigold / "depth" / "Log-stage2",
        marigold / "qwen_text_embeddings",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    for file in (
        qwen / "model_index.json",
        marigold / "depth" / "Log-stage2" / "trainables.safetensors",
        marigold
        / "qwen_text_embeddings"
        / "qwen_edit_2509_qwen_depth_realimg512_prompt_embeds.pt",
        marigold
        / "qwen_text_embeddings"
        / "qwen_edit_2509_qwen_depth_realimg512_prompt_mask.pt",
    ):
        file.touch()
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_inference_assets.py",
            "--assets-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Inference assets validated" in result.stdout
