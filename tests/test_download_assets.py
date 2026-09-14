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
