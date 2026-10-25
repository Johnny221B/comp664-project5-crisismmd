#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchvision


def main() -> None:
    output = Path("logs/m0")
    output.mkdir(parents=True, exist_ok=True)
    gpu_rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.strip().splitlines()
    payload = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "python": sys.version, "executable": sys.executable,
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(), "nvidia_smi": gpu_rows,
        "torch_devices": [
            {"index": i, "name": torch.cuda.get_device_name(i),
             "total_memory": torch.cuda.get_device_properties(i).total_memory}
            for i in range(torch.cuda.device_count())
        ],
    }
    (output / "environment.json").write_text(json.dumps(payload, indent=2) + "\n")
    frozen = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"], check=True, capture_output=True, text=True).stdout
    (output / "pip_freeze.txt").write_text(frozen)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
