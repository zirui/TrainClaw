#!/usr/bin/env python3
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone


def run_cmd(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return {"ok": True, "output": out.strip()}
    except subprocess.CalledProcessError as e:
        return {"ok": False, "output": e.output.strip(), "returncode": e.returncode}
    except FileNotFoundError:
        return {"ok": False, "output": "command not found", "returncode": 127}


def main():
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "cwd": os.getcwd(),
        "checks": {},
    }

    report["checks"]["nvidia_smi_exists"] = shutil.which("nvidia-smi") is not None
    if report["checks"]["nvidia_smi_exists"]:
        report["checks"]["nvidia_smi"] = run_cmd(["nvidia-smi", "-L"])
    else:
        report["checks"]["nvidia_smi"] = {"ok": False, "output": "nvidia-smi not found"}

    report["checks"]["torch_cuda"] = run_cmd(
        [
            sys.executable,
            "-c",
            (
                "import json, torch; "
                "print(json.dumps({'torch':torch.__version__,"
                "'cuda_available':torch.cuda.is_available(),"
                "'cuda_version':torch.version.cuda}))"
            ),
        ]
    )

    report["checks"]["nccl_env"] = {
        "ok": True,
        "output": {
            "NCCL_DEBUG": os.getenv("NCCL_DEBUG", ""),
            "NCCL_SOCKET_IFNAME": os.getenv("NCCL_SOCKET_IFNAME", ""),
            "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        },
    }

    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
