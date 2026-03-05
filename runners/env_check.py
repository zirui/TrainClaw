#!/usr/bin/env python3
import argparse
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


def detect_gpu_backend(preferred):
    if preferred and preferred != "auto":
        return preferred
    if shutil.which("nvidia-smi"):
        return "nvidia"
    if shutil.which("amd-smi") or shutil.which("rocm-smi"):
        return "amd"
    return "none"


def build_torch_check_cmd(args):
    snippet = (
        "import json, torch; "
        "print(json.dumps({'torch':torch.__version__,"
        "'cuda_available':torch.cuda.is_available(),"
        "'cuda_version':torch.version.cuda}))"
    )
    if args.torch_check_command:
        return ["bash", "-lc", args.torch_check_command]
    py_cmd = args.torch_python_cmd.strip() if args.torch_python_cmd else sys.executable
    if " " in py_cmd:
        return ["bash", "-lc", f"{py_cmd} -c \"{snippet}\""]
    return [py_cmd, "-c", snippet]


def main():
    ap = argparse.ArgumentParser(description="Environment sanity checks for TrainClaw launcher")
    ap.add_argument("--gpu-backend", choices=["auto", "nvidia", "amd", "none"], default="auto")
    ap.add_argument("--torch-python-cmd", default=None)
    ap.add_argument(
        "--torch-check-command",
        default=None,
        help="Full command that prints torch check JSON. If set, overrides --torch-python-cmd.",
    )
    args = ap.parse_args()

    gpu_backend = detect_gpu_backend(args.gpu_backend)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "cwd": os.getcwd(),
        "gpu_backend": gpu_backend,
        "checks": {},
    }

    report["checks"]["nvidia_smi_exists"] = shutil.which("nvidia-smi") is not None
    report["checks"]["amd_smi_exists"] = shutil.which("amd-smi") is not None
    report["checks"]["rocm_smi_exists"] = shutil.which("rocm-smi") is not None

    if gpu_backend == "nvidia":
        report["checks"]["nvidia_smi"] = run_cmd(["nvidia-smi", "-L"])
    elif gpu_backend == "amd" and report["checks"]["amd_smi_exists"]:
        report["checks"]["amd_smi"] = run_cmd(["amd-smi", "list"])
    elif gpu_backend == "amd" and report["checks"]["rocm_smi_exists"]:
        report["checks"]["rocm_smi"] = run_cmd(["rocm-smi", "-i"])
    else:
        report["checks"]["gpu_tool"] = {"ok": False, "output": "no supported gpu tool found"}

    report["checks"]["torch_cuda"] = run_cmd(build_torch_check_cmd(args))

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
