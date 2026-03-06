#!/usr/bin/env python3
import argparse
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


STEP_RE = re.compile(r"(?:step|iter)\s*[=: ]\s*(\d+)", re.IGNORECASE)
LOSS_RE = re.compile(r"\bloss\s*[=: ]\s*([-+0-9.eE]+)", re.IGNORECASE)
THROUGHPUT_RE = re.compile(r"(?:throughput|samples/s|tokens/s)\s*[=: ]\s*([-+0-9.eE]+)", re.IGNORECASE)
SITER_RE = re.compile(r"(?:s/iter|sec/iter|step\s*time)\s*[=: ]\s*([-+0-9.eE]+)", re.IGNORECASE)
KEYVAL_RE = re.compile(r"([A-Za-z][A-Za-z0-9_./-]*)=([^\s,]+)")
MEM_PAIR_RE = re.compile(r"^\s*([-+0-9.]+)\s*/\s*([-+0-9.]+)\s*([A-Za-z]+)?\s*$")
SBATCH_JOB_RE = re.compile(r"Submitted\s+batch\s+job\s+(\d+)", re.IGNORECASE)
SBATCH_OUT_RE = re.compile(r"^\s*#SBATCH\s+(?:--output(?:=|\s+)|-o\s+)(\S+)")
SLURM_TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "PREEMPTED",
    "OUT_OF_MEMORY",
    "BOOT_FAIL",
    "DEADLINE",
    "NODE_FAIL",
}


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        return json.loads(raw)

    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "YAML config requires PyYAML. Install via `pip install pyyaml` or use JSON config."
        ) from e
    return yaml.safe_load(raw)


def try_float(maybe):
    if maybe is None:
        return None
    try:
        return float(maybe)
    except (TypeError, ValueError):
        return None


def now_ts():
    return datetime.utcnow().isoformat() + "Z"


def detect_gpu_backend(config_backend):
    if config_backend and config_backend != "auto":
        return config_backend
    if shutil.which("nvidia-smi"):
        return "nvidia"
    if shutil.which("amd-smi") or shutil.which("rocm-smi"):
        return "amd"
    return "none"


def query_nvidia_snapshot():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []

    records = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        records.append(
            {
                "gpu_index": int(parts[0]),
                "utilization_gpu_pct": try_float(parts[1]),
                "memory_used_mb": try_float(parts[2]),
                "memory_total_mb": try_float(parts[3]),
                "temperature_c": try_float(parts[4]),
                "gpu_backend": "nvidia",
            }
        )
    return records


def query_amd_snapshot():
    # Prefer rocm-smi JSON when available because format is more stable for parsing.
    if shutil.which("rocm-smi"):
        try:
            out = subprocess.check_output(
                ["rocm-smi", "--showuse", "--showmemuse", "--json"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            data = json.loads(out)
            records = []
            for card_name, payload in data.items():
                idx = None
                m = re.search(r"(\d+)", str(card_name))
                if m:
                    idx = int(m.group(1))
                gpu_use = None
                mem_use_pct = None
                if isinstance(payload, dict):
                    for k, v in payload.items():
                        key = str(k).lower()
                        if "gpu use" in key:
                            gpu_use = try_float(str(v).replace("%", "").strip())
                        if "memory use" in key:
                            mem_use_pct = try_float(str(v).replace("%", "").strip())
                records.append(
                    {
                        "gpu_index": idx if idx is not None else -1,
                        "utilization_gpu_pct": gpu_use,
                        "memory_used_mb": None,
                        "memory_total_mb": None,
                        "memory_used_pct": mem_use_pct,
                        "temperature_c": None,
                        "gpu_backend": "amd",
                    }
                )
            return records
        except Exception:
            pass

    if shutil.which("amd-smi"):
        # Fallback: keep raw monitor snapshot for troubleshooting when parser is unavailable.
        try:
            out = subprocess.check_output(
                ["amd-smi", "list"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            records = []
            for line in out.splitlines():
                if not line.strip():
                    continue
                m = re.search(r"(\d+)", line)
                idx = int(m.group(1)) if m else -1
                records.append(
                    {
                        "gpu_index": idx,
                        "utilization_gpu_pct": None,
                        "memory_used_mb": None,
                        "memory_total_mb": None,
                        "temperature_c": None,
                        "gpu_backend": "amd",
                        "raw": line.strip(),
                    }
                )
            return records
        except Exception:
            pass

    return []


def query_gpu_snapshot(gpu_backend):
    if gpu_backend == "nvidia":
        return query_nvidia_snapshot()
    if gpu_backend == "amd":
        return query_amd_snapshot()
    return []


def is_useful_gpu_record(rec):
    metric_keys = [
        "utilization_gpu_pct",
        "memory_used_mb",
        "memory_total_mb",
        "memory_used_pct",
        "temperature_c",
    ]
    return any(rec.get(k) is not None for k in metric_keys)


def normalize_key(raw_key):
    key = raw_key.strip().lower()
    key = key.replace("/", "_per_").replace(".", "_").replace("-", "_")
    aliases = {
        "iter": "step",
        "iteration": "step",
        "steps": "step",
        "step_time": "s_per_iter",
        "iter_time": "s_per_iter",
        "sec_per_iter": "s_per_iter",
        "seconds_per_iter": "s_per_iter",
        "samples_per_s": "throughput",
        "tokens_per_s": "throughput",
        "gnorm": "grad_norm",
    }
    return aliases.get(key, key)


def parse_number_with_unit(raw_value):
    text = raw_value.strip().lower()
    if not text:
        return None

    mem_pair = MEM_PAIR_RE.match(text)
    if mem_pair:
        used = try_float(mem_pair.group(1))
        total = try_float(mem_pair.group(2))
        unit = (mem_pair.group(3) or "gb").lower()
        factor = 1024.0 if unit in {"gb", "gib"} else 1.0
        if unit in {"mb", "mib", "gb", "gib"} and used is not None and total is not None:
            return {
                "kind": "mem_pair",
                "used_mb": used * factor,
                "total_mb": total * factor,
                "ratio_pct": (used / total * 100.0) if total else None,
            }

    suffix_scale = {
        "ms": ("seconds", 1e-3),
        "s": ("seconds", 1.0),
        "m": ("seconds", 60.0),
        "h": ("seconds", 3600.0),
        "kb": ("mb", 1.0 / 1024.0),
        "kib": ("mb", 1.0 / 1024.0),
        "mb": ("mb", 1.0),
        "mib": ("mb", 1.0),
        "gb": ("mb", 1024.0),
        "gib": ("mb", 1024.0),
        "%": ("pct", 1.0),
    }

    m = re.match(r"^([-+0-9.eE]+)([A-Za-z%]+)?$", text)
    if not m:
        return None
    number = try_float(m.group(1))
    if number is None:
        return None
    suffix = (m.group(2) or "").lower()
    if suffix in suffix_scale:
        _, scale = suffix_scale[suffix]
        return {"kind": suffix, "value": number * scale}
    return {"kind": "number", "value": number}


def parse_key_values(line):
    parsed = {}
    for key, value in KEYVAL_RE.findall(line):
        nkey = normalize_key(key)
        converted = parse_number_with_unit(value)
        if converted is None:
            parsed[nkey] = value
            continue

        if converted.get("kind") == "mem_pair":
            parsed[f"{nkey}_used_mb"] = converted.get("used_mb")
            parsed[f"{nkey}_total_mb"] = converted.get("total_mb")
            parsed[f"{nkey}_ratio_pct"] = converted.get("ratio_pct")
            continue

        val = converted.get("value")
        if nkey in {"step"} and val is not None:
            parsed[nkey] = int(val)
        elif nkey in {"elapsed", "eta"} and val is not None and converted.get("kind") in {"ms", "s", "m", "h"}:
            parsed[f"{nkey}_sec"] = val
        elif nkey in {"s_per_iter"} and val is not None and converted.get("kind") in {"ms", "s", "m", "h"}:
            parsed[nkey] = val
        elif converted.get("kind") in {"kb", "kib", "mb", "mib", "gb", "gib"}:
            parsed[f"{nkey}_mb"] = val
        elif converted.get("kind") == "%":
            parsed[f"{nkey}_pct"] = val
        else:
            parsed[nkey] = val
    return parsed


def is_useful_train_metric(item):
    if not item:
        return False

    primary_keys = {
        "loss",
        "s_per_iter",
        "throughput",
        "lr",
        "learning_rate",
        "grad_norm",
        "mem_used_mb",
        "mem_total_mb",
        "mem_ratio_pct",
        "elapsed_sec",
        "eta_sec",
    }
    if any(k in item for k in primary_keys):
        return True

    # Keep explicit step updates only if they include at least one additional numeric signal.
    if "step" in item:
        numeric_count = 0
        for k, v in item.items():
            if k == "step":
                continue
            if isinstance(v, (int, float)):
                numeric_count += 1
        return numeric_count >= 1

    return False


def parse_train_line(line):
    item = parse_key_values(line)

    # Lightweight fallback for logs that do not use key=value style.
    if "step" not in item:
        m = STEP_RE.search(line)
        if m:
            item["step"] = int(m.group(1))
    if "loss" not in item:
        m = LOSS_RE.search(line)
        if m:
            item["loss"] = try_float(m.group(1))
    if "throughput" not in item:
        m = THROUGHPUT_RE.search(line)
        if m:
            item["throughput"] = try_float(m.group(1))
    if "s_per_iter" not in item:
        m = SITER_RE.search(line)
        if m:
            item["s_per_iter"] = try_float(m.group(1))

    if not is_useful_train_metric(item):
        return None
    return item


def summarize(train_records, gpu_records, warmup_steps):
    tputs = []
    siters = []
    losses = []

    for r in train_records:
        step = r.get("step")
        if step is not None and step <= warmup_steps:
            continue
        if r.get("throughput") is not None:
            tputs.append(r["throughput"])
        if r.get("s_per_iter") is not None:
            siters.append(r["s_per_iter"])
        if r.get("loss") is not None:
            losses.append(r["loss"])

    peak_mem = {}
    for g in gpu_records:
        idx = g.get("gpu_index")
        if idx is None:
            continue
        mem_used = g.get("memory_used_mb")
        if mem_used is None:
            continue
        peak_mem[idx] = max(peak_mem.get(idx, 0.0), mem_used)

    return {
        "train_points": len(train_records),
        "gpu_points": len(gpu_records),
        "avg_throughput": statistics.mean(tputs) if tputs else None,
        "avg_s_per_iter": statistics.mean(siters) if siters else None,
        "last_loss": losses[-1] if losses else None,
        "peak_gpu_mem_mb": {str(k): v for k, v in sorted(peak_mem.items())},
        "warmup_steps": warmup_steps,
    }


def maybe_parse_sbatch_log_template(sbatch_script_path):
    if not sbatch_script_path:
        return None
    p = Path(sbatch_script_path)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                m = SBATCH_OUT_RE.search(line)
                if m:
                    return m.group(1).strip()
    except Exception:
        return None
    return None


def resolve_slurm_log_path(workdir, template_path, job_id):
    if not template_path:
        return None
    rendered = template_path.replace("%j", str(job_id))
    p = Path(rendered)
    if not p.is_absolute():
        p = Path(workdir) / p
    return p.resolve()


def slurm_job_state(job_id):
    try:
        out = subprocess.check_output(
            ["squeue", "-h", "-j", str(job_id), "-o", "%T"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if out:
            return out.splitlines()[0].strip(), True
    except Exception:
        pass

    try:
        out = subprocess.check_output(
            ["sacct", "-j", str(job_id), "--format=State", "--noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        states = [x.strip() for x in out.splitlines() if x.strip()]
        if states:
            # Keep the first non-empty state as a coarse job state.
            state = states[0].split()[0].upper()
            return states[0], state not in SLURM_TERMINAL_STATES
    except Exception:
        pass

    return "UNKNOWN", False


def stream_external_log(
    log_path,
    job_id,
    log_fh,
    metrics_fh,
    train_records,
    gpu_records,
    gpu_backend,
    interval_sec,
    wait_timeout,
):
    start_wait = time.time()
    while not log_path.exists() and (time.time() - start_wait) < wait_timeout:
        time.sleep(1)

    if not log_path.exists():
        return

    offset = 0
    idle_rounds = 0

    while True:
        if log_path.exists():
            with open(log_path, "r", encoding="utf-8", errors="replace") as rf:
                rf.seek(offset)
                chunk = rf.read()
                if chunk:
                    idle_rounds = 0
                    for line in chunk.splitlines(keepends=True):
                        sys.stdout.write(line)
                        log_fh.write(line)
                        parsed = parse_train_line(line)
                        if parsed:
                            train_records.append(parsed)
                            metrics_fh.write(
                                json.dumps({"ts": now_ts(), "type": "train", **parsed}, ensure_ascii=True)
                                + "\n"
                            )
                    metrics_fh.flush()
                    log_fh.flush()
                    offset = rf.tell()
                else:
                    idle_rounds += 1

        time.sleep(interval_sec)

        _, active = slurm_job_state(job_id)
        if not active and idle_rounds >= 2:
            break


def main():
    ap = argparse.ArgumentParser(description="Unified local launcher for training experiments")
    ap.add_argument("--config", required=True, help="Path to JSON or YAML config")
    ap.add_argument("--exp-id", default=None)
    ap.add_argument("--output-root", default="results")
    ap.add_argument("--skip-env-check", action="store_true")
    args = ap.parse_args()

    config = load_config(args.config)
    name = config.get("name", "exp")
    exp_id = args.exp_id or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{name}"

    out_dir = Path(args.output_root) / exp_id
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    env_report_path = out_dir / "env_report.json"
    metrics_path = out_dir / "metrics.jsonl"
    summary_path = out_dir / "summary.json"
    log_path = logs_dir / "train.log"

    env = os.environ.copy()
    env.update(config.get("env") or {})

    env_check_cfg = config.get("env_check") or {}
    env_check_enabled = not args.skip_env_check and env_check_cfg.get("enabled", True)
    gpu_cfg = config.get("gpu") or {}
    gpu_backend = detect_gpu_backend(gpu_cfg.get("backend", "auto"))

    if env_check_enabled:
        env_cmd = [sys.executable, str(Path(__file__).with_name("env_check.py"),)]
        env_cmd += ["--gpu-backend", gpu_backend]
        if env_check_cfg.get("torch_python_cmd"):
            env_cmd += ["--torch-python-cmd", str(env_check_cfg["torch_python_cmd"])]
        if env_check_cfg.get("torch_check_command"):
            env_cmd += ["--torch-check-command", str(env_check_cfg["torch_check_command"])]

        env_out = subprocess.run(env_cmd, capture_output=True, text=True, env=env)
        env_report_path.write_text(env_out.stdout or env_out.stderr, encoding="utf-8")

    cmd = config.get("command")
    if not cmd:
        raise RuntimeError("config missing `command`")

    if isinstance(cmd, str):
        cmd_list = shlex.split(cmd)
    elif isinstance(cmd, list):
        cmd_list = [str(c) for c in cmd]
    else:
        raise RuntimeError("`command` must be a string or list")

    workdir = config.get("workdir", ".")
    warmup_steps = int(config.get("warmup_steps", 0))
    interval_sec = float(config.get("metrics_interval_sec", 2.0))
    slurm_cfg = config.get("slurm") or {}

    start = time.time()
    train_records = []
    gpu_records = []
    slurm_job_id = None
    slurm_log_path = None
    slurm_state = None

    with open(log_path, "w", encoding="utf-8") as log_fh, open(
        metrics_path, "w", encoding="utf-8"
    ) as metrics_fh:
        proc = subprocess.Popen(
            cmd_list,
            cwd=workdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        assert proc.stdout is not None
        submit_stdout_lines = []
        for line in proc.stdout:
            submit_stdout_lines.append(line.rstrip("\n"))
            sys.stdout.write(line)
            log_fh.write(line)

            parsed = parse_train_line(line)
            if parsed:
                item = {"ts": now_ts(), "type": "train", **parsed}
                metrics_fh.write(json.dumps(item, ensure_ascii=True) + "\n")
                train_records.append(parsed)

            metrics_fh.flush()
            log_fh.flush()

        exit_code = proc.wait()

        submit_stdout = "\n".join(submit_stdout_lines)
        m = SBATCH_JOB_RE.search(submit_stdout)
        if exit_code == 0 and m:
            slurm_job_id = m.group(1)

            sbatch_script = cmd_list[-1] if cmd_list else None
            if sbatch_script and not Path(sbatch_script).is_absolute():
                sbatch_script = str((Path(workdir) / sbatch_script).resolve())

            out_template = slurm_cfg.get("out_path_template") or maybe_parse_sbatch_log_template(sbatch_script)
            if out_template:
                slurm_log_path = str(resolve_slurm_log_path(workdir, out_template, slurm_job_id))

            if slurm_log_path:
                link_path = logs_dir / "slurm_job.log"
                try:
                    if link_path.exists() or link_path.is_symlink():
                        link_path.unlink()
                    os.symlink(slurm_log_path, link_path)
                except Exception:
                    pass

            if slurm_cfg.get("follow", False) and slurm_log_path:
                stream_external_log(
                    Path(slurm_log_path),
                    slurm_job_id,
                    log_fh,
                    metrics_fh,
                    train_records,
                    gpu_records,
                    gpu_backend,
                    float(slurm_cfg.get("poll_interval_sec", interval_sec)),
                    float(slurm_cfg.get("wait_for_log_timeout_sec", 900)),
                )
                slurm_state, _ = slurm_job_state(slurm_job_id)

        metrics_fh.flush()

    duration = time.time() - start
    summary = summarize(train_records, gpu_records, warmup_steps)
    status = "success" if exit_code == 0 else "failed"
    if slurm_cfg.get("follow", False) and slurm_state and slurm_state.upper() not in {
        "COMPLETED",
        "COMPLETING",
    }:
        status = "failed"

    summary.update(
        {
            "exp_id": exp_id,
            "name": name,
            "config": str(Path(args.config).resolve()),
            "output_dir": str(out_dir.resolve()),
            "command": cmd_list,
            "duration_sec": duration,
            "exit_code": exit_code,
            "status": status,
            "gpu_backend": gpu_backend,
            "slurm_job_id": slurm_job_id,
            "slurm_log_path": slurm_log_path,
            "slurm_state": slurm_state,
            "artifacts": {
                "log": str(log_path.resolve()),
                "metrics": str(metrics_path.resolve()),
                "env_report": str(env_report_path.resolve()),
            },
        }
    )

    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))

    sys.exit(0 if status == "success" else 1)


if __name__ == "__main__":
    main()
