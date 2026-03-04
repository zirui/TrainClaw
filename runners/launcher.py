#!/usr/bin/env python3
import argparse
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


STEP_RE = re.compile(r"(?:step|iter)\s*[=: ]\s*(\d+)", re.IGNORECASE)
LOSS_RE = re.compile(r"\bloss\s*[=: ]\s*([-+0-9.eE]+)", re.IGNORECASE)
THROUGHPUT_RE = re.compile(
    r"(?:throughput|samples/s|tokens/s)\s*[=: ]\s*([-+0-9.eE]+)", re.IGNORECASE
)
SITER_RE = re.compile(r"(?:s/iter|sec/iter|step\s*time)\s*[=: ]\s*([-+0-9.eE]+)", re.IGNORECASE)


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


def query_gpu_snapshot():
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
            }
        )
    return records


def parse_train_line(line):
    step = STEP_RE.search(line)
    loss = LOSS_RE.search(line)
    tput = THROUGHPUT_RE.search(line)
    siter = SITER_RE.search(line)

    item = {}
    if step:
        item["step"] = int(step.group(1))
    if loss:
        item["loss"] = try_float(loss.group(1))
    if tput:
        item["throughput"] = try_float(tput.group(1))
    if siter:
        item["s_per_iter"] = try_float(siter.group(1))

    return item if item else None


def gpu_sampler(stop_event, interval_sec, metrics_fh):
    while not stop_event.is_set():
        ts = now_ts()
        for rec in query_gpu_snapshot():
            metrics_fh.write(json.dumps({"ts": ts, "type": "gpu", **rec}, ensure_ascii=True) + "\n")
        metrics_fh.flush()
        stop_event.wait(interval_sec)


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
        idx = g["gpu_index"]
        peak_mem[idx] = max(peak_mem.get(idx, 0.0), g.get("memory_used_mb") or 0.0)

    return {
        "train_points": len(train_records),
        "gpu_points": len(gpu_records),
        "avg_throughput": statistics.mean(tputs) if tputs else None,
        "avg_s_per_iter": statistics.mean(siters) if siters else None,
        "last_loss": losses[-1] if losses else None,
        "peak_gpu_mem_mb": {str(k): v for k, v in sorted(peak_mem.items())},
        "warmup_steps": warmup_steps,
    }


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

    if not args.skip_env_check:
        env_out = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("env_check.py"))],
            capture_output=True,
            text=True,
        )
        env_report_path.write_text(env_out.stdout or env_out.stderr, encoding="utf-8")

    env = os.environ.copy()
    env.update(config.get("env") or {})

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

    start = time.time()
    train_records = []
    gpu_records = []

    with open(log_path, "w", encoding="utf-8") as log_fh, open(
        metrics_path, "w", encoding="utf-8"
    ) as metrics_fh:
        stop_event = threading.Event()
        t = threading.Thread(target=gpu_sampler, args=(stop_event, interval_sec, metrics_fh), daemon=True)
        t.start()

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
        for line in proc.stdout:
            sys.stdout.write(line)
            log_fh.write(line)
            parsed = parse_train_line(line)
            if parsed:
                item = {"ts": now_ts(), "type": "train", **parsed}
                metrics_fh.write(json.dumps(item, ensure_ascii=True) + "\n")
                train_records.append(parsed)
            metrics_fh.flush()
            log_fh.flush()

            # Collect a snapshot opportunistically to reduce missing data for short jobs.
            ts = now_ts()
            for rec in query_gpu_snapshot():
                metrics_fh.write(json.dumps({"ts": ts, "type": "gpu", **rec}, ensure_ascii=True) + "\n")
                gpu_records.append(rec)

        exit_code = proc.wait()
        stop_event.set()
        t.join(timeout=2)

        # Final sample
        ts = now_ts()
        for rec in query_gpu_snapshot():
            metrics_fh.write(json.dumps({"ts": ts, "type": "gpu", **rec}, ensure_ascii=True) + "\n")
            gpu_records.append(rec)
        metrics_fh.flush()

    duration = time.time() - start
    summary = summarize(train_records, gpu_records, warmup_steps)
    summary.update(
        {
            "exp_id": exp_id,
            "name": name,
            "config": str(Path(args.config).resolve()),
            "output_dir": str(out_dir.resolve()),
            "command": cmd_list,
            "duration_sec": duration,
            "exit_code": exit_code,
            "status": "success" if exit_code == 0 else "failed",
            "artifacts": {
                "log": str(log_path.resolve()),
                "metrics": str(metrics_path.resolve()),
                "env_report": str(env_report_path.resolve()),
            },
        }
    )

    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
