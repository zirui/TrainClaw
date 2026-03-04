#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Aggregate metrics.jsonl into a small summary")
    ap.add_argument("--metrics", required=True, help="Path to metrics.jsonl")
    ap.add_argument("--warmup-steps", type=int, default=0)
    args = ap.parse_args()

    train = []
    gpu = []

    with open(args.metrics, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("type") == "train":
                train.append(item)
            elif item.get("type") == "gpu":
                gpu.append(item)

    tputs = []
    siters = []
    for t in train:
        if t.get("step") is not None and t["step"] <= args.warmup_steps:
            continue
        if isinstance(t.get("throughput"), (int, float)):
            tputs.append(t["throughput"])
        if isinstance(t.get("s_per_iter"), (int, float)):
            siters.append(t["s_per_iter"])

    peak = {}
    for g in gpu:
        idx = str(g.get("gpu_index"))
        mem = g.get("memory_used_mb")
        if isinstance(mem, (int, float)):
            peak[idx] = max(peak.get(idx, 0.0), float(mem))

    summary = {
        "metrics_path": str(Path(args.metrics).resolve()),
        "train_points": len(train),
        "gpu_points": len(gpu),
        "avg_throughput": statistics.mean(tputs) if tputs else None,
        "avg_s_per_iter": statistics.mean(siters) if siters else None,
        "peak_gpu_mem_mb": peak,
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
