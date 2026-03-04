# TrainClaw Phase 1 (MVP)

This repository contains a minimal Phase 1 implementation for local OpenClaw + Claude workflow:

- Unified launcher: `runners/launcher.py`
- Environment check: `runners/env_check.py`
- Local run entry: `scripts/run_local.sh`
- Remote run wrapper over SSH: `scripts/run_remote.sh`
- Slurm submit wrapper: `scripts/submit_slurm.sh`
- Metrics aggregation helper: `scripts/collect_metrics.py`

## Quickstart

```bash
./scripts/run_local.sh configs/exp/phase1_smoke.yaml
```

OmniFlow (text-to-video) Slurm submit example:

```bash
./scripts/run_local.sh configs/exp/omniflow_slurm.yaml
```

Notes: this submits via `sbatch` to `/Users/zirui/code/OmniFlow-amd/examples/run_slurm.sh`; training logs will land in that repo under `logs/omniflow.<jobid>.out`. Phase 1 launcher records submission metadata but does not stream Slurm logs.

The launcher writes artifacts under `results/<exp_id>/`:

- `logs/train.log`
- `metrics.jsonl`
- `summary.json`
- `env_report.json`

## Notes

- Config supports JSON or YAML (`YAML` requires `PyYAML`).
- Training log parser is regex-based and framework-agnostic.
- GPU stats are sampled via `nvidia-smi` when available.
