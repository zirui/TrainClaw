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

Notes:
- This submits via `sbatch` to `/Users/zirui/code/OmniFlow-amd/examples/run_slurm.sh`.
- Launcher parses `jobid`, resolves Slurm log path (`logs/omniflow.<jobid>.out`), and creates `results/<exp_id>/logs/slurm_job.log` symlink.
- Set `slurm.follow: true` in config to tail job log into `results/<exp_id>/logs/train.log` and parse metrics from it.
- For Slurm jobs that run in container/other venv, set `env_check.enabled: false` or configure `env_check.torch_python_cmd` / `env_check.torch_check_command`.

The launcher writes artifacts under `results/<exp_id>/`:

- `logs/train.log`
- `metrics.jsonl`
- `summary.json`
- `env_report.json`

## Notes

- Config supports JSON or YAML (`YAML` requires `PyYAML`).
- Training log parser is regex-based and framework-agnostic.
- GPU stats backend supports auto-detect for `nvidia-smi`, `rocm-smi`, and basic `amd-smi` fallback.
