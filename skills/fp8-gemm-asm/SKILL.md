---
name: fp8-gemm-asm
description: Operate the fp8_gemm_asm auto-kernel project on the ssi cluster from OpenClaw or Feishu. Use when checking task status, launching or supervising optimization runs, inspecting results, or delegating coding/debugging work to the remote cursor-agent for fp8_gemm_asm.
metadata: {"openclaw":{"requires":{"bins":["ssh"]}}}
---

# fp8_gemm_asm on ssi

Use this skill for the remote `fp8_gemm_asm` kernel optimization project on `ssi`.

## Remote Defaults

- SSH alias: `ssi`
- Project directory: `/zirui/code/auto-kernel/fp8_gemm_asm`
- Cursor config: `CURSOR_CONFIG_DIR=/zirui/xutils/dotfiles/cursor`
- Cursor Agent is available on `ssi` through a login shell after exporting the Cursor config directory.

## Wrapper

Run the wrapper from this skill directory:

```bash
{baseDir}/scripts/run.sh status
```

The wrapper SSHes to `ssi`, starts a remote login shell, exports `CURSOR_CONFIG_DIR`, and enters the project directory before running commands.
By default it prints a stable JSON envelope:

```json
{"ok": true, "exit_code": 0, "summary": "...", "remote": null, "stdout": "...", "stderr": ""}
```

Use `--raw` only when the user explicitly wants raw remote output.

## Common Workflows

Check current repo state, Slurm jobs, recent result files, and remote Cursor Agent availability:

```bash
{baseDir}/scripts/run.sh status
```

Run a trusted shell command in the project directory:

```bash
{baseDir}/scripts/run.sh exec -- git status --short --branch
{baseDir}/scripts/run.sh exec -- bash -lc 'squeue -u "$USER"'
```

Delegate code or debugging work to the remote Cursor Agent:

```bash
{baseDir}/scripts/run.sh agent --prompt "检查当前 fp8_gemm_asm 任务状态并总结下一步"
```

Run remote Cursor Agent in read-only modes:

```bash
{baseDir}/scripts/run.sh agent --mode ask --prompt "解释当前 STATUS.md 中的最新进展"
{baseDir}/scripts/run.sh agent --mode plan --prompt "制定下一轮 kernel 优化计划"
```

For a long prompt, write it to a local file and use:

```bash
{baseDir}/scripts/run.sh agent --prompt-file /tmp/fp8_gemm_prompt.txt
```

## Safety Rules

- Treat `exec` commands as trusted remote shell commands.
- Prefer `status` before launching or changing long-running jobs.
- Use `agent --mode ask` or `agent --mode plan` for read-only inspection from Feishu/OpenClaw.
- Do not include secrets in prompts. Reference paths or environment variable names instead.
- Confirm destructive commands, process kills, or broad file deletion with the user first.

## Script Options

- `status`: show repository state, Slurm jobs, recent results, and Cursor Agent status.
- `exec -- <command...>`: run a trusted command in the remote project directory.
- `agent --prompt TEXT`: run remote `cursor-agent --print` in the project workspace.
- `agent --prompt-file PATH`: read the prompt from a local file.
- `agent --mode ask|plan`: run Cursor Agent in read-only ask or plan mode.
- `agent --model MODEL`: pass a Cursor model name.
- `agent --force` or `agent --yolo`: allow Cursor Agent commands unless denied by remote Cursor config.
- `agent --continue` or `agent --resume [CHAT_ID]`: continue or resume a remote Cursor Agent session.
