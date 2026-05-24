#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf '%s\n' "Usage:"
  printf '%s\n' "  run.sh status [--raw]"
  printf '%s\n' "  run.sh exec [--raw] -- <command...>"
  printf '%s\n' "  run.sh agent [options] --prompt TEXT"
  printf '%s\n' ""
  printf '%s\n' "Remote defaults can be overridden with environment variables:"
  printf '%s\n' "  FP8_GEMM_ASM_SSH_ALIAS      default: ssi"
  printf '%s\n' "  FP8_GEMM_ASM_REMOTE_DIR     default: /zirui/code/auto-kernel/fp8_gemm_asm"
  printf '%s\n' "  FP8_GEMM_ASM_CURSOR_CONFIG  default: /zirui/xutils/dotfiles/cursor"
  printf '%s\n' ""
  printf '%s\n' "Agent options:"
  printf '%s\n' "  --prompt TEXT                 Prompt to send"
  printf '%s\n' "  --prompt-file PATH            Read prompt from a local file"
  printf '%s\n' "  --mode ask|plan               Cursor Agent read-only mode"
  printf '%s\n' "  --model MODEL                 Cursor model name"
  printf '%s\n' "  --output-format FORMAT        text, json, or stream-json (default: json)"
  printf '%s\n' "  --raw                         Print raw remote output"
  printf '%s\n' "  --force, --yolo               Auto-approve commands unless denied"
  printf '%s\n' "  --sandbox enabled|disabled    Override Cursor sandbox mode"
  printf '%s\n' "  --continue                    Continue previous Cursor Agent session"
  printf '%s\n' "  --resume [CHAT_ID]            Resume a Cursor Agent session"
  printf '%s\n' "  --no-trust                    Do not pass --trust"
  printf '%s\n' "  --                            Remaining args are forwarded before prompt"
}

need_value() {
  local flag="$1"
  local value="${2-}"
  if [[ -z "$value" ]]; then
    printf 'Missing value for %s\n' "$flag" >&2
    exit 2
  fi
}

quote() {
  printf '%q' "$1"
}

join_quoted() {
  local out=""
  local item
  for item in "$@"; do
    out+=" $(quote "$item")"
  done
  printf '%s' "${out# }"
}

ssh_alias="${FP8_GEMM_ASM_SSH_ALIAS:-ssi}"
remote_dir="${FP8_GEMM_ASM_REMOTE_DIR:-/zirui/code/auto-kernel/fp8_gemm_asm}"
cursor_config="${FP8_GEMM_ASM_CURSOR_CONFIG:-/zirui/xutils/dotfiles/cursor}"

run_remote_script() {
  local script="$1"
  ssh -o BatchMode=yes "$ssh_alias" "bash -lc $(quote "$script")"
}

emit_envelope() {
  local exit_code="$1"
  local stdout_file="$2"
  local stderr_file="$3"
  python3 - "$exit_code" "$stdout_file" "$stderr_file" <<'PY'
import json
import sys
from pathlib import Path

exit_code = int(sys.argv[1])
stdout = Path(sys.argv[2]).read_text(encoding="utf-8", errors="replace")
stderr = Path(sys.argv[3]).read_text(encoding="utf-8", errors="replace")

def parse_json_or_none(text):
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None

parsed = parse_json_or_none(stdout)
summary = ""
if isinstance(parsed, dict):
    for key in ("result", "summary", "message", "text", "output"):
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            summary = value
            break
elif isinstance(parsed, list):
    summary = f"{len(parsed)} JSON records"

if not summary:
    summary = stdout.strip() or stderr.strip()
if len(summary) > 4000:
    summary = summary[:4000] + "\n... truncated ..."
if not summary:
    summary = "remote command completed" if exit_code == 0 else "remote command failed"

envelope = {
    "ok": exit_code == 0,
    "exit_code": exit_code,
    "summary": summary,
    "remote": parsed,
    "stdout": None if parsed is not None else stdout,
    "stderr": stderr,
}
print(json.dumps(envelope, ensure_ascii=True))
if exit_code != 0:
    raise SystemExit(exit_code)
PY
}

run_with_envelope() {
  local raw_output="$1"
  shift

  if [[ "$raw_output" -eq 1 ]]; then
    "$@"
    return
  fi

  local stdout_file
  local stderr_file
  stdout_file="$(mktemp "${TMPDIR:-/tmp}/fp8-gemm-asm-stdout.XXXXXX")"
  stderr_file="$(mktemp "${TMPDIR:-/tmp}/fp8-gemm-asm-stderr.XXXXXX")"
  cleanup() {
    rm -f "$stdout_file" "$stderr_file"
  }
  trap cleanup RETURN

  set +e
  "$@" >"$stdout_file" 2>"$stderr_file"
  local exit_code=$?
  set -e

  emit_envelope "$exit_code" "$stdout_file" "$stderr_file" "$@"
}

remote_preamble() {
  printf 'export CURSOR_CONFIG_DIR=%s; cd %s' "$(quote "$cursor_config")" "$(quote "$remote_dir")"
}

subcommand="${1:-status}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$subcommand" in
  -h|--help)
    usage
    exit 0
    ;;
  status)
    raw_output=0
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --raw)
          raw_output=1
          shift
          ;;
        -h|--help)
          usage
          exit 0
          ;;
        *)
          printf 'Unknown status option: %s\n' "$1" >&2
          exit 2
          ;;
      esac
    done

    status_script="$(remote_preamble)"$'
printf "== host ==\\n"
hostname
printf "\\n== repo ==\\n"
git status --short --branch | sed -n "1,80p"
printf "\\n== slurm jobs ==\\n"
squeue -u "$USER" 2>/dev/null | sed -n "1,50p" || true
printf "\\n== recent results ==\\n"
ls -lt results 2>/dev/null | sed -n "1,25p" || true
printf "\\n== cursor-agent ==\\n"
command -v cursor-agent || true
cursor-agent --version 2>/dev/null || true
'
    run_with_envelope "$raw_output" run_remote_script "$status_script"
    ;;

  exec|shell)
    raw_output=0
    if [[ "${1-}" == "--raw" ]]; then
      raw_output=1
      shift
    fi
    if [[ "${1-}" == "--" ]]; then
      shift
    fi
    if [[ $# -eq 0 ]]; then
      printf 'No command provided for exec.\n' >&2
      usage >&2
      exit 2
    fi
    exec_script="$(remote_preamble); $(join_quoted "$@")"
    run_with_envelope "$raw_output" run_remote_script "$exec_script"
    ;;

  agent)
    prompt=""
    prompt_file=""
    model=""
    mode=""
    output_format="${CURSOR_AGENT_OUTPUT_FORMAT:-json}"
    raw_output=0
    sandbox=""
    trust=1
    force=0
    continue_session=0
    resume_set=0
    resume_chat_id=""
    extra_args=()

    while [[ $# -gt 0 ]]; do
      case "$1" in
        --prompt)
          need_value "$1" "${2-}"
          prompt="$2"
          shift 2
          ;;
        --prompt-file)
          need_value "$1" "${2-}"
          prompt_file="$2"
          shift 2
          ;;
        --mode)
          need_value "$1" "${2-}"
          mode="$2"
          shift 2
          ;;
        --ask)
          mode="ask"
          shift
          ;;
        --plan)
          mode="plan"
          shift
          ;;
        --model)
          need_value "$1" "${2-}"
          model="$2"
          shift 2
          ;;
        --output-format)
          need_value "$1" "${2-}"
          output_format="$2"
          shift 2
          ;;
        --raw)
          raw_output=1
          shift
          ;;
        --force|--yolo)
          force=1
          shift
          ;;
        --sandbox)
          need_value "$1" "${2-}"
          sandbox="$2"
          shift 2
          ;;
        --continue)
          continue_session=1
          shift
          ;;
        --resume)
          resume_set=1
          if [[ $# -gt 1 && "${2:0:1}" != "-" ]]; then
            resume_chat_id="$2"
            shift 2
          else
            shift
          fi
          ;;
        --no-trust)
          trust=0
          shift
          ;;
        -h|--help)
          usage
          exit 0
          ;;
        --)
          shift
          while [[ $# -gt 0 ]]; do
            extra_args+=("$1")
            shift
          done
          ;;
        -*)
          printf 'Unknown agent option: %s\n' "$1" >&2
          usage >&2
          exit 2
          ;;
        *)
          if [[ -z "$prompt" ]]; then
            prompt="$1"
          else
            prompt="${prompt} ${1}"
          fi
          shift
          ;;
      esac
    done

    case "$mode" in
      ""|ask|plan) ;;
      *)
        printf 'Invalid --mode: %s (expected ask or plan)\n' "$mode" >&2
        exit 2
        ;;
    esac

    case "$output_format" in
      text|json|stream-json) ;;
      *)
        printf 'Invalid --output-format: %s (expected text, json, or stream-json)\n' "$output_format" >&2
        exit 2
        ;;
    esac

    case "$sandbox" in
      ""|enabled|disabled) ;;
      *)
        printf 'Invalid --sandbox: %s (expected enabled or disabled)\n' "$sandbox" >&2
        exit 2
        ;;
    esac

    if [[ -n "$prompt_file" ]]; then
      if [[ ! -f "$prompt_file" ]]; then
        printf 'Prompt file not found: %s\n' "$prompt_file" >&2
        exit 2
      fi
      prompt="$(<"$prompt_file")"
    fi

    if [[ -z "$prompt" && ! -t 0 ]]; then
      prompt="$(</dev/stdin)"
    fi

    if [[ -z "$prompt" ]]; then
      printf 'No prompt provided. Use --prompt, --prompt-file, positional text, or stdin.\n' >&2
      usage >&2
      exit 2
    fi

    agent_cmd=("cursor-agent" "--print" "--output-format" "$output_format")
    if [[ "$trust" -eq 1 ]]; then
      agent_cmd+=("--trust")
    fi
    agent_cmd+=("--workspace" "$remote_dir")
    if [[ -n "$model" ]]; then
      agent_cmd+=("--model" "$model")
    fi
    if [[ -n "$mode" ]]; then
      agent_cmd+=("--mode" "$mode")
    fi
    if [[ "$force" -eq 1 ]]; then
      agent_cmd+=("--force")
    fi
    if [[ -n "$sandbox" ]]; then
      agent_cmd+=("--sandbox" "$sandbox")
    fi
    if [[ "$continue_session" -eq 1 ]]; then
      agent_cmd+=("--continue")
    fi
    if [[ "$resume_set" -eq 1 ]]; then
      if [[ -n "$resume_chat_id" ]]; then
        agent_cmd+=("--resume" "$resume_chat_id")
      else
        agent_cmd+=("--resume")
      fi
    fi
    if [[ "${extra_args[*]-}" != "" ]]; then
      agent_cmd+=("${extra_args[@]}")
    fi
    agent_cmd+=("--" "$prompt")

    agent_script="$(remote_preamble); $(join_quoted "${agent_cmd[@]}")"
    run_with_envelope "$raw_output" run_remote_script "$agent_script"
    ;;

  *)
    printf 'Unknown subcommand: %s\n' "$subcommand" >&2
    usage >&2
    exit 2
    ;;
esac
