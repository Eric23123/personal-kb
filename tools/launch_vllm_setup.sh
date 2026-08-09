#!/usr/bin/env bash
set -euo pipefail

setup_script="${1:-${PERSONAL_KB_VLLM_SETUP_SCRIPT:-$HOME/setup_vllm.sh}}"
log_file="${PERSONAL_KB_VLLM_SETUP_LOG:-/tmp/personal-kb-vllm-setup.log}"

if [[ ! -f "$setup_script" ]]; then
  printf 'Setup script not found: %s\n' "$setup_script" >&2
  printf 'Pass its path as the first argument or set PERSONAL_KB_VLLM_SETUP_SCRIPT.\n' >&2
  exit 1
fi

nohup bash "$setup_script" >"$log_file" 2>&1 &
printf 'Started PID %s; log: %s\n' "$!" "$log_file"
