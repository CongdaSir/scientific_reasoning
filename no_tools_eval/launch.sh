#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
python_bin="$project_dir/.venv/bin/python"
export GPU="${GPU:-2}"
export PORT="${PORT:-18041}"
export CONCURRENCY="${CONCURRENCY:-16}"
export MODEL_PATH="${MODEL_PATH:-/data/shared/model/Qwen3-4B}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"
run_dir=${1:-"$project_dir/outputs/no-tools-qwen3-4b-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)"}
if [ "$#" -gt 0 ]; then shift; fi
mkdir -p "$run_dir"
run_dir=$(cd "$run_dir" && pwd)

# Reuse only a healthy endpoint that advertises the requested model.
if "$python_bin" - "$PORT" "$MODEL_NAME" <<'PY'
import sys, httpx
try:
    response = httpx.get(f"http://127.0.0.1:{sys.argv[1]}/v1/models", timeout=3, trust_env=False)
    response.raise_for_status()
    assert sys.argv[2] in {item['id'] for item in response.json()['data']}
except Exception:
    raise SystemExit(1)
PY
then
  server_args=()
else
  # An occupied port must not cause the evaluator to use an unrelated service.
  "$python_bin" - "$PORT" <<'PY'
import socket, sys
with socket.socket() as sock:
    if sock.connect_ex(('127.0.0.1', int(sys.argv[1]))) == 0:
        raise SystemExit('Port is occupied by an unhealthy or different service; choose another PORT.')
PY
  "$python_bin" - "$project_dir/no_tools_eval/serve.sh" "$run_dir" <<'PY'
from pathlib import Path
import subprocess, sys
run = Path(sys.argv[2])
with (run / 'server.log').open('ab', buffering=0) as log:
    process = subprocess.Popen(['nohup', 'bash', sys.argv[1]], stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
(run / 'server.pid').write_text(str(process.pid) + '\n')
PY
  server_args=(--server-pid-file "$run_dir/server.pid")
fi

"$python_bin" - "$run_dir" "$python_bin" \
  --output-dir "$run_dir" \
  --base-url "http://127.0.0.1:$PORT/v1" \
  --model "$MODEL_NAME" \
  --tokenizer-path "$MODEL_PATH" \
  --concurrency "$CONCURRENCY" \
  "${server_args[@]}" "$@" <<'PY'
from pathlib import Path
import os, subprocess, sys
run = Path(sys.argv[1])
pid_path = run / 'evaluator.pid'
if pid_path.exists():
    previous = int(pid_path.read_text())
    command = Path(f'/proc/{previous}/cmdline')
    if command.exists() and b'no_tools_eval' in command.read_bytes():
        raise SystemExit(f'Evaluator {previous} is already running; refusing a duplicate launch.')
with (run / 'nohup.log').open('ab', buffering=0) as log:
    process = subprocess.Popen(['nohup', sys.argv[2], '-u', '-m', 'no_tools_eval', *sys.argv[3:]],
                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
pid_path.write_text(str(process.pid) + '\n')
PY
printf 'Output: %s\nEvaluator PID: %s\n' "$run_dir" "$(cat "$run_dir/evaluator.pid")"
