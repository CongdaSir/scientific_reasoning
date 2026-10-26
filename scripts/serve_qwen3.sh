#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=6

# Triton needs Python headers when compiling its CUDA runtime helpers.
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_dev_include="$project_dir/.venv/python-dev/usr/include"
if [ -d "$python_dev_include" ]; then
  python_version=$("$project_dir/.venv/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  export CPATH="$python_dev_include/python${python_version}:$python_dev_include${CPATH:+:$CPATH}"
fi

# First argument: model name/path; remaining arguments are forwarded to vLLM.
model_path=${1:-/data/shared/model/Qwen3-8B}
if [ "$#" -gt 0 ]; then shift; fi
vllm serve "$model_path" \
  --served-model-name "${SERVED_MODEL_NAME:-Qwen/Qwen3-8B}" \
  --tensor-parallel-size 1 \
  --host 127.0.0.1 \
  --port 8001 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  "$@"
