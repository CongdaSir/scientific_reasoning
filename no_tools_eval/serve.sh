#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin="$project_dir/.venv/bin/python"
export PATH="$project_dir/.venv/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${GPU:-2}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# Use the Python development headers already installed for this environment.
python_dev_include="$project_dir/.venv/python-dev/usr/include"
if [ -d "$python_dev_include" ]; then
  python_version=$("$python_bin" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  export CPATH="$python_dev_include/python${python_version}:$python_dev_include${CPATH:+:$CPATH}"
fi

exec "$python_bin" -u -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH:-/data/shared/model/Qwen3-4B}" \
  --served-model-name "${MODEL_NAME:-Qwen/Qwen3-4B}" \
  --host 127.0.0.1 \
  --port "${PORT:-18041}" \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.55}" \
  --max-model-len 40960 \
  --max-num-seqs "${CONCURRENCY:-16}" \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --reasoning-parser qwen3 \
  --generation-config vllm \
  --seed 42 \
  "$@"
