#!/usr/bin/env bash
set -e

# First argument: model name/path; remaining arguments are forwarded to vLLM.
model_path=${1:-Qwen/Qwen3-8B}
if [ "$#" -gt 0 ]; then shift; fi
vllm serve "$model_path" \
  --served-model-name "${SERVED_MODEL_NAME:-Qwen/Qwen3-8B}" \
  --host 127.0.0.1 \
  --port 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  "$@"
