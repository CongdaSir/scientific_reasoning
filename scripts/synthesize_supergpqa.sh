#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin="$project_dir/.venv/bin/python"

if [ ! -x "$python_bin" ]; then
  echo "Project Python environment not found: $python_bin" >&2
  exit 1
fi

cd -- "$project_dir"

# Questions retain their original zero-based index and option order.
# Default: all questions, beginning at index 0, with the DeepSeek API config.
# Examples: --start-index 100; --start-index 100 --limit 20.
# Resume with the same range: --resume --output-dir outputs/<existing-run>.
exec "$python_bin" -u -m src.evaluation.supergpqa \
  --input "$project_dir/datasets/supergpqa/SuperGPQA-all.jsonl" \
  --config "$project_dir/configs/deepseek.yaml" \
  --env-file "$project_dir/.env" \
  "$@"
