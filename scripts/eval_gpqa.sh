#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin="$project_dir/.venv/bin/python"

if [ ! -x "$python_bin" ]; then
  echo "Project Python environment not found: $python_bin" >&2
  exit 1
fi

cd -- "$project_dir"

# Default: the entire GPQA Diamond dataset, with option shuffle seed 42.
# Extra arguments override these defaults, e.g. --limit 20 --task-name gpqa-test.
# Resume: --resume --output-dir outputs/<existing-task-timestamp>.
exec "$python_bin" -u -m src.evaluation.gpqa \
  --csv "$project_dir/datasets/gpqa/gpqa_diamond.csv" \
  --config "$project_dir/configs/default.yaml" \
  --env-file "$project_dir/.env" \
  --seed 42 \
  "$@"
