#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"

python_executable="$project_root/.venv/bin/python"

if [[ ! -x "$python_executable" ]]; then
  python_executable="python"
fi

"$python_executable" scripts/python/05-generate_statistical_inference_figures.py
