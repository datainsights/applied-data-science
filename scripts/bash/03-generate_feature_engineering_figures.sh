#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
PLOT_SCRIPT="${PROJECT_ROOT}/scripts/python/03-generate_feature_engineering_figures.py"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Error: repository Python environment not found at ${PYTHON_BIN}" >&2
  echo "Create and install the repository-specific .venv before running this script." >&2
  exit 1
fi

if [[ ! -f "${PLOT_SCRIPT}" ]]; then
  echo "Error: Chapter 03 plotting script not found at ${PLOT_SCRIPT}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" "${PLOT_SCRIPT}"

echo "Chapter 03 figures created in results/figures/."
