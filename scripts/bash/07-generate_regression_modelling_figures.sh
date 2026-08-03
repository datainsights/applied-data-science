#!/usr/bin/env bash

set -euo pipefail

project_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$project_root"

python scripts/python/07-generate_regression_modelling_figures.py
