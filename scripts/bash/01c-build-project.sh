#!/usr/bin/env bash

set -euo pipefail

echo "Checking Python environment..."
python scripts/python/01b_check_environment.py

echo
echo "Rendering Quarto project..."
quarto render

echo
echo "Build complete."
