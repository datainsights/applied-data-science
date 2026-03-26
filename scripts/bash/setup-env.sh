#!/usr/bin/env bash

set -euo pipefail

echo "==> CDI Data Science Foundations Premium: environment setup"

# Move to project root if script is run from scripts/bash/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

echo "Project root: ${PROJECT_ROOT}"

# Check Python
if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 is not installed or not available in PATH."
  exit 1
fi

echo "Python found: $(python3 --version)"

# Create virtual environment if missing
if [ ! -d ".venv" ]; then
  echo "==> Creating virtual environment in .venv"
  python3 -m venv .venv
else
  echo "==> Virtual environment already exists: .venv"
fi

# Activate environment
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Active Python: $(which python)"
python --version

# Upgrade packaging tools
echo "==> Upgrading pip, setuptools, and wheel"
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
if [ -f "requirements.txt" ]; then
  echo "==> Installing packages from requirements.txt"
  pip install -r requirements.txt
else
  echo "Error: requirements.txt not found in project root."
  exit 1
fi

# Optional: register Jupyter kernel
if python -c "import ipykernel" >/dev/null 2>&1; then
  echo "==> Registering Jupyter kernel: ds-premium"
  python -m ipykernel install --user --name ds-premium --display-name "Python (.venv) - DS Premium" || true
else
  echo "==> ipykernel not available; skipping kernel registration"
fi

# Check Quarto separately, since the project uses Quarto-first workflow
if command -v quarto >/dev/null 2>&1; then
  echo "==> Quarto found: $(quarto --version)"
else
  echo "==> Warning: Quarto is not installed or not in PATH."
  echo "   Install Quarto from https://quarto.org"
fi

echo "==> Environment setup complete"
echo
echo "Next steps:"
echo "  source .venv/bin/activate"
echo "  bash scripts/bash/build.sh"