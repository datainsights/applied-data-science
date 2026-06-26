#!/usr/bin/env bash

set -euo pipefail

echo "Creating Applied Data Science System project structure..."

mkdir -p data/raw data/processed data/reference
mkdir -p scripts/bash scripts/python
mkdir -p notebooks models reports docs

echo "Project folders created."
echo
echo "Current project structure:"
find . -maxdepth 2 -type d | sort
