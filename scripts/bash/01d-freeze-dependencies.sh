#!/usr/bin/env bash

set -euo pipefail

echo "Freezing Python dependency versions..."

python -m pip freeze > requirements-lock.txt

echo "Dependency lock file written to requirements-lock.txt"
