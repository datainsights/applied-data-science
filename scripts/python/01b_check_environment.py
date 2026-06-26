from pathlib import Path
import sys

required_dirs = [
    "data/raw",
    "data/processed",
    "data/reference",
    "scripts/bash",
    "scripts/python",
    "notebooks",
    "models",
    "reports",
    "docs",
]

required_files = [
    "requirements.txt",
    "_quarto.yml",
]

missing_dirs = [path for path in required_dirs if not Path(path).exists()]
missing_files = [path for path in required_files if not Path(path).exists()]

print("Python executable:", sys.executable)
print("Python version:", sys.version.split()[0])
print()

if missing_dirs:
    print("Missing directories:")
    for path in missing_dirs:
        print(f"- {path}")
    print()

if missing_files:
    print("Missing files:")
    for path in missing_files:
        print(f"- {path}")
    print()

if missing_dirs or missing_files:
    raise SystemExit("Environment check failed.")

print("Environment check passed.")
