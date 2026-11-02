#!/usr/bin/env bash
# One-command setup: creates an isolated venv with uv and installs deps.
set -euo pipefail
cd "$(dirname "$0")"

echo "==> Creating venv (.venv) and installing dependencies with uv..."
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt

echo
echo "Setup complete. Run with a webcam connected:"
echo "  .venv/bin/python 'human detection.py'              # detection only, prints bounding boxes"
echo "  .venv/bin/python human_detection_switch_toggle.py   # detection + on-screen toggle indicator"
echo "  .venv/bin/python Simulation.py                      # detection + separate PyQt5 status window"
