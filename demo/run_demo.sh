#!/usr/bin/env bash
# run_demo.sh — launch the CT -> Photoreal local demo server.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec venv/bin/python demo/server.py --open "$@"
