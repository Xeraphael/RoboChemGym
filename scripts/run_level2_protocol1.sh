#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
exec conda run -n labutopia python main.py \
  --config-name Level2_Protocol1 \
  --headless \
  --/rtx/verifyDriverVersion/enabled=false "$@"
