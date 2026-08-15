#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
exec conda run -n labutopia python main.py \
  --config-name example_protocol \
  --headless \
  --/rtx/verifyDriverVersion/enabled=false "$@"
