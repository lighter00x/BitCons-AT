#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Backward-compatible entry point. Configure DATASET, MODEL, METHODS, DESC,
# BITCONS and BITCONS_CONTRAST through the same environment variables as
# run_eval.sh.
exec "$ROOT_DIR/run_eval.sh" "$@"
