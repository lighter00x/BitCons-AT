#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Timeline: compatibility alias for the cross-phase run_eval.sh utility.
# Purpose: preserve older commands/documentation that used run_batch_eval.sh;
# it defines no experiment matrix and immediately delegates to run_eval.sh.
# Status: legacy alias; new commands should call run_eval.sh directly.
# Configure DATASET, MODEL, METHODS, DESC,
# BITCONS and BITCONS_CONTRAST through the same environment variables as
# run_eval.sh.
exec "$ROOT_DIR/run_eval.sh" "$@"
