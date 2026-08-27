#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Backward-compatible masked-evaluation entry point. The attack remains the
# documented attack-original-then-mask evaluation implemented by eval.py.
export BITCONS="${BITCONS:-true}"
export ENABLE_BITCONS_TEST="${ENABLE_BITCONS_TEST:-1}"
export BITCONS_PLANES="${BITCONS_PLANES:-3 4 5}"

exec "$ROOT_DIR/run_eval.sh" "$@"
