#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Timeline: Phase 1/2 masked-view diagnostic compatibility entry point.
# Purpose: reproduce the historical attack-original-then-mask metric for the
# legacy BitCons masking stream. This is a mechanism diagnostic, not an adaptive
# attack against a masked decision function and not the current main protocol.
# Status: historical reproduction only. RA-WC-BitCons uses ordinary inference
# and must be evaluated without ENABLE_BITCONS_TEST.
# The attack remains the documented attack-original-then-mask evaluation in
# eval.py.
export BITCONS="${BITCONS:-true}"
export ENABLE_BITCONS_TEST="${ENABLE_BITCONS_TEST:-1}"
export BITCONS_PLANES="${BITCONS_PLANES:-3 4 5}"

exec "$ROOT_DIR/run_eval.sh" "$@"
