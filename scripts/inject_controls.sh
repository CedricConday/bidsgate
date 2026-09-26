#!/usr/bin/env bash
# Inject lesions into ds007908 subjects (default: the eight controls) from this checkout.
# Usage: scripts/inject_controls.sh [sub-XXXX ...]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
SRC=${SRC:-/home/ubuntu/repos/ms-lesion-networks/sourcedata/ds007908}
PY=${PY:-$HOME/.venvs/bidsgate/bin/python}
args=()
for s in "$@"; do args+=(--subject "$s"); done
[ ${#args[@]} -eq 0 ] && for d in "$SRC"/sub-9*/; do args+=(--subject "$(basename "$d")"); done
"$PY" -m bidsgate.cli inject-lesions "$SRC" --out derivatives/bidsgate-lesions --seed 0 "${args[@]}"
