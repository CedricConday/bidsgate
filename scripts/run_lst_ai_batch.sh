#!/usr/bin/env bash
# Run LST-AI v2 on every injected subject that has no segmentation yet, one at a time.
# Usage: nohup setsid scripts/run_lst_ai_batch.sh > work/logs/lst_batch.log 2>&1 &
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DERIV=${1:-derivatives/bidsgate-lesions}
OUT=${2:-derivatives/lst-ai}
for d in "$ROOT/$DERIV"/sub-*/; do
  S=$(basename "$d")
  if ls "$ROOT/$OUT/$S"/*seg-lst.nii.gz >/dev/null 2>&1; then echo "$S already segmented"; continue; fi
  "$ROOT/scripts/run_lst_ai.sh" "$S" "$DERIV" "$OUT"
done
echo "batch done $(date -Is)"
