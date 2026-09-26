#!/usr/bin/env bash
# Run LST-AI v2 (arm64/amd64 CPU image) on one injected subject of the bidsgate lesion derivative.
# Usage: scripts/run_lst_ai.sh sub-XXXX [derivative-dir] [out-dir]
set -uo pipefail
S=$1
DERIV=${2:-derivatives/bidsgate-lesions}
OUT=${3:-derivatives/lst-ai}
IMG=jqmcginnis/lst-ai:v2.0.0rc1-cpu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IN="$ROOT/$DERIV/$S/anat"
mkdir -p "$ROOT/$OUT/$S" "$ROOT/work/lst-tmp/$S" "$ROOT/work/logs"
start=$(date +%s)
docker run --rm -v "$IN":/in:ro -v "$ROOT/$OUT/$S":/out -v "$ROOT/work/lst-tmp/$S":/tmpd "$IMG" \
  --t1 "/in/${S}_T1w.nii.gz" --flair "/in/${S}_FLAIR.nii.gz" --output /out --temp /tmpd \
  --device cpu --segment_only --fast-mode --threads 4 > "$ROOT/work/logs/lst_$S.log" 2>&1
rc=$?
docker run --rm -v "$ROOT":/p --entrypoint chown "$IMG" -R "$(id -u):$(id -g)" "/p/$OUT/$S" "/p/work/lst-tmp/$S" >/dev/null 2>&1
echo "$S exit=$rc seconds=$(( $(date +%s) - start ))" | tee -a "$ROOT/work/logs/lst_runs.log"
