#!/usr/bin/env bash
# Run LST-AI v2 (arm64/amd64 CPU image) on one injected subject of the bidsgate lesion derivative.
# Usage: scripts/run_lst_ai.sh sub-XXXX [derivative-dir] [out-dir]
# The container runs detached, so it survives the calling shell; the log is copied out when it exits.
set -uo pipefail
S=$1
DERIV=${2:-derivatives/bidsgate-lesions}
OUT=${3:-derivatives/lst-ai}
IMG=jqmcginnis/lst-ai:v2.0.0rc1-cpu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IN=$(ls -d "$ROOT/$DERIV/$S"/ses-*/anat 2>/dev/null | head -1)
[ -z "$IN" ] && IN="$ROOT/$DERIV/$S/anat"
T1=$(basename "$(ls "$IN"/*_T1w.nii.gz | head -1)")
FL=$(basename "$(ls "$IN"/*_FLAIR.nii.gz | head -1)")
mkdir -p "$ROOT/$OUT/$S" "$ROOT/work/lst-tmp/$S" "$ROOT/work/logs"
NAME="bidsgate-lst-$S"
docker rm -f "$NAME" >/dev/null 2>&1
start=$(date +%s)
docker run -d --name "$NAME" -v "$IN":/in:ro -v "$ROOT/$OUT/$S":/out -v "$ROOT/work/lst-tmp/$S":/tmpd "$IMG" \
  --t1 "/in/$T1" --flair "/in/$FL" --output /out --temp /tmpd \
  --device cpu --segment_only --fast-mode --threads 4 >/dev/null || { echo "$S exit=start-failed" | tee -a "$ROOT/work/logs/lst_runs.log"; exit 1; }
rc=$(docker wait "$NAME")
docker logs "$NAME" > "$ROOT/work/logs/lst_$S.log" 2>&1
docker rm "$NAME" >/dev/null 2>&1
docker run --rm -v "$ROOT":/p --entrypoint chown "$IMG" -R "$(id -u):$(id -g)" "/p/$OUT/$S" "/p/work/lst-tmp/$S" >/dev/null 2>&1
echo "$S exit=$rc seconds=$(( $(date +%s) - start ))" | tee -a "$ROOT/work/logs/lst_runs.log"
