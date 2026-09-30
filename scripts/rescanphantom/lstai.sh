#!/usr/bin/env bash
# lstai.sh T1 FLAIR OUT_MASK: LST-AI v2 (CPU container, segment only, fast mode) as a rescanphantom segmenter
set -euo pipefail
T1=$(readlink -f "$1"); FL=$(readlink -f "$2"); OUT=$(readlink -f "$3"); D=$(dirname "$OUT"); W=$(mktemp -d -p "$D")
docker run --rm -v "$(dirname $T1)":/in:ro -v "$W":/out jqmcginnis/lst-ai:v2.0.0rc1-cpu \
  --t1 /in/$(basename $T1) --flair /in/$(basename $FL) --output /out --temp /out/tmp --device cpu --segment_only --fast-mode --threads 4 >/dev/null 2>&1
docker run --rm -v "$W":/p --entrypoint chown jqmcginnis/lst-ai:v2.0.0rc1-cpu -R "$(id -u):$(id -g)" /p
cp "$W/space-flair_seg-lst.nii.gz" "$OUT"; mv "$W" "$D/.done-$(basename $OUT .nii.gz)"
