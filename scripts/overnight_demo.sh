#!/usr/bin/env bash
# The whole LST-AI demo, unattended: inject lesions into the ds007908 controls,
# segment every injected subject with LST-AI v2 (CPU, one at a time), score, scorecard.
# Safe to re-run: subjects that are already injected or segmented are skipped.
#
# Usage (detached, survives logout):
#   nohup setsid scripts/overnight_demo.sh > work/logs/overnight.log 2>&1 &
# Progress:  tail -f ~/repos/bidsgate/work/logs/overnight.log
# Result:    ~/repos/bidsgate/results/lst-ai-v2/scorecard_lesions.html
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
SRC=${SRC:-/home/ubuntu/repos/ms-lesion-networks/sourcedata/ds007908}
PY=${PY:-$HOME/.venvs/bidsgate/bin/python}
SUBJECTS=${SUBJECTS:-sub-9*}   # the eight healthy controls of ds007908; the patients are sub-0*
DERIV=derivatives/bidsgate-lesions
OUT=derivatives/lst-ai
mkdir -p work/logs results/lst-ai-v2

echo "== $(date -Is) inject (skips subjects that already have a truth file)"
for d in "$SRC"/$SUBJECTS/; do
  S=$(basename "$d")
  if ls "$DERIV/$S"/*/anat/*desc-lesion_truth.json >/dev/null 2>&1; then echo "$S already injected"; continue; fi
  "$PY" -m bidsgate.cli inject-lesions "$SRC" --out "$DERIV" --subject "$S" --seed 0 2>&1
done

echo "== $(date -Is) LST-AI v2, one subject at a time"
scripts/run_lst_ai_batch.sh "$DERIV" "$OUT"

echo "== $(date -Is) score"
SEG=$(ls "$OUT"/sub-*/*seg-lst.nii.gz 2>/dev/null | head -1)
if [ -z "$SEG" ]; then echo "no segmentation found under $OUT; nothing to score"; exit 1; fi
# LST-AI names its output after the input: <base>_space-flair_seg-lst.nii.gz (fallback: any seg-lst file in the folder)
"$PY" -m bidsgate.cli score-lesions --truth "$DERIV" \
  --pred "$OUT/{subject}/{base}_space-flair_seg-lst.nii.gz" \
  --pipeline "LST-AI v2.0.0rc1 (CPU, fast mode)" --out results/lst-ai-v2 2>&1 \
|| "$PY" - "$OUT" "$DERIV" <<'EOF'
import glob, os, subprocess, sys
out, deriv = sys.argv[1], sys.argv[2]
# fall back: link whatever seg-lst file each subject has to the expected name, then score again
for seg in glob.glob(f"{out}/sub-*/*seg-lst.nii.gz"):
    sub = seg.split("/")[-2]
    truth = glob.glob(f"{deriv}/{sub}/*/anat/*desc-lesion_truth.json")
    if not truth:
        continue
    base = os.path.basename(truth[0]).replace("_desc-lesion_truth.json", "")
    want = f"{out}/{sub}/{base}_space-flair_seg-lst.nii.gz"
    if not os.path.exists(want):
        os.symlink(os.path.basename(seg), want)
sys.exit(subprocess.call([sys.executable, "-m", "bidsgate.cli", "score-lesions", "--truth", deriv,
                          "--pred", f"{out}/{{subject}}/{{base}}_space-flair_seg-lst.nii.gz",
                          "--pipeline", "LST-AI v2.0.0rc1 (CPU, fast mode)", "--out", "results/lst-ai-v2"]))
EOF
echo "== $(date -Is) done; scorecard at results/lst-ai-v2/scorecard_lesions.html"
