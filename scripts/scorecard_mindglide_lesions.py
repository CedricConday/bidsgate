"""Scorecard #3: MindGlide's lesion label against the injected lesions of scorecard #1.

Unattended and resumable: every step skips when its output exists, so a re-run picks up
where the last one stopped. Run with the bidsgate venv's python from the repo root:

    nohup setsid nice -n 10 ~/.venvs/bidsgate/bin/python scripts/scorecard_mindglide_lesions.py \
        --after work/scorecard2/run.log >> work/scorecard3/run.log 2>&1 < /dev/null &

Design (see work/scorecard3/RUN.md):
* the same six ds007908 controls and the same injected images as the LST-AI scorecard
  (derivatives/bidsgate-lesions, seed 0, twelve lesions each), scored by the same
  `score-lesions` with the same defaults, so the two tools are directly comparable;
* MindGlide 1.3.0 takes ONE image of any sequence; it gets the injected FLAIR, at its
  defaults (CPU, largest-component cleanup on, internal reorientation on); label 18 is
  "Lesion" and is the prediction;
* condition `flair`: the injected FLAIR as is (0.8 mm isotropic);
* condition `flair-thick`: the same FLAIR averaged into thick axial slabs (6 slices,
  4.8 mm), the "any quality" claim; MindGlide returns its labels on that thick grid and
  they are repeated back onto the truth grid slice by slice before scoring;
* per condition: scores_lesions.json + scorecard_lesions.html from score-lesions, a
  per-lesion TSV, and a height/size table next to LST-AI v2's numbers on the same lesions.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
TRUTH = REPO / "derivatives" / "bidsgate-lesions"
LST = REPO / "results" / "lst-ai-v2" / "scores_lesions.json"
MG_PY = Path(os.environ.get("MG_PY", "/home/ubuntu/.venvs/mindglide/bin/python"))
PY = sys.executable
W = REPO / "work" / "scorecard3"
OUT = REPO / "results" / "mindglide-lesions"
SUBJECTS = ["sub-9000", "sub-9001", "sub-9002", "sub-9003", "sub-9004", "sub-9008"]
SESSION = "ses-1"
LESION = 18  # MindGlide 1.3.0 label table: 18 = Lesion
SLAB = 6  # flair-thick: slices averaged per slab along the superior-inferior axis
CONDITIONS = ["flair", "flair-thick"]
THREADS = int(os.environ.get("THREADS", "4"))
GO = W / "GO"  # touch this file to start without waiting for scorecard #2

SEGMENT = r'''
import sys, time, torch
torch.set_num_threads(int(sys.argv[3])); torch.set_num_interop_threads(1)
from mindglide import segment
t = time.time()
segment(sys.argv[1], sys.argv[2], device="cpu")
print(f"MINDGLIDE_SECONDS {time.time() - t:.0f}")
'''


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S UTC ", time.gmtime()) + msg, flush=True)


def wait_for(logfile: Path) -> None:
    """Block until scorecard #2 has written its last line (it owns the CPU until then)."""
    warned = False
    while True:
        if GO.exists():
            log(f"{GO} exists; starting")
            return
        if logfile.exists() and "scorecard2 done" in logfile.read_text(errors="replace"):
            log(f"'scorecard2 done' found in {logfile}; starting")
            return
        alive = subprocess.run(["pgrep", "-f", "scorecard_fastsurfer_atrophy"], capture_output=True,
                               check=False).stdout.strip()
        if not alive and not warned:
            log(f"WARNING scorecard #2 is not running and has not finished; still waiting. "
                f"Touch {GO} to start anyway.")
            warned = True
        time.sleep(300)


def flair(sub: str) -> Path:
    return TRUTH / sub / SESSION / "anat" / f"{sub}_{SESSION}_FLAIR.nii.gz"


def si_axis(img: nib.Nifti1Image) -> int:
    return [c in "SI" for c in nib.aff2axcodes(img.affine)].index(True)


def thicken(src: Path, dst: Path) -> None:
    """Average SLAB consecutive slices along the S-I axis; the last slab takes what is left."""
    img = nib.load(src)
    data = np.asarray(img.dataobj, dtype=np.float32)
    ax = si_axis(img)
    n = data.shape[ax]
    slabs = [data.take(range(i, min(i + SLAB, n)), axis=ax).mean(axis=ax) for i in range(0, n, SLAB)]
    thick = np.stack(slabs, axis=ax)
    aff = img.affine.copy()
    aff[:3, 3] += aff[:3, ax] * (SLAB - 1) / 2  # centre of the first slab
    aff[:3, ax] *= SLAB
    out = nib.Nifti1Image(thick, aff)
    out.header.set_xyzt_units(*img.header.get_xyzt_units())
    dst.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, dst)


def unthicken(seg: np.ndarray, ref: nib.Nifti1Image) -> np.ndarray:
    """Repeat each thick slab's labels over the SLAB fine slices it was averaged from."""
    ax = si_axis(ref)
    return np.repeat(seg, SLAB, axis=ax).take(range(ref.shape[ax]), axis=ax)


def mindglide_input(cond: str, sub: str) -> Path:
    if cond == "flair":
        return flair(sub)
    p = W / "inputs" / cond / f"{sub}_{SESSION}_FLAIR.nii.gz"
    if not p.exists():
        thicken(flair(sub), p)
    return p


def segment(cond: str, sub: str) -> Path:
    """MindGlide on one scan; returns its full label map. Skips when it exists."""
    seg = W / "seg" / cond / f"{sub}_{SESSION}_mindglide_dseg.nii.gz"
    if seg.exists():
        return seg
    seg.parent.mkdir(parents=True, exist_ok=True)
    inp = mindglide_input(cond, sub)
    env = dict(os.environ, OMP_NUM_THREADS=str(THREADS), MKL_NUM_THREADS=str(THREADS),
               OPENBLAS_NUM_THREADS=str(THREADS))
    log(f"MindGlide start {sub} {cond}")
    t0 = time.time()
    with open(seg.parent / f"{sub}_{SESSION}.mindglide.log", "w") as fh:
        r = subprocess.run([str(MG_PY), "-c", SEGMENT, str(inp), str(seg), str(THREADS)], env=env,
                           stdout=fh, stderr=subprocess.STDOUT, check=False)
    if r.returncode != 0 or not seg.exists():
        raise SystemExit(f"MindGlide failed for {sub} {cond} (exit {r.returncode}); see {fh.name}")
    log(f"MindGlide done {sub} {cond}: {(time.time() - t0) / 60:.1f} min")
    return seg


def prediction(cond: str, sub: str, seg: Path) -> Path:
    """Binary lesion mask (label 18) on the truth grid, where score-lesions looks for it."""
    pred = W / "pred" / cond / sub / f"{sub}_{SESSION}_desc-mindglide_mask.nii.gz"
    if pred.exists():
        return pred
    ref = nib.load(flair(sub))
    img = nib.load(seg)
    labels = np.asarray(img.dataobj)
    if not np.allclose(img.affine, nib.load(mindglide_input(cond, sub)).affine, atol=1e-3):
        raise SystemExit(f"{sub} {cond}: label map affine differs from MindGlide's input")
    if cond == "flair-thick":
        labels = unthicken(labels, ref)
    if labels.shape != ref.shape[:3]:
        raise SystemExit(f"{sub} {cond}: label map {labels.shape} is not on the truth grid {ref.shape}")
    mask = (labels == LESION).astype(np.uint8)
    pred.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mask, ref.affine), pred)
    vox = float(np.prod(ref.header.get_zooms()[:3]))
    log(f"{sub} {cond}: lesion label {mask.sum() * vox:.0f} mm3")
    return pred


def score(cond: str) -> dict:
    d = OUT / cond
    thick = f", FLAIR in {SLAB}-slice axial slabs" if cond == "flair-thick" else ""
    subprocess.run([PY, "-m", "bidsgate.cli", "score-lesions", "--truth", str(TRUTH),
                    "--pred", str(W / "pred" / cond / "{subject}" / "{base}_desc-mindglide_mask.nii.gz"),
                    "--pipeline", f"MindGlide 1.3.0 (CPU, defaults, FLAIR only{thick})", "--out", str(d)],
                   cwd=REPO, check=True)
    return json.loads((d / "scores_lesions.json").read_text())


PER_LESION_COLS = ["id", "volume_mm3", "bin", "height", "height_frac", "depth_mm", "detected",
                   "overlap_fraction"]


def per_lesion_tsv(scores: dict, path: Path) -> None:
    lines = ["subject\t" + "\t".join(PER_LESION_COLS)]
    for r in scores["results"]:
        for les in r["score"]["per_lesion"]:
            lines.append(r["subject"] + "\t" + "\t".join(str(les[c]) for c in PER_LESION_COLS))
    path.write_text("\n".join(lines) + "\n")


def counts(scores: dict) -> dict[str, tuple[int, int]]:
    """Detected / total per height band, size bin and overall, from the per-lesion lists."""
    c: dict[str, list[int]] = {}
    for r in scores["results"]:
        for les in r["score"]["per_lesion"]:
            for key in ("all", f"height:{les['height']}", f"size:{les['bin']}"):
                c.setdefault(key, [0, 0])
                c[key][0] += bool(les["detected"])
                c[key][1] += 1
    return {k: (v[0], v[1]) for k, v in c.items()}


def compare(all_scores: dict[str, dict]) -> None:
    """One table: detected of n per height and size, MindGlide conditions next to LST-AI v2."""
    tools = {"lst-ai-v2": json.loads(LST.read_text())} | all_scores
    for name, s in tools.items():  # same subjects and lesions, or the comparison is void
        got = sorted(r["subject"] for r in s["results"])
        if got != sorted(f"{x}_{SESSION}" for x in SUBJECTS):
            log(f"WARNING {name} scored {got}, not the six controls")
    table = {name: counts(s) for name, s in tools.items()}
    dice = {name: float(np.mean([r["score"]["dice"] for r in s["results"]])) for name, s in tools.items()}
    fp = {name: sum(r["score"]["false_positive_components"] for r in s["results"]) for name, s in tools.items()}
    keys = ["all", "height:lower", "height:middle", "height:upper"] + sorted(k for k in table["lst-ai-v2"]
                                                                            if k.startswith("size:"))
    lines = ["stratum\t" + "\t".join(tools)]
    for k in keys:
        lines.append(k + "\t" + "\t".join(f"{table[t].get(k, (0, 0))[0]}/{table[t].get(k, (0, 0))[1]}"
                                          for t in tools))
    lines.append("mean_dice\t" + "\t".join(f"{dice[t]:.3f}" for t in tools))
    lines.append("fp_components_total\t" + "\t".join(str(fp[t]) for t in tools))
    (OUT / "comparison.tsv").write_text("\n".join(lines) + "\n")
    (OUT / "comparison.json").write_text(json.dumps(
        {"detected_of_n": {t: {k: list(v) for k, v in table[t].items()} for t in tools},
         "mean_dice": dice, "false_positive_components_total": fp}, indent=2))
    log("comparison:\n" + "\n".join(lines))


def provenance() -> None:
    code = ("import mindglide, torch, monai; from mindglide import infer; "
            "print(mindglide.__version__, torch.__version__, monai.__version__, "
            "infer.HF_MODEL_FILENAME, infer.HF_MODEL_REVISION)")
    r = subprocess.run([str(MG_PY), "-c", code], capture_output=True, text=True, check=False)
    v = r.stdout.split()
    keys = ["mindglide", "torch", "monai", "checkpoint", "checkpoint_revision"]
    info = dict(zip(keys, v)) | {"lesion_label": LESION, "threads": THREADS, "slab_slices": SLAB,
                                 "settings": "device=cpu, defaults (largest-component cleanup, reorient)"}
    times = {}
    for cond in CONDITIONS:
        for sub in SUBJECTS:
            lg = W / "seg" / cond / f"{sub}_{SESSION}.mindglide.log"
            for line in lg.read_text().splitlines() if lg.exists() else []:
                if line.startswith("MINDGLIDE_SECONDS"):
                    times[f"{cond}/{sub}"] = int(line.split()[1])
    info["seconds_per_scan"] = times
    (OUT / "provenance.json").write_text(json.dumps(info, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--after", type=Path, help="wait until this log contains 'scorecard2 done'")
    a = ap.parse_args()
    W.mkdir(parents=True, exist_ok=True)
    if a.after:
        log(f"scorecard3 armed; waiting for 'scorecard2 done' in {a.after} (checks every 5 min)")
        wait_for(a.after if a.after.is_absolute() else REPO / a.after)
    log(f"scorecard3 start: {len(SUBJECTS)} subjects, conditions {CONDITIONS}, {THREADS} threads")
    for cond in CONDITIONS:
        for sub in SUBJECTS:
            prediction(cond, sub, segment(cond, sub))
    all_scores = {}
    for cond in CONDITIONS:
        all_scores[f"mindglide-{cond}"] = s = score(cond)
        per_lesion_tsv(s, OUT / cond / "per_lesion.tsv")
    compare(all_scores)
    provenance()
    log(f"results in {OUT}")
    log("scorecard3 done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
