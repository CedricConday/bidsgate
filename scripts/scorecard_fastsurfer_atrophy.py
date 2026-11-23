"""Scorecard #2: FastSurfer's hippocampal volume against injected regional atrophy.

Unattended and resumable: every step skips when its output exists, so a re-run picks up
where the last one stopped. Run with the bidsgate venv's python from the repo root:

    nohup setsid ~/.venvs/bidsgate/bin/python scripts/scorecard_fastsurfer_atrophy.py \
        > work/scorecard2/run.log 2>&1 &

Design (see work/scorecard2/RUN.md):
* six ds007908 healthy controls, the ones the LST-AI scorecard used;
* FastSurfer seg-only (VINN, 1 mm conformed, 4 threads, CPU) on each original T1w: the baseline;
* the baseline segmentation, resampled to the T1w grid, is the region image: bidsgate contracts
  the LEFT hippocampus (label 17) by 2, 5 and 10 % (factor 0.98, 0.95, 0.90);
* FastSurfer on every contracted image; left hippocampus is the target, RIGHT hippocampus
  (label 53, untouched, checked to lie beyond the 12 mm falloff) is the noise reference;
* volumes two ways: FastSurfer's partial-volume-corrected stats (what it reports, headline)
  and plain voxel counts of its segmentation; score-atrophy per level, summarise-atrophy;
* light and dark figures to work/scorecard2/figures (not committed).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parents[1]
SRC = Path(os.environ.get("SRC", "/home/ubuntu/repos/ms-lesion-networks/sourcedata/ds007908"))
FS_HOME = Path(os.environ.get("FS_HOME", "/home/ubuntu/opt/FastSurfer-2.6.1"))
FS_PY = Path(os.environ.get("FS_PY", "/home/ubuntu/.venvs/fastsurfer/bin/python"))
PY = sys.executable
W = REPO / "work" / "scorecard2"
OUT = REPO / "results" / "fastsurfer-atrophy"
SUBJECTS = ["sub-9000", "sub-9001", "sub-9002", "sub-9003", "sub-9004", "sub-9008"]
SESSION = "ses-1"
FACTORS = ["0.98", "0.95", "0.90"]
TARGET, REFERENCE = 17, 53  # Left-, Right-Hippocampus (FreeSurfer LUT)
NAMES = {17: "Left-Hippocampus", 53: "Right-Hippocampus"}
FALLOFF_MM = 12.0
THREADS = "4"


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S UTC ", time.gmtime()) + msg, flush=True)


def wait_for_other_fastsurfer() -> None:
    """The timing run (or anything else under this repo's work dir) must finish first: 4 threads total."""
    while True:
        r = subprocess.run(["pgrep", "-f", "FastSurferCNN/run_prediction.py"], capture_output=True, text=True, check=False)
        if not r.stdout.strip():
            return
        log("another FastSurfer prediction is running; waiting 120 s")
        time.sleep(120)


def fastsurfer(t1: Path, sd: Path, sid: str) -> Path:
    """Run FastSurfer seg-only; return the subject dir. Skips when a previous run exited 0."""
    subj = sd / sid
    done = sd / f"{sid}.done"
    if done.exists():
        return subj
    if subj.exists():  # an interrupted run: move it aside, FastSurfer refuses to reuse half a dir
        shutil.move(str(subj), str(sd / f"{sid}.interrupted-{int(time.time())}"))
    wait_for_other_fastsurfer()
    sd.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PATH=f"{FS_PY.parent}:{os.environ['PATH']}", VIRTUAL_ENV=str(FS_PY.parent.parent))
    cmd = [str(FS_HOME / "run_fastsurfer.sh"), "--t1", str(t1), "--sid", sid, "--sd", str(sd), "--seg_only",
           "--no_cereb", "--no_hypothal", "--no_cc", "--device", "cpu", "--threads", THREADS,
           "--vox_size", "1", "--batch_size", "4", "--py", "python"]
    t0 = time.time()
    log(f"FastSurfer start {sid} in {sd.name}")
    with open(sd / f"{sid}.fastsurfer.log", "w") as fh:
        r = subprocess.run(cmd, cwd=FS_HOME, env=env, stdout=fh, stderr=subprocess.STDOUT, check=False)
    if r.returncode != 0 or not (subj / "mri" / "aparc.DKTatlas+aseg.deep.mgz").exists():
        raise SystemExit(f"FastSurfer failed for {sid} (exit {r.returncode}); see {sd / (sid + '.fastsurfer.log')}")
    if stats_file(subj) is None:
        log(f"WARNING {sid}: no aseg+DKT stats table; scoring will fall back on voxel counts only")
    done.write_text(f"{time.time() - t0:.0f} s\n")
    log(f"FastSurfer done {sid} in {sd.name}: {(time.time() - t0) / 60:.1f} min")
    return subj


def stats_file(subj: Path) -> Path | None:
    hits = sorted((subj / "stats").glob("aseg+DKT*.stats")) if (subj / "stats").exists() else []
    return hits[0] if hits else None


def stats_volumes(subj: Path) -> dict[int, float]:
    """Volume_mm3 by SegId from FastSurfer's partial-volume-corrected stats table."""
    cols, vols = None, {}
    for line in stats_file(subj).read_text().splitlines():
        if line.startswith("# ColHeaders"):
            cols = line.split()[2:]
        elif line and not line.startswith("#") and cols:
            row = dict(zip(cols, line.split()))
            vols[int(row["SegId"])] = float(row["Volume_mm3"])
    return vols


def seg_volumes(subj: Path) -> dict[int, float]:
    img = nib.load(subj / "mri" / "aparc.DKTatlas+aseg.deep.mgz")
    seg = np.asarray(img.dataobj)
    vox = float(np.prod(img.header.get_zooms()[:3]))
    return {lab: float((seg == lab).sum() * vox) for lab in (TARGET, REFERENCE)}


def t1_path(sub: str) -> Path:
    return SRC / sub / SESSION / "anat" / f"{sub}_{SESSION}_T1w.nii.gz"


def region_image(sub: str, subj: Path) -> Path:
    """Baseline segmentation resampled (nearest neighbour) from FastSurfer's conformed space to the T1w grid."""
    out = W / "regions" / f"{sub}_{SESSION}_dseg.nii.gz"
    meta = out.with_suffix("").with_suffix(".json")
    if out.exists() and meta.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    t1 = nib.load(t1_path(sub))
    seg_img = nib.load(subj / "mri" / "aparc.DKTatlas+aseg.deep.mgz")
    m = np.linalg.inv(seg_img.affine) @ t1.affine
    seg = ndi.affine_transform(np.asarray(seg_img.dataobj).astype(np.int16), m[:3, :3], m[:3, 3],
                               output_shape=t1.shape[:3], order=0, mode="constant", cval=0)
    nib.save(nib.Nifti1Image(seg.astype(np.int16), t1.affine), out)
    zooms = t1.header.get_zooms()[:3]
    dist = ndi.distance_transform_edt(seg != TARGET, sampling=zooms)
    gap = float(dist[seg == REFERENCE].min())
    vox = float(np.prod(zooms))
    info = {"target_label": TARGET, "reference_label": REFERENCE,
            "target_ml_on_t1_grid": float((seg == TARGET).sum() * vox / 1000),
            "reference_ml_on_t1_grid": float((seg == REFERENCE).sum() * vox / 1000),
            "reference_min_distance_from_target_mm": gap, "falloff_mm": FALLOFF_MM,
            "reference_beyond_falloff": gap > FALLOFF_MM}
    meta.write_text(json.dumps(info, indent=2))
    if gap <= FALLOFF_MM:
        log(f"WARNING {sub}: right hippocampus is {gap:.1f} mm from the left, inside the falloff")
    return out


def inject(sub: str, factor: str) -> Path:
    out = W / "inj" / f"f{factor}"
    truth = out / sub / SESSION / "anat" / f"{sub}_{SESSION}_desc-atrophy_truth.json"
    t1 = out / sub / SESSION / "anat" / f"{sub}_{SESSION}_T1w.nii.gz"
    if truth.exists() and t1.exists():
        return t1
    r = subprocess.run([PY, "-m", "bidsgate.cli", "inject-atrophy", str(SRC), "--out", str(out), "--subject", sub,
                        "--factor", factor, "--falloff", str(FALLOFF_MM),
                        "--region", str(W / "regions" / "{base}_dseg.nii.gz"), "--label", str(TARGET)],
                       cwd=REPO, capture_output=True, text=True, check=False)
    log(f"inject {sub} f{factor}: {r.stdout.strip().splitlines()[0] if r.stdout.strip() else ''} {r.stderr.strip()}")
    if r.returncode != 0 or not truth.exists():
        raise SystemExit(f"injection failed for {sub} at {factor}")
    return t1


def score() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for kind, reader in (("voxels", seg_volumes), ("pv", stats_volumes)):
        score_files = []
        for factor in FACTORS:
            lines = ["subject\tvolume_before_mm3\tvolume_after_mm3\treference_before_mm3\treference_after_mm3"]
            for sub in SUBJECTS:
                b = reader(W / "fs" / "base" / f"{sub}_{SESSION}")
                a = reader(W / "fs" / f"f{factor}" / f"{sub}_{SESSION}")
                lines.append(f"{sub}_{SESSION}\t{b[TARGET]:.2f}\t{a[TARGET]:.2f}\t{b[REFERENCE]:.2f}\t{a[REFERENCE]:.2f}")
                rows.append({"volumes": kind, "factor": factor, "subject": sub, "target_before": b[TARGET],
                             "target_after": a[TARGET], "reference_before": b[REFERENCE], "reference_after": a[REFERENCE]})
            d = OUT / kind / f"f{factor}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "volumes.tsv").write_text("\n".join(lines) + "\n")
            pipeline = f"FastSurfer 2.6.1 seg-only, {'partial-volume stats' if kind == 'pv' else 'voxel count'}"
            subprocess.run([PY, "-m", "bidsgate.cli", "score-atrophy", "--truth", str(W / "inj" / f"f{factor}"),
                            "--volumes", str(d / "volumes.tsv"), "--pipeline", pipeline, "--out", str(d)],
                           cwd=REPO, check=True)
            score_files.append(str(d / "scores_atrophy.json"))
        subprocess.run([PY, "-m", "bidsgate.cli", "summarise-atrophy", *score_files, "--out", str(OUT / kind)],
                       cwd=REPO, check=True)
    regions = {sub: json.loads((W / "regions" / f"{sub}_{SESSION}_dseg.json").read_text()) for sub in SUBJECTS}
    (OUT / "regions.json").write_text(json.dumps(regions, indent=2))
    # determinism: the timing run of sub-9000 and the batch baseline are the same input
    timing = REPO / "work" / "fastsurfer-timing" / "sub-9000_timing" / "mri" / "aparc.DKTatlas+aseg.deep.mgz"
    base = W / "fs" / "base" / f"sub-9000_{SESSION}" / "mri" / "aparc.DKTatlas+aseg.deep.mgz"
    if timing.exists() and base.exists():
        a, b = np.asarray(nib.load(timing).dataobj), np.asarray(nib.load(base).dataobj)
        (OUT / "rerun_sub-9000.json").write_text(json.dumps({
            "same_shape": a.shape == b.shape,
            "voxels_differing": int((a != b).sum()) if a.shape == b.shape else None,
            "left_hippocampus_voxels": [int((a == TARGET).sum()), int((b == TARGET).sum())]}, indent=2))


FIGURE = r'''
import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
out, figdir = sys.argv[1], sys.argv[2]
df = pd.read_csv(f"{out}/pv/scores_atrophy_all.tsv", sep="\t")
summ = json.load(open(f"{out}/pv/summary_atrophy.json"))
for theme, fg, bg, grid, pt, ref in (("light", "#1f2328", "#ffffff", "#d0d7de", "#0969da", "#8c959f"),
                                     ("dark", "#e6edf3", "#0d1117", "#30363d", "#4493f8", "#6e7681")):
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    lim = 12
    ax.plot([0, -lim], [0, -lim], color=grid, lw=1, label="perfect recovery")
    ax.scatter(df["injected_change_pct"], df["reference_change_pct"], s=16, color=ref, marker="x",
               label="right hippocampus (untouched)", zorder=2)
    ax.scatter(df["injected_change_pct"], df["measured_change_pct"], s=22, color=pt,
               label="left hippocampus (contracted)", zorder=3)
    if summ["noise_floor_pct"] is not None:
        ax.axhspan(-summ["noise_floor_pct"], summ["noise_floor_pct"], color=grid, alpha=0.35, lw=0, label="noise floor")
    ax.set_xlim(1, -lim); ax.set_ylim(-lim, 2)
    ax.set_xlabel("injected volume change (%)", color=fg); ax.set_ylabel("FastSurfer volume change (%)", color=fg)
    ax.tick_params(colors=fg)
    for s in ax.spines.values(): s.set_color(grid)
    ax.grid(color=grid, lw=0.5, alpha=0.6)
    leg = ax.legend(frameon=False, fontsize=8, loc="lower left")
    for t in leg.get_texts(): t.set_color(fg)
    ax.set_title("FastSurfer 2.6.1, six healthy controls, synthetic contraction", color=fg, fontsize=10)
    fig.tight_layout(); fig.savefig(f"{figdir}/fastsurfer-atrophy-{theme}.png", facecolor=bg); plt.close(fig)
'''


def figures() -> None:
    figdir = W / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    script = W / "figure.py"
    script.write_text(FIGURE)
    subprocess.run([str(FS_PY), str(script), str(OUT), str(figdir)], check=True)


def main() -> int:
    W.mkdir(parents=True, exist_ok=True)
    os.environ.update(OMP_NUM_THREADS=THREADS, MKL_NUM_THREADS=THREADS, OPENBLAS_NUM_THREADS=THREADS)
    log(f"scorecard2 start: {len(SUBJECTS)} subjects, factors {FACTORS}")
    for sub in SUBJECTS:
        subj = fastsurfer(t1_path(sub), W / "fs" / "base", f"{sub}_{SESSION}")
        region_image(sub, subj)
        n = seg_volumes(subj)
        try:  # an unreadable stats table must not stop 24 segmentations; scoring reports it
            v = stats_volumes(subj)
            log(f"{sub} baseline hippocampus L/R: stats {v[TARGET]:.0f}/{v[REFERENCE]:.0f} mm3, "
                f"voxels {n[TARGET]:.0f}/{n[REFERENCE]:.0f} mm3")
        except Exception as e:  # noqa: BLE001
            log(f"WARNING {sub}: stats table unreadable ({e!r}); voxels {n[TARGET]:.0f}/{n[REFERENCE]:.0f} mm3")
    for factor in FACTORS:
        for sub in SUBJECTS:
            fastsurfer(inject(sub, factor), W / "fs" / f"f{factor}", f"{sub}_{SESSION}")
    score()
    figures()
    log(f"results in {OUT}, figures in {W / 'figures'}")
    log("scorecard2 done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
