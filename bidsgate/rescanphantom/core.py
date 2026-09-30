"""Pseudo-rescans of one scan and the lesion changes a segmenter reports between them.

A rescan on the same scanner changes gain, bias field and noise but not anatomy. rescanphantom writes
``k`` copies of T1w and FLAIR with independent smooth multiplicative bias fields (``bias`` peak-to-peak
fraction), global gain (``gain``) and Gaussian noise (``noise`` of the 99th percentile), on the original
grid, so masks compare voxel to voxel. Any lesion in one copy's mask with no overlap in another's is a
change the segmenter invented: the floor under every "new lesion" it will report on real follow-ups.
"""

from __future__ import annotations

import shlex
import subprocess
from itertools import combinations
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi


def bias_field(shape, rng, strength, smooth_vox=40.0):
    f = ndi.gaussian_filter(rng.standard_normal(shape).astype(np.float32), smooth_vox, mode="reflect")
    f = (f - f.min()) / (np.ptp(f) + 1e-9) - 0.5
    return 1.0 + strength * f


def perturb(t1: Path, flair: Path, out: Path, k: int = 3, gain=0.05, bias=0.10, noise=0.015, seed=0) -> list[tuple]:
    rng = np.random.default_rng(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    imgs = {"T1w": nib.load(t1), "FLAIR": nib.load(flair)}
    data = {n: np.asarray(i.dataobj, dtype=np.float32) for n, i in imgs.items()}
    copies = []
    for c in range(k):
        paths = {}
        for n, d in data.items():
            head = d > 0
            x = d * bias_field(d.shape, rng, bias) * (1 + rng.uniform(-gain, gain))
            x[head] += rng.normal(0, noise * np.percentile(d[head], 99), int(head.sum())).astype(np.float32)
            x = np.where(head, np.maximum(x, 0), 0).astype(np.float32)
            p = out / f"rescan{c}_{n}.nii.gz"
            nib.save(nib.Nifti1Image(x, imgs[n].affine, imgs[n].header), p)
            paths[n] = p
        copies.append((paths["T1w"], paths["FLAIR"]))
    return copies


def segment(command: str, t1: Path, flair: Path, out_mask: Path) -> Path:
    """``command`` is a template with {t1}, {flair}, {out}; it must write a binary mask to {out}."""
    cmd = command.format(t1=shlex.quote(str(t1)), flair=shlex.quote(str(flair)), out=shlex.quote(str(out_mask)))
    subprocess.run(cmd, shell=True, check=True)
    if not Path(out_mask).exists():
        raise RuntimeError(f"segmenter wrote no mask: {cmd}")
    return Path(out_mask)


def compare(a: Path, b: Path, min_mm3: float = 10.0) -> dict:
    ia = nib.load(a)
    ma, mb = np.asarray(ia.dataobj) > 0.5, np.asarray(nib.load(b).dataobj) > 0.5
    vox = float(np.prod(ia.header.get_zooms()[:3]))
    st = ndi.generate_binary_structure(3, 2)
    def only(x, y):
        lab, n = ndi.label(x, st)
        if not n:
            return 0, 0.0
        hit = np.bincount(lab[y].ravel(), minlength=n + 1) > 0
        size = np.bincount(lab.ravel(), minlength=n + 1) * vox
        sel = [i for i in range(1, n + 1) if not hit[i] and size[i] >= min_mm3]
        return len(sel), float(sum(size[i] for i in sel))
    nb, vb = only(mb, ma)
    na, va = only(ma, mb)
    inter = (ma & mb).sum()
    return {"lesions_a": int(ndi.label(ma, st)[1]), "lesions_b": int(ndi.label(mb, st)[1]),
            "dice": float(2 * inter / (ma.sum() + mb.sum())) if (ma.sum() + mb.sum()) else None,
            "invented_new": nb, "invented_new_mm3": vb, "invented_resolved": na, "invented_resolved_mm3": va,
            "volume_change_pct": float((mb.sum() - ma.sum()) / ma.sum() * 100) if ma.sum() else None}


def run(t1, flair, command, out, k=3, seed=0, min_mm3=10.0) -> tuple[pd.DataFrame, dict]:
    out = Path(out)
    masks = []
    for c, (a, b) in enumerate(perturb(t1, flair, out, k, seed=seed)):
        m = out / f"rescan{c}_mask.nii.gz"
        masks.append(m if m.exists() else segment(command, a, b, m))
    rows = [{"a": i, "b": j, **compare(masks[i], masks[j], min_mm3)} for i, j in combinations(range(k), 2)]
    df = pd.DataFrame(rows)
    summ = {"copies": k, "pairs": len(df), "mean_dice": float(df["dice"].mean()),
            "invented_new_per_pair": float(df["invented_new"].mean()),
            "invented_resolved_per_pair": float(df["invented_resolved"].mean()),
            "abs_volume_change_pct_mean": float(df["volume_change_pct"].abs().mean())}
    return df, summ
