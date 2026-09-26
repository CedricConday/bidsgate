"""Score what a pipeline recovered against the injected truth."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

SIZE_BINS = [(0, 100), (100, 500), (500, 10**9)]


def _bin(vol: float) -> str:
    for lo, hi in SIZE_BINS:
        if lo <= vol < hi:
            return f"{lo}-{hi if hi < 10**9 else 'inf'}mm3"
    return "?"


def score_lesions(truth_json: Path, truth_mask: Path, pred_mask: Path, threshold: float = 0.5) -> dict:
    """Voxel and lesion-wise agreement between a predicted mask and the injected one.

    A predicted mask may be probabilistic; it is thresholded at ``threshold``.
    A truth lesion counts as detected when any predicted voxel overlaps it;
    a predicted component counts as a false positive when it overlaps no
    truth lesion.
    """
    with open(truth_json) as fh:
        truth = json.load(fh)
    t_img = nib.load(truth_mask)
    t = np.asarray(t_img.dataobj).astype(np.int32)
    p_img = nib.load(pred_mask)
    p = np.asarray(p_img.dataobj, dtype=np.float32)
    if p.shape != t.shape:
        raise ValueError(f"prediction {p.shape} is not on the truth grid {t.shape}; resample it first")
    pb = p >= threshold
    tb = t > 0
    inter = float((pb & tb).sum())
    dice = 2 * inter / (pb.sum() + tb.sum()) if (pb.sum() + tb.sum()) else float("nan")
    voxel_mm3 = float(np.prod(t_img.header.get_zooms()[:3]))
    rows = []
    for les in truth["lesions"]:
        sel = t == les["id"]
        hit = bool(pb[sel].any())
        rows.append({"id": les["id"], "volume_mm3": les["volume_mm3"], "voxels": int(sel.sum()),
                     "bin": _bin(les["volume_mm3"]), "detected": hit,
                     "overlap_fraction": float(pb[sel].mean()) if sel.any() else float("nan")})
    per = pd.DataFrame(rows)
    comp, n = ndi.label(pb, structure=ndi.generate_binary_structure(3, 2))
    fp = 0
    fp_volume = 0.0
    for k in range(1, n + 1):
        sel = comp == k
        if not tb[sel].any():
            fp += 1
            fp_volume += float(sel.sum() * voxel_mm3)
    by_bin = per.groupby("bin")["detected"].agg(["count", "mean"]).rename(columns={"count": "n", "mean": "sensitivity"})
    return {
        "dice": dice,
        "sensitivity": float(per["detected"].mean()) if len(per) else float("nan"),
        "lesions": len(per),
        "detected": int(per["detected"].sum()),
        "false_positive_components": fp,
        "false_positive_volume_mm3": fp_volume,
        "predicted_volume_mm3": float(pb.sum() * voxel_mm3),
        "truth_volume_mm3": float(tb.sum() * voxel_mm3),
        "volume_ratio": float(pb.sum() / tb.sum()) if tb.sum() else float("nan"),
        "by_size": {k: {"n": int(v["n"]), "sensitivity": float(v["sensitivity"])} for k, v in by_bin.iterrows()},
        "per_lesion": rows,
    }


def score_atrophy(truth_json: Path, volume_before_mm3: float, volume_after_mm3: float) -> dict:
    """Compare a tool's reported volumes before and after injection with the injected factor."""
    with open(truth_json) as fh:
        truth = json.load(fh)
    injected = truth["volume_factor"]
    measured = volume_after_mm3 / volume_before_mm3 if volume_before_mm3 else float("nan")
    injected_change = (injected - 1) * 100
    measured_change = (measured - 1) * 100
    return {
        "injected_factor": injected,
        "measured_factor": measured,
        "injected_change_pct": injected_change,
        "measured_change_pct": measured_change,
        "recovery": measured_change / injected_change if injected_change else float("nan"),
        "volume_before_mm3": volume_before_mm3,
        "volume_after_mm3": volume_after_mm3,
    }
