"""Score what a pipeline recovered against the injected truth."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

SIZE_BINS = [(0, 100), (100, 500), (500, 10**9)]
HEIGHT_BINS = [("lower", 0.0, 1 / 3), ("middle", 1 / 3, 2 / 3), ("upper", 2 / 3, 1.01)]


def _height_bin(frac) -> str:
    if frac is None:
        return "unknown"
    for name, lo, hi in HEIGHT_BINS:
        if lo <= frac < hi:
            return name
    return "unknown"


def _bin(vol: float) -> str:
    for lo, hi in SIZE_BINS:
        if lo <= vol < hi:
            return f"{lo}-{hi if hi < 10**9 else 'inf'}mm3"
    return "?"


def score_lesions(truth_json: Path, truth_mask: Path, pred_mask: Path, threshold: float = 0.5,
                  fp_margin_mm: float = 2.0, regions: Path | None = None,
                  region_names: dict | None = None) -> dict:
    """Voxel and lesion-wise agreement between a predicted mask and the injected one.

    A predicted mask may be probabilistic; it is thresholded at ``threshold``.
    A truth lesion counts as detected when any predicted voxel overlaps it.
    False-positive volume is every predicted voxel farther than ``fp_margin_mm``
    from any truth lesion, and false-positive components are the connected
    pieces (18-connectivity) of that residual, so over-segmentation that
    happens to touch a true lesion is still counted.

    Sensitivity is also broken down by the lesion's height in the brain
    (thirds of the brain mask's superior-inferior extent, recorded at
    injection) and, when ``regions`` is given (a label image on the truth
    grid), by the label under each lesion's centre.
    """
    with open(truth_json) as fh:
        truth = json.load(fh)
    t_img = nib.load(truth_mask)
    t = np.asarray(t_img.dataobj).astype(np.int32)
    p_img = nib.load(pred_mask)
    p = np.asarray(p_img.dataobj, dtype=np.float32)
    if p.shape != t.shape or not np.allclose(p_img.affine, t_img.affine, atol=1e-3):
        raise ValueError(f"prediction {p.shape} is not on the truth grid {t.shape} (shape or affine differs); "
                         "resample it onto the truth image first")
    pb = p >= threshold
    tb = t > 0
    reg = None
    if regions is not None:
        r_img = nib.load(regions)
        if r_img.shape != t.shape:
            raise ValueError(f"regions {r_img.shape} is not on the truth grid {t.shape}")
        reg = np.asarray(r_img.dataobj).astype(np.int64)
    inter = float((pb & tb).sum())
    dice = 2 * inter / (pb.sum() + tb.sum()) if (pb.sum() + tb.sum()) else float("nan")
    voxel_mm3 = float(np.prod(t_img.header.get_zooms()[:3]))
    rows = []
    for les in truth["lesions"]:
        sel = t == les["id"]
        hit = bool(pb[sel].any())
        row = {"id": les["id"], "volume_mm3": les["volume_mm3"], "voxels": int(sel.sum()),
               "bin": _bin(les["volume_mm3"]), "detected": hit,
               "overlap_fraction": float(pb[sel].mean()) if sel.any() else float("nan"),
               "height_frac": les.get("height_frac"), "height": _height_bin(les.get("height_frac")),
               "depth_mm": les.get("depth_mm")}
        if reg is not None:
            c = tuple(int(np.rint(v)) for v in les["centre_vox"])
            label = int(reg[c])
            row["region"] = (region_names or {}).get(label, str(label))
        rows.append(row)
    per = pd.DataFrame(rows)
    near_truth = ndi.distance_transform_edt(~tb, sampling=t_img.header.get_zooms()[:3]) <= fp_margin_mm
    residual = pb & ~near_truth
    _, fp = ndi.label(residual, structure=ndi.generate_binary_structure(3, 2))
    fp_volume = float(residual.sum() * voxel_mm3)
    by_bin = per.groupby("bin")["detected"].agg(["count", "mean"]).rename(columns={"count": "n", "mean": "sensitivity"})
    by_height = per.groupby("height")["detected"].agg(["count", "mean"]).rename(columns={"count": "n", "mean": "sensitivity"})
    by_region = per.groupby("region")["detected"].agg(["count", "mean"]).rename(columns={"count": "n", "mean": "sensitivity"}) if "region" in per else None
    return {
        "dice": dice,
        "sensitivity": float(per["detected"].mean()) if len(per) else float("nan"),
        "lesions": len(per),
        "detected": int(per["detected"].sum()),
        "false_positive_components": int(fp),
        "false_positive_volume_mm3": fp_volume,
        "fp_margin_mm": fp_margin_mm,
        "predicted_volume_mm3": float(pb.sum() * voxel_mm3),
        "truth_volume_mm3": float(tb.sum() * voxel_mm3),
        "volume_ratio": float(pb.sum() / tb.sum()) if tb.sum() else float("nan"),
        "by_size": {k: {"n": int(v["n"]), "sensitivity": float(v["sensitivity"])} for k, v in by_bin.iterrows()},
        "by_height": {k: {"n": int(v["n"]), "sensitivity": float(v["sensitivity"])} for k, v in by_height.iterrows()},
        "by_region": ({k: {"n": int(v["n"]), "sensitivity": float(v["sensitivity"])} for k, v in by_region.iterrows()} if by_region is not None else None),
        "per_lesion": rows,
    }


def score_atrophy(truth_json: Path, volume_before_mm3: float, volume_after_mm3: float,
                  reference_before_mm3: float | None = None, reference_after_mm3: float | None = None) -> dict:
    """Compare a tool's reported volumes before and after injection with the injected factor.

    ``reference_*`` are the same tool's volumes for a structure the injection did not touch
    (farther than the falloff from the target), measured in the same two runs. Its change is
    what the tool reports when nothing changed: the noise a real change has to exceed.
    """
    with open(truth_json) as fh:
        truth = json.load(fh)
    injected = truth["volume_factor"]
    measured = volume_after_mm3 / volume_before_mm3 if volume_before_mm3 else float("nan")
    injected_change = (injected - 1) * 100
    measured_change = (measured - 1) * 100
    before_mask = truth.get("target_volume_mm3_before")
    after_mask = truth.get("target_volume_mm3_after_measured")
    out = {
        "target": truth.get("target", "brain"),
        "injected_factor": injected,
        "measured_factor": measured,
        "injected_change_pct": injected_change,
        "measured_change_pct": measured_change,
        "recovery": measured_change / injected_change if injected_change else float("nan"),
        "bias_pp": measured_change - injected_change,
        "volume_before_mm3": volume_before_mm3,
        "volume_after_mm3": volume_after_mm3,
        "truth_mask_change_pct": ((after_mask / before_mask - 1) * 100) if before_mask and after_mask is not None else None,
    }
    if reference_before_mm3 is not None and reference_after_mm3 is not None:
        out["reference_change_pct"] = ((reference_after_mm3 / reference_before_mm3 - 1) * 100
                                       if reference_before_mm3 else float("nan"))
    return out


def summarise_atrophy(scores: list[dict], k: float = 2.0) -> dict:
    """Pool score_atrophy results over subjects and injected levels.

    Per level: n, mean measured change, recovery (mean and SD), bias in percentage points
    (mean and SD; the SD is the noise), and how many subjects measured a change of the
    injected sign larger than the noise floor. The noise floor is ``k`` times the RMS of
    every reference-structure change in ``scores`` (expected zero); without reference
    volumes it is undefined and no level is called separated. ``smallest_separated_pct``
    is the smallest injected change at which every subject, at that level and every larger
    one, cleared the floor.
    """
    ref = np.array([s["reference_change_pct"] for s in scores if s.get("reference_change_pct") is not None], float)
    ref = ref[np.isfinite(ref)]
    floor = float(k * np.sqrt(np.mean(ref ** 2))) if ref.size else None
    levels = []
    for inj in sorted({round(s["injected_change_pct"], 6) for s in scores}, key=abs):
        grp = [s for s in scores if round(s["injected_change_pct"], 6) == inj]
        meas = np.array([s["measured_change_pct"] for s in grp], float)
        rec = np.array([s["recovery"] for s in grp], float)
        bias = meas - inj
        sd = (lambda x: float(np.std(x, ddof=1)) if x.size > 1 else float("nan"))
        row = {"injected_change_pct": inj, "n": len(grp),
               "measured_change_pct_mean": float(meas.mean()), "measured_change_pct_sd": sd(meas),
               "recovery_mean": float(rec.mean()), "recovery_sd": sd(rec),
               "bias_pp_mean": float(bias.mean()), "bias_pp_sd": sd(bias)}
        if floor is not None and inj != 0:
            hit = (np.sign(meas) == np.sign(inj)) & (np.abs(meas) > floor)
            row["n_beyond_floor"] = int(hit.sum())
            row["separated"] = bool(hit.all())
        levels.append(row)
    smallest = None
    for row in reversed(levels):  # largest change first; stop at the first level that fails
        if not row.get("separated"):
            break
        smallest = abs(row["injected_change_pct"])
    return {"noise_floor_pct": floor, "k": k, "n_reference": int(ref.size),
            "reference_change_pct_rms": float(np.sqrt(np.mean(ref ** 2))) if ref.size else None,
            "levels": levels, "smallest_separated_pct": smallest}
