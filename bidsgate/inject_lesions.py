"""Insert synthetic white-matter lesions with a known mask into T1w and FLAIR images.

The lesions are ellipsoids with Gaussian edges, placed inside a white-matter
estimate derived from the subject's own T1w (bright tissue away from the
brain edge), given FLAIR hyperintensity and T1w hypointensity relative to
the surrounding white matter. Sizes, count, contrasts and the random seed
are recorded next to the mask, so the truth is exact and reproducible.

This is a test of software, not a model of pathology: the point is that a
segmenter's recovery of these lesions is measurable, not that they look
like any particular disease.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


@dataclass
class LesionSpec:
    n: int = 12
    volume_mm3: tuple = (30.0, 80.0, 200.0, 600.0, 1500.0)  # cycled, then shuffled
    flair_contrast: float = 0.6  # FLAIR intensity gain over local white matter at the core
    t1_contrast: float = -0.2  # T1w change over local white matter at the core
    edge_mm: float = 1.0  # Gaussian edge width
    elongation: tuple = (1.0, 2.0)  # axis ratio range
    seed: int = 0
    min_gap_mm: float = 6.0  # between lesions
    extra: dict = field(default_factory=dict)


def brain_and_wm(t1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crude brain mask and white-matter estimate from T1w intensities alone.

    Good enough to place lesions inside bright, deep tissue; not a segmentation.
    """
    sm = ndi.gaussian_filter(t1, sigma=1.0)
    nz = sm[sm > 0]
    thr = np.percentile(nz, 20)
    brain = ndi.binary_fill_holes(ndi.binary_opening(sm > thr, iterations=2))
    lab, n = ndi.label(brain)
    if n > 1:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        brain = lab == sizes.argmax()
    inside = sm[brain]
    wm_thr = np.percentile(inside, 70)  # T1w: white matter is the bright part
    wm = brain & (sm >= wm_thr)
    wm = ndi.binary_opening(wm, iterations=1)
    core = ndi.binary_erosion(brain, iterations=6)  # away from the skull and cortex edge
    return brain, wm & core


def place_lesions(wm: np.ndarray, zooms: tuple, spec: LesionSpec, rng: np.random.Generator) -> list[dict]:
    """Choose centres inside white matter, at least min_gap apart; returns per-lesion records."""
    candidates = np.argwhere(wm)
    if len(candidates) == 0:
        raise ValueError("no white-matter estimate; is this a T1w image?")
    vols = [spec.volume_mm3[i % len(spec.volume_mm3)] for i in range(spec.n)]
    rng.shuffle(vols)
    chosen: list[dict] = []
    tries = 0
    vox_mm = np.array(zooms[:3], dtype=float)
    while len(chosen) < spec.n and tries < 20000:
        tries += 1
        c = candidates[rng.integers(len(candidates))].astype(float)
        if any(np.linalg.norm((c - np.array(o["centre_vox"])) * vox_mm) < spec.min_gap_mm for o in chosen):
            continue
        vol = vols[len(chosen)]
        elong = rng.uniform(*spec.elongation)
        # ellipsoid semi-axes (mm) with volume vol: 4/3 pi a b c, a = elong * b, b = c
        b = (3 * vol / (4 * np.pi * elong)) ** (1 / 3)
        axes = np.array([elong * b, b, b])
        rng.shuffle(axes)
        chosen.append({"id": len(chosen) + 1, "centre_vox": c.tolist(), "axes_mm": axes.tolist(), "volume_mm3": float(vol)})
    if len(chosen) < spec.n:
        raise ValueError(f"could only place {len(chosen)} of {spec.n} lesions {spec.min_gap_mm} mm apart")
    return chosen


def render(shape: tuple, zooms: tuple, lesions: list[dict], edge_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Soft lesion field in [0, 1] and the integer label map (voxel inside the ellipsoid)."""
    grids = np.indices(shape, dtype=np.float32)
    field_ = np.zeros(shape, np.float32)
    labels = np.zeros(shape, np.int16)
    vox_mm = np.array(zooms[:3], dtype=np.float32)
    for les in lesions:
        c = np.array(les["centre_vox"], np.float32)
        axes = np.array(les["axes_mm"], np.float32)
        d = (grids - c[:, None, None, None]) * vox_mm[:, None, None, None] / axes[:, None, None, None]
        r = np.sqrt((d**2).sum(0))  # 1.0 on the ellipsoid surface
        # distance outside the surface in mm, along the smallest axis as a scale
        outside = np.clip(r - 1.0, 0, None) * axes.min()
        soft = np.exp(-0.5 * (outside / max(edge_mm, 1e-3)) ** 2)
        field_ = np.maximum(field_, soft.astype(np.float32))
        labels[(r <= 1.0) & (labels == 0)] = les["id"]
    return field_, labels


def inject(t1_path: Path, flair_path: Path | None, out_t1: Path, out_flair: Path | None, out_mask: Path,
           out_truth: Path, spec: LesionSpec) -> dict:
    rng = np.random.default_rng(spec.seed)
    t1_img = nib.load(t1_path)
    t1 = np.asarray(t1_img.dataobj, dtype=np.float32)
    zooms = t1_img.header.get_zooms()
    _, wm = brain_and_wm(t1)
    lesions = place_lesions(wm, zooms, spec, rng)
    field_, labels = render(t1.shape, zooms, lesions, spec.edge_mm)
    # local white-matter reference intensity: median of WM near each lesion
    wm_ref_t1 = float(np.median(t1[wm]))
    t1_out = t1 * (1.0 + spec.t1_contrast * field_)
    nib.save(nib.Nifti1Image(t1_out.astype(t1_img.get_data_dtype() if np.issubdtype(t1_img.get_data_dtype(), np.floating) else np.float32), t1_img.affine, t1_img.header), out_t1)
    flair_note = None
    if flair_path is not None and out_flair is not None:
        fl_img = nib.load(flair_path)
        fl = np.asarray(fl_img.dataobj, dtype=np.float32)
        if fl.shape != t1.shape:
            raise ValueError(f"FLAIR shape {fl.shape} differs from T1w {t1.shape}; inject expects co-registered images")
        wm_ref_fl = float(np.median(fl[wm]))
        fl_out = fl + spec.flair_contrast * wm_ref_fl * field_
        nib.save(nib.Nifti1Image(fl_out.astype(np.float32), fl_img.affine, fl_img.header), out_flair)
        flair_note = {"wm_reference": wm_ref_fl, "contrast": spec.flair_contrast}
    nib.save(nib.Nifti1Image(labels, t1_img.affine), out_mask)
    truth = {
        "kind": "lesions", "seed": spec.seed, "n": len(lesions), "edge_mm": spec.edge_mm,
        "t1": {"wm_reference": wm_ref_t1, "contrast": spec.t1_contrast}, "flair": flair_note,
        "voxel_mm": [float(z) for z in zooms[:3]],
        "lesions": [{**les, "voxels": int((labels == les["id"]).sum())} for les in lesions],
        "total_volume_mm3": float(sum(les["volume_mm3"] for les in lesions)),
        "mask_voxels": int((labels > 0).sum()),
    }
    with open(out_truth, "w") as fh:
        json.dump(truth, fh, indent=2)
    return truth
