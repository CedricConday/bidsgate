"""Shrink the brain by a known factor to create a ground-truth volume change.

A smooth radial contraction about the brain's centroid, uniform inside the
brain mask and fading to identity over ``falloff_mm`` outside it, is
applied to the T1w (and FLAIR when present). The brain's volume changes by
exactly the requested factor inside the uniform region, so any morphometry
tool's reported total brain volume can be checked against the truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

from .inject_lesions import brain_and_wm


@dataclass
class AtrophySpec:
    volume_factor: float = 0.95  # 0.95 = 5 % brain volume loss
    falloff_mm: float = 12.0
    seed: int = 0


def _warp(image: np.ndarray, disp: np.ndarray, order: int, cval: float = 0.0) -> np.ndarray:
    grids = np.indices(image.shape, dtype=np.float32)
    return ndi.map_coordinates(image, grids + disp, order=order, mode="constant", cval=cval)


def displacement(shape: tuple, zooms: tuple, brain: np.ndarray, factor: float, falloff_mm: float) -> np.ndarray:
    """Sampling displacement (3, X, Y, Z): output(x) = input(x + u(x)).

    Inside the brain the map is a radial scaling by ``factor ** (1/3)`` about
    the centroid; a shrunk brain samples from farther out, so u points away
    from the centre. The weight fades to zero over ``falloff_mm`` outside.
    """
    lam = factor ** (1.0 / 3.0)
    c = np.array(ndi.center_of_mass(brain), np.float32)
    vox = np.array(zooms[:3], np.float32)
    grids = np.indices(shape, dtype=np.float32)
    d = grids - c[:, None, None, None]
    dist_out_mm = ndi.distance_transform_edt(~brain, sampling=zooms[:3]).astype(np.float32)
    w = np.clip(1.0 - dist_out_mm / max(falloff_mm, 1e-3), 0.0, 1.0)
    w[brain] = 1.0
    # output(x) = input(c + (x - c) / lam): points inside map outward for lam < 1
    _ = vox  # voxel sizes are isotropic-agnostic here: the scaling is about the centroid in voxel space
    return d * ((1.0 / lam - 1.0) * w)[None]


def inject(t1_path: Path, flair_path: Path | None, out_t1: Path, out_flair: Path | None,
           out_truth: Path, spec: AtrophySpec) -> dict:
    t1_img = nib.load(t1_path)
    t1 = np.asarray(t1_img.dataobj, dtype=np.float32)
    zooms = t1_img.header.get_zooms()
    brain, _ = brain_and_wm(t1)
    disp = displacement(t1.shape, zooms, brain, spec.volume_factor, spec.falloff_mm)
    nib.save(nib.Nifti1Image(_warp(t1, disp, 1).astype(np.float32), t1_img.affine, t1_img.header), out_t1)
    if flair_path is not None and out_flair is not None:
        fl_img = nib.load(flair_path)
        fl = np.asarray(fl_img.dataobj, dtype=np.float32)
        if fl.shape != t1.shape:
            raise ValueError("FLAIR and T1w must share a grid")
        nib.save(nib.Nifti1Image(_warp(fl, disp, 1).astype(np.float32), fl_img.affine, fl_img.header), out_flair)
    voxel_mm3 = float(np.prod(zooms[:3]))
    brain_after = _warp(brain.astype(np.float32), disp, 1) >= 0.5
    truth = {
        "kind": "atrophy", "seed": spec.seed, "volume_factor": spec.volume_factor,
        "falloff_mm": spec.falloff_mm,
        "brain_volume_mm3_before": float(brain.sum() * voxel_mm3),
        "brain_volume_mm3_after_measured": float(brain_after.sum() * voxel_mm3),
        "voxel_mm": [float(z) for z in zooms[:3]],
    }
    with open(out_truth, "w") as fh:
        json.dump(truth, fh, indent=2)
    return truth
