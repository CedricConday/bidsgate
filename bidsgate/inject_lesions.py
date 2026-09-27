"""Insert synthetic white-matter lesions with a known mask into T1w and FLAIR images.

The lesions are ellipsoids with soft edges, placed inside a white-matter
estimate derived from the subject's own T1w and FLAIR (bright T1w tissue
deep inside a morphological brain mask, or inside a mask you supply), given
FLAIR hyperintensity and T1w hypointensity relative to the median of that
white-matter estimate. Sizes, count, contrasts and the random seed
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
from scipy.special import erf


@dataclass
class LesionSpec:
    n: int = 12
    volume_mm3: tuple = (30.0, 80.0, 200.0, 600.0, 1500.0)  # cycled, then shuffled
    flair_contrast: float = 0.6  # FLAIR gain over the white-matter median at the core
    t1_contrast: float = -0.2  # T1w change over the white-matter median at the core
    edge_mm: float = 1.0  # erf edge width, centred on the surface
    elongation: tuple = (1.0, 2.0)  # axis ratio range
    seed: int = 0
    min_gap_mm: float = 6.0  # between lesion surfaces
    extra: dict = field(default_factory=dict)


def _otsu(x: np.ndarray) -> float:
    hist, edges = np.histogram(x, bins=256)
    mids = 0.5 * (edges[:-1] + edges[1:])
    w1 = np.cumsum(hist).astype(float)
    w2 = w1[-1] - w1
    hm = np.cumsum(hist * mids)
    m1 = hm / np.maximum(w1, 1)
    m2 = (hm[-1] - hm) / np.maximum(w2, 1)
    var = w1[:-1] * w2[:-1] * (m1[:-1] - m2[:-1]) ** 2
    return float(mids[var.argmax()])


def _si_axis(affine: np.ndarray) -> int:
    """Voxel axis that runs superior-inferior (largest |z| component in the affine)."""
    return int(np.argmax(np.abs(affine[2, :3])))


def _extent(mask: np.ndarray, axis: int) -> tuple[float, float]:
    idx = np.flatnonzero(mask.any(axis=tuple(a for a in range(3) if a != axis)))
    return float(idx.min()), float(idx.max())


def estimate_brain(t1: np.ndarray, zooms: tuple, erode_mm: float = 8.0,
                   volume_ml: tuple = (700.0, 2000.0), min_core_ml: float = 100.0,
                   affine: np.ndarray | None = None) -> np.ndarray:
    """Brain mask from a whole-head T1w by morphology, no atlas.

    Tissue is what lies above an Otsu threshold of the smoothed image. The
    brain core is every piece of tissue more than ``erode_mm`` from any
    non-tissue voxel that is at least ``min_core_ml`` in size: the erosion
    cuts the thin scalp, the optic nerves and the spinal cord, and the size
    rule keeps both hemispheres when a deep fissure splits the core but drops
    muscle and tongue. The core is grown back by the same distance inside
    tissue and holes (ventricles) are filled. The result must have a
    plausible brain volume or a ValueError is raised: it is a placement
    mask, not a segmentation, and a wrong one would put lesions in the neck.
    """
    vox = np.array(zooms[:3], dtype=float)
    sm = ndi.gaussian_filter(t1, sigma=1.0 / vox)
    if not (sm > 0).any():
        raise ValueError("empty image")
    tissue = sm > _otsu(sm.ravel())  # background and bone below, soft tissue above
    core = ndi.distance_transform_edt(tissue, sampling=vox) > erode_mm
    lab, n = ndi.label(core)
    if n == 0:
        raise ValueError("no tissue thicker than the erosion radius; supply a brain mask")
    sizes = np.bincount(lab.ravel()) * float(np.prod(vox)) / 1000.0
    sizes[0] = 0
    largest = int(sizes.argmax())
    # Keep a second large piece only if it sits level with the largest one along the
    # superior-inferior axis: a hemisphere split off by a deep fissure does, neck and face
    # tissue below the skull base does not.
    si = _si_axis(affine) if affine is not None else 2
    lo, hi = _extent(lab == largest, si)
    keep = [largest]
    for k in np.flatnonzero(sizes >= min_core_ml):
        if k == largest:
            continue
        c = ndi.center_of_mass(lab == k)[si]
        if lo <= c <= hi:
            keep.append(int(k))
    core = np.isin(lab, keep)
    brain = tissue & (ndi.distance_transform_edt(~core, sampling=vox) <= erode_mm)
    brain = ndi.binary_fill_holes(brain)
    for axis in range(3):  # ventricles open to the outside through narrow channels: fill them slice-wise too
        brain = np.moveaxis(np.array([ndi.binary_fill_holes(sl) for sl in np.moveaxis(brain, axis, 0)]), 0, axis)
    ml = brain.sum() * float(np.prod(vox)) / 1000.0
    if not volume_ml[0] <= ml <= volume_ml[1]:
        raise ValueError(f"brain estimate is {ml:.0f} ml, outside {volume_ml[0]:.0f}-{volume_ml[1]:.0f} ml; "
                         "supply a brain mask with --mask")
    return brain


def brain_and_wm(t1: np.ndarray, zooms: tuple, flair: np.ndarray | None = None,
                 mask: np.ndarray | None = None, depth_mm: float = 6.0,
                 affine: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Brain mask (estimated, or the one given) and a white-matter placement estimate.

    White matter is bright T1w tissue deeper than ``depth_mm`` inside the
    brain; when a FLAIR is given it must also be within 0.6-1.4 of the FLAIR
    white-matter median, which excludes CSF and anything outside the FLAIR
    field of view. Good enough to place lesions in deep tissue; not a
    segmentation.
    """
    vox = np.array(zooms[:3], dtype=float)
    brain = mask.astype(bool) if mask is not None else estimate_brain(t1, zooms, affine=affine)
    sm = ndi.gaussian_filter(t1, sigma=1.0 / vox)
    deep = ndi.distance_transform_edt(brain, sampling=vox) > depth_mm
    wm_thr = np.percentile(sm[brain], 65)  # T1w: white matter is the bright part of the brain
    wm = deep & (sm >= wm_thr)
    if flair is not None:
        fs = ndi.gaussian_filter(flair, sigma=1.0 / vox)
        med = float(np.median(fs[wm])) if wm.any() else 0.0
        wm &= (fs > 0.6 * med) & (fs < 1.4 * med)
    wm = ndi.binary_opening(wm, iterations=1)
    if wm.sum() * float(np.prod(vox)) < 20_000:
        raise ValueError("white-matter estimate under 20 ml; is this a whole-head T1w?")
    return brain, wm


def place_lesions(wm: np.ndarray, depth_mm: np.ndarray, zooms: tuple, spec: LesionSpec,
                  rng: np.random.Generator) -> list[dict]:
    """Choose centres inside white matter so that no lesion touches another or leaves the brain.

    ``depth_mm`` is the distance from each voxel to the brain edge; a lesion's
    centre must be deeper than its longest semi-axis. Two lesions must be at
    least the sum of their longest semi-axes plus ``min_gap_mm`` apart.
    """
    candidates = np.argwhere(wm)
    if len(candidates) == 0:
        raise ValueError("no white-matter estimate; is this a T1w image?")
    vols = [spec.volume_mm3[i % len(spec.volume_mm3)] for i in range(spec.n)]
    rng.shuffle(vols)
    chosen: list[dict] = []
    vox_mm = np.array(zooms[:3], dtype=float)
    for vol in vols:
        elong = rng.uniform(*spec.elongation)
        # ellipsoid semi-axes (mm) with volume vol: 4/3 pi a b c, a = elong * b, b = c
        b = (3 * vol / (4 * np.pi * elong)) ** (1 / 3)
        axes = np.array([elong * b, b, b])
        rng.shuffle(axes)
        reach = float(axes.max())
        for _ in range(5000):
            c = candidates[rng.integers(len(candidates))]
            if depth_mm[tuple(c)] < reach + 1.0:
                continue
            c = c.astype(float)
            if any(np.linalg.norm((c - np.array(o["centre_vox"])) * vox_mm) < reach + max(o["axes_mm"]) + spec.min_gap_mm
                   for o in chosen):
                continue
            chosen.append({"id": len(chosen) + 1, "centre_vox": c.tolist(), "axes_mm": axes.tolist(),
                           "volume_mm3_nominal": float(vol)})
            break
        else:
            raise ValueError(f"could only place {len(chosen)} of {spec.n} lesions without overlap; fewer or smaller lesions")
    return chosen


def render(shape: tuple, zooms: tuple, lesions: list[dict], edge_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Soft lesion field in [0, 1] and the integer label map.

    The field is 0.5 on the ellipsoid surface and falls off over ``edge_mm``
    on either side (an erf profile of the signed distance, scaled along the
    shortest axis), so the label, which is the voxels inside the surface, is
    exactly what a half-maximum segmenter would recover.
    """
    grids = np.indices(shape, dtype=np.float32)
    field_ = np.zeros(shape, np.float32)
    labels = np.zeros(shape, np.int16)
    vox_mm = np.array(zooms[:3], dtype=np.float32)
    for les in lesions:
        c = np.array(les["centre_vox"], np.float32)
        axes = np.array(les["axes_mm"], np.float32)
        d = (grids - c[:, None, None, None]) * vox_mm[:, None, None, None] / axes[:, None, None, None]
        r = np.sqrt((d**2).sum(0))  # 1.0 on the ellipsoid surface
        signed_mm = (r - 1.0) * axes.min()  # negative inside
        soft = 0.5 * (1.0 - erf(signed_mm / max(edge_mm, 1e-3)))
        field_ = np.maximum(field_, soft.astype(np.float32))
        labels[(r <= 1.0) & (labels == 0)] = les["id"]
    return field_, labels


def _float_header(img):
    hdr = img.header.copy()
    hdr.set_data_dtype(np.float32)
    hdr.set_slope_inter(1.0, 0.0)
    return hdr


def inject(t1_path: Path, flair_path: Path | None, out_t1: Path, out_flair: Path | None, out_mask: Path,
           out_truth: Path, spec: LesionSpec, mask_path: Path | None = None) -> dict:
    """Write the injected T1w (and FLAIR), the truth label map and the truth JSON.

    Nothing is written until every input has been checked, so a failure
    leaves no partial subject behind.
    """
    rng = np.random.default_rng(spec.seed)
    t1_img = nib.load(t1_path)
    t1 = np.asarray(t1_img.dataobj, dtype=np.float32)
    zooms = t1_img.header.get_zooms()
    fl_img = fl = None
    if flair_path is not None and out_flair is not None:
        fl_img = nib.load(flair_path)
        if fl_img.shape != t1_img.shape or not np.allclose(fl_img.affine, t1_img.affine, atol=1e-3):
            raise ValueError(f"FLAIR grid {fl_img.shape} differs from T1w {t1_img.shape} (or the affines differ); "
                             "inject expects co-registered images on one grid")
        fl = np.asarray(fl_img.dataobj, dtype=np.float32)
    mask = None
    if mask_path is not None:
        m_img = nib.load(mask_path)
        if m_img.shape != t1_img.shape:
            raise ValueError(f"brain mask grid {m_img.shape} differs from T1w {t1_img.shape}")
        mask = np.asarray(m_img.dataobj) > 0
    brain, wm = brain_and_wm(t1, zooms, fl, mask, affine=t1_img.affine)
    depth = ndi.distance_transform_edt(brain, sampling=zooms[:3])
    lesions = place_lesions(wm, depth, zooms, spec, rng)
    field_, labels = render(t1.shape, zooms, lesions, spec.edge_mm)
    # reference intensity: median of the white-matter estimate over the whole brain
    wm_ref_t1 = float(np.median(t1[wm]))
    t1_out = t1 * (1.0 + spec.t1_contrast * field_)
    nib.save(nib.Nifti1Image(t1_out.astype(np.float32), t1_img.affine, _float_header(t1_img)), out_t1)
    flair_note = None
    if fl is not None:
        wm_ref_fl = float(np.median(fl[wm]))
        fl_out = fl + spec.flair_contrast * wm_ref_fl * field_
        nib.save(nib.Nifti1Image(fl_out.astype(np.float32), fl_img.affine, _float_header(fl_img)), out_flair)
        flair_note = {"wm_reference": wm_ref_fl, "contrast": spec.flair_contrast}
    nib.save(nib.Nifti1Image(labels, t1_img.affine), out_mask)
    voxel_mm3 = float(np.prod(zooms[:3]))
    si = _si_axis(t1_img.affine)
    lo, hi = _extent(brain, si)
    up = 1.0 if t1_img.affine[2, si] > 0 else -1.0  # which way along the axis is superior
    for les in lesions:
        les["voxels"] = int((labels == les["id"]).sum())
        les["volume_mm3"] = les["voxels"] * voxel_mm3  # the label's volume, which is what is scored
        c = les["centre_vox"]
        frac = (c[si] - lo) / max(hi - lo, 1.0)
        les["height_frac"] = float(frac if up > 0 else 1.0 - frac)  # 0 = bottom of the brain mask, 1 = top
        les["depth_mm"] = float(depth[tuple(int(round(v)) for v in c)])
    truth = {
        "kind": "lesions", "seed": spec.seed, "n": len(lesions), "edge_mm": spec.edge_mm,
        "edge": "field is 0.5 on the ellipsoid surface, erf falloff over edge_mm; label = inside the surface",
        "t1": {"wm_reference": wm_ref_t1, "contrast": spec.t1_contrast}, "flair": flair_note,
        "brain_mask": "given" if mask is not None else "estimated",
        "brain_volume_mm3": float(brain.sum() * voxel_mm3),
        "height": "height_frac is the lesion centre's position along the superior-inferior axis of the brain mask, 0 bottom to 1 top; depth_mm its distance from the mask edge",
        "voxel_mm": [float(z) for z in zooms[:3]],
        "lesions": lesions,
        "total_volume_mm3": float(sum(les["volume_mm3"] for les in lesions)),
        "mask_voxels": int((labels > 0).sum()),
    }
    with open(out_truth, "w") as fh:
        json.dump(truth, fh, indent=2)
    return truth
