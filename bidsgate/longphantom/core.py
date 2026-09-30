"""Build a known-truth longitudinal series from one real baseline scan.

Fates are drawn once per series (``Design``): among baseline lesions of at least ``min_voxels``, some
enlarge (volume x ``grow`` per year, radial expansion around the lesion as in lesiontrack.synth), some
shrink (x ``shrink`` per year), some small ones resolve at a chosen visit (voxels refilled with local
normal-appearing tissue intensity), the rest stay. New lesions (bidsgate inject-lesions, placed at least
``exclusion_mm`` from existing ones) appear at chosen visits and persist. The whole brain contracts by
``atrophy`` per year (bidsgate displacement). Every visit adds a rigid repositioning, gain and noise.
Truth is not the design: it is measured on the warped label map of each visit, so what the phantom
contains and what the truth table says cannot drift apart.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi


@dataclass(frozen=True)
class Design:
    times: tuple = (0.0, 0.5, 1.0, 2.0)
    n_enlarge: int = 4
    n_shrink: int = 3
    n_resolve: int = 3
    n_new: int = 6
    grow: float = 1.30        # volume factor per year
    shrink: float = 0.75
    atrophy: float = 0.99     # brain volume factor per year
    min_voxels: int = 30
    resolve_max_voxels: int = 400
    exclusion_mm: float = 6.0
    rigid_rot_deg: float = 1.0
    rigid_trans_vox: float = 1.0
    gain: tuple = (0.92, 1.08)
    noise_frac: float = 0.02
    seed: int = 0
    extra: dict = field(default_factory=dict)


def _sdf_warp(mask, disp, zooms):
    from bidsgate.inject_atrophy import _warp

    sd = (ndi.distance_transform_edt(~mask, sampling=zooms) - ndi.distance_transform_edt(mask, sampling=zooms)).astype(np.float32)
    return _warp(sd, disp, 1, cval=1e3) <= 0


def _warp_labels(labels, disp, zooms):
    from bidsgate.inject_atrophy import _warp

    support = _sdf_warp(labels > 0, disp, zooms)
    nn = _warp(labels, disp, 0)
    miss = support & (nn == 0)
    if miss.any():
        idx = ndi.distance_transform_edt(nn == 0, return_distances=False, return_indices=True)
        nn = np.where(miss, nn[tuple(idx)], nn)
    return np.where(support, nn, 0).astype(np.int32)


def build(t1_path, flair_path, mask_path, brain_path, out, d: Design | None = None) -> pd.DataFrame:
    from lesiontrack.synth import _radial_displacement, _rigid_field
    from lesiontrack.tracking import label_lesions

    from bidsgate.inject_atrophy import _warp, displacement
    from bidsgate.inject_lesions import LesionSpec, inject

    d = d or Design()
    out = Path(out).resolve()  # the manifest carries absolute paths
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(d.seed)
    img = nib.load(t1_path)
    aff = img.affine
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    t1 = np.asarray(img.dataobj, dtype=np.float32)
    fl = np.asarray(nib.load(flair_path).dataobj, dtype=np.float32)
    mask = np.asarray(nib.load(mask_path).dataobj) > 0.5
    brain = np.asarray(nib.load(brain_path).dataobj) > 0
    lab = label_lesions(mask, 2)
    sizes = np.bincount(lab.ravel())
    ids = [k for k in range(1, sizes.size) if sizes[k] >= d.min_voxels]
    rng.shuffle(ids)
    enl, shr = ids[:d.n_enlarge], ids[d.n_enlarge:d.n_enlarge + d.n_shrink]
    taken = set(enl) | set(shr)
    small = [k for k in rng.permutation(range(1, sizes.size)).tolist() if k not in taken and 10 <= sizes[k] <= d.resolve_max_voxels]
    res = small[:d.n_resolve]
    res_time = {k: float(rng.choice(d.times[1:])) for k in res}
    fate = {k: "stable" for k in range(1, sizes.size)}
    cents0 = dict(zip(range(1, sizes.size), ndi.center_of_mass(mask, lab, range(1, sizes.size))))
    for k in enl + shr:
        reach = (3 * sizes[k] / (4 * np.pi)) ** (1 / 3) + 2.0 + 6.0  # radial field radius + margin + falloff (voxels)
        for j in range(1, sizes.size):
            if j != k and np.linalg.norm(np.subtract(cents0[j], cents0[k])) <= reach + (3 * sizes[j] / (4 * np.pi)) ** (1 / 3):
                fate[j] = "near a deformed lesion"  # compressed or stretched by the neighbouring field: not a stable control
    fate.update({k: "enlarging" for k in enl}); fate.update({k: "shrinking" for k in shr}); fate.update({k: "resolving" for k in res})
    cents = dict(zip(range(1, sizes.size), ndi.center_of_mass(mask, lab, range(1, sizes.size))))
    # normal-appearing tissue intensity for refilling resolved lesions: median of a shell around each lesion
    def refill(arr, k):
        m = lab == k
        shell = ndi.binary_dilation(m, iterations=3) & ~ndi.binary_dilation(mask, iterations=1) & brain
        return float(np.median(arr[shell])) if shell.any() else float(np.median(arr[brain & ~mask]))
    fill = {k: (refill(t1, k), refill(fl, k)) for k in res}
    # new lesions: one bidsgate injection on the baseline, away from existing lesions
    tmp = out / "work"
    tmp.mkdir(exist_ok=True)
    it = max(1, round(d.exclusion_mm / min(zooms)))
    place = brain & ~ndi.binary_dilation(mask, ndi.generate_binary_structure(3, 2), iterations=it)
    nib.save(nib.Nifti1Image(place.astype(np.uint8), aff), tmp / "place.nii.gz")
    inj = inject(Path(t1_path), Path(flair_path), tmp / "t1i.nii.gz", tmp / "fli.nii.gz", tmp / "newlab.nii.gz",
                 tmp / "new.json", LesionSpec(n=d.n_new, seed=int(rng.integers(0, 2**31 - 1))), mask_path=tmp / "place.nii.gz")
    t1i = np.asarray(nib.load(tmp / "t1i.nii.gz").dataobj, dtype=np.float32)
    fli = np.asarray(nib.load(tmp / "fli.nii.gz").dataobj, dtype=np.float32)
    newlab = np.asarray(nib.load(tmp / "newlab.nii.gz").dataobj).astype(np.int32)
    new_ids = sorted(int(x["id"]) for x in inj["lesions"])
    new_time = {i: float(rng.choice(d.times[1:])) for i in new_ids}
    rows, manifest = [], []
    for v, t in enumerate(d.times):
        a1, af = t1.copy(), fl.copy()
        truth = np.where(mask, lab, 0).astype(np.int32)
        for k in res:
            if res_time[k] <= t:
                m = ndi.binary_dilation(lab == k, iterations=1) & ~(truth.astype(bool) & (lab != k))
                a1[m], af[m] = fill[k]
                truth[lab == k] = 0
        for i in new_ids:
            if new_time[i] <= t:
                region = ndi.binary_dilation(newlab == i, iterations=2)
                a1[region], af[region] = t1i[region], fli[region]
                truth[newlab == i] = 1000 + i
        # every visit, the first included, is resampled through a rigid repositioning, so the label resampling
        # bias (small lesions lose a little volume) is the same at every visit and cancels in the change
        if t == 0:
            disp = np.zeros((3, *t1.shape), np.float32)
        else:
            disp = displacement(t1.shape, zooms, brain, d.atrophy ** t, 12.0)
        for k in enl + shr:
            if fate[k] == "resolving":
                continue
            f = (d.grow if k in enl else d.shrink) ** t
            r = (3 * sizes[k] / (4 * np.pi)) ** (1 / 3) + 2.0
            if t > 0:
                disp += _radial_displacement(t1.shape, np.array(cents[k], np.float32), r, f, 6.0)
        disp += _rigid_field(t1.shape, rng.uniform(-1, 1, 3) * d.rigid_rot_deg, rng.uniform(-1, 1, 3) * d.rigid_trans_vox)
        w1, wf = _warp(a1, disp, 1), _warp(af, disp, 1)
        wl = _warp_labels(truth, disp, zooms)
        wb = _sdf_warp(brain, disp, zooms)
        for arr in (w1, wf):
            arr *= rng.uniform(*d.gain)
            sd = d.noise_frac * float(np.percentile(arr[wb], 99)) if wb.any() else 0.0
            arr[wb] += rng.normal(0, sd, int(wb.sum())).astype(np.float32)
            arr[~wb] = 0
        vd = out / f"visit-{v:02d}"
        vd.mkdir(exist_ok=True)
        p = {n: vd / f"{n}.nii.gz" for n in ("T1w", "FLAIR", "mask", "brainmask", "truth_labels")}
        nib.save(nib.Nifti1Image(w1.astype(np.float32), aff), p["T1w"])
        nib.save(nib.Nifti1Image(wf.astype(np.float32), aff), p["FLAIR"])
        nib.save(nib.Nifti1Image((wl > 0).astype(np.uint8), aff), p["mask"])
        nib.save(nib.Nifti1Image(wb.astype(np.uint8), aff), p["brainmask"])
        nib.save(nib.Nifti1Image(wl, aff), p["truth_labels"])
        vol = np.bincount(wl.ravel()) * float(np.prod(zooms))
        for k in range(1, sizes.size):
            rows.append({"visit": v, "time_years": t, "lesion": k, "origin": "baseline", "fate": fate[k],
                         "volume_mm3": float(vol[k]) if k < vol.size else 0.0})
        for i in new_ids:
            rows.append({"visit": v, "time_years": t, "lesion": 1000 + i, "origin": "new", "fate": f"new at {new_time[i]:g} y",
                         "volume_mm3": float(vol[1000 + i]) if 1000 + i < vol.size else 0.0})
        manifest.append({"subject": out.name, "session": f"V{v:02d}", "time_years": t, "t1": p["T1w"], "flair": p["FLAIR"],
                         "mask": p["mask"], "brainmask": p["brainmask"], "brain_volume_mm3": float(wb.sum() * np.prod(zooms))})
    truth_df = pd.DataFrame(rows)
    truth_df.to_csv(out / "truth.tsv", sep="\t", index=False)
    pd.DataFrame(manifest).to_csv(out / "manifest.tsv", sep="\t", index=False)
    (out / "design.json").write_text(json.dumps({**d.__dict__, "enlarging": enl, "shrinking": shr, "resolving": res_time,
                                                 "new": new_time, "source": [str(t1_path), str(flair_path), str(mask_path)]},
                                                indent=2, default=str))
    for f in tmp.glob("*"):
        f.unlink()
    tmp.rmdir()
    return truth_df
