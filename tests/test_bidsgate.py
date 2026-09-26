import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from bidsgate.cli import main
from bidsgate.inject_atrophy import AtrophySpec
from bidsgate.inject_atrophy import inject as inject_atrophy
from bidsgate.inject_lesions import LesionSpec, brain_and_wm, estimate_brain
from bidsgate.inject_lesions import inject as inject_lesions
from bidsgate.score import score_atrophy, score_lesions


def phantom(tmp_path: Path, shape=(134, 150, 134), zoom=1.5, subject="sub-01"):
    """A BIDS dataset with one subject: a head-shaped phantom at 1.5 mm.

    Brain: ellipsoid with semi-axes 60/70/60 mm (about 1.05 l), grey shell
    and brighter white-matter core; a dark skull gap; a bright scalp shell
    outside it, as in a real T1w. FLAIR: brain mid-grey, scalp bright.
    """
    g = np.indices(shape).astype(float) * zoom
    c = np.array(shape) / 2 * zoom
    r = np.sqrt(((g[0] - c[0]) / 60) ** 2 + ((g[1] - c[1]) / 70) ** 2 + ((g[2] - c[2]) / 60) ** 2)
    t1 = np.zeros(shape, np.float32)
    t1[(r > 1.12) & (r <= 1.25)] = 380  # scalp
    t1[r <= 1.0] = 300  # grey
    t1[r <= 0.8] = 450  # white matter core
    rng = np.random.default_rng(0)
    t1 += (t1 > 0) * rng.normal(0, 5, shape).astype(np.float32)
    fl = np.zeros(shape, np.float32)
    fl[(r > 1.12) & (r <= 1.25)] = 260
    fl[r <= 1.0] = 200
    fl[r <= 0.8] = 180
    d = tmp_path / "bids" / subject / "anat"
    d.mkdir(parents=True, exist_ok=True)
    aff = np.diag([zoom, zoom, zoom, 1.0])
    nib.save(nib.Nifti1Image(t1, aff), d / f"{subject}_T1w.nii.gz")
    nib.save(nib.Nifti1Image(fl, aff), d / f"{subject}_FLAIR.nii.gz")
    (tmp_path / "bids" / "dataset_description.json").write_text('{"Name":"phantom","BIDSVersion":"1.9.0"}')
    return tmp_path / "bids"


def _load(p):
    img = nib.load(p)
    return np.asarray(img.dataobj), img.header.get_zooms()


def test_brain_estimate_excludes_the_scalp_and_wm_is_deep_white_matter(tmp_path):
    root = phantom(tmp_path)
    t1, zooms = _load(root / "sub-01/anat/sub-01_T1w.nii.gz")
    fl, _ = _load(root / "sub-01/anat/sub-01_FLAIR.nii.gz")
    brain, wm = brain_and_wm(t1, zooms, fl)
    ml = brain.sum() * float(np.prod(zooms)) / 1000
    assert 900 < ml < 1200  # the ellipsoid is 1.05 l; the scalp (another 0.9 l) is not in it
    scalp = (t1 > 360) & (t1 < 400)  # scalp is 380, white matter 450, grey 300
    assert (brain & scalp).sum() < 0.01 * scalp.sum()
    assert brain.sum() > wm.sum() > 1000
    assert (t1[wm] > 400).mean() > 0.99
    with pytest.raises(ValueError):
        estimate_brain(t1[:40], zooms)  # a slab has no plausible brain volume


def test_head_mask_alone_is_refused(tmp_path):
    """A T1w with no skull gap (brain fused to a thick scalp) gives an implausible volume and is refused."""
    root = phantom(tmp_path)
    t1, zooms = _load(root / "sub-01/anat/sub-01_T1w.nii.gz")
    t1 = t1.copy()
    t1[t1 == 0] = 300  # fill the whole field of view with tissue
    with pytest.raises(ValueError, match="ml"):
        estimate_brain(t1, zooms)


def test_lesion_injection_truth_matches_mask_and_contrast(tmp_path):
    root = phantom(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    spec = LesionSpec(n=6, volume_mm3=(100.0, 300.0, 1000.0), seed=3)
    truth = inject_lesions(root / "sub-01/anat/sub-01_T1w.nii.gz", root / "sub-01/anat/sub-01_FLAIR.nii.gz",
                           out / "T1w.nii.gz", out / "FLAIR.nii.gz", out / "mask.nii.gz", out / "truth.json", spec)
    mask = np.asarray(nib.load(out / "mask.nii.gz").dataobj)
    assert truth["n"] == 6 and mask.max() == 6
    vox_mm3 = float(np.prod(truth["voxel_mm"]))
    for les in truth["lesions"]:
        vox = (mask == les["id"]).sum()
        assert vox == les["voxels"] and abs(vox * vox_mm3 - les["volume_mm3"]) < 1e-6  # truth volume is the label's
        assert abs(les["volume_mm3"] - les["volume_mm3_nominal"]) / les["volume_mm3_nominal"] < 0.35  # discretised ellipsoid
    t10 = np.asarray(nib.load(root / "sub-01/anat/sub-01_T1w.nii.gz").dataobj)
    assert (t10[mask > 0] > 400).all()  # every labelled voxel is in the white-matter core, none in scalp or air
    fl0 = np.asarray(nib.load(root / "sub-01/anat/sub-01_FLAIR.nii.gz").dataobj)
    fl1 = np.asarray(nib.load(out / "FLAIR.nii.gz").dataobj)
    core = mask > 0
    assert (fl1[core] > fl0[core] * 1.15).mean() > 0.9  # lesions are FLAIR-bright (edge voxels carry at least half the contrast)
    t11 = np.asarray(nib.load(out / "T1w.nii.gz").dataobj)
    assert (t11[core] < t10[core]).mean() > 0.9  # and T1w-dark
    # the added FLAIR contrast at half maximum is the label: a half-maximum segmenter recovers volume ratio ~1
    added = fl1 - fl0
    half = added >= 0.5 * spec.flair_contrast * truth["flair"]["wm_reference"]
    assert 0.85 < half.sum() / core.sum() < 1.15
    # no two lesions touch
    from scipy import ndimage as ndi

    _, ncomp = ndi.label(core, structure=ndi.generate_binary_structure(3, 3))
    assert ncomp == 6


def test_scoring_perfect_and_dilated_predictions(tmp_path):
    root = phantom(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    spec = LesionSpec(n=5, volume_mm3=(100.0, 300.0), seed=1)
    inject_lesions(root / "sub-01/anat/sub-01_T1w.nii.gz", None, out / "T1w.nii.gz", None, out / "mask.nii.gz", out / "truth.json", spec)
    s = score_lesions(out / "truth.json", out / "mask.nii.gz", out / "mask.nii.gz")
    assert s["dice"] == 1.0 and s["sensitivity"] == 1.0 and s["false_positive_components"] == 0 and s["volume_ratio"] == 1.0
    # over-segmentation that touches a true lesion is still a false positive
    ids = np.asarray(nib.load(out / "mask.nii.gz").dataobj)
    c = np.argwhere(ids == 1).mean(0).astype(int)
    slab = ids > 0
    slab[c[0], :, :] = True  # a plane through lesion 1 across the whole volume
    nib.save(nib.Nifti1Image(slab.astype(np.uint8), np.diag([1.5, 1.5, 1.5, 1.0])), out / "slab.nii.gz")
    s = score_lesions(out / "truth.json", out / "mask.nii.gz", out / "slab.nii.gz")
    assert s["false_positive_components"] >= 1 and s["false_positive_volume_mm3"] > 0.9 * (slab.sum() - (ids > 0).sum()) * 1.5**3
    # a prediction on a shifted grid is refused
    nib.save(nib.Nifti1Image((ids > 0).astype(np.uint8), np.diag([1.5, 1.5, 1.5, 1.0]) + np.array([[0, 0, 0, 5.0], [0] * 4, [0] * 4, [0] * 4])), out / "shifted.nii.gz")
    with pytest.raises(ValueError, match="grid"):
        score_lesions(out / "truth.json", out / "mask.nii.gz", out / "shifted.nii.gz")
    from scipy import ndimage as ndi

    m = np.asarray(nib.load(out / "mask.nii.gz").dataobj) > 0
    half = m.copy()
    ids = np.asarray(nib.load(out / "mask.nii.gz").dataobj)
    half[ids == 1] = False  # miss one lesion
    fp = np.zeros_like(m)
    fp[2:6, 2:6, 2:6] = True  # a blob far away
    pred = ndi.binary_dilation(half, iterations=1) | fp
    nib.save(nib.Nifti1Image(pred.astype(np.uint8), np.diag([1.5, 1.5, 1.5, 1.0])), out / "pred.nii.gz")
    s = score_lesions(out / "truth.json", out / "mask.nii.gz", out / "pred.nii.gz")
    assert s["detected"] == 4 and s["lesions"] == 5 and s["false_positive_components"] == 1  # one-voxel dilation is within the 2 mm margin
    assert 0.5 < s["dice"] < 1.0 and s["volume_ratio"] > 1.0


def test_atrophy_shrinks_by_the_requested_factor(tmp_path):
    root = phantom(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    t = inject_atrophy(root / "sub-01/anat/sub-01_T1w.nii.gz", None, out / "T1w.nii.gz", None, out / "truth.json", AtrophySpec(volume_factor=0.90))
    measured = t["brain_volume_mm3_after_measured"] / t["brain_volume_mm3_before"]
    assert abs(measured - 0.90) < 0.02
    # a tool that reports the true volumes scores recovery 1
    s = score_atrophy(out / "truth.json", 1000.0, 900.0)
    assert abs(s["recovery"] - 1.0) < 1e-9
    s = score_atrophy(out / "truth.json", 1000.0, 950.0)
    assert abs(s["recovery"] - 0.5) < 1e-9


def test_cli_end_to_end(tmp_path, capsys):
    root = phantom(tmp_path)
    out = tmp_path / "derivatives" / "bidsgate-lesions"
    assert main(["inject-lesions", str(root), "--out", str(out), "--n", "4", "--seed", "5"]) == 0
    mask = out / "sub-01" / "anat" / "sub-01_desc-lesionTruth_mask.nii.gz"
    assert mask.exists() and (out / "dataset_description.json").exists()
    assert (out / "sub-01" / "anat" / "sub-01_T1w.json").exists()
    with open(out / "sub-01" / "anat" / "sub-01_desc-lesion_truth.json") as fh:
        seed_full = json.load(fh)["seed"]
    # the same subject gets the same seed whether or not it is selected alone
    out_sel = tmp_path / "derivatives" / "sel"
    assert main(["inject-lesions", str(root), "--out", str(out_sel), "--n", "4", "--seed", "5", "--subject", "sub-01"]) == 0
    with open(out_sel / "sub-01" / "anat" / "sub-01_desc-lesion_truth.json") as fh:
        assert json.load(fh)["seed"] == seed_full
    # a supplied brain mask is honoured and recorded
    brain_mask = tmp_path / "brain.nii.gz"
    t1_img = nib.load(root / "sub-01/anat/sub-01_T1w.nii.gz")
    nib.save(nib.Nifti1Image((np.asarray(t1_img.dataobj) > 250).astype(np.uint8), t1_img.affine), brain_mask)
    out_m = tmp_path / "derivatives" / "masked"
    assert main(["inject-lesions", str(root), "--out", str(out_m), "--n", "3", "--mask", str(brain_mask)]) == 0
    with open(out_m / "sub-01" / "anat" / "sub-01_desc-lesion_truth.json") as fh:
        assert json.load(fh)["brain_mask"] == "given"
    pred_dir = tmp_path / "derivatives" / "mytool" / "sub-01"
    pred_dir.mkdir(parents=True)
    import shutil

    shutil.copy(mask, pred_dir / "sub-01_seg.nii.gz")
    assert main(["score-lesions", "--truth", str(out), "--pred", str(tmp_path / "derivatives/mytool/{subject}/{base}_seg.nii.gz"),
                 "--pipeline", "mytool", "--out", str(tmp_path / "scores")]) == 0
    page = (tmp_path / "scores" / "scorecard_lesions.html").read_text()
    assert "mytool" in page and "1.00" in page
    with open(tmp_path / "scores" / "scores_lesions.json") as fh:
        data = json.load(fh)
    assert data["results"][0]["score"]["dice"] == 1.0
    atro = tmp_path / "derivatives" / "bidsgate-atrophy"
    assert main(["inject-atrophy", str(root), "--out", str(atro), "--factor", "0.9"]) == 0
    vols = tmp_path / "vols.tsv"
    vols.write_text("subject\tvolume_before_mm3\tvolume_after_mm3\nsub-01\t1000\t910\n")
    assert main(["score-atrophy", "--truth", str(atro), "--volumes", str(vols), "--pipeline", "mytool", "--out", str(tmp_path / "scores2")]) == 0
    assert "0.90" in (tmp_path / "scores2" / "scorecard_atrophy.html").read_text()


def test_run_entities_are_kept_in_derivative_names(tmp_path):
    root = phantom(tmp_path)
    d = root / "sub-01" / "anat"
    for run in ("1", "2"):
        for suf in ("T1w", "FLAIR"):
            (d / f"sub-01_run-{run}_{suf}.nii.gz").write_bytes((d / f"sub-01_{suf}.nii.gz").read_bytes())
        (d / f"sub-01_{suf}.nii.gz").unlink() if False else None
    (d / "sub-01_T1w.nii.gz").unlink()
    (d / "sub-01_FLAIR.nii.gz").unlink()
    out = tmp_path / "out"
    assert main(["inject-lesions", str(root), "--out", str(out), "--n", "3"]) == 0
    names = sorted(p.name for p in (out / "sub-01" / "anat").glob("*truth.json"))
    assert names == ["sub-01_run-1_desc-lesion_truth.json", "sub-01_run-2_desc-lesion_truth.json"]
    seeds = {json.loads((out / "sub-01" / "anat" / n).read_text())["seed"] for n in names}
    assert len(seeds) == 2


def test_partial_subject_is_not_written_when_flair_grid_differs(tmp_path):
    root = phantom(tmp_path)
    d = root / "sub-01" / "anat"
    fl = nib.load(d / "sub-01_FLAIR.nii.gz")
    nib.save(nib.Nifti1Image(np.asarray(fl.dataobj)[:-10], fl.affine), d / "sub-01_FLAIR.nii.gz")
    out = tmp_path / "out"
    assert main(["inject-lesions", str(root), "--out", str(out), "--n", "3"]) == 0
    assert not list((out / "sub-01" / "anat").glob("*.nii.gz"))


def test_brain_estimate_keeps_both_hemispheres_and_fills_ventricles(tmp_path):
    root = phantom(tmp_path)
    t1_img = nib.load(root / "sub-01/anat/sub-01_T1w.nii.gz")
    t1 = np.asarray(t1_img.dataobj).copy()
    zooms = t1_img.header.get_zooms()
    mid = t1.shape[0] // 2
    t1[mid - 1:mid + 2, :, :] = np.where(t1[mid - 1:mid + 2] > 250, 0, t1[mid - 1:mid + 2])  # a 4.5 mm fissure through the brain
    ref = estimate_brain(np.asarray(t1_img.dataobj), zooms)
    # a hollow (ventricle) connected to the outside by a thin channel
    c = np.array(t1.shape) // 2
    t1[c[0] + 10:c[0] + 16, c[1] - 6:c[1] + 6, c[2] - 6:c[2] + 6] = 0
    t1[c[0] + 10:c[0] + 16, c[1], c[2]:] = 0
    brain = estimate_brain(t1, zooms)
    assert brain.sum() > 0.9 * ref.sum()  # the fissure itself is not tissue; everything else is kept
    assert brain[c[0] - 25, c[1], c[2]] and brain[c[0] + 25, c[1], c[2]]  # both halves
    assert brain[c[0] + 12, c[1], c[2]]
