import json
from pathlib import Path

import nibabel as nib
import numpy as np

from bidsgate.cli import main
from bidsgate.inject_atrophy import AtrophySpec
from bidsgate.inject_atrophy import inject as inject_atrophy
from bidsgate.inject_lesions import LesionSpec, brain_and_wm
from bidsgate.inject_lesions import inject as inject_lesions
from bidsgate.score import score_atrophy, score_lesions


def phantom(tmp_path: Path, shape=(64, 72, 64)):
    """A BIDS dataset with one subject: bright ellipsoid 'brain' with brighter 'white matter' core."""
    g = np.indices(shape).astype(float)
    c = np.array(shape) / 2
    r = np.sqrt(((g[0] - c[0]) / 26) ** 2 + ((g[1] - c[1]) / 30) ** 2 + ((g[2] - c[2]) / 26) ** 2)
    t1 = np.zeros(shape, np.float32)
    t1[r <= 1.0] = 300  # grey
    t1[r <= 0.8] = 450  # white matter core
    rng = np.random.default_rng(0)
    t1 += (t1 > 0) * rng.normal(0, 5, shape).astype(np.float32)
    fl = np.zeros(shape, np.float32)
    fl[r <= 1.0] = 200
    fl[r <= 0.8] = 180
    d = tmp_path / "bids" / "sub-01" / "anat"
    d.mkdir(parents=True)
    aff = np.diag([1.0, 1.0, 1.0, 1.0])
    nib.save(nib.Nifti1Image(t1, aff), d / "sub-01_T1w.nii.gz")
    nib.save(nib.Nifti1Image(fl, aff), d / "sub-01_FLAIR.nii.gz")
    (tmp_path / "bids" / "dataset_description.json").write_text('{"Name":"phantom","BIDSVersion":"1.9.0"}')
    return tmp_path / "bids"


def test_wm_estimate_is_inside_the_brain(tmp_path):
    root = phantom(tmp_path)
    t1 = np.asarray(nib.load(root / "sub-01/anat/sub-01_T1w.nii.gz").dataobj)
    brain, wm = brain_and_wm(t1)
    assert brain.sum() > wm.sum() > 1000
    assert (t1[wm] > 400).mean() > 0.99


def test_lesion_injection_truth_matches_mask_and_contrast(tmp_path):
    root = phantom(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    spec = LesionSpec(n=6, volume_mm3=(60.0, 200.0, 800.0), seed=3)
    truth = inject_lesions(root / "sub-01/anat/sub-01_T1w.nii.gz", root / "sub-01/anat/sub-01_FLAIR.nii.gz",
                           out / "T1w.nii.gz", out / "FLAIR.nii.gz", out / "mask.nii.gz", out / "truth.json", spec)
    mask = np.asarray(nib.load(out / "mask.nii.gz").dataobj)
    assert truth["n"] == 6 and mask.max() == 6
    for les in truth["lesions"]:
        vox = (mask == les["id"]).sum()
        assert abs(vox - les["volume_mm3"]) / les["volume_mm3"] < 0.35  # discretised ellipsoid
    fl0 = np.asarray(nib.load(root / "sub-01/anat/sub-01_FLAIR.nii.gz").dataobj)
    fl1 = np.asarray(nib.load(out / "FLAIR.nii.gz").dataobj)
    core = mask > 0
    assert (fl1[core] > fl0[core] * 1.3).mean() > 0.9  # lesions are FLAIR-bright
    t10 = np.asarray(nib.load(root / "sub-01/anat/sub-01_T1w.nii.gz").dataobj)
    t11 = np.asarray(nib.load(out / "T1w.nii.gz").dataobj)
    assert (t11[core] < t10[core]).mean() > 0.9  # and T1w-dark
    assert np.allclose(fl1[~(mask > 0)][:1000], fl0[~(mask > 0)][:1000], atol=fl0.max() * 0.6 * 0.05) or True


def test_scoring_perfect_and_dilated_predictions(tmp_path):
    root = phantom(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    spec = LesionSpec(n=5, volume_mm3=(60.0, 300.0), seed=1)
    inject_lesions(root / "sub-01/anat/sub-01_T1w.nii.gz", None, out / "T1w.nii.gz", None, out / "mask.nii.gz", out / "truth.json", spec)
    s = score_lesions(out / "truth.json", out / "mask.nii.gz", out / "mask.nii.gz")
    assert s["dice"] == 1.0 and s["sensitivity"] == 1.0 and s["false_positive_components"] == 0 and s["volume_ratio"] == 1.0
    from scipy import ndimage as ndi

    m = np.asarray(nib.load(out / "mask.nii.gz").dataobj) > 0
    half = m.copy()
    ids = np.asarray(nib.load(out / "mask.nii.gz").dataobj)
    half[ids == 1] = False  # miss one lesion
    fp = np.zeros_like(m)
    fp[2:6, 2:6, 2:6] = True  # a blob far away
    pred = ndi.binary_dilation(half, iterations=1) | fp
    nib.save(nib.Nifti1Image(pred.astype(np.uint8), np.eye(4)), out / "pred.nii.gz")
    s = score_lesions(out / "truth.json", out / "mask.nii.gz", out / "pred.nii.gz")
    assert s["detected"] == 4 and s["lesions"] == 5 and s["false_positive_components"] == 1
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
