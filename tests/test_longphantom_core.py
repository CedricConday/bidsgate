import nibabel as nib
import numpy as np

from bidsgate.longphantom.core import Design, build


def test_series(tmp_path):
    sh = (96, 112, 96)
    z, y, x = np.indices(sh)
    c = np.array(sh) / 2
    r = np.sqrt(((z - c[0]) / 40) ** 2 + ((y - c[1]) / 48) ** 2 + ((x - c[2]) / 40) ** 2)
    brain = r < 1
    t1 = np.where(brain, 300.0, 0).astype(np.float32)
    t1[brain & (r > 0.85)] = 200
    fl = np.where(brain, 80.0, 0).astype(np.float32)
    mask = np.zeros(sh, np.uint8)
    for i, (a, b, cc) in enumerate(((34, 44, 34), (56, 44, 34), (34, 64, 56), (56, 64, 56), (46, 40, 62), (46, 74, 30))):
        s = 7 if i < 4 else 3
        mask[a:a + s, b:b + s, cc:cc + s] = 1
    t1[mask > 0] = 220
    fl[mask > 0] = 150
    for n, arr in (("t1", t1), ("fl", fl), ("m", mask), ("b", brain.astype(np.uint8))):
        nib.save(nib.Nifti1Image(arr, np.eye(4)), tmp_path / f"{n}.nii.gz")
    d = Design(times=(0.0, 1.0), n_enlarge=2, n_shrink=1, n_resolve=1, n_new=2, min_voxels=100, grow=2.0, shrink=0.5,
               rigid_rot_deg=0, rigid_trans_vox=0)
    t = build(tmp_path / "t1.nii.gz", tmp_path / "fl.nii.gz", tmp_path / "m.nii.gz", tmp_path / "b.nii.gz", tmp_path / "out", d)
    v0, v1 = t[t.visit == 0].set_index("lesion"), t[t.visit == 1].set_index("lesion")
    enl = v0[v0.fate == "enlarging"].index
    shr = v0[v0.fate == "shrinking"].index
    assert (v1.loc[enl, "volume_mm3"] > v0.loc[enl, "volume_mm3"]).all()
    assert (v1.loc[shr, "volume_mm3"] < v0.loc[shr, "volume_mm3"]).all()
    assert (v1[v1.fate == "resolving"]["volume_mm3"] == 0).all()
    assert (v0[v0.origin == "new"]["volume_mm3"] == 0).all() and (v1[v1.origin == "new"]["volume_mm3"] > 0).all()
