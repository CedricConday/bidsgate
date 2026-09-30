import sys

import nibabel as nib
import numpy as np

from bidsgate.rescanphantom.core import run

SEG = (f"{sys.executable} -c \"import nibabel as n,numpy as p,sys;i=n.load(sys.argv[1]);d=p.asarray(i.dataobj);"
       "h=d>0;t=d[h].mean()+2.5*d[h].std();n.save(n.Nifti1Image((d>t).astype(p.uint8),i.affine),sys.argv[2])\" {flair} {out}")


def test_threshold_segmenter_invents_change(tmp_path):
    rng = np.random.default_rng(0)
    sh = (40, 40, 40)
    fl = np.where(np.indices(sh).sum(0) > 0, 100.0, 0).astype(np.float32) + rng.normal(0, 3, sh).astype(np.float32)
    fl[18:22, 18:22, 18:22] = 180          # a clear lesion
    fl[5:7, 30:32, 10:12] = 112            # a borderline one: the threshold will flip on it
    for n in ("t1", "fl"):
        nib.save(nib.Nifti1Image(fl, np.eye(4)), tmp_path / f"{n}.nii.gz")
    _df, s = run(tmp_path / "t1.nii.gz", tmp_path / "fl.nii.gz", SEG, tmp_path / "out", k=3, min_mm3=1)
    assert s["pairs"] == 3 and 0 < s["mean_dice"] <= 1
