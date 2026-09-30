"""rescanphantom T1w FLAIR --cmd 'segmenter {t1} {flair} {out}' --out dir [--k 3]"""

import argparse
import json

from .core import run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rescanphantom", description=__doc__)
    ap.add_argument("t1")
    ap.add_argument("flair")
    ap.add_argument("--cmd", required=True, help="template with {t1} {flair} {out}; must write a binary mask to {out}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-mm3", type=float, default=10.0)
    a = ap.parse_args(argv)
    df, s = run(a.t1, a.flair, a.cmd, a.out, a.k, a.seed, a.min_mm3)
    df.to_csv(f"{a.out}/pairs.tsv", sep="\t", index=False)
    print(json.dumps(s, indent=2))
    return 0
