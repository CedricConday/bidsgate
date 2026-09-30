"""longphantom T1w FLAIR lesion_mask brainmask --out DIR [--times 0 0.5 1 2] [--seed 0]"""

import argparse

from .core import Design, build


def main(argv=None) -> int:
    import sys

    if len(sys.argv if argv is None else argv) > 1 and (sys.argv if argv is None else argv)[1 if argv is None else 0] == "score":
        return _score(sys.argv[2:] if argv is None else argv[1:])
    ap = argparse.ArgumentParser(prog="longphantom", description=__doc__)
    ap.add_argument("t1")
    ap.add_argument("flair")
    ap.add_argument("mask")
    ap.add_argument("brainmask")
    ap.add_argument("--out", required=True)
    ap.add_argument("--times", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--atrophy", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    t = build(a.t1, a.flair, a.mask, a.brainmask, a.out, Design(times=tuple(a.times), atrophy=a.atrophy, seed=a.seed))
    last = t[t.visit == t.visit.max()]
    print(last.groupby("fate")["volume_mm3"].agg(["count", "sum"]).round(0).to_string())
    print(f"{a.out}/manifest.tsv")
    return 0


def _score(argv) -> int:
    from .score import score

    ap = argparse.ArgumentParser(prog="longphantom score", description="score a lesiontrack output dir against a phantom")
    ap.add_argument("phantom")
    ap.add_argument("lesiontrack_dir")
    a = ap.parse_args(argv)
    df, missed, conf = score(a.phantom, a.lesiontrack_dir)
    print(conf.to_string())
    print(f"groups {len(df)}, correct {int(df['correct'].sum())}, false {int((df['truth'] == 'none (false)').sum())}, "
          f"truth lesions not found {len(missed)}")
    return 0
