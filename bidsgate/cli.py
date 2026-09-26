"""bidsgate: inject a known truth into BIDS data, then score what a pipeline recovered."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import __version__
from .bids import copy_json_sidecar, derivative_path, find_anat, sibling, write_dataset_description
from .inject_atrophy import AtrophySpec
from .inject_atrophy import inject as inject_atrophy
from .inject_lesions import LesionSpec
from .inject_lesions import inject as inject_lesions
from .report import scorecard_atrophy, scorecard_lesions
from .score import score_atrophy, score_lesions


def cmd_inject_lesions(a) -> int:
    root, out = Path(a.bids), Path(a.out)
    anats = find_anat(root, "T1w", a.subject)
    if not anats:
        raise SystemExit(f"no T1w images under {root}")
    write_dataset_description(out, "bidsgate lesions", root, "lesion")
    for i, anat in enumerate(anats):
        flair = sibling(anat, "FLAIR")
        spec = LesionSpec(n=a.n, seed=a.seed + i, flair_contrast=a.flair_contrast, t1_contrast=a.t1_contrast)
        out_t1 = derivative_path(out, anat, "T1w")
        out_fl = derivative_path(out, anat, "FLAIR") if flair else None
        mask = derivative_path(out, anat, "mask", desc="lesionTruth")
        truth = derivative_path(out, anat, "truth", desc="lesion", ext=".json")
        t = inject_lesions(anat.path, flair, out_t1, out_fl, mask, truth, spec)
        note = {"BidsgateInjection": "lesions", "BidsgateSeed": spec.seed, "BidsgateTruth": truth.name}
        copy_json_sidecar(anat.path, out_t1, note)
        if flair and out_fl:
            copy_json_sidecar(flair, out_fl, note)
        print(f"{anat.base}: {t['n']} lesions, {t['total_volume_mm3']:.0f} mm3, {'T1w+FLAIR' if flair else 'T1w only'} -> {out_t1.parent}")
    print(f"derivative dataset written to {out}; run your pipeline on it, then `bidsgate score-lesions`")
    return 0


def cmd_inject_atrophy(a) -> int:
    root, out = Path(a.bids), Path(a.out)
    anats = find_anat(root, "T1w", a.subject)
    if not anats:
        raise SystemExit(f"no T1w images under {root}")
    write_dataset_description(out, "bidsgate atrophy", root, "atrophy")
    for anat in anats:
        flair = sibling(anat, "FLAIR")
        out_t1 = derivative_path(out, anat, "T1w")
        out_fl = derivative_path(out, anat, "FLAIR") if flair else None
        truth = derivative_path(out, anat, "truth", desc="atrophy", ext=".json")
        t = inject_atrophy(anat.path, flair, out_t1, out_fl, truth, AtrophySpec(volume_factor=a.factor, falloff_mm=a.falloff))
        copy_json_sidecar(anat.path, out_t1, {"BidsgateInjection": "atrophy", "BidsgateVolumeFactor": a.factor, "BidsgateTruth": truth.name})
        if flair and out_fl:
            copy_json_sidecar(flair, out_fl, {"BidsgateInjection": "atrophy", "BidsgateVolumeFactor": a.factor})
        print(f"{anat.base}: brain {t['brain_volume_mm3_before']/1000:.0f} ml -> factor {a.factor} (measured on the mask: {t['brain_volume_mm3_after_measured']/t['brain_volume_mm3_before']:.3f})")
    print(f"derivative dataset written to {out}; run your morphometry on {root} and on {out}, then `bidsgate score-atrophy`")
    return 0


def cmd_score_lesions(a) -> int:
    truth_root = Path(a.truth)
    results = []
    for truth_json in sorted(truth_root.glob("sub-*/**/anat/*desc-lesion_truth.json")):
        mask = Path(str(truth_json).replace("desc-lesion_truth.json", "desc-lesionTruth_mask.nii.gz"))
        base = truth_json.name.replace("_desc-lesion_truth.json", "")
        pred = Path(a.pred.format(base=base, subject=base.split("_")[0]))
        if not pred.exists():
            print(f"{base}: prediction not found at {pred}", file=sys.stderr)
            continue
        s = score_lesions(truth_json, mask, pred, a.threshold)
        results.append({"subject": base, "score": s})
        print(f"{base}: Dice {s['dice']:.2f}  detected {s['detected']}/{s['lesions']}  FP {s['false_positive_components']}  volume ratio {s['volume_ratio']:.2f}")
    if not results:
        raise SystemExit("nothing scored")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "scores_lesions.json", "w") as fh:
        json.dump({"pipeline": a.pipeline, "results": results}, fh, indent=2)
    page = scorecard_lesions(results, a.pipeline, out / "scorecard_lesions.html")
    print(f"scorecard: {page}")
    return 0


def cmd_score_atrophy(a) -> int:
    """Volumes come from the user's tool as a TSV: subject, volume_before_mm3, volume_after_mm3."""
    truth_root = Path(a.truth)
    vols = pd.read_csv(a.volumes, sep="\t")
    results = []
    for _, row in vols.iterrows():
        base = str(row["subject"])
        hits = list(truth_root.glob(f"{base.split('_')[0]}/**/anat/{base}*desc-atrophy_truth.json"))
        if not hits:
            print(f"{base}: no truth found", file=sys.stderr)
            continue
        s = score_atrophy(hits[0], float(row["volume_before_mm3"]), float(row["volume_after_mm3"]))
        results.append({"subject": base, "score": s})
        print(f"{base}: injected {s['injected_change_pct']:+.1f}%  measured {s['measured_change_pct']:+.1f}%  recovery {s['recovery']:.2f}")
    if not results:
        raise SystemExit("nothing scored")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "scores_atrophy.json", "w") as fh:
        json.dump({"pipeline": a.pipeline, "results": results}, fh, indent=2)
    print(f"scorecard: {scorecard_atrophy(results, a.pipeline, out / 'scorecard_atrophy.html')}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bidsgate", description=__doc__)
    ap.add_argument("--version", action="version", version=f"bidsgate {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inject-lesions", help="write a derivative dataset with synthetic lesions and their truth")
    p.add_argument("bids")
    p.add_argument("--out", required=True)
    p.add_argument("--subject", action="append", help="restrict to these subjects (repeatable)")
    p.add_argument("--n", type=int, default=12, help="lesions per subject")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--flair-contrast", type=float, default=0.6, help="FLAIR gain over local white matter at the core")
    p.add_argument("--t1-contrast", type=float, default=-0.2, help="T1w change over local white matter at the core")
    p.set_defaults(func=cmd_inject_lesions)

    p = sub.add_parser("inject-atrophy", help="write a derivative dataset with a known brain-volume change")
    p.add_argument("bids")
    p.add_argument("--out", required=True)
    p.add_argument("--subject", action="append")
    p.add_argument("--factor", type=float, default=0.95, help="brain volume factor, 0.95 = 5 %% loss")
    p.add_argument("--falloff", type=float, default=12.0, help="mm over which the deformation fades outside the brain")
    p.set_defaults(func=cmd_inject_atrophy)

    p = sub.add_parser("score-lesions", help="score predicted lesion masks against the injected truth")
    p.add_argument("--truth", required=True, help="the inject-lesions output directory")
    p.add_argument("--pred", required=True, help="path pattern with {base} or {subject}, e.g. derivatives/lst/{subject}/{base}_seg.nii.gz")
    p.add_argument("--pipeline", required=True, help="name for the scorecard")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out", default="bidsgate-scores")
    p.set_defaults(func=cmd_score_lesions)

    p = sub.add_parser("score-atrophy", help="score a tool's reported volumes against the injected change")
    p.add_argument("--truth", required=True)
    p.add_argument("--volumes", required=True, help="TSV: subject, volume_before_mm3, volume_after_mm3")
    p.add_argument("--pipeline", required=True)
    p.add_argument("--out", default="bidsgate-scores")
    p.set_defaults(func=cmd_score_atrophy)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
