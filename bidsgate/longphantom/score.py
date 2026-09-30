"""Score a lesiontrack run (baseline against the last visit) against the phantom truth.

lesiontrack works in a halfway space. The truth label maps of the first and last visits are resliced into
it with the same transforms lesiontrack used (baseline: the frame matrix inverted; last visit: its own
halfway matrix), then every tracked group is matched to the truth lesion it overlaps most.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

EXPECT = {"enlarging": {"enlarging", "trend_up"}, "shrinking": {"shrinking", "trend_down"}, "resolving": {"resolved"},
          "stable": {"stable", "trend_up", "trend_down"}, "new": {"new", "adjacent_fragment"}}


def _ids(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    return [int(float(x)) for x in str(v).split(",") if str(x).strip()]


def score(phantom: Path, lt_dir: Path, threads: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    from lesiontrack.registration import run_greedy

    phantom, lt_dir = Path(phantom), Path(lt_dir)
    man = pd.read_csv(phantom / "manifest.tsv", sep="\t")
    first, last = man.iloc[0], man.iloc[-1]
    fu = str(last["session"])
    reg = lt_dir / "reg"
    t0 = lt_dir / "truth_first_halfway.nii.gz"
    t1 = lt_dir / "truth_last_halfway.nii.gz"
    v0 = phantom / Path(first["t1"]).parent.name / "truth_labels.nii.gz"
    v1 = phantom / Path(last["t1"]).parent.name / "truth_labels.nii.gz"
    run_greedy(f"-d 3 -threads {threads} -rf {first['t1']} -ri NN -rt short -rm {v0} {t0} -r {reg / (fu + '_halfway.mat')},-1", None)
    run_greedy(f"-d 3 -threads {threads} -rf {first['t1']} -ri NN -rt short -rm {v1} {t1} -r {reg / (fu + '_halfway.mat')}", None)
    tb = np.asarray(nib.load(t0).dataobj).astype(np.int64)
    tl = np.asarray(nib.load(t1).dataobj).astype(np.int64)
    lb = np.asarray(nib.load(lt_dir / "baseline_lesion_labels.nii.gz").dataobj).astype(np.int64)
    lf = np.asarray(nib.load(lt_dir / f"lesion_labels_{fu}.nii.gz").dataobj).astype(np.int64)
    track = pd.read_csv(lt_dir / "lesion_tracking.tsv", sep="\t", dtype={"baseline_ids": str, "followup_ids": str, "follow_up": str})
    track = track[track["follow_up"] == fu]
    truth = pd.read_csv(phantom / "truth.tsv", sep="\t")
    fates = truth.drop_duplicates("lesion").set_index("lesion")["fate"].to_dict()
    # The truth class of a baseline lesion is what its measured truth volume did, judged with lesiontrack's
    # own bands; the design only says what was attempted (a large confluent lesion barely grows under a
    # local radial field).
    from lesiontrack.config import TrackParams
    from lesiontrack.tracking import classify_rate

    tp = TrackParams()
    v_first = truth[truth.visit == truth.visit.min()].set_index("lesion")["volume_mm3"]
    v_last = truth[truth.visit == truth.visit.max()].set_index("lesion")
    dt = float(v_last["time_years"].iloc[0] - truth["time_years"].min())
    measured = {}
    for k in fates:
        if int(k) >= 1000:
            continue
        a, b = float(v_first[k]), float(v_last.loc[k, "volume_mm3"])
        if b == 0:
            measured[k] = "resolving"
        elif a > 0:
            c = classify_rate((b - a) / a * 100 / dt, round(b - a), tp)
            measured[k] = {"enlarging": "enlarging", "shrinking": "shrinking"}.get(c, "stable")
    design = dict(fates)
    fates = {**fates, **measured}
    rows = []
    for _, g in track.iterrows():
        if _ids(g["baseline_ids"]):
            sel = np.isin(lb, _ids(g["baseline_ids"]))
            lab = tb[sel]
        else:
            sel = np.isin(lf, _ids(g["followup_ids"]))
            lab = tl[sel]
        lab = lab[lab > 0]
        tid = int(np.bincount(lab).argmax()) if lab.size else 0
        fate = fates.get(tid, "none (false)") if tid else "none (false)"
        fate = "new" if str(fate).startswith("new") else fate
        rows.append({"group": int(g["group_id"]), "called": g["class"], "truth_lesion": tid, "truth": fate,
                     "design": design.get(tid, ""),
                     "correct": g["class"] in EXPECT.get(fate, set())})
    df = pd.DataFrame(rows)
    # When resampling splits a small piece off a truth lesion, two tracked groups map to it; the largest is the
    # lesion, the others are fragments and are listed but not scored.
    vol = track.set_index(track["group_id"].astype(int))[["volume_baseline_mm3", "volume_followup_mm3"]].max(axis=1)
    df["group_volume_mm3"] = df["group"].map(vol)
    main = df[df["truth_lesion"] > 0].sort_values("group_volume_mm3", ascending=False).drop_duplicates("truth_lesion")["group"]
    df["fragment"] = (df["truth_lesion"] > 0) & ~df["group"].isin(main)
    df.loc[df["fragment"], "correct"] = True
    seen = set(df["truth_lesion"])
    last_v = truth[truth.visit == truth.visit.max()]
    missed = [{"truth_lesion": int(k), "truth": ("new" if str(fates[k]).startswith("new") else fates[k])}
              for k in fates if k not in seen and (k < 1000 or last_v.set_index("lesion").loc[k, "volume_mm3"] > 0)]
    conf = pd.crosstab(df[~df["fragment"]]["truth"], df[~df["fragment"]]["called"])
    return df, pd.DataFrame(missed), conf
