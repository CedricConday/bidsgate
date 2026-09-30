"""Aggregate a segmenter's evidence into one card (JSON + self-contained HTML).

Inputs, all optional but at least one: a bidsgate ``scores_lesions.json``; one or more rescanphantom
summaries (the JSON it prints, saved to a file); a TSV of Dice against expert masks (subject, dice).
Nothing is re-computed; the card pools what the tools measured and says which evidence is missing.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd


def pool_bidsgate(path) -> dict:
    d = json.loads(Path(path).read_text())
    rows = [r["score"] for r in d["results"]]
    det = sum(r["detected"] for r in rows)
    n = sum(r["lesions"] for r in rows)
    def pool(key):
        acc = {}
        for r in rows:
            for k, v in (r.get(key) or {}).items():
                a = acc.setdefault(k, [0, 0.0])
                a[0] += v["n"]
                a[1] += v["n"] * v["sensitivity"]
        return {k: {"n": a[0], "sensitivity": a[1] / a[0]} for k, a in sorted(acc.items())}
    return {"pipeline": d.get("pipeline"), "scans": len(rows), "injected": n, "detected": det,
            "sensitivity": det / n if n else None, "dice_injected_mean": sum(r["dice"] for r in rows) / len(rows),
            "fp_components_per_scan": sum(r["false_positive_components"] for r in rows) / len(rows),
            "by_size": pool("by_size"), "by_height": pool("by_height")}


def build(name: str, bidsgate=None, rescans=(), reference=None) -> dict:
    card = {"segmenter": name, "evidence": {}, "missing": []}
    if bidsgate:
        card["evidence"]["injection"] = pool_bidsgate(bidsgate)
    else:
        card["missing"].append("injected-lesion sensitivity (bidsgate)")
    if rescans:
        rs = [json.loads(Path(p).read_text()) for p in rescans]
        card["evidence"]["rescan"] = {"scans": len(rs), "pairs": sum(r["pairs"] for r in rs),
                                      "invented_new_per_pair": sum(r["invented_new_per_pair"] * r["pairs"] for r in rs) / sum(r["pairs"] for r in rs),
                                      "invented_resolved_per_pair": sum(r["invented_resolved_per_pair"] * r["pairs"] for r in rs) / sum(r["pairs"] for r in rs),
                                      "mean_dice": sum(r["mean_dice"] for r in rs) / len(rs)}
    else:
        card["missing"].append("scan-rescan stability (rescanphantom)")
    if reference:
        t = pd.read_csv(reference, sep="\t")
        card["evidence"]["reference"] = {"scans": len(t), "dice_mean": float(t["dice"].mean()), "dice_median": float(t["dice"].median())}
    else:
        card["missing"].append("agreement with expert masks")
    return card


def _pct(x):
    return "–" if x is None else f"{x * 100:.0f} %"


def render(card: dict) -> str:
    e = card["evidence"]
    rows = []
    if "injection" in e:
        i = e["injection"]
        rows.append(("Injected lesions found", f"{i['detected']} of {i['injected']} ({_pct(i['sensitivity'])}) on {i['scans']} scans"))
        rows += [(f"… {k}", f"{_pct(v['sensitivity'])} of {v['n']}") for k, v in i["by_size"].items()]
        rows += [(f"… {k} of the brain", f"{_pct(v['sensitivity'])} of {v['n']}") for k, v in i["by_height"].items()]
        rows.append(("False-positive components per scan", f"{i['fp_components_per_scan']:.1f}"))
    if "rescan" in e:
        r = e["rescan"]
        rows.append(("New lesions invented between rescans", f"{r['invented_new_per_pair']:.1f} per pair ({r['pairs']} pairs)"))
        rows.append(("Lesions lost between rescans", f"{r['invented_resolved_per_pair']:.1f} per pair"))
        rows.append(("Dice between rescans", f"{r['mean_dice']:.2f}"))
    if "reference" in e:
        rows.append(("Dice against expert masks", f"{e['reference']['dice_mean']:.2f} mean over {e['reference']['scans']} scans"))
    body = "".join(f"<tr><td>{html.escape(a)}</td><td>{html.escape(b)}</td></tr>" for a, b in rows)
    miss = "".join(f"<li>{html.escape(m)}</li>" for m in card["missing"])
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>segcard {html.escape(card['segmenter'])}</title><style>body{{font-family:system-ui,sans-serif;max-width:640px;margin:24px auto;padding:0 16px}}"
            f"td{{padding:6px 10px;border-bottom:1px solid #ddd}}td:last-child{{text-align:right;font-variant-numeric:tabular-nums}}</style></head><body>"
            f"<h1>{html.escape(card['segmenter'])}</h1><p>Benchmark card from bidsgate.segcard. Every line is measured by an open tool; nothing is self-reported.</p>"
            f"<table>{body}</table>" + (f"<h3>Not yet measured</h3><ul>{miss}</ul>" if miss else "") + "</body></html>")
