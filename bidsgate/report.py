"""A self-contained HTML scorecard, one file, no scripts, readable in any browser."""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path


def _pct(x) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "-"
    return "-" if v != v else f"{100 * v:.0f}%"


def _f(x, d=2) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "-"
    return "-" if v != v else f"{v:.{d}f}"


def scorecard_lesions(results: list[dict], pipeline: str, out: Path) -> Path:
    """``results`` is a list of {"subject": ..., "score": score_lesions(...)}."""
    rows = []
    for r in results:
        s = r["score"]
        rows.append(f"<tr><td>{html.escape(r['subject'])}</td><td>{_f(s['dice'])}</td><td>{s['detected']}/{s['lesions']} ({_pct(s['sensitivity'])})</td>"
                    f"<td>{s['false_positive_components']}</td><td>{_f(s['volume_ratio'])}</td>"
                    + "".join(f"<td>{_pct(v['sensitivity'])} (n={v['n']})</td>" for _, v in sorted(s['by_size'].items()))
                    + "".join(f"<td>{_pct(s['by_height'].get(h, {}).get('sensitivity'))} (n={s['by_height'].get(h, {}).get('n', 0)})</td>" for h in ('lower', 'middle', 'upper'))
                    + "</tr>")
    n = len(results)
    dice = sum(r["score"]["dice"] for r in results) / n if n else float("nan")
    sens = sum(r["score"]["sensitivity"] for r in results) / n if n else float("nan")
    fp = sum(r["score"]["false_positive_components"] for r in results)
    bins = sorted(results[0]["score"]["by_size"].keys()) if results else []
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>bidsgate scorecard: {html.escape(pipeline)}</title>
<style>body{{font:15px/1.4 system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#222}}
table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #ddd;padding:.4rem .5rem;text-align:left}}
th{{background:#f4f4f4}}.big{{font-size:2rem;margin:.2rem 0}}.grid{{display:flex;gap:2rem;flex-wrap:wrap}}.note{{color:#555}}</style></head><body>
<h1>bidsgate scorecard: {html.escape(pipeline)}</h1>
<p class="note">Synthetic lesions with a known mask were injected into real T1w/FLAIR images; the pipeline was run on the result; this is what it recovered. Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. Nothing here is evidence about any disease; it is a test of software.</p>
<div class="grid"><div><div class="big">{_f(dice)}</div>mean Dice</div><div><div class="big">{_pct(sens)}</div>lesion-wise sensitivity</div><div><div class="big">{fp}</div>false-positive components</div><div><div class="big">{n}</div>subjects</div></div>
<table><tr><th>subject</th><th>Dice</th><th>detected</th><th>FP comps</th><th>volume ratio</th>{''.join(f'<th>sens {html.escape(b)}</th>' for b in bins)}<th>sens lower third</th><th>sens middle third</th><th>sens upper third</th></tr>{''.join(rows)}</table>
<p>Height thirds are of the brain mask's superior-inferior extent, recorded at injection; a miss pattern by height is the first thing to check against a per-region atlas (<code>--regions</code>).</p>
<p class="note">Sensitivity by injected lesion volume shows the detection floor: the size below which the pipeline stops seeing lesions. Volume ratio is predicted over injected volume; under 1 means the pipeline under-segments what it does find.</p>
</body></html>"""
    out.write_text(doc)
    return out


def scorecard_atrophy(results: list[dict], pipeline: str, out: Path) -> Path:
    rows = "".join(f"<tr><td>{html.escape(r['subject'])}</td><td>{_f(r['score']['injected_change_pct'],1)}%</td><td>{_f(r['score']['measured_change_pct'],1)}%</td><td>{_f(r['score']['recovery'])}</td></tr>" for r in results)
    n = len(results)
    rec = sum(r["score"]["recovery"] for r in results) / n if n else float("nan")
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>bidsgate scorecard: {html.escape(pipeline)}</title>
<style>body{{font:15px/1.4 system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#222}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #ddd;padding:.4rem .5rem;text-align:left}}th{{background:#f4f4f4}}.big{{font-size:2rem}}.note{{color:#555}}</style></head><body>
<h1>bidsgate scorecard: {html.escape(pipeline)}</h1>
<p class="note">A known brain-volume change was injected into real T1w images; the pipeline measured volumes before and after. Recovery 1.0 means it reported exactly the injected change.</p>
<div class="big">{_f(rec)}</div>mean recovery over {n} subjects
<table><tr><th>subject</th><th>injected change</th><th>measured change</th><th>recovery</th></tr>{rows}</table></body></html>"""
    out.write_text(doc)
    return out
