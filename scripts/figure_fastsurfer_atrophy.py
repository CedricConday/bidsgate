"""Light and dark SVG of scorecard #2.

    python scripts/figure_fastsurfer_atrophy.py results/fastsurfer-atrophy figures
"""

import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["svg.hashsalt"] = "bidsgate"

THEMES = (("light", "#1f2328", "#ffffff", "#d0d7de", "#0969da", "#8c959f"),
          ("dark", "#e6edf3", "#0d1117", "#30363d", "#4493f8", "#6e7681"))


def main(out: str, figdir: str) -> None:
    df = pd.read_csv(f"{out}/pv/scores_atrophy_all.tsv", sep="\t")
    summ = json.loads(Path(f"{out}/pv/summary_atrophy.json").read_text())
    lim = 12
    for theme, fg, bg, grid, pt, ref in THEMES:
        fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        ax.plot([0, -lim], [0, -lim], color=grid, lw=1, label="perfect recovery")
        ax.scatter(df["injected_change_pct"], df["reference_change_pct"], s=16, color=ref, marker="x",
                   label="right hippocampus (untouched)", zorder=2)
        ax.scatter(df["injected_change_pct"], df["measured_change_pct"], s=22, color=pt,
                   label="left hippocampus (contracted)", zorder=3)
        if summ["noise_floor_pct"] is not None:
            ax.axhspan(-summ["noise_floor_pct"], summ["noise_floor_pct"], color=grid, alpha=0.35, lw=0,
                       label="noise floor (2 x RMS of right hippocampus)")
        ax.set_xlim(1, -lim)
        ax.set_ylim(-lim, 2)
        ax.set_xlabel("injected volume change (%)", color=fg)
        ax.set_ylabel("FastSurfer volume change (%)", color=fg)
        ax.tick_params(colors=fg)
        for s in ax.spines.values():
            s.set_color(grid)
        ax.grid(color=grid, lw=0.5, alpha=0.6)
        leg = ax.legend(frameon=False, fontsize=8, loc="lower left")
        for t in leg.get_texts():
            t.set_color(fg)
        ax.set_title("FastSurfer 2.6.1, six healthy controls, synthetic contraction", color=fg, fontsize=10)
        fig.tight_layout()
        fig.savefig(f"{figdir}/fastsurfer-atrophy-{theme}.svg", facecolor=bg)
        plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
