"""Revision (Minor 3): one-glance summary of identity vs relational effects on both tasks.
Dots are seed-averaged mean yearly differences, bars 95% bootstrap intervals over test years, the grey band the
+/-0.005 equivalence margin. Inputs: out/equivalence.csv, out/xs_contrasts.csv. Output: paper/figures/decomposition.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(__file__).resolve().parent / "out"
FIG = Path(__file__).resolve().parents[1] / "paper" / "figures"
MARKETS = [("kospi", "KOSPI", "#2a78d6", "o"), ("kosdaq", "KOSDAQ", "#eb6834", "s"), ("nasdaq", "Nasdaq", "#1baf7a", "^")]
ROWS = ["Asset embedding\n(Temporal+AE $-$ Temporal)", "CorrVote $-$ Temporal+AE", "Cross $-$ Temporal+AE",
        "Real $-$ permuted $P$", "MASTER inter-stock\nattention"]
REGIME = ["Identity: Temporal+AE - Temporal", "CorrVote over identity", "Cross over identity", "Real vs permuted P",
          "MASTER inter-stock attention"]
RANK = ["MART-Temporal+AE - MART-Temporal", "MART-CorrVote - MART-Temporal+AE", "MART-Cross - MART-Temporal+AE",
        "MART-CorrVote - MART-CorrVote (permuted P)", "MASTER - MASTER w/o inter"]


def panel(ax, df, keys, key_col, title, xlabel):
    ax.axvspan(-0.005, 0.005, color="#e6e5e1", zorder=0, lw=0)
    ax.axvline(0, color="#8a8984", lw=0.8, zorder=1)
    for k, (m, name, color, marker) in enumerate(MARKETS):
        for r, key in enumerate(keys):
            x = df[(df.market == m) & (df[key_col] == key)].iloc[0]
            y = r + (k - 1) * 0.22
            ax.plot([x.ci_lo, x.ci_hi], [y, y], color=color, lw=1.6, zorder=2, solid_capstyle="round")
            ax.plot(x.mean_diff, y, marker=marker, ms=5.5, color=color, mec="white", mew=0.8, zorder=3,
                    label=name if r == 0 else None)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels(ROWS, fontsize=7.5)
    ax.set_title(title, fontsize=8.5, loc="left")
    ax.set_xlabel(xlabel, fontsize=7.5)
    ax.tick_params(axis="x", labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#ecebe7", lw=0.6, zorder=0)


def main() -> None:
    eq = pd.read_csv(OUT / "equivalence.csv")
    xs = pd.read_csv(OUT / "xs_contrasts.csv")
    xs = xs[xs.horizon == 1]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), sharey=True)
    panel(axes[0], eq, REGIME, "contrast", "(a) Regime task, per-asset labels", "Difference in macro-F1")
    panel(axes[1], xs, RANK, "contrast", "(b) Ranking task, cross-sectional labels", "Difference in daily RankIC")
    axes[0].invert_yaxis()  # shared y: invert once
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=7, frameon=False, ncol=3, loc="upper center")
    fig.tight_layout(w_pad=1.5, rect=(0, 0, 1, 0.92))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "decomposition.pdf")
    fig.savefig(OUT / "decomposition.png", dpi=160)


if __name__ == "__main__":
    main()
