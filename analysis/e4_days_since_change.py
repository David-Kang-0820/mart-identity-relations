"""Paper 2 E4 figure: accuracy by how long the current regime has lasted.

k = number of consecutive days, up to and including t, on which the current regime y(t-1) has held. The first
run of each asset in each test year is dropped because its start is unknown. Persistence accuracy at k equals
the probability that the regime survives one more day.

Outputs: analysis/out/e4_days_since_change.csv, paper/figures/days_since_change.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from e4_change_calibration import FIRST, OUT, SEEDS, add_prev_label, load_base, load_mart  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "paper" / "figures"
MODELS = [("temporal_only", "MART-Temporal"), ("temporal_asset_emb", "MART-Temporal+AE"),
          ("temporal_graph_feature_corr_classwise_r3", "MART-CorrVote"), ("tcn", "TCN")]
BINS = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 20), (21, 10_000)]
LABELS = ["1", "2", "3", "4–5", "6–10", "11–20", ">20"]
# reference categorical palette, slots 1-4 in fixed order; persistence is neutral ink
STYLE = {"Persistence": ("#52514e", "--", "s"), "MART-Temporal": ("#2a78d6", "-", "o"),
         "MART-Temporal+AE": ("#eb6834", "-", "^"), "MART-CorrVote": ("#1baf7a", "-", "D"), "TCN": ("#eda100", "-", "v")}


def run_length(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["code", "date"]).copy()
    run = (df.y_prev != df.groupby("code").y_prev.shift()).groupby(df.code).cumsum()
    df["k"] = df.groupby([df.code, run]).cumcount() + 1
    return df[run > 1]


def bin_of(k: pd.Series) -> pd.Series:
    out = pd.Series(index=k.index, dtype=object)
    for (lo, hi), lab in zip(BINS, LABELS):
        out[(k >= lo) & (k <= hi)] = lab
    return out


def main() -> None:
    rows = []
    for market, first in FIRST.items():
        for seed in SEEDS:
            for year in range(first, 2026):
                for key, name in MODELS:
                    df = load_base(market, seed, key, year) if key == "tcn" else load_mart(market, seed, key, year)
                    if df is None:
                        continue
                    df = run_length(add_prev_label(df))
                    df["bin"] = bin_of(df.k)
                    df["correct"] = df.pred == df.y_true
                    df["persist"] = df.y_prev == df.y_true
                    g = df.groupby("bin").agg(n=("correct", "size"), acc=("correct", "mean"), persist=("persist", "mean"))
                    rows += [dict(market=market, seed=seed, year=year, model=name, bin=b, **r) for b, r in g.iterrows()]
                    if key == "tcn":
                        rows += [dict(market=market, seed=seed, year=year, model="Persistence", bin=b, n=r.n, acc=r.persist,
                                      persist=r.persist) for b, r in g.iterrows()]
    cells = pd.DataFrame(rows)
    # pool by sample count over markets, seeds and years
    cells["hits"] = cells.acc * cells.n
    agg = cells.groupby(["model", "bin"]).agg(hits=("hits", "sum"), n=("n", "sum")).reset_index()
    agg["acc"] = agg.hits / agg.n
    agg.to_csv(OUT / "e4_days_since_change.csv", index=False)
    tab = agg.pivot(index="bin", columns="model", values="acc").reindex(LABELS)
    share = agg[agg.model == "TCN"].set_index("bin").n.reindex(LABELS)
    print(tab.round(3).to_string())
    print("share of days per bin:", (share / share.sum()).round(3).to_dict())

    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": "#52514e",
                         "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = range(len(LABELS))
    for name, (color, ls, mk) in STYLE.items():
        ax.plot(x, tab[name], color=color, ls=ls, marker=mk, ms=4, lw=1.5, label=name)
    ax.set_xticks(list(x), LABELS)
    ax.set_xlabel("Days the current regime has lasted")
    ax.set_ylabel("Next-day accuracy")
    ax.grid(axis="y", color="#e4e3df", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "days_since_change.pdf")
    fig.savefig(OUT / "days_since_change.png", dpi=200)


if __name__ == "__main__":
    main()
