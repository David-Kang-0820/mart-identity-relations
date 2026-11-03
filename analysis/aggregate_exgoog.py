"""Paper 2 E6 sensitivity: Nasdaq 2020-2025 with and without the duplicate share class GOOG.

Compares the key contrasts on the same six test years between the original universe (GOOG and GOOGL) and the
universe with GOOG removed before selection (the 21st stock enters). TCN/DLinear use the original 16-epoch
budget in both universes. With six years the Wilcoxon test cannot go below p=0.031, so mean differences and
win counts are the main read-out.

Output: analysis/out/exgoog_contrasts.csv
"""
from __future__ import annotations

import json
import re

import pandas as pd
from scipy.stats import wilcoxon

from aggregate_e1_e2 import BASE_NAMES, MART_NAMES, RES, SEEDS

OUT = RES.parents[2] / "analysis" / "out"
YEARS = range(2020, 2026)
CONTRASTS = [
    ("MART-Temporal+AE", "MART-Temporal"), ("MART-CorrVote", "MART-Temporal+AE"), ("MART-Cross", "MART-Temporal+AE"),
    ("MART-CorrVote", "MART-CorrVote w/o AE"), ("MART-CorrVote", "MART-CorrVote (permuted P)"),
    ("MART-CorrVote", "MART-CorrVote (uniform P)"), ("MART-Temporal+AE", "TCN"), ("MART-Temporal+AE", "DLinear"),
]


def collect() -> pd.DataFrame:
    rows = []
    for d in RES.glob("top20_amount_ablation_nasdaq_19_1*_seed*"):
        m = re.fullmatch(r"top20_amount_ablation_nasdaq_19_1(_exGOOG)?_seed(\d+)", d.name)
        if not m or int(m[2]) not in SEEDS:
            continue
        for f in d.glob("*/projected_d32/epoch_40/test_*/summary.json"):
            if f.parts[-5] in MART_NAMES:
                s = json.loads(f.read_text(encoding="utf-8"))
                rows.append(dict(universe="exGOOG" if m[1] else "original", seed=int(m[2]), year=int(s["test_year"]),
                                 model=MART_NAMES[f.parts[-5]], macro_f1=s["test_metrics"]["overall_macro_f1"]))
    for d in RES.glob("figure4_baseline_comparison_nasdaq_19_1*_seed*"):
        m = re.fullmatch(r"figure4_baseline_comparison_nasdaq_19_1(_exGOOG)?_seed(\d+)", d.name)
        if not m or int(m[2]) not in SEEDS:
            continue
        for f in d.glob("baselines/test_*/*/*_summary.json"):
            if f.parent.name in ("persistence", "tcn", "dlinear"):
                s = json.loads(f.read_text(encoding="utf-8"))
                rows.append(dict(universe="exGOOG" if m[1] else "original", seed=int(m[2]),
                                 year=int(f.parts[-3].split("_")[1]), model=BASE_NAMES[f.parent.name],
                                 macro_f1=s["test_metrics"]["macro_f1"]))
    df = pd.DataFrame(rows)
    return df[df.year.isin(YEARS)]


def main() -> None:
    df = collect()
    print("cells (years x seeds):\n", df.groupby(["model", "universe"]).size().unstack().to_string())
    lvl = df.groupby(["universe", "model", "seed"]).macro_f1.mean().groupby(["universe", "model"]).mean().unstack(0)
    print("\nMEAN MACRO-F1 2020-2025\n", lvl.assign(diff=lvl.exGOOG - lvl.original).round(4).to_string())
    ym = df.groupby(["universe", "model", "year"]).macro_f1.mean()
    rows = []
    for uni in ("original", "exGOOG"):
        for a, b in CONTRASTS:
            diff = (ym.loc[(uni, a)] - ym.loc[(uni, b)]).dropna()
            p = wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
            rows.append(dict(universe=uni, contrast=f"{a} - {b}", mean_diff=diff.mean(), wins=int((diff > 0).sum()),
                             n=len(diff), p=p))
    con = pd.DataFrame(rows)
    con.to_csv(OUT / "exgoog_contrasts.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nCONTRASTS 2020-2025 (year-level, seed-averaged; unadjusted Wilcoxon)")
    print(con.pivot(index="contrast", columns="universe", values=["mean_diff", "wins", "p"]).round(4).to_string())


if __name__ == "__main__":
    main()
