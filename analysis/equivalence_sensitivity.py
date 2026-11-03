"""Revision: sensitivity of the relational equivalence claims to the margin and to dependence between test years.

For the relational contrasts of Table tab:equiv (regime task) we report, per market,
  - TOST at margins 0.0025, 0.005, 0.0075 and 0.01 macro-F1,
  - the smallest margin (grid step 0.0005) at which TOST rejects non-equivalence,
  - a moving-block bootstrap (block length 3 years) 90% interval, the interval that corresponds to a 5% TOST,
    which allows for dependence between neighbouring test years (overlapping training periods),
  - the lag-1 autocorrelation of the yearly differences.
Output: analysis/out/equiv_sensitivity.csv and a printed summary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from equivalence import OUT, load, one_sided

RNG = np.random.default_rng(1)
MARGINS = [0.0025, 0.005, 0.0075, 0.01]
BLOCK = 3
RELATIONAL = [
    ("CorrVote over identity", "base", "MART-CorrVote", "base", "MART-Temporal+AE"),
    ("Cross over identity", "base", "MART-Cross", "base", "MART-Temporal+AE"),
    ("Real vs permuted P", "base", "MART-CorrVote", "base", "MART-CorrVote (permuted P)"),
    ("Real vs uniform P", "base", "MART-CorrVote", "base", "MART-CorrVote (uniform P)"),
    ("5 seeds: CorrVote over identity", "seeds5", "MART-CorrVote", "seeds5", "MART-Temporal+AE"),
    ("5 seeds: Cross over identity", "seeds5", "MART-Cross", "seeds5", "MART-Temporal+AE"),
    ("Val-selected HP: CorrVote over identity", "hp_valsel", "MART-CorrVote", "hp_valsel", "MART-Temporal+AE"),
    ("Val-selected HP: Cross over identity", "hp_valsel", "MART-Cross", "hp_valsel", "MART-Temporal+AE"),
    ("MASTER inter-stock attention", "base", "MASTER", "base", "MASTER w/o inter"),
    ("MASTER inter-stock attention (+AE)", "base", "MASTER+AE", "base", "MASTER w/o inter+AE"),
    ("Top-50: CorrVote over identity", "top50", "MART-CorrVote", "top50", "MART-Temporal+AE"),
    ("Top-50: Cross over identity", "top50", "MART-Cross", "top50", "MART-Temporal+AE"),
    ("Top-50: real vs permuted P", "top50", "MART-CorrVote", "top50", "MART-CorrVote (permuted P)"),
]


def p_tost(x: np.ndarray, delta: float) -> float:
    return max(one_sided(x + delta, "greater"), one_sided(x - delta, "less"))


def block_boot(x: np.ndarray, n_boot: int = 10_000) -> np.ndarray:
    n = len(x)
    starts = RNG.integers(0, n - BLOCK + 1, size=(n_boot, int(np.ceil(n / BLOCK))))
    idx = (starts[:, :, None] + np.arange(BLOCK)).reshape(n_boot, -1)[:, :n]
    return x[idx].mean(axis=1)


def main() -> None:
    df = load()
    rows = []
    for label, pa, a, pb, b in RELATIONAL:
        for market in sorted(df.market.unique()):
            xa = df[(df.protocol == pa) & (df.market == market) & (df.model == a)]
            xb = df[(df.protocol == pb) & (df.market == market) & (df.model == b)]
            seeds = sorted(set(xa.seed) & set(xb.seed))
            ya = xa[xa.seed.isin(seeds)].groupby("year").macro_f1.mean()
            yb = xb[xb.seed.isin(seeds)].groupby("year").macro_f1.mean()
            d = (ya - yb).dropna().sort_index().to_numpy()
            if len(d) < 6:
                continue
            grid = np.arange(0.0005, 0.0305, 0.0005)
            min_margin = next((g for g in grid if p_tost(d, g) < 0.05), np.nan)
            bb = block_boot(d)
            row = dict(contrast=label, market=market, n_years=len(d), mean_diff=d.mean(), min_margin=min_margin,
                       bb90_lo=np.percentile(bb, 5), bb90_hi=np.percentile(bb, 95),
                       ac1=pd.Series(d).autocorr(1))
            for m in MARGINS:
                row[f"eq_{m}"] = p_tost(d, m) < 0.05
            rows.append(row)
    res = pd.DataFrame(rows)
    res["bb_within_005"] = (res.bb90_lo > -0.005) & (res.bb90_hi < 0.005)
    res.to_csv(OUT / "equiv_sensitivity.csv", index=False)
    pd.set_option("display.width", 220)
    print(res.round(4).to_string(index=False))
    print("\ncells:", len(res))
    for m in MARGINS:
        print(f"equivalent at +/-{m}: {int(res[f'eq_{m}'].sum())}")
    print("block-bootstrap 90% CI inside +/-0.005:", int(res.bb_within_005.sum()))
    print("max |bb90 bound|:", round(float(res[["bb90_lo", "bb90_hi"]].abs().max().max()), 4))
    print("median / max min_margin:", res.min_margin.median(), res.min_margin.max())
    print("mean lag-1 autocorrelation of yearly differences:", round(float(res.ac1.mean()), 3))


if __name__ == "__main__":
    main()
