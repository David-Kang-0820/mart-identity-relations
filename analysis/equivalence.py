"""Paper 2 Q5: are the null effects small, not just non-significant?

For each contrast, differences are taken per test year after averaging the seeds both models share. We report
  - the mean difference with a 95% percentile bootstrap CI over test years (10,000 resamples),
  - a two one-sided tests (TOST) equivalence p-value with margin DELTA, using one-sided Wilcoxon signed-rank tests
    of (diff + DELTA) > 0 and (diff - DELTA) < 0; p_tost = max of the two. p_tost < 0.05 means the effect lies
    within +/- DELTA.
DELTA = 0.005 macro-F1, about a fifth of the identity effect.

Inputs: analysis/out/yearly_long.csv (main), followup_yearly.csv (X1-X4), r_yearly.csv (R1-R5), qyearly.csv (Q1-Q4,
optional). Output: analysis/out/equivalence.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

OUT = Path(__file__).resolve().parent / "out"
DELTA = 0.005
RNG = np.random.default_rng(0)

# (label, protocol A, model A, protocol B, model B)
CONTRASTS = [
    ("Identity: Temporal+AE - Temporal", "base", "MART-Temporal+AE", "base", "MART-Temporal"),
    ("CorrVote over identity", "base", "MART-CorrVote", "base", "MART-Temporal+AE"),
    ("Cross over identity", "base", "MART-Cross", "base", "MART-Temporal+AE"),
    ("Real vs permuted P", "base", "MART-CorrVote", "base", "MART-CorrVote (permuted P)"),
    ("Real vs uniform P", "base", "MART-CorrVote", "base", "MART-CorrVote (uniform P)"),
    ("CorrVote w/o AE - Temporal", "base", "MART-CorrVote w/o AE", "base", "MART-Temporal"),
    ("Temporal+AE - TCN", "base", "MART-Temporal+AE", "base", "TCN (eq. budget)"),
    ("MASTER inter-stock attention", "base", "MASTER", "base", "MASTER w/o inter"),
    ("MASTER inter-stock attention (+AE)", "base", "MASTER+AE", "base", "MASTER w/o inter+AE"),
    ("MASTER: identity", "base", "MASTER+AE", "base", "MASTER"),
    ("Top-50: CorrVote over identity", "top50", "MART-CorrVote", "top50", "MART-Temporal+AE"),
    ("Top-50: Cross over identity", "top50", "MART-Cross", "top50", "MART-Temporal+AE"),
    ("Top-50: real vs permuted P", "top50", "MART-CorrVote", "top50", "MART-CorrVote (permuted P)"),
    ("X1: Temporal (per-asset std) - Temporal+AE (orig.)", "stdasset", "MART-Temporal", "base", "MART-Temporal+AE"),
    ("Asset bias - Temporal+AE", "base", "MART-Temporal+asset bias", "base", "MART-Temporal+AE"),
    ("Permuted identity - Temporal", "base", "MART-Temporal+AE (permuted)", "base", "MART-Temporal"),
    ("5 seeds: CorrVote over identity", "seeds5", "MART-CorrVote", "seeds5", "MART-Temporal+AE"),
    ("5 seeds: Cross over identity", "seeds5", "MART-Cross", "seeds5", "MART-Temporal+AE"),
    ("5 seeds: real vs permuted P", "seeds5", "MART-CorrVote", "seeds5", "MART-CorrVote (permuted P)"),
    ("Val-selected HP: CorrVote over identity", "hp_valsel", "MART-CorrVote", "hp_valsel", "MART-Temporal+AE"),
    ("Val-selected HP: Cross over identity", "hp_valsel", "MART-Cross", "hp_valsel", "MART-Temporal+AE"),
    ("Unseen assets: CorrVote - Temporal", "xfer", "MART-CorrVote (unseen)", "xfer", "MART-Temporal (unseen)"),
    ("Unseen assets: Cross - Temporal", "xfer", "MART-Cross (unseen)", "xfer", "MART-Temporal (unseen)"),
]


def load() -> pd.DataFrame:
    parts = []
    main = pd.read_csv(OUT / "yearly_long.csv").assign(protocol="base")
    parts.append(main[["protocol", "market", "model", "seed", "year", "macro_f1"]])
    for name in ("followup_yearly.csv", "r_yearly.csv", "q_yearly.csv"):
        f = OUT / name
        if f.exists():
            parts.append(pd.read_csv(f)[["protocol", "market", "model", "seed", "year", "macro_f1"]])
    df = pd.concat(parts).drop_duplicates(["protocol", "market", "model", "seed", "year"])
    return df[df.year < 2026]


def one_sided(x: np.ndarray, alternative: str) -> float:
    return wilcoxon(x, alternative=alternative).pvalue if (x != 0).any() else 1.0


def main() -> None:
    df = load()
    rows = []
    for label, pa, a, pb, b in CONTRASTS:
        for market in sorted(df.market.unique()):
            xa = df[(df.protocol == pa) & (df.market == market) & (df.model == a)]
            xb = df[(df.protocol == pb) & (df.market == market) & (df.model == b)]
            seeds = sorted(set(xa.seed) & set(xb.seed))
            if not seeds:
                continue
            ya = xa[xa.seed.isin(seeds)].groupby("year").macro_f1.mean()
            yb = xb[xb.seed.isin(seeds)].groupby("year").macro_f1.mean()
            diff = (ya - yb).dropna().to_numpy()
            if len(diff) < 6:
                continue
            boot = RNG.choice(diff, size=(10_000, len(diff)), replace=True).mean(axis=1)
            p_tost = max(one_sided(diff + DELTA, "greater"), one_sided(diff - DELTA, "less"))
            rows.append(dict(contrast=label, market=market, seeds=len(seeds), n_years=len(diff), mean_diff=diff.mean(),
                             ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5), p_tost=p_tost,
                             equivalent=p_tost < 0.05))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "equivalence.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"EQUIVALENCE (margin +/-{DELTA}); CI = 95% bootstrap over test years")
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
