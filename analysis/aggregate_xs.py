"""Paper 2 standard task: cross-sectional return ranking on the top-50 universe (S1).

Target: the H-day forward return (H = 1, 5), rank-normalized across assets per date; MSE loss; evaluated by the
daily RankIC (Spearman), IC (Pearson, returns clipped at +/-50%), their information ratios and the mean
top-minus-bottom-quintile forward return. Model-free references from the same summaries: 1-day reversal,
20-day momentum, and each asset's mean training-period target (identity only).

Outputs (analysis/out/): xs_yearly.csv, xs_table.csv, xs_contrasts.csv (Wilcoxon over test years after seed
averaging, Holm within market x horizon), xs_equivalence.csv (TOST, margin DELTA on RankIC).
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from aggregate_e1_e2 import HOLDOUT, MART_NAMES, RES, SEEDS, holm

OUT = RES.parents[2] / "analysis" / "out"
NAMES = {**MART_NAMES, "temporal_asset_bias": "MART-Temporal+asset bias", "master": "MASTER",
         "master_asset_emb": "MASTER+AE", "master_no_inter": "MASTER w/o inter",
         "temporal_cross_asset_no_ae": "MART-Cross w/o AE", "temporal_ctxmean": "MART-PanelMean w/o AE"}
FACTORS = {"reversal_1d": "1-day reversal", "momentum_20d": "20-day momentum", "asset_mean_train": "Asset mean (train)"}
METRICS = ["rank_ic", "ic", "rank_icir", "icir", "ls_spread"]
DELTA = 0.005
RNG = np.random.default_rng(0)


def collect() -> pd.DataFrame:
    rows = []
    for d in RES.glob("top20_amount_ablation_*_labxsret_top50_seed*"):
        m = re.fullmatch(r"top20_amount_ablation_(kospi|kosdaq|nasdaq)_(\d+)_(\d+)_labxsret_top50_seed(\d+)", d.name)
        if not m or int(m[4]) not in SEEDS:
            continue
        market, horizon, seed = m[1], int(m[3]), int(m[4])
        for f in d.glob("*/projected_d32/epoch_40/test_*/summary.json"):
            variant = f.parts[-5]
            if variant not in NAMES:
                continue
            s = json.loads(f.read_text(encoding="utf-8"))
            base = dict(market=market, horizon=horizon, seed=seed, year=int(s["test_year"]))
            rows.append(dict(base, model=NAMES[variant], **{k: s["test_metrics"][k] for k in METRICS}))
            if variant == "temporal_only":
                for key, name in FACTORS.items():
                    rows.append(dict(base, model=name, **{k: s["factor_baselines"][key][k] for k in METRICS}))
    return pd.DataFrame(rows)


CONTRASTS = [
    ("Identity", "MART-Temporal+AE", "MART-Temporal"),
    ("Identity", "MART-Temporal+asset bias", "MART-Temporal"),
    ("Identity", "MASTER+AE", "MASTER"),
    ("Relation", "MART-CorrVote", "MART-Temporal+AE"),
    ("Relation", "MART-Cross", "MART-Temporal+AE"),
    ("Relation", "MART-CorrVote", "MART-CorrVote (permuted P)"),
    ("Relation", "MASTER", "MASTER w/o inter"),
    ("Reference", "MART-Temporal", "1-day reversal"),
    ("Reference", "MART-Temporal+AE", "1-day reversal"),
    ("Reference", "MASTER", "MART-Temporal"),
]


def one_sided(x: np.ndarray, alternative: str) -> float:
    return wilcoxon(x, alternative=alternative).pvalue if (x != 0).any() else 1.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = collect()
    df.to_csv(OUT / "xs_yearly.csv", index=False)
    dev = df[df.year < HOLDOUT]
    pd.set_option("display.width", 220)
    print("cells (years x seeds):\n", dev.groupby(["horizon", "model", "market"]).size().unstack("market").to_string())
    seed_mean = dev.groupby(["horizon", "market", "model", "seed"])[METRICS].mean()
    table = seed_mean.groupby(["horizon", "market", "model"]).agg(["mean", "std"])
    table.columns = [f"{a}_{b}" for a, b in table.columns]
    table.reset_index().to_csv(OUT / "xs_table.csv", index=False)
    for metric in ("rank_ic", "rank_icir", "ls_spread"):
        print(f"\nMEAN {metric}\n", table[f"{metric}_mean"].unstack("market").round(4).to_string())

    ym = dev.groupby(["horizon", "market", "model", "year"])[METRICS].mean()
    con = contrast_rows(dev, ym, CONTRASTS)
    con.to_csv(OUT / "xs_contrasts.csv", index=False)
    print(f"\nRANKIC CONTRASTS (seed-averaged years; Wilcoxon, Holm within market x horizon; TOST margin {DELTA})")
    print(con.round(4).to_string(index=False))
    # Revision check, kept out of the main Holm family so that Table xs_contrasts is unchanged.
    noae = contrast_rows(dev, ym, NOAE_CONTRASTS)
    if len(noae):
        noae.to_csv(OUT / "xs_noae.csv", index=False)
        print("\nCROSS WITHOUT EMBEDDING (Holm within market x horizon over these contrasts)")
        print(noae.round(4).to_string(index=False))
    ctx = contrast_rows(dev, ym, [("Context", "MART-PanelMean w/o AE", "MART-Temporal"),
                                  ("Relation", "MART-Cross w/o AE", "MART-PanelMean w/o AE")])
    if len(ctx):
        ctx.to_csv(OUT / "xs_ctx.csv", index=False)
        print("\nPANELMEAN WITHOUT EMBEDDING\n", ctx.round(4).to_string(index=False))


NOAE_CONTRASTS = [
    ("Relation", "MART-Cross w/o AE", "MART-Temporal"),
    ("Identity", "MART-Cross", "MART-Cross w/o AE"),
]


def contrast_rows(dev: pd.DataFrame, ym: pd.DataFrame, contrasts) -> pd.DataFrame:
    rows = []
    for (h, market), _ in dev.groupby(["horizon", "market"]):
        for kind, a, b in contrasts:
            try:
                diff = (ym.loc[(h, market, a), "rank_ic"] - ym.loc[(h, market, b), "rank_ic"]).dropna()
            except KeyError:
                continue
            if len(diff) < 6:
                continue
            x = diff.to_numpy()
            boot = RNG.choice(x, size=(10_000, len(x)), replace=True).mean(axis=1)
            rows.append(dict(horizon=h, market=market, kind=kind, contrast=f"{a} - {b}", mean_diff=x.mean(),
                             wins=int((x > 0).sum()), n=len(x), p=one_sided(x, "two-sided"),
                             ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5),
                             p_tost=max(one_sided(x + DELTA, "greater"), one_sided(x - DELTA, "less"))))
    con = pd.DataFrame(rows)
    if con.empty:
        return con
    con["p_holm"] = con.groupby(["horizon", "market"]).p.transform(holm)
    con["equivalent"] = con.p_tost < 0.05
    return con


if __name__ == "__main__":
    main()
