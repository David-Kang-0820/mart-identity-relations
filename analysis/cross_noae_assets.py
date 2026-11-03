"""Revision: what does identity-free cross attention gain on the regime task, and for which assets?

For every asset-year, the seed-averaged accuracy gain of MART-Cross w/o AE over MART-Temporal (both without an
asset embedding) is related to the same training-period characteristics as in residual_identity.py. If cross
attention helps by giving each asset a cross-sectional reference (a partial substitute for the per-asset
calibration that identity provides), its gain should grow with the label-location offset, like the embedding's.
Output: analysis/out/cross_noae_assets.csv, cross_noae_ols.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from residual_identity import CHARS, MARKETS, OUT, RES, SEEDS, asset_accuracy, characteristics

UNIVERSES = {"amount": "", "corr": "_ucorr", "leadlag": "_uleadlag"}


def gain(market: str, year: int, suffix: str, a: str, b: str) -> np.ndarray | None:
    diffs = []
    for seed in SEEDS:
        base = RES / f"top20_amount_ablation_{market}_19_1{suffix}_seed{seed}"
        f = {v: base / v / "projected_d32" / "epoch_40" / f"test_{year}" / "test_predictions.csv" for v in (a, b)}
        if all(p.exists() for p in f.values()):
            diffs.append(asset_accuracy(f[a]) - asset_accuracy(f[b]))
    return np.mean(diffs, axis=0) if len(diffs) == len(SEEDS) else None


def main() -> None:
    rows = []
    for universe, suffix in UNIVERSES.items():
        for market, years in MARKETS.items():
            for year in years:
                ch = characteristics(market, year, suffix)
                g_x = gain(market, year, suffix, "temporal_cross_asset_no_ae", "temporal_only")
                g_ae = gain(market, year, suffix, "temporal_asset_emb", "temporal_only")
                if ch is None or g_x is None or g_ae is None or len(g_x) != len(ch):
                    continue
                rows.append(ch.assign(universe=universe, market=market, year=year, gain_cross=g_x, gain_ae=g_ae))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(OUT / "cross_noae_assets.csv", index=False)
    for c in CHARS:
        ref = df[df.universe == "amount"].groupby("market")[c].agg(["mean", "std"])
        df[f"z_{c}"] = (df[c] - df.market.map(ref["mean"])) / df.market.map(ref["std"])
    out = []
    for target in ("gain_cross", "gain_ae"):
        formula = f"{target} ~ " + " + ".join(f"z_{c}" for c in CHARS)
        for (universe, market), sub in df.groupby(["universe", "market"]):
            fit = smf.ols(formula, data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub.year})
            for c in CHARS:
                out.append(dict(target=target, universe=universe, market=market, term=c, coef=fit.params[f"z_{c}"],
                                p=fit.pvalues[f"z_{c}"], r2=fit.rsquared, mean=sub[target].mean(), n=len(sub)))
    ols = pd.DataFrame(out)
    ols.to_csv(OUT / "cross_noae_ols.csv", index=False)
    pd.set_option("display.width", 200)
    print(ols[ols.term == "loc"].round(4).to_string(index=False))
    df["loc_q"] = df.groupby(["universe", "market"]).loc.transform(lambda x: pd.qcut(x, 4, labels=False))
    print("\nmean gain of Cross w/o AE over Temporal by loc quartile:\n",
          df.groupby(["universe", "market", "loc_q"]).gain_cross.mean().unstack().round(4))
    # does the cross gain track the embedding gain asset by asset?
    print("\nasset-level correlation of the two gains:\n",
          df.groupby(["universe", "market"])[["gain_cross", "gain_ae"]].corr().iloc[0::2, 1].round(3))


if __name__ == "__main__":
    main()
