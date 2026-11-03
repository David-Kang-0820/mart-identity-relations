"""Revision (Major 3): which assets does identity help, before and after per-asset standardization (X1)?

For every asset-year we take the test accuracy of MART-Temporal+AE minus MART-Temporal (seed-averaged, seeds 42-44)
under the original protocol ("base") and under per-asset standardization ("stdasset"), and relate it to
pretrain-period characteristics of the asset, each relative to the pooled 20-asset distribution of that window:
  loc     |median_i(s) - median_pool(s)| / IQR_pool(s)   label-location offset (s = 20-day mean daily return)
  scale   |log(IQR_i(s) / IQR_pool(s))|                   label-scale offset
  vol     log(std_i(r) / std_pool(r))                     relative volatility of daily returns
  kurt    excess kurtosis of daily returns
  switch  share of pretrain days whose next-day regime differs from the current one
Regressions: gain ~ standardized characteristics + market fixed effects, OLS with standard errors clustered by
test year. Output: analysis/out/residual_identity_assets.csv, residual_identity_ols.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1] / "code" / "paper_final2"
RES, DATA = ROOT / "results", ROOT / "data"
OUT = Path(__file__).resolve().parent / "out"
SEEDS = [42, 43, 44]
MARKETS = {"kospi": range(2006, 2026), "kosdaq": range(2006, 2026), "nasdaq": range(2014, 2026)}
# protocol -> (data-cache suffix, result-directory suffix). The universe protocols replicate the analysis on the
# relation-rich universes and the random placebo universe (all drawn from the 100 most traded stocks).
PROTOCOLS = {"base": ("", ""), "stdasset": ("", "_stdper_asset"), "ucorr": ("_ucorr", "_ucorr"),
             "uleadlag": ("_uleadlag", "_uleadlag"), "urandom": ("_urandom", "_urandom")}
CHARS = ["loc", "scale", "vol", "kurt", "switch"]


def asset_accuracy(path: Path) -> np.ndarray:
    p = pd.read_csv(path)
    yt = np.array([json.loads(s) for s in p.y_true])
    yp = np.array([json.loads(s) for s in p.y_pred])
    ok = yt >= 0
    return np.where(ok, yt == yp, 0).sum(axis=0) / np.maximum(ok.sum(axis=0), 1)


def characteristics(market: str, year: int, data_suffix: str = "") -> pd.DataFrame | None:
    root = DATA / f"{market}_top20_amount{data_suffix}"
    if not (root / "selected_top20" / f"test_{year}_top20_amount.csv").exists():
        return None
    codes = pd.read_csv(root / "selected_top20" / f"test_{year}_top20_amount.csv", dtype={"code": str}).code
    codes = [c if market == "nasdaq" else c.zfill(6) for c in codes]
    frames = {}
    for c in codes:
        df = pd.read_csv(root / f"test_{year}" / "stage1" / c / "asset_tables" / f"{c}.csv", usecols=["ret1", "split"])
        frames[c] = df[df.split == "pretrain"].ret1.dropna()
    s = {c: r.rolling(20).mean().dropna() for c, r in frames.items()}
    s_pool = pd.concat(s.values())
    r_pool = pd.concat(frames.values())
    iqr = lambda x: x.quantile(0.75) - x.quantile(0.25)
    rows = []
    for k, c in enumerate(codes):
        si, ri = s[c], frames[c]
        q1, q2 = si.quantile([1 / 3, 2 / 3])
        reg = np.digitize(si.to_numpy(), [q1, q2])
        rows.append(dict(slot=k, code=c,
                         loc=abs(si.median() - s_pool.median()) / iqr(s_pool),
                         scale=abs(np.log(iqr(si) / iqr(s_pool))),
                         vol=np.log(ri.std() / r_pool.std()),
                         kurt=ri.kurt(),
                         switch=float((reg[1:] != reg[:-1]).mean())))
    return pd.DataFrame(rows)


def gains(market: str, year: int, suffix: str) -> np.ndarray | None:
    diffs = []
    for seed in SEEDS:
        base = RES / f"top20_amount_ablation_{market}_19_1{suffix}_seed{seed}"
        f = {v: base / v / "projected_d32" / "epoch_40" / f"test_{year}" / "test_predictions.csv"
             for v in ("temporal_asset_emb", "temporal_only")}
        if not all(p.exists() for p in f.values()):
            continue
        diffs.append(asset_accuracy(f["temporal_asset_emb"]) - asset_accuracy(f["temporal_only"]))
    return np.mean(diffs, axis=0) if diffs else None


def main() -> None:
    rows = []
    for market, years in MARKETS.items():
        for year in years:
            for proto, (data_suffix, res_suffix) in PROTOCOLS.items():
                ch = characteristics(market, year, data_suffix)
                g = gains(market, year, res_suffix)
                if ch is None or g is None or len(g) != len(ch):
                    continue
                rows.append(ch.assign(market=market, year=year, protocol=proto, gain=g))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(OUT / "residual_identity_assets.csv", index=False)
    present = [p for p in PROTOCOLS if (df.protocol == p).any()]
    # standardize within market on the base protocol, so coefficients are per base-protocol standard deviation
    # and comparable across protocols
    for c in CHARS:
        ref = df[df.protocol == "base"].groupby("market")[c].agg(["mean", "std"])
        df[f"z_{c}"] = (df[c] - df.market.map(ref["mean"])) / df.market.map(ref["std"])
    out = []
    formula = "gain ~ " + " + ".join(f"z_{c}" for c in CHARS) + " + C(market)"
    for proto in present:
        for scope, sub in [("all", df[df.protocol == proto])] + [
                (m, df[(df.protocol == proto) & (df.market == m)]) for m in MARKETS]:
            f = formula if scope == "all" else formula.replace(" + C(market)", "")
            fit = smf.ols(f, data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub.year.astype(str) + sub.market})
            for c in CHARS:
                out.append(dict(protocol=proto, scope=scope, term=c, coef=fit.params[f"z_{c}"],
                                se=fit.bse[f"z_{c}"], p=fit.pvalues[f"z_{c}"], r2=fit.rsquared,
                                mean_gain=sub.gain.mean(), n=len(sub)))
    ols = pd.DataFrame(out)
    ols.to_csv(OUT / "residual_identity_ols.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.groupby(["protocol", "market"]).gain.describe()[["count", "mean", "50%"]].round(4))
    print(ols.round(4).to_string(index=False))
    # share of the gain carried by the assets with the largest label-location offset (top quartile)
    for proto in present:
        sub = df[df.protocol == proto].copy()
        sub["loc_q"] = sub.groupby("market").loc.transform(lambda x: pd.qcut(x, 4, labels=False))
        print(proto, "mean gain by loc quartile:\n", sub.groupby(["market", "loc_q"]).gain.mean().unstack().round(4))

    # Out-of-sample check of the account: fit gain ~ loc + scale (+ market) on the base protocol only, predict the
    # mean identity gain of every universe-window from its assets' characteristics, compare with the observed one.
    fit = smf.ols("gain ~ loc + scale + C(market)", data=df[df.protocol == "base"]).fit()
    df["pred"] = fit.predict(df)
    win = df.groupby(["protocol", "market", "year"])[["gain", "pred", "loc"]].mean().reset_index()
    win.to_csv(OUT / "residual_identity_windows.csv", index=False)
    summ = win.groupby(["protocol", "market"]).agg(obs=("gain", "mean"), pred=("pred", "mean"), loc=("loc", "mean"),
                                                  n=("year", "size")).reset_index()
    summ.to_csv(OUT / "residual_identity_universe_pred.csv", index=False)
    print("\nbase-protocol fit:", fit.params.round(4).to_dict())
    print("universe-level identity gain, observed vs predicted from loc/scale:\n", summ.round(4).to_string(index=False))
    oos = win[win.protocol.str.startswith("u")]
    if len(oos):
        r = np.corrcoef(oos.gain, oos.pred)[0, 1]
        print(f"window-level correlation (universe protocols, n={len(oos)}): {r:.3f}; "
              f"mean abs error {np.abs(oos.gain - oos.pred).mean():.4f}")


if __name__ == "__main__":
    main()
