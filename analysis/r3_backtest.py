"""Paper 2 R3: economic value of the (19,1) regime forecasts.

Each test day t a model's predicted regime for every asset sets a position held over the next day's
close-to-close return r_{t+1} (features use data through the close of t):
  long-short  +1/N for predicted regime 2 (up), -1/N for 0 (down), 0 otherwise (N = assets with a return)
  long-only   equal weight over assets predicted 2, cash otherwise
Net returns subtract COST per unit of one-way turnover. Benchmarks: equal-weight buy-and-hold of the universe and
persistence (the current regime, i.e. a 20-day mean-return momentum rule against each asset's own tertiles).
Metrics are computed per seed over all test years of a market and then averaged over seeds; yearly paired tests
use seed-averaged annual net Sharpe ratios.

Outputs: analysis/out/r3_backtest_summary.csv, r3_backtest_yearly.csv, r3_backtest_contrasts.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import sys

from e4_change_calibration import FIRST, OUT, SEEDS, load_base, load_mart

DATA = Path(__file__).resolve().parents[1] / "code" / "paper_final2" / "data"
COST = 0.001  # 10 bp per unit of one-way turnover
MODELS = [("temporal_only", "MART-Temporal"), ("temporal_asset_emb", "MART-Temporal+AE"),
          ("temporal_cross_asset", "MART-Cross"), ("temporal_graph_feature_corr_classwise_r3", "MART-CorrVote"),
          ("tcn", "TCN"), ("dlinear", "DLinear")]
CONTRASTS = [("MART-Temporal+AE", "MART-Temporal"), ("MART-CorrVote", "MART-Temporal+AE"),
             ("MART-Cross", "MART-Temporal+AE"), ("MART-Temporal+AE", "TCN"), ("MART-Temporal+AE", "Persistence"),
             ("TCN", "Persistence"), ("Persistence", "Buy and hold")]


def next_returns(market: str, year: int, codes: list[str]) -> pd.DataFrame:
    """r_{t+1} indexed by (date t, code), only within the test year."""
    rows = []
    for code in codes:
        f = DATA / f"{market}_top20_amount" / f"test_{year}" / "stage1" / code / "asset_tables" / f"{code}.csv"
        df = pd.read_csv(f, usecols=["date", "close", "split"])
        df = df[df.split == "test"].copy()
        df["date"] = pd.to_datetime(df.date).dt.strftime("%Y-%m-%d")
        df["r_next"] = df.close.shift(-1) / df.close - 1.0
        rows.append(df.assign(code=code)[["date", "code", "r_next"]])
    return pd.concat(rows).dropna()


def daily_portfolio(pred: pd.DataFrame, r: pd.DataFrame) -> pd.DataFrame:
    """Daily gross/net returns of the long-short and long-only books from a (date, code, pred) frame."""
    d = pred.merge(r, on=["date", "code"], how="inner")
    d["ls"] = np.select([d.pred == 2, d.pred == 0], [1.0, -1.0], 0.0)
    d["lo"] = (d.pred == 2).astype(float)
    n = d.groupby("date").code.transform("size")
    d["w_ls"] = d.ls / n
    d["w_lo"] = d.lo / d.groupby("date").lo.transform("sum").replace(0, np.nan)
    d["w_lo"] = d.w_lo.fillna(0.0)
    w = d.pivot(index="date", columns="code", values=["w_ls", "w_lo"]).fillna(0.0).sort_index()
    rr = d.pivot(index="date", columns="code", values="r_next").reindex(w.index).fillna(0.0)
    out = pd.DataFrame(index=w.index)
    for book in ("ls", "lo"):
        wb = w[f"w_{book}"]
        gross = (wb * rr[wb.columns]).sum(axis=1)
        turnover = wb.diff().abs().sum(axis=1)
        turnover.iloc[0] = wb.iloc[0].abs().sum()
        out[f"{book}_gross"] = gross
        out[f"{book}_net"] = gross - COST * turnover
        out[f"{book}_turnover"] = turnover
    return out


def stats(x: pd.Series) -> dict:
    ann = x.mean() * 252
    vol = x.std() * np.sqrt(252)
    eq = (1 + x).cumprod()
    return dict(ann_return=ann, ann_vol=vol, sharpe=ann / vol if vol > 0 else np.nan, max_dd=(eq / eq.cummax() - 1).min())


def main() -> None:
    # TASK "19_1" (default) or e.g. "0_20": MART roots *_<TASK>_seed*, baselines at the original budget for other tasks
    task = sys.argv[1] if len(sys.argv) > 1 else "19_1"
    horizon = int(task.split("_")[1])
    base_root = "19_1_eq40" if task == "19_1" else task
    tag = "" if task == "19_1" else f"_{task}"
    daily = []
    for market, first in FIRST.items():
        for seed in SEEDS:
            for year in range(first, 2026):
                preds = {}
                for key, name in MODELS:
                    df = (load_base(market, seed, key, year, base_root) if key in ("tcn", "dlinear")
                          else load_mart(market, seed, key, year, task))
                    if df is not None:
                        preds[name] = df
                if "MART-Temporal" not in preds:
                    continue
                # persistence: the regime observable at t, i.e. the label of t-H (y(t-H) is the regime at t)
                ref = preds["MART-Temporal"][lambda d: d.y_true >= 0].sort_values(["code", "date"]).copy()
                ref["pred"] = ref.groupby("code").y_true.shift(horizon)
                preds["Persistence"] = ref.dropna(subset=["pred"]).astype({"pred": int})
                codes = sorted(preds["MART-Temporal"].code.unique())
                r = next_returns(market, year, codes)
                for name, df in preds.items():
                    p = daily_portfolio(df[["date", "code", "pred"]], r)
                    daily.append(p.assign(market=market, seed=seed, year=year, model=name).reset_index())
                bh = r.groupby("date").r_next.mean()
                bh = pd.DataFrame({"lo_gross": bh, "lo_net": bh, "lo_turnover": 0.0,
                                   "ls_gross": np.nan, "ls_net": np.nan, "ls_turnover": np.nan})
                daily.append(bh.assign(market=market, seed=seed, year=year, model="Buy and hold").reset_index())
    daily = pd.concat(daily, ignore_index=True)

    rows = []
    for (market, model, seed), g in daily.groupby(["market", "model", "seed"]):
        for book in ("ls", "lo"):
            if g[f"{book}_net"].isna().all():
                continue
            s_net, s_gross = stats(g[f"{book}_net"]), stats(g[f"{book}_gross"])
            rows.append(dict(market=market, model=model, seed=seed, book=book, sharpe_gross=s_gross["sharpe"],
                             turnover=g[f"{book}_turnover"].mean(), **{f"{k}_net": v for k, v in s_net.items()}))
    summ = pd.DataFrame(rows).groupby(["market", "book", "model"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    summ.to_csv(OUT / f"r3_backtest{tag}_summary.csv", index=False)

    yearly = (daily.groupby(["market", "model", "seed", "year"])
              .agg(**{f"{b}_sharpe": (f"{b}_net", lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else np.nan)
                      for b in ("ls", "lo")}).reset_index()
              .groupby(["market", "model", "year"]).mean(numeric_only=True).drop(columns="seed").reset_index())
    yearly.to_csv(OUT / f"r3_backtest{tag}_yearly.csv", index=False)
    con = []
    for market, g in yearly.groupby("market"):
        for book in ("ls", "lo"):
            w = g.pivot(index="year", columns="model", values=f"{book}_sharpe")
            for a, b in CONTRASTS:
                if a not in w or b not in w:
                    continue
                diff = (w[a] - w[b]).dropna()
                if diff.empty:
                    continue
                p = wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
                con.append(dict(market=market, book=book, contrast=f"{a} - {b}", mean_diff=diff.mean(),
                                wins=int((diff > 0).sum()), n=len(diff), p=p))
    con = pd.DataFrame(con)
    con.to_csv(OUT / f"r3_backtest{tag}_contrasts.csv", index=False)

    pd.set_option("display.width", 220)
    for book, title in (("ls", "LONG-SHORT"), ("lo", "LONG-ONLY")):
        print(f"\n{title} (net of {COST * 1e4:.0f} bp, seed mean over all test years)")
        s = summ[summ.book == book]
        print(s.pivot(index="model", columns="market", values="sharpe_net").round(3).to_string())
        print("ann. return net:\n", s.pivot(index="model", columns="market", values="ann_return_net").round(4).to_string())
        print("daily turnover:\n", s.pivot(index="model", columns="market", values="turnover").round(3).to_string())
    print("\nYEARLY NET-SHARPE CONTRASTS (seed-averaged; unadjusted Wilcoxon)")
    print(con.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
