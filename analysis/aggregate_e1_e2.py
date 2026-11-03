"""Paper 2: aggregate E1 (MART decomposition) + E2/E3 (baselines) over 3 seeds x 3 markets.

Outputs (paper2_mart/analysis/out/):
  yearly_long.csv        one row per market/model/seed/year (macro-F1, accuracy)
  main_table.csv         mean over 2006(2014)-2025 test years; seed-mean and seed-std
  holdout_2026.csv       2026 Jan-Apr holdout, seed mean/std
  contrasts.csv          paired contrasts (year-level, seed-averaged), Wilcoxon + Holm per market
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RES = Path(__file__).resolve().parents[1] / "code" / "paper_final2" / "results"
OUT = Path(__file__).resolve().parent / "out"
SEEDS = [42, 43, 44]
HOLDOUT = 2026

MART_NAMES = {
    "temporal_only": "MART-Temporal",
    "temporal_asset_emb": "MART-Temporal+AE",
    "temporal_asset_emb_ctxmean": "MART-Temporal+AE+PanelMean",
    "temporal_cross_asset": "MART-Cross",
    "temporal_graph_feature_corr_classwise_r3": "MART-CorrVote",
    "corrvote_no_asset_emb": "MART-CorrVote w/o AE",
    "corrvote_perm_p": "MART-CorrVote (permuted P)",
    "corrvote_uniform_p": "MART-CorrVote (uniform P)",
}
BASE_NAMES = {
    "persistence": "Persistence", "markov": "Markov", "logreg": "LogReg", "xgboost": "XGBoost",
    "lightgbm": "LightGBM", "lstm": "LSTM", "tcn": "TCN", "dlinear": "DLinear", "patchtst": "PatchTST",
}


def collect() -> pd.DataFrame:
    rows = []
    for d in RES.glob("top20_amount_ablation_*_19_1_seed*"):
        m = re.fullmatch(r"top20_amount_ablation_(\w+?)_19_1_seed(\d+)", d.name)
        if not m or int(m[2]) not in SEEDS:
            continue
        for f in d.glob("*/projected_d32/epoch_40/test_*/summary.json"):
            s = json.loads(f.read_text(encoding="utf-8"))
            variant = f.parts[-5]
            if variant not in MART_NAMES:
                continue
            rows.append(dict(market=m[1], seed=int(m[2]), year=int(s["test_year"]), model=MART_NAMES[variant],
                             family="MART", budget="mart",
                             macro_f1=s["test_metrics"]["overall_macro_f1"], accuracy=s["test_metrics"]["overall_accuracy"]))
    for d in RES.glob("figure4_baseline_comparison_*_19_1*_seed*"):
        m = re.fullmatch(r"figure4_baseline_comparison_(\w+?)_19_1(_eq40)?_seed(\d+)", d.name)
        if not m or int(m[3]) not in SEEDS:
            continue
        for f in d.glob("baselines/test_*/*/*_summary.json"):
            model = f.parent.name
            s = json.loads(f.read_text(encoding="utf-8"))
            name = BASE_NAMES[model] + (" (eq. budget)" if m[2] else "")
            rows.append(dict(market=m[1], seed=int(m[3]), year=int(f.parts[-3].split("_")[1]), model=name,
                             family="baseline", budget="eq40" if m[2] else "orig",
                             macro_f1=s["test_metrics"]["macro_f1"], accuracy=s["test_metrics"]["accuracy"]))
    return pd.DataFrame(rows)


def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values().index
    m = len(p)
    adj, running = {}, 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[idx]))
        adj[idx] = running
    return pd.Series(adj)[p.index]


CONTRASTS = [
    # (A, B): report A - B
    ("MART-Temporal+AE", "MART-Temporal"),
    ("MART-CorrVote", "MART-Temporal"),
    ("MART-CorrVote", "MART-Temporal+AE"),
    ("MART-Cross", "MART-Temporal+AE"),
    ("MART-Temporal+AE+PanelMean", "MART-Temporal+AE"),
    ("MART-CorrVote", "MART-CorrVote w/o AE"),
    ("MART-CorrVote", "MART-CorrVote (permuted P)"),
    ("MART-CorrVote", "MART-CorrVote (uniform P)"),
    ("MART-Temporal+AE", "TCN"),
    ("MART-Temporal+AE", "TCN (eq. budget)"),
    ("MART-Temporal+AE", "DLinear"),
    ("MART-Temporal+AE", "DLinear (eq. budget)"),
    ("MART-Temporal+AE", "PatchTST (eq. budget)"),
    ("MART-Temporal+AE", "LSTM (eq. budget)"),
    ("MART-CorrVote", "TCN (eq. budget)"),
    ("MART-Cross", "TCN (eq. budget)"),
    ("TCN (eq. budget)", "TCN"),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = collect()
    df.to_csv(OUT / "yearly_long.csv", index=False)
    dev = df[df.year < HOLDOUT]
    # completeness check: every (market, model) should have the same years for all seeds
    cnt = dev.groupby(["market", "model"]).size().unstack(0)
    print("cells per market/model (years x seeds):\n", cnt.to_string())

    seed_means = dev.groupby(["market", "model", "family", "seed"]).macro_f1.mean().reset_index()
    table = (seed_means.groupby(["market", "model", "family"]).macro_f1
             .agg(mean="mean", seed_std="std").reset_index())
    table["n_years"] = dev.groupby(["market", "model"]).year.nunique().reindex(
        pd.MultiIndex.from_frame(table[["market", "model"]])).values
    table = table.sort_values(["market", "mean"], ascending=[True, False])
    table.to_csv(OUT / "main_table.csv", index=False)
    wide = table.pivot(index="model", columns="market", values="mean")
    wide["avg"] = wide.mean(axis=1)
    print("\nMEAN MACRO-F1 (dev years, seed-mean)\n", wide.sort_values("avg", ascending=False).round(4).to_string())

    ho = df[df.year == HOLDOUT].groupby(["market", "model"]).macro_f1.agg(["mean", "std"]).reset_index()
    ho.to_csv(OUT / "holdout_2026.csv", index=False)
    print("\n2026 HOLDOUT (seed mean)\n", ho.pivot(index="model", columns="market", values="mean").round(4).to_string())

    # year-level paired contrasts: average over seeds first, so n = number of test years
    ym = dev.groupby(["market", "model", "year"]).macro_f1.mean().unstack("model")
    rows = []
    for market, g in ym.groupby(level=0):
        for a, b in CONTRASTS:
            if a not in g or b not in g:
                continue
            diff = (g[a] - g[b]).dropna()
            stat = wilcoxon(diff) if (diff != 0).any() else None
            rows.append(dict(market=market, contrast=f"{a} - {b}", mean_diff=diff.mean(),
                             wins=int((diff > 0).sum()), n=len(diff), p=stat.pvalue if stat else 1.0))
    con = pd.DataFrame(rows)
    con["p_holm"] = con.groupby("market").p.transform(holm)
    con.to_csv(OUT / "contrasts.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nPAIRED CONTRASTS (year-level, seed-averaged; Wilcoxon, Holm within market)")
    print(con.assign(mean_diff=con.mean_diff.round(4), p=con.p.round(4), p_holm=con.p_holm.round(4)).to_string(index=False))


if __name__ == "__main__":
    main()
