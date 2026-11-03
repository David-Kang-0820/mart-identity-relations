"""Paper 2 E4: where do models differ? Regime-change vs stable days, and probability calibration.

For the (19,1) task the label y(t) is the tertile regime at t+1, so the current regime is y(t-1) of the
same asset. A "change day" is one where y(t) != y(t-1); persistence is correct on every stable day and on
no change day by construction.

Outputs (paper2_mart/analysis/out/):
  e4_cells.csv   one row per market/model/seed/year
  e4_summary.csv seed- and year-averaged metrics per market/model
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

RES = Path(__file__).resolve().parents[1] / "code" / "paper_final2" / "results"
OUT = Path(__file__).resolve().parent / "out"
SEEDS = [42, 43, 44]
FIRST = {"kospi": 2006, "kosdaq": 2006, "nasdaq": 2014}
MART = {
    "temporal_only": "MART-Temporal",
    "temporal_asset_emb": "MART-Temporal+AE",
    "temporal_cross_asset": "MART-Cross",
    "temporal_graph_feature_corr_classwise_r3": "MART-CorrVote",
}
BASE = {"tcn": "TCN", "dlinear": "DLinear", "lstm": "LSTM", "patchtst": "PatchTST"}


def load_mart(market: str, seed: int, variant: str, year: int, task: str = "19_1") -> pd.DataFrame | None:
    d = RES / f"top20_amount_ablation_{market}_{task}_seed{seed}" / variant / "projected_d32" / "epoch_40" / f"test_{year}"
    f = d / "test_predictions.csv"
    if not f.exists():
        return None
    codes = json.loads((d / "summary.json").read_text(encoding="utf-8"))["selected_codes"]
    raw = pd.read_csv(f)
    rows = []
    for date, yt, yp, pr in zip(raw.date, raw.y_true, raw.y_pred, raw.y_prob):
        yt, yp, pr = ast.literal_eval(yt), ast.literal_eval(yp), ast.literal_eval(pr)
        for a, code in enumerate(codes):
            rows.append((str(date), str(code), int(yt[a]), int(yp[a]), *pr[a]))
    return pd.DataFrame(rows, columns=["date", "code", "y_true", "pred", "prob_0", "prob_1", "prob_2"])


def load_base(market: str, seed: int, model: str, year: int, root: str = "19_1_eq40") -> pd.DataFrame | None:
    f = (RES / f"figure4_baseline_comparison_{market}_{root}_seed{seed}" / "baselines" / f"test_{year}"
         / model / f"{model}_test_predictions.csv")
    if not f.exists():
        return None
    df = pd.read_csv(f, dtype={"code": str})
    return df if "prob_0" in df.columns else None


def add_prev_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df.y_true >= 0].sort_values(["code", "date"]).copy()
    df["y_prev"] = df.groupby("code").y_true.shift(1)
    return df.dropna(subset=["y_prev"]).astype({"y_prev": int})


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return total


def metrics(df: pd.DataFrame) -> dict:
    change = df.y_true != df.y_prev
    prob = df[["prob_0", "prob_1", "prob_2"]].to_numpy(dtype=float)
    prob = prob / prob.sum(axis=1, keepdims=True)
    onehot = np.eye(3)[df.y_true.to_numpy()]
    correct = (df.pred == df.y_true).to_numpy()
    return dict(
        n=len(df), change_rate=change.mean(),
        acc_all=correct.mean(), acc_stable=correct[~change].mean(), acc_change=correct[change].mean(),
        macro_f1_change=f1_score(df.y_true[change], df.pred[change], average="macro", labels=[0, 1, 2], zero_division=0),
        # how often the model departs from persistence, and how often such a departure is right
        departs=(df.pred != df.y_prev).mean(),
        departs_correct=correct[(df.pred != df.y_prev).to_numpy()].mean() if (df.pred != df.y_prev).any() else np.nan,
        ece=ece(prob.max(axis=1), correct), brier=((prob - onehot) ** 2).sum(axis=1).mean(),
        nll=-np.log(np.clip(prob[np.arange(len(df)), df.y_true.to_numpy()], 1e-7, 1)).mean(),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for market, first in FIRST.items():
        for seed in SEEDS:
            for year in range(first, 2026):
                for key, name in [*MART.items(), *BASE.items()]:
                    df = load_mart(market, seed, key, year) if key in MART else load_base(market, seed, key, year)
                    if df is None:
                        continue
                    df = add_prev_label(df)
                    rows.append(dict(market=market, seed=seed, year=year, model=name, **metrics(df)))
                    if key == "tcn":  # persistence on the identical sample set
                        p = df.assign(pred=df.y_prev, prob_0=0.0, prob_1=0.0, prob_2=0.0)
                        p[["prob_0", "prob_1", "prob_2"]] = np.eye(3)[p.y_prev.to_numpy()]
                        rows.append(dict(market=market, seed=seed, year=year, model="Persistence", **metrics(p)))
    cells = pd.DataFrame(rows)
    cells.to_csv(OUT / "e4_cells.csv", index=False)
    cols = ["acc_all", "acc_stable", "acc_change", "macro_f1_change", "departs", "departs_correct", "ece", "brier", "nll", "change_rate"]
    summ = cells.groupby(["market", "model"])[cols].mean().reset_index()
    summ.to_csv(OUT / "e4_summary.csv", index=False)
    pd.set_option("display.width", 220)
    for market, g in summ.groupby("market"):
        print(f"\n== {market}")
        print(g.drop(columns="market").sort_values("acc_change", ascending=False).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
