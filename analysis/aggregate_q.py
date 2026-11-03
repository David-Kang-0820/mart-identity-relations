"""Paper 2 rigor checks Q1-Q4.

  Q1/Q2 base    identity controls next to MART-Temporal(+AE), seeds 42-44: a per-asset logit bias (3 parameters per
                asset) and the asset embedding with rows permuted on every forward pass (placebo)
  Q3   seeds5   the main MART variants and TCN (matched budget) over seeds 42-46
  Q4   hp_*     AE / Cross / CorrVote at seed 42 under d_model x lr = {32, 64} x {1e-3, 3e-4}; hp_valsel picks, per
                variant, market and test year, the setting with the best validation macro-F1

Outputs (analysis/out/): q_yearly.csv, q_table.csv, q_contrasts.csv. Holm adjustment is within each market and block.
"""
from __future__ import annotations

import json
import re

import pandas as pd
from scipy.stats import wilcoxon

from aggregate_e1_e2 import BASE_NAMES, HOLDOUT, MART_NAMES, RES, holm

OUT = RES.parents[2] / "analysis" / "out"
NAMES = {**MART_NAMES, "temporal_asset_bias": "MART-Temporal+asset bias",
         "temporal_asset_emb_perm": "MART-Temporal+AE (permuted)"}
Q12 = {"temporal_asset_bias", "temporal_asset_emb_perm", "temporal_only", "temporal_asset_emb"}
SEEDS5 = [42, 43, 44, 45, 46]
DEFAULT_HP = "d32_lr1e-3"


def mart_rows(d, seed, protocol, variants=None, hp=None):
    rows = []
    for f in d.glob("*/projected_d32/epoch_40/test_*/summary.json"):
        variant = f.parts[-5]
        if variant not in NAMES or (variants and variant not in variants):
            continue
        s = json.loads(f.read_text(encoding="utf-8"))
        rows.append(dict(market=s["market"], seed=seed, year=int(s["test_year"]), protocol=protocol, hp=hp,
                         model=NAMES[variant], macro_f1=s["test_metrics"]["overall_macro_f1"],
                         val_f1=s["val_metrics"]["overall_macro_f1"]))
    return rows


def collect() -> pd.DataFrame:
    rows = []
    for d in RES.glob("top20_amount_ablation_*_19_1*_seed*"):
        m = re.fullmatch(r"top20_amount_ablation_(kospi|kosdaq|nasdaq)_19_1(?:_(d\d+_lr[\de.-]+))?_seed(\d+)", d.name)
        if not m:
            continue
        seed = int(m[3])
        if m[2]:
            rows += mart_rows(d, seed, "hp_" + m[2], hp=m[2])
            continue
        if seed in SEEDS5:
            rows += mart_rows(d, seed, "seeds5", variants=set(MART_NAMES) - {"temporal_asset_emb_ctxmean"})
        if seed <= 44:
            rows += mart_rows(d, seed, "base", variants=Q12)
        if seed == 42:
            rows += mart_rows(d, seed, "hp_" + DEFAULT_HP, hp=DEFAULT_HP,
                              variants={"temporal_asset_emb", "temporal_cross_asset",
                                        "temporal_graph_feature_corr_classwise_r3"})
    for d in RES.glob("figure4_baseline_comparison_*_19_1_eq40_seed*"):
        m = re.fullmatch(r"figure4_baseline_comparison_(kospi|kosdaq|nasdaq)_19_1_eq40_seed(\d+)", d.name)
        if not m or int(m[2]) not in SEEDS5:
            continue
        for f in d.glob("baselines/test_*/tcn/tcn_summary.json"):
            s = json.loads(f.read_text(encoding="utf-8"))
            rows.append(dict(market=m[1], seed=int(m[2]), year=int(f.parts[-3].split("_")[1]), protocol="seeds5",
                             model=BASE_NAMES["tcn"], macro_f1=s["test_metrics"]["macro_f1"]))
    df = pd.DataFrame(rows)
    hp = df[df.protocol.str.startswith("hp_")]
    best = hp.loc[hp.groupby(["market", "model", "year"]).val_f1.idxmax()].assign(protocol="hp_valsel")
    return pd.concat([df, best], ignore_index=True)


BLOCKS = {
    "Q1/Q2 identity controls": [
        ("base", "MART-Temporal+asset bias", "MART-Temporal"),
        ("base", "MART-Temporal+asset bias", "MART-Temporal+AE"),
        ("base", "MART-Temporal+AE (permuted)", "MART-Temporal"),
        ("base", "MART-Temporal+AE", "MART-Temporal+AE (permuted)"),
    ],
    "Q3 five seeds": [
        ("seeds5", "MART-Temporal+AE", "MART-Temporal"),
        ("seeds5", "MART-CorrVote", "MART-Temporal+AE"),
        ("seeds5", "MART-Cross", "MART-Temporal+AE"),
        ("seeds5", "MART-CorrVote", "MART-CorrVote (permuted P)"),
        ("seeds5", "MART-Temporal+AE", "TCN"),
    ],
    **{f"Q4 {p}": [(p, "MART-CorrVote", "MART-Temporal+AE"), (p, "MART-Cross", "MART-Temporal+AE")]
       for p in ["hp_d32_lr1e-3", "hp_d32_lr3e-4", "hp_d64_lr1e-3", "hp_d64_lr3e-4", "hp_valsel"]},
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = collect()
    df.to_csv(OUT / "q_yearly.csv", index=False)
    dev = df[df.year < HOLDOUT]
    pd.set_option("display.width", 220)
    print("cells (years x seeds):\n", dev.groupby(["protocol", "model", "market"]).size().unstack("market").to_string())
    table = (dev.groupby(["protocol", "market", "model", "seed"]).macro_f1.mean()
             .groupby(["protocol", "market", "model"]).agg(mean="mean", seed_std="std").reset_index())
    table.to_csv(OUT / "q_table.csv", index=False)
    print("\nMEAN MACRO-F1\n", table.pivot(index=["protocol", "model"], columns="market", values="mean").round(4).to_string())
    print("\nSEED STD (seeds5)\n", table[table.protocol == "seeds5"].pivot(index="model", columns="market",
                                                                            values="seed_std").round(4).to_string())
    sel = dev[dev.protocol == "hp_valsel"].groupby(["market", "model"]).hp.value_counts().unstack("hp").fillna(0)
    print("\nVALIDATION-SELECTED SETTING (years)\n", sel.astype(int).to_string())

    ym = dev.groupby(["protocol", "market", "model", "year"]).macro_f1.mean()
    rows = []
    for block, items in BLOCKS.items():
        for market in sorted(dev.market.unique()):
            for p, a, b in items:
                try:
                    diff = (ym.loc[(p, market, a)] - ym.loc[(p, market, b)]).dropna()
                except KeyError:
                    continue
                if diff.empty:
                    continue
                pv = wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
                rows.append(dict(block=block, market=market, contrast=f"{a} - {b}", mean_diff=diff.mean(),
                                 wins=int((diff > 0).sum()), n=len(diff), p=pv))
    con = pd.DataFrame(rows)
    con["p_holm"] = con.groupby(["block", "market"]).p.transform(holm)
    con.to_csv(OUT / "q_contrasts.csv", index=False)
    print("\nPAIRED CONTRASTS (year-level, seed-averaged; Wilcoxon, Holm within market and block)")
    print(con.assign(mean_diff=con.mean_diff.round(4), p=con.p.round(4), p_holm=con.p_holm.round(4)).to_string(index=False))


if __name__ == "__main__":
    main()
