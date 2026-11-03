"""Paper 2: aggregate follow-ups X1-X4 next to the default protocol.

Protocols (directory suffix after the market):
  base     _19_1                 per-asset labels, pooled standardization (E1/E2)
  stdasset _19_1_stdper_asset    X1 per-asset feature standardization
  labpool  _19_1_labpooled       X2 pooled (cross-asset) label thresholds
  h5       _15_5                 X4 horizon 5
  h10      _10_10                X4 horizon 10
Baselines with tag "noid" drop the asset one-hot (X3).

Outputs (paper2_mart/analysis/out/):
  followup_yearly.csv     one row per protocol/market/model/seed/year
  followup_table.csv      dev-year mean macro-F1 (seed-mean) and seed std
  followup_contrasts.csv  year-level paired contrasts, Wilcoxon, Holm within market over all follow-up contrasts
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

from aggregate_e1_e2 import BASE_NAMES, HOLDOUT, MART_NAMES, RES, SEEDS, holm

OUT = Path(__file__).resolve().parent / "out"
PROTOCOLS = {"_19_1": "base", "_19_1_stdper_asset": "stdasset", "_19_1_labpooled": "labpool",
             "_15_5": "h5", "_10_10": "h10"}
SUFFIX_RE = "|".join(re.escape(s) for s in sorted(PROTOCOLS, key=len, reverse=True))


def collect() -> pd.DataFrame:
    rows = []
    for d in RES.glob("top20_amount_ablation_*_seed*"):
        m = re.fullmatch(rf"top20_amount_ablation_(kospi|kosdaq|nasdaq)({SUFFIX_RE})_seed(\d+)", d.name)
        if not m or int(m[3]) not in SEEDS:
            continue
        for f in d.glob("*/projected_d32/epoch_40/test_*/summary.json"):
            variant = f.parts[-5]
            if variant not in MART_NAMES:
                continue
            s = json.loads(f.read_text(encoding="utf-8"))
            rows.append(dict(protocol=PROTOCOLS[m[2]], market=m[1], seed=int(m[3]), year=int(s["test_year"]),
                             model=MART_NAMES[variant], macro_f1=s["test_metrics"]["overall_macro_f1"]))
    for d in RES.glob("figure4_baseline_comparison_*_seed*"):
        m = re.fullmatch(rf"figure4_baseline_comparison_(kospi|kosdaq|nasdaq)({SUFFIX_RE})(_noid)?_seed(\d+)", d.name)
        if not m or int(m[4]) not in SEEDS:
            continue
        for f in d.glob("baselines/test_*/*/*_summary.json"):
            s = json.loads(f.read_text(encoding="utf-8"))
            name = BASE_NAMES[f.parent.name] + (" w/o one-hot" if m[3] else "")
            rows.append(dict(protocol=PROTOCOLS[m[2]], market=m[1], seed=int(m[4]),
                             year=int(f.parts[-3].split("_")[1]), model=name, macro_f1=s["test_metrics"]["macro_f1"]))
    return pd.DataFrame(rows)


# (protocol A, model A, protocol B, model B): report A - B
WITHIN = [("MART-Temporal+AE", "MART-Temporal"), ("MART-CorrVote", "MART-Temporal+AE"),
          ("MART-Cross", "MART-Temporal+AE"), ("MART-Temporal+AE", "TCN"), ("MART-Temporal", "TCN w/o one-hot"),
          ("TCN", "TCN w/o one-hot"), ("DLinear", "DLinear w/o one-hot"),
          ("MART-Temporal+AE", "Persistence"), ("TCN", "Persistence")]
CONTRASTS = [(p, a, p, b) for p in PROTOCOLS.values() for a, b in WITHIN] + [
    ("stdasset", "MART-Temporal", "base", "MART-Temporal"),
    ("stdasset", "MART-Temporal", "base", "MART-Temporal+AE"),
    ("stdasset", "MART-Temporal+AE", "base", "MART-Temporal+AE"),
    ("stdasset", "TCN w/o one-hot", "base", "TCN"),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = collect()
    df.to_csv(OUT / "followup_yearly.csv", index=False)
    dev = df[df.year < HOLDOUT]
    cnt = dev.groupby(["protocol", "market", "model"]).size().unstack("market")
    pd.set_option("display.width", 220)
    print("cells (years x seeds):\n", cnt.to_string())

    seed_means = dev.groupby(["protocol", "market", "model", "seed"]).macro_f1.mean().reset_index()
    table = seed_means.groupby(["protocol", "market", "model"]).macro_f1.agg(mean="mean", seed_std="std").reset_index()
    table.to_csv(OUT / "followup_table.csv", index=False)
    print("\nMEAN MACRO-F1 (dev years, seed-mean)\n",
          table.pivot(index=["protocol", "model"], columns="market", values="mean").round(4).to_string())

    ym = dev.groupby(["protocol", "market", "model", "year"]).macro_f1.mean()
    rows = []
    for market in sorted(dev.market.unique()):
        for pa, a, pb, b in CONTRASTS:
            try:
                xa, xb = ym.loc[(pa, market, a)], ym.loc[(pb, market, b)]
            except KeyError:
                continue
            diff = (xa - xb).dropna()
            if diff.empty:
                continue
            stat = wilcoxon(diff) if (diff != 0).any() else None
            la = a if pa == pb else f"{a} [{pa}]"
            lb = b if pa == pb else f"{b} [{pb}]"
            rows.append(dict(market=market, protocol=pa if pa == pb else "cross", contrast=f"{la} - {lb}",
                             mean_diff=diff.mean(), wins=int((diff > 0).sum()), n=len(diff),
                             p=stat.pvalue if stat else 1.0))
    con = pd.DataFrame(rows)
    con["p_holm"] = con.groupby("market").p.transform(holm)
    con.to_csv(OUT / "followup_contrasts.csv", index=False)
    print("\nPAIRED CONTRASTS (year-level, seed-averaged; Wilcoxon, Holm within market)")
    print(con.assign(mean_diff=con.mean_diff.round(4), p=con.p.round(4), p_holm=con.p_holm.round(4)).to_string(index=False))


if __name__ == "__main__":
    main()
