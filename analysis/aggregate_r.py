"""Paper 2 weakness checks R1, R2, R4, R5.

  R1 base   MASTER re-implementation (+/- inter-stock attention, +/- asset embedding) next to MART and TCN
  R2 h20    task (0,20): target window does not overlap the input
  R4 top50  universe of 50 assets per market
  R5 xfer / xfer_std  models trained on the top 20 and tested on ranks 21-40 (unseen assets), under the original
            and the per-asset standardization; "(unseen)" rows are those transfer scores

Outputs (analysis/out/): r_yearly.csv, r_table.csv, r_contrasts.csv. Holm adjustment is within each
market and experiment block.
"""
from __future__ import annotations

import json
import re

import pandas as pd
from scipy.stats import wilcoxon

from aggregate_e1_e2 import BASE_NAMES, HOLDOUT, MART_NAMES, RES, SEEDS, holm

OUT = RES.parents[2] / "analysis" / "out"
NAMES = {**MART_NAMES, "master": "MASTER", "master_asset_emb": "MASTER+AE",
         "master_no_inter": "MASTER w/o inter", "master_no_inter_asset_emb": "MASTER w/o inter+AE"}
PROTOCOLS = {"_19_1": "base", "_0_20": "h20", "_19_1_top50": "top50", "_19_1_xfer": "xfer",
             "_19_1_stdper_asset_xfer": "xfer_std"}
SUFFIX_RE = "|".join(re.escape(s) for s in sorted(PROTOCOLS, key=len, reverse=True))


def collect() -> pd.DataFrame:
    rows = []
    for d in RES.glob("top20_amount_ablation_*_seed*"):
        m = re.fullmatch(rf"top20_amount_ablation_(kospi|kosdaq|nasdaq)({SUFFIX_RE})_seed(\d+)", d.name)
        if not m or int(m[3]) not in SEEDS:
            continue
        proto = PROTOCOLS[m[2]]
        for f in d.glob("*/projected_d32/epoch_40/test_*/summary.json"):
            variant = f.parts[-5]
            if variant not in NAMES:
                continue
            s = json.loads(f.read_text(encoding="utf-8"))
            base = dict(market=m[1], seed=int(m[3]), year=int(s["test_year"]))
            rows.append(dict(base, protocol=proto, model=NAMES[variant], macro_f1=s["test_metrics"]["overall_macro_f1"]))
            if s.get("transfer"):
                rows.append(dict(base, protocol=proto, model=NAMES[variant] + " (unseen)",
                                 macro_f1=s["transfer"]["test_metrics"]["overall_macro_f1"]))
    for d in RES.glob("figure4_baseline_comparison_*_seed*"):
        m = re.fullmatch(rf"figure4_baseline_comparison_(kospi|kosdaq|nasdaq)({SUFFIX_RE})(_eq40|_noid)?_seed(\d+)", d.name)
        if not m or int(m[4]) not in SEEDS:
            continue
        proto = PROTOCOLS[m[2]]
        if proto == "base" and m[3] != "_eq40":  # the main comparison uses the matched budget
            continue
        for f in d.glob("baselines/test_*/*/*_summary.json"):
            s = json.loads(f.read_text(encoding="utf-8"))
            name = BASE_NAMES[f.parent.name] + (" w/o one-hot" if m[3] == "_noid" else "")
            rows.append(dict(market=m[1], seed=int(m[4]), year=int(f.parts[-3].split("_")[1]), protocol=proto,
                             model=name, macro_f1=s["test_metrics"]["macro_f1"]))
    return pd.DataFrame(rows)


# block -> list of (protocol A, model A, protocol B, model B); reported as A - B
BLOCKS = {
    "R1 MASTER": [
        ("base", "MASTER+AE", "base", "MASTER"),
        ("base", "MASTER", "base", "MASTER w/o inter"),
        ("base", "MASTER+AE", "base", "MASTER w/o inter+AE"),
        ("base", "MASTER", "base", "MART-Temporal"),
        ("base", "MASTER", "base", "MART-Temporal+AE"),
        ("base", "MASTER+AE", "base", "MART-Temporal+AE"),
        ("base", "MASTER+AE", "base", "TCN"),
    ],
    "R2 (0,20)": [
        ("h20", "MART-Temporal+AE", "h20", "MART-Temporal"),
        ("h20", "MART-CorrVote", "h20", "MART-Temporal+AE"),
        ("h20", "MART-Cross", "h20", "MART-Temporal+AE"),
        ("h20", "MART-Temporal+AE", "h20", "TCN"),
        ("h20", "MART-Temporal+AE", "h20", "Persistence"),
        ("h20", "TCN", "h20", "Persistence"),
        ("h20", "MART-Temporal+AE", "h20", "Markov"),
    ],
    "R4 top-50": [
        ("top50", "MART-Temporal+AE", "top50", "MART-Temporal"),
        ("top50", "MART-CorrVote", "top50", "MART-Temporal+AE"),
        ("top50", "MART-Cross", "top50", "MART-Temporal+AE"),
        ("top50", "MART-CorrVote", "top50", "MART-CorrVote (permuted P)"),
        ("top50", "MART-Temporal+AE", "top50", "TCN"),
        ("top50", "TCN", "top50", "TCN w/o one-hot"),
        ("top50", "MART-Temporal", "top50", "TCN w/o one-hot"),
    ],
    "R5 unseen assets": [
        ("xfer", "MART-Temporal+AE (unseen)", "xfer", "MART-Temporal (unseen)"),
        ("xfer", "MART-Cross (unseen)", "xfer", "MART-Temporal (unseen)"),
        ("xfer", "MART-CorrVote (unseen)", "xfer", "MART-Temporal (unseen)"),
        ("xfer_std", "MART-Temporal (unseen)", "xfer", "MART-Temporal (unseen)"),
        ("xfer_std", "MART-Temporal+AE (unseen)", "xfer_std", "MART-Temporal (unseen)"),
        ("xfer_std", "MART-Temporal (unseen)", "xfer", "MART-Temporal+AE"),
        ("xfer", "MART-Temporal (unseen)", "xfer", "MART-Temporal"),
        ("xfer", "MART-Temporal+AE (unseen)", "xfer", "MART-Temporal+AE"),
        ("xfer_std", "MART-Temporal (unseen)", "xfer_std", "MART-Temporal"),
    ],
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = collect()
    df.to_csv(OUT / "r_yearly.csv", index=False)
    dev = df[df.year < HOLDOUT]
    pd.set_option("display.width", 220)
    cnt = dev.groupby(["protocol", "model", "market"]).size().unstack("market")
    print("cells (years x seeds):\n", cnt.to_string())
    table = (dev.groupby(["protocol", "market", "model", "seed"]).macro_f1.mean()
             .groupby(["protocol", "market", "model"]).agg(mean="mean", seed_std="std").reset_index())
    table.to_csv(OUT / "r_table.csv", index=False)
    print("\nMEAN MACRO-F1\n", table.pivot(index=["protocol", "model"], columns="market", values="mean").round(4).to_string())

    ym = dev.groupby(["protocol", "market", "model", "year"]).macro_f1.mean()
    rows = []
    for block, items in BLOCKS.items():
        for market in sorted(dev.market.unique()):
            for pa, a, pb, b in items:
                try:
                    diff = (ym.loc[(pa, market, a)] - ym.loc[(pb, market, b)]).dropna()
                except KeyError:
                    continue
                if diff.empty:
                    continue
                p = wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
                la, lb = (a, b) if pa == pb else (f"{a} [{pa}]", f"{b} [{pb}]")
                rows.append(dict(block=block, market=market, protocol=pa if pa == pb else f"{pa}/{pb}",
                                 contrast=f"{la} - {lb}", mean_diff=diff.mean(),
                                 wins=int((diff > 0).sum()), n=len(diff), p=p))
    con = pd.DataFrame(rows)
    con["p_holm"] = con.groupby(["block", "market"]).p.transform(holm)
    con.to_csv(OUT / "r_contrasts.csv", index=False)
    print("\nPAIRED CONTRASTS (year-level, seed-averaged; Wilcoxon, Holm within market and block)")
    print(con.assign(mean_diff=con.mean_diff.round(4), p=con.p.round(4), p_holm=con.p_holm.round(4)).to_string(index=False))


if __name__ == "__main__":
    main()
