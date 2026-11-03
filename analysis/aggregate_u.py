"""Revision experiments.

U  relation-rich universes (Major 2): 20 of the 100 most traded stocks, chosen greedily for the highest mean
   pairwise same-day return correlation ("corr") or lag-1 cross-correlation ("leadlag") over the training period.
   Regime task (macro-F1) and ranking task at H=1 (RankIC); MART variants and MASTER.
P  MASTER at the published width and lookback (d=256, 8 days) on the 5-day ranking task, top-50 (Major 4),
   next to our default MASTER (d=64, 60 days).
Statistics as elsewhere: seed-averaged yearly differences, two-sided Wilcoxon with Holm within market x block,
TOST with margin 0.005, 95% bootstrap interval over test years.
Outputs: analysis/out/u_table.csv, u_contrasts.csv, u_universe_stats.csv, p_contrasts.csv
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from aggregate_e1_e2 import HOLDOUT, RES, SEEDS, holm

OUT = RES.parents[2] / "analysis" / "out"
DATA = RES.parent / "data"
DELTA = 0.005
NAMES = {"temporal_only": "MART-Temporal", "temporal_asset_emb": "MART-Temporal+AE", "temporal_cross_asset": "MART-Cross",
         "temporal_graph_feature_corr_classwise_r3": "MART-CorrVote", "corrvote_perm_p": "MART-CorrVote (permuted P)",
         "master": "MASTER", "master_no_inter": "MASTER w/o inter", "master_asset_emb": "MASTER+AE",
         "master_no_inter_asset_emb": "MASTER w/o inter+AE", "temporal_cross_asset_no_ae": "MART-Cross w/o AE",
         "temporal_ctxmean": "MART-PanelMean w/o AE", "corrvote_no_asset_emb": "MART-CorrVote w/o AE"}
# Revision checks, Holm-adjusted separately so that the five-contrast family of Table universes is unchanged:
# does cross attention help without the embedding, and what does the embedding do inside the cross model?
NOAE_CONTRASTS = [("Relation", "MART-Cross w/o AE", "MART-Temporal"),
                  ("Identity", "MART-Cross", "MART-Cross w/o AE")]
# relation-free market context without identity (cross-sectional feature mean), the control for Cross w/o AE;
# Holm-adjusted on its own so that the Cross w/o AE family above does not change
CTX_CONTRASTS = [("Context", "MART-PanelMean w/o AE", "MART-Temporal"),
                 ("Relation", "MART-Cross w/o AE", "MART-PanelMean w/o AE"),
                 ("Relation", "MART-CorrVote w/o AE", "MART-Temporal"),
                 ("Relation", "MASTER", "MASTER w/o inter")]
U_CONTRASTS = {
    "regime": [("Identity", "MART-Temporal+AE", "MART-Temporal"),
               ("Relation", "MART-CorrVote", "MART-Temporal+AE"),
               ("Relation", "MART-Cross", "MART-Temporal+AE"),
               ("Relation", "MART-CorrVote", "MART-CorrVote (permuted P)"),
               ("Relation", "MASTER+AE", "MASTER w/o inter+AE")],
    "rank": [("Identity", "MART-Temporal+AE", "MART-Temporal"),
             ("Relation", "MART-CorrVote", "MART-Temporal+AE"),
             ("Relation", "MART-Cross", "MART-Temporal+AE"),
             ("Relation", "MART-CorrVote", "MART-CorrVote (permuted P)"),
             ("Relation", "MASTER", "MASTER w/o inter")],
}


def collect(pattern: str, fields: dict, seeds=SEEDS) -> pd.DataFrame:
    rows = []
    for d in RES.glob("top20_amount_ablation_*"):
        m = re.fullmatch(pattern, d.name)
        if not m or int(m["seed"]) not in seeds:
            continue
        for f in d.glob("*/projected_d*/epoch_40/test_*/summary.json"):
            variant = f.parts[-5]
            if variant not in NAMES:
                continue
            s = json.loads(f.read_text(encoding="utf-8"))
            rows.append(dict(**{k: m[k] for k in fields}, seed=int(m["seed"]), year=int(s["test_year"]),
                             model=NAMES[variant],
                             score=s["test_metrics"].get("rank_ic", s["test_metrics"].get("overall_macro_f1"))))
    return pd.DataFrame(rows)


def one_sided(x: np.ndarray, alternative: str) -> float:
    return wilcoxon(x, alternative=alternative).pvalue if (x != 0).any() else 1.0


def contrasts(df: pd.DataFrame, keys: list, specs) -> pd.DataFrame:
    ym = df.groupby(keys + ["model", "year"]).score.mean()
    rows = []
    for g, _ in df.groupby(keys):
        g = g if isinstance(g, tuple) else (g,)
        spec = specs(dict(zip(keys, g))) if callable(specs) else specs
        for kind, a, b in spec:
            try:
                diff = (ym.loc[(*g, a)] - ym.loc[(*g, b)]).dropna()
            except KeyError:
                continue
            if len(diff) < 6:
                continue
            x = diff.to_numpy()
            # fresh generator per contrast, so an interval does not depend on which contrasts were computed before
            boot = np.random.default_rng(0).choice(x, size=(10_000, len(x)), replace=True).mean(axis=1)
            rows.append(dict(zip(keys, g), kind=kind, contrast=f"{a} - {b}", mean_diff=x.mean(), wins=int((x > 0).sum()),
                             n=len(x), p=one_sided(x, "two-sided"), ci_lo=np.percentile(boot, 2.5),
                             ci_hi=np.percentile(boot, 97.5),
                             p_tost=max(one_sided(x + DELTA, "greater"), one_sided(x - DELTA, "less"))))
    con = pd.DataFrame(rows)
    if con.empty:
        return con
    con["p_holm"] = con.groupby(keys).p.transform(holm)
    con["equivalent"] = con.p_tost < 0.05
    return con


def universe_stats() -> pd.DataFrame:
    rows = []
    for d in DATA.glob("*_top20_amount_u*"):
        market, universe = d.name.split("_")[0], d.name.split("_u")[-1]
        for f in (d / "manifests").glob("test_*_manifest.json"):
            j = json.loads(f.read_text(encoding="utf-8"))
            if "mean_corr_selected" in j:
                rows.append(dict(market=market, universe=universe, year=j["test_year"],
                                 **{k: j[k] for k in j if k.startswith("mean_")}))
    return pd.DataFrame(rows)


def main() -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    us = universe_stats()
    us.to_csv(OUT / "u_universe_stats.csv", index=False)
    print(us.groupby(["universe", "market"])[[c for c in us.columns if c.startswith("mean_")]].mean().round(3))

    reg_pat = r"top20_amount_ablation_(?P<market>kospi|kosdaq|nasdaq)_19_1_u(?P<universe>corr|leadlag|random)_seed(?P<seed>\d+)"
    rnk_pat = r"top20_amount_ablation_(?P<market>kospi|kosdaq|nasdaq)_19_1_labxsret_u(?P<universe>corr|leadlag|random)_seed(?P<seed>\d+)"
    keys = {"market": 0, "universe": 0}
    u = pd.concat([collect(reg_pat, keys).assign(task="regime"), collect(rnk_pat, keys).assign(task="rank")])
    u = u[u.year < HOLDOUT]
    u.to_csv(OUT / "u_yearly.csv", index=False)
    print("cells:\n", u.groupby(["task", "universe", "model", "market"]).size().unstack("market").to_string())
    tab = u.groupby(["task", "universe", "market", "model", "seed"]).score.mean().groupby(
        ["task", "universe", "market", "model"]).agg(["mean", "std"]).reset_index()
    tab.to_csv(OUT / "u_table.csv", index=False)
    print(tab.pivot_table(index=["task", "universe", "model"], columns="market", values="mean").round(4).to_string())
    con = contrasts(u, ["task", "universe", "market"], lambda k: U_CONTRASTS[k["task"]])
    # Holm over all relational contrasts of a task in the two selected universes (the section-wide family).
    sel = (con.kind == "Relation") & con.universe.isin(["corr", "leadlag"])
    con.loc[sel, "p_holm_section"] = con[sel].groupby("task").p.transform(holm)
    con.to_csv(OUT / "u_contrasts.csv", index=False)
    print(con.round(4).to_string(index=False))
    for task in ("regime", "rank"):
        rel = con[sel & (con.task == task)]
        kr = rel[rel.market != "nasdaq"]
        print(f"\n{task} relational (corr+leadlag): {len(rel)} cells, Holm-significant {int((rel.p_holm < 0.05).sum())}, "
              f"section-Holm-significant {int((rel.p_holm_section < 0.05).sum())}, equivalent {int(rel.equivalent.sum())}"
              f" (Korean {int(kr.equivalent.sum())}/{len(kr)}), max upper CI {rel.ci_hi.max():.4f}")
    # regime task outside the selected universes: top-20 (amount), per-asset standardization (X1), top-50
    base = pd.concat([collect(rf"top20_amount_ablation_(?P<market>kospi|kosdaq|nasdaq)_19_1{sfx}_seed(?P<seed>\d+)",
                              {"market": 0}).assign(task="regime", universe=name)
                      for name, sfx in (("amount", ""), ("stdasset", "_stdper_asset"), ("top50", "_top50"))])
    base = base[base.year < HOLDOUT]
    noae = contrasts(pd.concat([u, base]), ["task", "universe", "market"], NOAE_CONTRASTS)
    noae.to_csv(OUT / "u_noae.csv", index=False)
    print("\nCROSS WITHOUT EMBEDDING\n", noae.round(4).to_string(index=False))
    ctx = contrasts(pd.concat([u, base]), ["task", "universe", "market"], CTX_CONTRASTS)
    if len(ctx):
        ctx.to_csv(OUT / "u_ctx.csv", index=False)
        print("\nPANELMEAN WITHOUT EMBEDDING\n", ctx.round(4).to_string(index=False))

    # Five seeds for the ranking task in the selected universes (seeds 45, 46 added in the revision).
    u5 = collect(rnk_pat, keys, seeds=[42, 43, 44, 45, 46]).assign(task="rank")
    u5 = u5[(u5.year < HOLDOUT) & u5.universe.isin(["corr", "leadlag"]) & ~u5.model.str.startswith("MASTER")]
    rows = u5.groupby(["universe", "market", "seed"]).size()
    complete = rows[rows == rows.groupby(["universe", "market"]).transform("max")].reset_index()
    n_seeds = complete.groupby(["universe", "market"]).seed.nunique()
    print("\nseeds per cell (rank):\n", n_seeds.unstack())
    if (n_seeds == 5).all():
        c5 = contrasts(u5, ["task", "universe", "market"], U_CONTRASTS["rank"][:4] + NOAE_CONTRASTS)
        c5.to_csv(OUT / "u5_contrasts.csv", index=False)
        print("\nRANK, FIVE SEEDS\n", c5.round(4).to_string(index=False))

    p_new = collect(r"top20_amount_ablation_(?P<market>kospi|kosdaq|nasdaq)_15_5_labxsret_top50_pubcfg_seed(?P<seed>\d+)",
                    {"market": 0}).assign(config="published")
    p_old = collect(r"top20_amount_ablation_(?P<market>kospi|kosdaq|nasdaq)_15_5_labxsret_top50_seed(?P<seed>\d+)",
                    {"market": 0}).assign(config="default")
    p_old = p_old[p_old.model.str.startswith("MASTER")]
    p = pd.concat([p_new, p_old])
    p = p[p.year < HOLDOUT]
    if len(p_new):
        print("\nP cells:\n", p.groupby(["config", "model", "market"]).size().unstack("market"))
        print(p.groupby(["config", "model", "market"]).score.mean().unstack("market").round(4))
        spec = [("Relation", "MASTER", "MASTER w/o inter"), ("Identity", "MASTER+AE", "MASTER")]
        pc = contrasts(p, ["config", "market"], spec)
        p2 = p.assign(model=p.model + " [" + p.config + "]", config="cross")
        pc2 = contrasts(p2, ["config", "market"], [("Config", "MASTER [published]", "MASTER [default]")])
        pc = pd.concat([pc, pc2])
        pc.to_csv(OUT / "p_contrasts.csv", index=False)
        print(pc.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
