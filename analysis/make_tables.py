"""Write LaTeX tables for the paper from analysis/out/*.csv (run aggregate_e1_e2.py and e4_change_calibration.py first)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "out"
TAB = Path(__file__).resolve().parents[1] / "paper" / "tables"
MARKETS = [("kospi", "KOSPI"), ("kosdaq", "KOSDAQ"), ("nasdaq", "Nasdaq")]


def fmt_p(p: float) -> str:
    return "$p_\\text{H}<0.001$" if p < 0.0005 else f"$p_\\text{{H}}$={p:.3f}"


def fmt_mean_std(m: float, s: float) -> str:
    return f"{m:.4f}\\,{{\\scriptsize$\\pm${s:.4f}}}"


def bold_best(col: pd.Series, cells: pd.Series) -> pd.Series:
    best = col.max()
    return cells.where(col != best, "\\textbf{" + cells + "}")


def main_table() -> None:
    t = pd.read_csv(OUT / "main_table.csv")
    ho = pd.read_csv(OUT / "holdout_2026.csv")
    order = [
        ("Persistence", "Persistence"), ("Markov", "Markov"), ("LogReg", "Logistic regression"),
        ("XGBoost", "XGBoost"), ("LightGBM", "LightGBM"), ("LSTM (eq. budget)", "LSTM"),
        ("TCN (eq. budget)", "TCN"), ("DLinear (eq. budget)", "DLinear"), ("PatchTST (eq. budget)", "PatchTST"),
        ("MART-Temporal", "MART-Temporal"), ("MART-Temporal+AE", "MART-Temporal+AE"),
        ("MART-Cross", "MART-Cross"), ("MART-CorrVote", "MART-CorrVote"),
    ]
    rows = []
    for key, label in order:
        r = [label]
        for m, _ in MARKETS:
            x = t[(t.market == m) & (t.model == key)].iloc[0]
            r.append(fmt_mean_std(x["mean"], x["seed_std"]))
        for m, _ in MARKETS:
            r.append(f"{ho[(ho.market == m) & (ho.model == key)]['mean'].iloc[0]:.4f}")
        rows.append(r)
    lines = [
        "\\begin{table*}[t]\\centering\\small",
        "\\caption{Macro-F1 on the $(19,1)$ task. Development columns average the yearly rolling test windows "
        "(KOSPI/KOSDAQ 2006--2025, Nasdaq 2014--2025) and then the three seeds ($\\pm$ standard deviation across seeds). "
        "Holdout columns are the January--April 2026 window (seed mean). Neural baselines use the MART-matched "
        "training budget; the shorter original budget changes every neural baseline by at most 0.001 "
        "(Section~\\ref{sec:budget}).}\\label{tab:main}",
        "\\begin{tabular}{lcccccc}\\toprule",
        " & \\multicolumn{3}{c}{Development (rolling)} & \\multicolumn{3}{c}{Holdout 2026} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
        "Model & " + " & ".join(n for _, n in MARKETS) * 1 + " & " + " & ".join(n for _, n in MARKETS) + " \\\\\\midrule",
    ]
    for i, r in enumerate(rows):
        if i == 9:
            lines.append("\\midrule")
        lines.append(" & ".join(r) + " \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table*}"]
    (TAB / "main_results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def contrast_table() -> None:
    c = pd.read_csv(OUT / "contrasts.csv")
    keep = [
        ("MART-Temporal+AE - MART-Temporal", "Asset embedding (Temporal+AE $-$ Temporal)"),
        ("MART-CorrVote - MART-Temporal+AE", "CorrVote over identity (CorrVote $-$ Temporal+AE)"),
        ("MART-Cross - MART-Temporal+AE", "Cross attention over identity (Cross $-$ Temporal+AE)"),
        ("MART-Temporal+AE+PanelMean - MART-Temporal+AE", "Panel-mean context (+PanelMean $-$ Temporal+AE)"),
        ("MART-CorrVote - MART-CorrVote w/o AE", "Remove AE from CorrVote (CorrVote $-$ CorrVote w/o AE)"),
        ("MART-CorrVote - MART-CorrVote (permuted P)", "Real vs.\\ permuted correlation $P$"),
        ("MART-CorrVote - MART-CorrVote (uniform P)", "Real vs.\\ uniform $P$"),
        ("MART-Temporal+AE - TCN (eq. budget)", "Temporal+AE $-$ TCN"),
        ("MART-Temporal+AE - DLinear (eq. budget)", "Temporal+AE $-$ DLinear"),
        ("TCN (eq. budget) - TCN", "TCN: matched $-$ original budget"),
    ]
    lines = [
        "\\begin{table*}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Paired contrasts of macro-F1 (row model minus reference). Differences are computed per test year "
        "after averaging the three seeds; wins count years with a positive difference; $p$ is the two-sided "
        "Wilcoxon signed-rank test, Holm-adjusted within each market over the 17 contrasts we report in the "
        "supplementary CSV. Bold: $p_\\text{Holm}<0.05$.}\\label{tab:contrasts}",
        "\\begin{tabular}{lccc}\\toprule",
        "Contrast & " + " & ".join(n for _, n in MARKETS) + " \\\\\\midrule",
    ]
    for key, label in keep:
        cells = []
        for m, _ in MARKETS:
            x = c[(c.market == m) & (c.contrast == key)].iloc[0]
            s = f"{x.mean_diff:+.4f} ({int(x.wins)}/{int(x.n)}, {fmt_p(x.p_holm)})"
            cells.append("\\textbf{" + s + "}" if x.p_holm < 0.05 else s)
        lines.append(label + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table*}"]
    (TAB / "contrasts.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def e4_table() -> None:
    e = pd.read_csv(OUT / "e4_summary.csv")
    models = ["Persistence", "LSTM", "TCN", "DLinear", "PatchTST", "MART-Temporal", "MART-Temporal+AE", "MART-Cross", "MART-CorrVote"]
    cols = [("acc_stable", "Stable acc."), ("acc_change", "Change acc."), ("departs_correct", "Prec.\\ of switch"), ("ece", "ECE"), ("brier", "Brier")]
    lines = [
        "\\begin{table}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}",
        "\\caption{Where the models differ, averaged over the three markets, seeds and test years. "
        "Stable/change accuracy is accuracy on days where the next-day regime equals/differs from the current one "
        "(change days are 17--18\\% of test days). Precision of switch is the accuracy of predictions that depart "
        "from the current regime. ECE uses 15 confidence bins.}\\label{tab:e4}",
        "\\begin{tabular}{l" + "c" * len(cols) + "}\\toprule",
        "Model & " + " & ".join(n for _, n in cols) + " \\\\\\midrule",
    ]
    avg = e.groupby("model")[[c for c, _ in cols]].mean()
    for mdl in models:
        r = avg.loc[mdl]
        cells = ["--" if (mdl == "Persistence" and c in ("departs_correct",)) else f"{r[c]:.3f}" for c, _ in cols]
        lines.append(mdl + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table}"]
    (TAB / "change_calibration.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def followup_table() -> None:
    c = pd.read_csv(OUT / "followup_contrasts.csv")
    blocks = [
        ("Original protocol (per-asset thresholds, pooled standardization)", "base", [
            ("MART-Temporal+AE - MART-Temporal", "Asset embedding"),
            ("TCN - TCN w/o one-hot", "TCN: one-hot $-$ none"),
            ("DLinear - DLinear w/o one-hot", "DLinear: one-hot $-$ none"),
            ("MART-Temporal - TCN w/o one-hot", "Temporal $-$ TCN w/o one-hot"),
        ]),
        ("X1: per-asset standardization", "stdasset", [
            ("MART-Temporal+AE - MART-Temporal", "Asset embedding"),
            ("TCN - TCN w/o one-hot", "TCN: one-hot $-$ none"),
            ("DLinear - DLinear w/o one-hot", "DLinear: one-hot $-$ none"),
            ("MART-Temporal [stdasset] - MART-Temporal [base]", "Temporal: X1 $-$ original"),
            ("MART-Temporal [stdasset] - MART-Temporal+AE [base]", "Temporal (X1) $-$ Temporal+AE (orig.)"),
        ]),
        ("X2: pooled thresholds", "labpool", [
            ("MART-Temporal+AE - MART-Temporal", "Asset embedding"),
            ("TCN - TCN w/o one-hot", "TCN: one-hot $-$ none"),
            ("DLinear - DLinear w/o one-hot", "DLinear: one-hot $-$ none"),
        ]),
        ("X4: task $(15,5)$", "h5", [
            ("MART-Temporal+AE - MART-Temporal", "Asset embedding"),
            ("MART-CorrVote - MART-Temporal+AE", "CorrVote $-$ Temporal+AE"),
            ("MART-Cross - MART-Temporal+AE", "Cross $-$ Temporal+AE"),
            ("MART-Temporal+AE - TCN", "Temporal+AE $-$ TCN"),
            ("MART-Temporal+AE - Persistence", "Temporal+AE $-$ Persistence"),
        ]),
        ("X4: task $(10,10)$", "h10", [
            ("MART-Temporal+AE - MART-Temporal", "Asset embedding"),
            ("MART-CorrVote - MART-Temporal+AE", "CorrVote $-$ Temporal+AE"),
            ("MART-Cross - MART-Temporal+AE", "Cross $-$ Temporal+AE"),
            ("MART-Temporal+AE - TCN", "Temporal+AE $-$ TCN"),
            ("MART-Temporal+AE - Persistence", "Temporal+AE $-$ Persistence"),
        ]),
    ]
    lines = [
        "\\begin{table*}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Mechanism interventions and longer horizons: paired macro-F1 contrasts as in "
        "Table~\\ref{tab:contrasts}. Holm adjustment is within each market over all 38 follow-up contrasts in the "
        "supplementary CSV. TCN and DLinear use the original budget. Levels under X2 and X4 are not comparable with "
        "the main task because the labels change. Bold: $p_\\text{Holm}<0.05$.}\\label{tab:followup}",
        "\\begin{tabular}{lccc}\\toprule",
        "Contrast & " + " & ".join(n for _, n in MARKETS) + " \\\\",
    ]
    for title, proto, keep in blocks:
        lines += ["\\midrule", f"\\multicolumn{{4}}{{l}}{{\\emph{{{title}}}}} \\\\"]
        for key, label in keep:
            cells = []
            for m, _ in MARKETS:
                x = c[(c.market == m) & (c.contrast == key) & (c.protocol.isin([proto, "cross"]))].iloc[0]
                s = f"{x.mean_diff:+.4f} ({int(x.wins)}/{int(x.n)}, {fmt_p(x.p_holm)})"
                cells.append("\\textbf{" + s + "}" if x.p_holm < 0.05 else s)
            lines.append("\\quad " + label + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table*}"]
    (TAB / "followups.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def contrast_cell(x) -> str:
    s = f"{x.mean_diff:+.4f} ({int(x.wins)}/{int(x.n)}, {fmt_p(x.p_holm)})"
    return "\\textbf{" + s + "}" if x.p_holm < 0.05 else s


def robust_table() -> None:
    r = pd.read_csv(OUT / "r_contrasts.csv")
    q = pd.read_csv(OUT / "q_contrasts.csv").assign(protocol=lambda d: d.block)
    c = pd.concat([r, q])
    blocks = [
        ("Q1/Q2: identity controls (MART, 3 seeds)", "Q1/Q2 identity controls", [
            ("MART-Temporal+asset bias - MART-Temporal", "Asset bias $-$ Temporal"),
            ("MART-Temporal+asset bias - MART-Temporal+AE", "Asset bias $-$ Temporal+AE"),
            ("MART-Temporal+AE (permuted) - MART-Temporal", "Permuted AE $-$ Temporal"),
        ]),
        ("R1: MASTER re-implementation (3 seeds)", "base", [
            ("MASTER+AE - MASTER", "Asset embedding"),
            ("MASTER - MASTER w/o inter", "Inter-stock attention (no AE)"),
            ("MASTER+AE - MASTER w/o inter+AE", "Inter-stock attention (with AE)"),
            ("MASTER+AE - MART-Temporal+AE", "MASTER+AE $-$ MART-Temporal+AE"),
        ]),
        ("Q3: five seeds (42--46)", "Q3 five seeds", [
            ("MART-Temporal+AE - MART-Temporal", "Asset embedding"),
            ("MART-CorrVote - MART-Temporal+AE", "CorrVote $-$ Temporal+AE"),
            ("MART-Cross - MART-Temporal+AE", "Cross $-$ Temporal+AE"),
            ("MART-Temporal+AE - TCN", "Temporal+AE $-$ TCN"),
        ]),
        ("Q4: hyperparameters selected on validation (seed 42)", "Q4 hp_valsel", [
            ("MART-CorrVote - MART-Temporal+AE", "CorrVote $-$ Temporal+AE"),
            ("MART-Cross - MART-Temporal+AE", "Cross $-$ Temporal+AE"),
        ]),
        ("R4: top-50 universe (3 seeds)", "top50", [
            ("MART-Temporal+AE - MART-Temporal", "Asset embedding"),
            ("MART-CorrVote - MART-Temporal+AE", "CorrVote $-$ Temporal+AE"),
            ("MART-Cross - MART-Temporal+AE", "Cross $-$ Temporal+AE"),
            ("TCN - TCN w/o one-hot", "TCN: one-hot $-$ none"),
        ]),
        ("R2: task $(0,20)$, no overlap between input and target window (3 seeds)", "h20", [
            ("MART-Temporal+AE - MART-Temporal", "Asset embedding"),
            ("MART-CorrVote - MART-Temporal+AE", "CorrVote $-$ Temporal+AE"),
            ("MART-Cross - MART-Temporal+AE", "Cross $-$ Temporal+AE"),
            ("MART-Temporal+AE - Persistence", "Temporal+AE $-$ Persistence"),
        ]),
        ("R5: unseen assets, ranks 21--40 (3 seeds)", None, [
            ("xfer", "MART-Temporal+AE (unseen) - MART-Temporal (unseen)", "Temporal+AE $-$ Temporal (pooled std.)"),
            ("xfer", "MART-CorrVote (unseen) - MART-Temporal (unseen)", "CorrVote $-$ Temporal (pooled std.)"),
            ("xfer", "MART-Temporal (unseen) - MART-Temporal", "Temporal: unseen $-$ seen (pooled std.)"),
            ("xfer", "MART-Temporal+AE (unseen) - MART-Temporal+AE", "Temporal+AE: unseen $-$ seen (pooled std.)"),
            ("xfer_std/xfer", "MART-Temporal (unseen) [xfer_std] - MART-Temporal (unseen) [xfer]",
             "Temporal unseen: per-asset $-$ pooled std."),
            ("xfer_std/xfer", "MART-Temporal (unseen) [xfer_std] - MART-Temporal+AE [xfer]",
             "Temporal unseen (per-asset) $-$ Temporal+AE seen"),
        ]),
    ]
    lines = [
        "\\begin{table*}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Identity controls and robustness checks on the $(19,1)$ task unless stated: paired macro-F1 "
        "contrasts as in Table~\\ref{tab:contrasts}. Holm adjustment is within each market and block over all "
        "contrasts of the block in the supplementary CSV. In R5, models are trained on the top 20 assets and tested "
        "on the assets ranked 21--40 in the same test year; ``seen'' is the same model on the top 20. "
        "Bold: $p_\\text{Holm}<0.05$.}\\label{tab:robust}",
        "\\begin{tabular}{lccc}\\toprule",
        "Contrast & " + " & ".join(n for _, n in MARKETS) + " \\\\",
    ]
    for title, proto, keep in blocks:
        lines += ["\\midrule", f"\\multicolumn{{4}}{{l}}{{\\emph{{{title}}}}} \\\\"]
        for item in keep:
            p, key, label = item if proto is None else (proto, *item)
            cells = [contrast_cell(c[(c.market == m) & (c.protocol == p) & (c.contrast == key)].iloc[0])
                     for m, _ in MARKETS]
            lines.append("\\quad " + label + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table*}"]
    (TAB / "robustness.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def equivalence_table() -> None:
    e = pd.read_csv(OUT / "equivalence.csv")
    keep = [
        ("CorrVote over identity", "CorrVote $-$ Temporal+AE"),
        ("Cross over identity", "Cross $-$ Temporal+AE"),
        ("Real vs permuted P", "Real $-$ permuted $P$"),
        ("Real vs uniform P", "Real $-$ uniform $P$"),
        ("5 seeds: CorrVote over identity", "CorrVote $-$ Temporal+AE, 5 seeds"),
        ("5 seeds: Cross over identity", "Cross $-$ Temporal+AE, 5 seeds"),
        ("Val-selected HP: CorrVote over identity", "CorrVote $-$ Temporal+AE, val.-selected HP"),
        ("Val-selected HP: Cross over identity", "Cross $-$ Temporal+AE, val.-selected HP"),
        ("MASTER inter-stock attention", "MASTER: inter-stock attention"),
        ("MASTER inter-stock attention (+AE)", "MASTER+AE: inter-stock attention"),
        ("Top-50: CorrVote over identity", "Top-50: CorrVote $-$ Temporal+AE"),
        ("Top-50: Cross over identity", "Top-50: Cross $-$ Temporal+AE"),
        ("Top-50: real vs permuted P", "Top-50: real $-$ permuted $P$"),
        ("Identity: Temporal+AE - Temporal", "Reference: asset embedding"),
    ]
    lines = [
        "\\begin{table*}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Equivalence of the relational contrasts. Mean macro-F1 difference with a 95\\% bootstrap "
        "interval over test years (seed-averaged) and the TOST $p$-value for the margin $\\pm0.005$, from one-sided "
        "Wilcoxon signed-rank tests. $^\\ast$: equivalent at the 5\\% level. The last row, the identity effect, "
        "is shown for scale.}\\label{tab:equiv}",
        "\\begin{tabular}{lccc}\\toprule",
        "Contrast & " + " & ".join(n for _, n in MARKETS) + " \\\\\\midrule",
    ]
    for key, label in keep:
        cells = []
        for m, _ in MARKETS:
            x = e[(e.market == m) & (e.contrast == key)].iloc[0]
            star = "$^\\ast$" if x.equivalent else ""
            pt = "$<$0.001" if x.p_tost < 0.0005 else f"{x.p_tost:.3f}"
            cells.append(f"{x.mean_diff:+.4f} [{x.ci_lo:+.4f}, {x.ci_hi:+.4f}] {pt}{star}")
        if key.startswith("Identity"):
            lines.append("\\midrule")
        lines.append(label + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table*}"]
    (TAB / "equivalence.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


XS_MODELS = [
    ("1-day reversal", "1-day reversal"), ("20-day momentum", "20-day momentum"),
    ("Asset mean (train)", "Asset mean (train, identity only)"),
    ("MART-Temporal", "MART-Temporal"), ("MART-Temporal+AE", "MART-Temporal+AE"),
    ("MART-Temporal+asset bias", "MART-Temporal+asset bias"), ("MART-Cross", "MART-Cross"),
    ("MART-Cross w/o AE", "MART-Cross w/o AE"),
    ("MART-CorrVote", "MART-CorrVote"), ("MART-CorrVote (permuted P)", "MART-CorrVote (permuted $P$)"),
    ("MASTER w/o inter", "MASTER w/o inter"), ("MASTER", "MASTER"), ("MASTER+AE", "MASTER+AE"),
]


def xs_level_table() -> None:
    t = pd.read_csv(OUT / "xs_table.csv")
    lines = [
        "\\begin{table*}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Standard task: cross-sectional ranking of $H$-day forward returns in the top-50 universe. Mean "
        "daily RankIC over the rolling test years, averaged over three seeds ($\\pm$ standard deviation across "
        "seeds; factors are deterministic). The last column group gives the mean top-minus-bottom-quintile "
        "$H$-day forward return in percent, before costs.}\\label{tab:xs}",
        "\\begin{tabular}{lcccccccc}\\toprule",
        " & \\multicolumn{3}{c}{RankIC, $H=1$} & \\multicolumn{3}{c}{RankIC, $H=5$} & \\multicolumn{2}{c}{Q5$-$Q1 (\\%), KOSPI} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-9}",
        "Model & " + " & ".join(n for _, n in MARKETS) * 1 + " & " + " & ".join(n for _, n in MARKETS)
        + " & $H=1$ & $H=5$ \\\\\\midrule",
    ]
    for i, (key, label) in enumerate(XS_MODELS):
        if i == 3:
            lines.append("\\midrule")
        cells = []
        for h in (1, 5):
            for m, _ in MARKETS:
                x = t[(t.horizon == h) & (t.market == m) & (t.model == key)].iloc[0]
                cells.append(f"{x.rank_ic_mean:.4f}" if i < 3 else fmt_mean_std(x.rank_ic_mean, x.rank_ic_std))
        for h in (1, 5):
            x = t[(t.horizon == h) & (t.market == "kospi") & (t.model == key)].iloc[0]
            cells.append(f"{100 * x.ls_spread_mean:.2f}")
        lines.append(label + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table*}"]
    (TAB / "xs_levels.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def xs_contrast_table() -> None:
    c = pd.read_csv(OUT / "xs_contrasts.csv")
    keep = [
        ("Identity", "MART-Temporal+AE - MART-Temporal", "Asset embedding (MART)"),
        ("Identity", "MASTER+AE - MASTER", "Asset embedding (MASTER)"),
        ("Identity", "MART-Temporal+asset bias - MART-Temporal", "Asset bias $-$ Temporal"),
        ("Relation", "MART-CorrVote - MART-Temporal+AE", "CorrVote $-$ Temporal+AE"),
        ("Relation", "MART-Cross - MART-Temporal+AE", "Cross $-$ Temporal+AE"),
        ("Relation", "MART-CorrVote - MART-CorrVote (permuted P)", "Real $-$ permuted $P$"),
        ("Relation", "MASTER - MASTER w/o inter", "MASTER: inter-stock attention"),
        ("Reference", "MART-Temporal - 1-day reversal", "MART-Temporal $-$ 1-day reversal"),
        ("Reference", "MASTER - MART-Temporal", "MASTER $-$ MART-Temporal"),
    ]
    lines = [
        "\\begin{table*}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Standard task: paired RankIC contrasts (row model minus reference), computed per test year after "
        "averaging the three seeds; wins count years with a positive difference; $p_\\text{H}$ is the two-sided "
        "Wilcoxon test, Holm-adjusted within each market and horizon over the ten contrasts in the supplementary "
        "CSV. Bold: $p_\\text{Holm}<0.05$; $^\\ast$: equivalent to zero within $\\pm0.005$ (TOST, 5\\%).}"
        "\\label{tab:xscontrasts}",
        "\\begin{tabular}{lccc}\\toprule",
        "Contrast & " + " & ".join(n for _, n in MARKETS) + " \\\\",
    ]
    for h in (1, 5):
        lines += ["\\midrule", f"\\multicolumn{{4}}{{l}}{{\\emph{{Horizon $H={h}$}}}} \\\\"]
        for kind, key, label in keep:
            cells = []
            for m, _ in MARKETS:
                x = c[(c.horizon == h) & (c.market == m) & (c.contrast == key)].iloc[0]
                cells.append(contrast_cell(x) + ("$^\\ast$" if x.equivalent else ""))
            lines.append("\\quad " + label + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table*}"]
    (TAB / "xs_contrasts.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def u_table() -> None:
    c = pd.read_csv(OUT / "u_contrasts.csv")
    keep = [
        ("MART-Temporal+AE - MART-Temporal", "Asset embedding (MART)"),
        ("MART-CorrVote - MART-Temporal+AE", "CorrVote $-$ Temporal+AE"),
        ("MART-Cross - MART-Temporal+AE", "Cross $-$ Temporal+AE"),
        ("MART-CorrVote - MART-CorrVote (permuted P)", "Real $-$ permuted $P$"),
        ("MASTER+AE - MASTER w/o inter+AE", "MASTER+AE: inter-stock attention"),
        ("MASTER - MASTER w/o inter", "MASTER (no AE): inter-stock attention"),
    ]
    cols = [(u, m) for u in ("corr", "leadlag") for m, _ in MARKETS]
    lines = [
        "\\begin{table*}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Relation-rich universes: paired contrasts (row model minus reference; macro-F1 for the regime "
        "task, RankIC at $H=1$ for the ranking task), mean over test years after averaging the three seeds, with "
        "the 95\\% bootstrap interval. Bold: Holm-adjusted $p<0.05$ (within universe, market and task); "
        "$^\\ast$: equivalent to zero within $\\pm0.005$ (TOST, 5\\%). The ranking task uses the 20 selected "
        "stocks. $^\\dagger$: identity-free check added in the revision, Holm-adjusted within its own pair of "
        "contrasts.}\\label{tab:universes}",
        "\\begin{tabular}{l" + "c" * len(cols) + "}\\toprule",
        " & \\multicolumn{3}{c}{Co-movement universe} & \\multicolumn{3}{c}{Lead--lag universe} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
        "Contrast & " + " & ".join(n for _ in range(2) for _, n in MARKETS) + " \\\\",
    ]
    noae_path = OUT / "u_noae.csv"
    if noae_path.exists():
        c = pd.concat([c, pd.read_csv(noae_path)])
        keep = keep + [("MART-Cross w/o AE - MART-Temporal", "Cross w/o AE $-$ Temporal$^\\dagger$"),
                       ("MART-Cross - MART-Cross w/o AE", "Embedding inside Cross$^\\dagger$")]
    for task, title in (("regime", "Regime task, $(19,1)$, macro-F1"), ("rank", "Ranking task, $H=1$, RankIC")):
        lines += ["\\midrule", f"\\multicolumn{{7}}{{l}}{{\\emph{{{title}}}}} \\\\"]
        for key, label in keep:
            sub = c[(c.task == task) & (c.contrast == key) & c.universe.isin(["corr", "leadlag"])]
            if sub.empty:
                continue
            cells = []
            for u, m in cols:
                x = sub[(sub.universe == u) & (sub.market == m)].iloc[0]
                s = f"{x.mean_diff:+.4f}"
                s = "\\textbf{" + s + "}" if x.p_holm < 0.05 else s
                cells.append("\\begin{tabular}[t]{@{}c@{}}" + s + ("$^\\ast$" if x.equivalent else "")
                             + f"\\\\[-1pt]{{\\tiny[{x.ci_lo:+.4f}, {x.ci_hi:+.4f}]}}\\end{{tabular}}")
            lines.append("\\quad " + label + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table*}"]
    (TAB / "universes.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def idfree_table() -> None:
    """Identity-free comparisons: relational (and relation-free context) modules without the embedding against
    MART-Temporal, on both tasks and in every universe where they were run."""
    frames = [pd.read_csv(OUT / f) for f in ("u_noae.csv", "u_ctx.csv") if (OUT / f).exists()]
    if not frames:
        return
    u = pd.concat(frames)
    xs = pd.concat([pd.read_csv(OUT / f) for f in ("xs_noae.csv", "xs_ctx.csv") if (OUT / f).exists()])
    xs = xs[xs.horizon == 1].assign(task="rank", universe="top50")
    c = pd.concat([u, xs])
    rows = [("MART-Cross w/o AE - MART-Temporal", "Cross w/o AE"),
            ("MART-CorrVote w/o AE - MART-Temporal", "CorrVote w/o AE"),
            ("MART-PanelMean w/o AE - MART-Temporal", "PanelMean w/o AE"),
            ("MASTER - MASTER w/o inter", "MASTER inter-stock (no AE)")]
    settings = {"regime": [("amount", "Top 20"), ("top50", "Top 50"), ("corr", "Co-movement"),
                           ("leadlag", "Lead--lag"), ("random", "Random (placebo)"),
                           ("stdasset", "Top 20, per-asset std.")],
                "rank": [("top50", "Top 50"), ("corr", "Co-movement"), ("leadlag", "Lead--lag"),
                         ("random", "Random (placebo)")]}
    lines = [
        "\\begin{table*}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Identity-free comparisons: modules without the asset embedding minus MART-Temporal (macro-F1 "
        "for the regime task, RankIC at $H=1$ for the ranking task), mean over test years after averaging three "
        "seeds, with the 95\\% bootstrap interval. PanelMean w/o AE adds only the cross-sectional mean of the "
        "features, a relation-free market context. Bold: Holm-adjusted $p<0.05$ within the cell; $^\\ast$: "
        "equivalent to zero within $\\pm0.005$; --: not run.}\\label{tab:idfree}",
        "\\begin{tabular}{llccc}\\toprule",
        "Universe & Module & " + " & ".join(n for _, n in MARKETS) + " \\\\",
    ]
    for task, title in (("regime", "Regime task, $(19,1)$, macro-F1"), ("rank", "Ranking task, $H=1$, RankIC")):
        lines += ["\\midrule", f"\\multicolumn{{5}}{{l}}{{\\emph{{{title}}}}} \\\\"]
        for universe, uname in settings[task]:
            first = True
            for key, label in rows:
                sub = c[(c.task == task) & (c.universe == universe) & (c.contrast == key)]
                if sub.empty:
                    continue
                cells = []
                for m, _ in MARKETS:
                    x = sub[sub.market == m]
                    if x.empty:
                        cells.append("--")
                        continue
                    x = x.iloc[0]
                    s = f"{x.mean_diff:+.4f}"
                    s = "\\textbf{" + s + "}" if x.p_holm < 0.05 else s
                    cells.append(s + ("$^\\ast$" if x.equivalent else "")
                                 + f" {{\\tiny[{x.ci_lo:+.4f}, {x.ci_hi:+.4f}]}}")
                lines.append((uname if first else "") + " & " + label + " & " + " & ".join(cells) + " \\\\")
                first = False
    lines += ["\\bottomrule\\end{tabular}\\end{table*}"]
    (TAB / "idfree.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    TAB.mkdir(parents=True, exist_ok=True)
    idfree_table()
    u_table()
    xs_level_table()
    xs_contrast_table()
    main_table()
    contrast_table()
    e4_table()
    followup_table()
    robust_table()
    equivalence_table()
    print("wrote", *sorted(p.name for p in TAB.glob("*.tex")))
