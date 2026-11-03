#!/usr/bin/env python3
"""Figure-4 baseline comparison on the paper_final2 `(19,1)` protocol.

This script keeps the final paper_final2 data protocol:

- KOSPI/Nasdaq raw OHLCV -> local `paper_final2/data` cache
- top20 stocks by pretrain-period average trading amount
- final 45-channel feature set (`legacy60feat_v1`)
- yearly 5-train / 1-val / 1-test rolling windows

It combines the three main paper MART variants from
`run_top20_amount_ablation.py` with task-native baselines trained on the same
yearly panels. Legacy graph diagnostics are intentionally excluded from this
Figure-4 runner.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
COMPARE_ROOT = PROJECT_ROOT / "code" / "compare_algo"
if str(COMPARE_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPARE_ROOT))

import run_compare_algos as comp  # noqa: E402
import run_top20_amount_ablation as pf  # noqa: E402


RESULT_ROOT = HERE / "results" / "figure4_baseline_comparison_kospi_19_1"
MART_RESULT_ROOT = HERE / "results" / "top20_amount_ablation_kospi_19_1"
DEFAULT_MART_VARIANTS = list(pf.MAIN_PAPER_VARIANTS)
MART_VARIANT_CHOICES = list(pf.MAIN_PAPER_VARIANTS)
BASELINE_MODELS = ["persistence", "markov", "logreg", "lstm", "xgboost", "lightgbm", "tcn", "dlinear", "patchtst"]
DISPLAY_NAMES = {
    "persistence": "Persistence",
    "markov": "Markov",
    "logreg": "Logistic Regression",
    "lstm": "LSTM",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "tcn": "TCN",
    "dlinear": "DLinear",
    "patchtst": "PatchTST",
    "temporal_only": "MART-Temporal",
    "temporal_cross_asset": "MART-Cross",
    "temporal_graph_feature_corr_classwise_r3": "MART-CorrVote",
}


def configure_market(market: str, seed: int | None = None, tag: str = "") -> None:
    global RESULT_ROOT, MART_RESULT_ROOT
    pf.configure_market(market)
    suffix = (f"_{tag}" if tag else "") + (f"_seed{seed}" if seed is not None else "")
    task = pf.protocol_suffix()
    RESULT_ROOT = HERE / "results" / f"figure4_baseline_comparison_{pf.MARKET}{task}{suffix}"
    mart_root = pf.RESULT_ROOT.parent / pf.RESULT_ROOT.name.replace("_19_1", task)
    MART_RESULT_ROOT = mart_root if seed is None else mart_root.parent / f"{mart_root.name}_seed{seed}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def panel_from_asset_map_with_current(asset_map: Dict[str, pd.DataFrame], codes: Sequence[str]):
    dates, split_arr, x_panel, ret_panel, y_panel = pf.panel_from_asset_map(asset_map, list(codes))
    date_index = {d: i for i, d in enumerate(dates)}
    y_cur_panel = np.full_like(y_panel, -1)
    for a_idx, code in enumerate(codes):
        df = asset_map[str(code)].copy().sort_values("date").reset_index(drop=True)
        score = pd.to_numeric(df["ret1"], errors="coerce").rolling(pf.REGIME_WINDOW).mean()
        pretrain_score = score[df["split"] == "pretrain"].dropna()
        if pretrain_score.empty:
            raise RuntimeError(f"empty pretrain regime score: {code}")
        q33 = float(pretrain_score.quantile(0.3333))
        q66 = float(pretrain_score.quantile(0.6667))
        regime = pd.Series(np.nan, index=df.index, dtype=float)
        valid = score.notna()
        regime.loc[valid & (score <= q33)] = 0
        regime.loc[valid & (score > q33) & (score <= q66)] = 1
        regime.loc[valid & (score > q66)] = 2
        for idx, row in df.iterrows():
            pos = date_index.get(str(row["date"]))
            if pos is not None and not pd.isna(regime.iloc[idx]):
                y_cur_panel[pos, a_idx] = int(regime.iloc[idx])
    feature_cols = pf.FEATURE_COLS + ["avail_mask"]
    return dates, split_arr, x_panel, y_panel, y_cur_panel, feature_cols


def persistence_summary(dates, split_arr, y_panel, y_cur_panel, codes, out_dir: Path, force: bool) -> Dict:
    summary_path = out_dir / "persistence_summary.json"
    if summary_path.exists() and not force:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    y_true, y_pred, rows = [], [], []
    for t, split in enumerate(split_arr):
        if str(split) != "test":
            continue
        valid = (y_panel[t] >= 0) & (y_cur_panel[t] >= 0)
        for a_idx in np.where(valid)[0]:
            true = int(y_panel[t, a_idx])
            pred = int(y_cur_panel[t, a_idx])
            y_true.append(true)
            y_pred.append(pred)
            rows.append({"date": str(dates[t]), "code": str(codes[a_idx]), "y_true": true, "pred": pred})
    summary = {
        "model": "persistence",
        "input": "current observed regime y(t)",
        "test_size": len(y_true),
        "test_metrics": comp.metric_dict(np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)),
    }
    save_json(summary, summary_path)
    pd.DataFrame(rows).to_csv(out_dir / "persistence_test_predictions.csv", index=False, encoding="utf-8-sig")
    return summary


def mart_summary_path(variant: str, year: int, epochs: int) -> Path:
    return MART_RESULT_ROOT / variant / "projected_d32" / f"epoch_{int(epochs)}" / f"test_{int(year)}" / "summary.json"


def load_mart_rows(years: Sequence[int], epochs: int, mart_variants: Sequence[str]) -> List[Dict]:
    rows = []
    for year in years:
        for variant in mart_variants:
            path = mart_summary_path(variant, int(year), int(epochs))
            if not path.exists():
                raise FileNotFoundError(f"missing MART result; run run_top20_amount_ablation.py first: {path}")
            s = json.loads(path.read_text(encoding="utf-8"))
            rows.append({
                "market": pf.MARKET,
                "test_year": int(year),
                "model": variant,
                "display_name": DISPLAY_NAMES[variant],
                "family": "MART",
                "test_macro_f1": float(s["test_metrics"]["overall_macro_f1"]),
                "test_accuracy": float(s["test_metrics"]["overall_accuracy"]),
                "test_balanced_accuracy": float(s["test_metrics"]["overall_balanced_accuracy"]),
                "test_weighted_f1": float(s["test_metrics"]["overall_weighted_f1"]),
                "selected_epoch": int(s.get("selected_epoch", epochs)),
            })
    return rows


def run_baselines_for_year(year: int, args) -> List[Dict]:
    out_base = ensure_dir(RESULT_ROOT / "baselines" / f"test_{int(year)}")
    codes, asset_map, _stage1 = pf.build_or_load_stage1(int(year), args)
    dates, split_arr, x_panel, y_panel, y_cur_panel, feature_cols = panel_from_asset_map_with_current(asset_map, codes)
    samples = comp.build_fair_samples(x_panel, y_panel, split_arr, dates, codes, pf.LOOKBACK,
                                      asset_onehot=not args.no_asset_id)
    rows = []
    run_args = SimpleNamespace(
        seed=int(args.seed),
        force=bool(args.force),
        epochs=int(args.baseline_epochs),
        batch_size=int(args.baseline_batch_size),
        patience=int(args.baseline_patience),
        lr=float(args.baseline_lr),
        lookback=pf.LOOKBACK,
        # PatchTST panel settings: compare_algo defaults; same date-batch as MART
        panel_batch_size=int(args.panel_batch_size),
        d_model=128, n_heads=8, e_layers=2, d_ff=256, dropout=0.1, patch_len=12, patch_stride=6,
    )
    for model_name in args.models:
        model_out = out_base / model_name
        if model_name == "persistence":
            summary = persistence_summary(dates, split_arr, y_panel, y_cur_panel, codes, model_out, args.force)
        elif model_name == "markov":
            summary = comp.run_markov(dates, split_arr, y_panel, y_cur_panel, codes, model_out, args.force)
        elif model_name == "logreg":
            summary = comp.run_logreg(samples, model_out, args.force)
        elif model_name == "lstm":
            summary = comp.run_sequence(samples, model_out, "lstm", lambda: comp.LSTMClassifier(samples["seq_input_dim"]), run_args)
        elif model_name == "xgboost":
            def builder():
                from xgboost import XGBClassifier
                return XGBClassifier(
                    objective="multi:softmax", num_class=3, n_estimators=300, max_depth=4,
                    learning_rate=0.05, subsample=0.85, colsample_bytree=0.85,
                    reg_lambda=1.0, min_child_weight=1.0, tree_method="hist",
                    random_state=args.seed, n_jobs=4,
                )
            summary = comp.run_tree(samples, model_out, "xgboost", builder, args.force)
        elif model_name == "lightgbm":
            def builder():
                from lightgbm import LGBMClassifier
                return LGBMClassifier(
                    objective="multiclass", num_class=3, n_estimators=400, learning_rate=0.05,
                    num_leaves=31, max_depth=-1, min_child_samples=20, subsample=0.85,
                    colsample_bytree=0.85, reg_lambda=1.0, random_state=args.seed,
                    n_jobs=4, verbosity=-1,
                )
            summary = comp.run_tree(samples, model_out, "lightgbm", builder, args.force)
        elif model_name == "tcn":
            summary = comp.run_sequence(samples, model_out, "tcn", lambda: comp.TCNClassifier(samples["seq_input_dim"]), run_args)
        elif model_name == "dlinear":
            summary = comp.run_sequence(
                samples,
                model_out,
                "dlinear",
                lambda: comp.DLinearClassifier(pf.LOOKBACK, samples["seq_input_dim"]),
                run_args,
            )
        elif model_name == "patchtst":
            summary = comp.run_official_panel(
                "patchtst", dates, split_arr, x_panel, y_panel, codes, feature_cols, model_out, run_args,
            )
        else:
            raise ValueError(model_name)
        metrics = summary["test_metrics"]
        rows.append({
            "market": pf.MARKET,
            "test_year": int(year),
            "model": model_name,
            "display_name": DISPLAY_NAMES[model_name],
            "family": "baseline",
            "test_macro_f1": float(metrics.get("macro_f1", np.nan)),
            "test_accuracy": float(metrics.get("accuracy", np.nan)),
            "test_balanced_accuracy": float(metrics.get("balanced_accuracy", np.nan)),
            "test_weighted_f1": float(metrics.get("weighted_f1", np.nan)),
            "selected_epoch": int(summary.get("best_epoch", 0) or 0),
        })
        print(f"[figure4] test_{year} {model_name} macro_f1={rows[-1]['test_macro_f1']:.6f}", flush=True)
    return rows


def write_outputs(rows: List[Dict], selected_years: List[int], full_years: List[int]) -> None:
    ensure_dir(RESULT_ROOT)
    yearly = pd.DataFrame(rows).sort_values(["family", "model", "test_year"]).reset_index(drop=True)
    summary = (
        yearly.groupby(["model", "display_name", "family"], as_index=False)
        .agg(
            n_windows=("test_year", "nunique"),
            mean_macro_f1=("test_macro_f1", "mean"),
            mean_accuracy=("test_accuracy", "mean"),
            mean_balanced_accuracy=("test_balanced_accuracy", "mean"),
            mean_weighted_f1=("test_weighted_f1", "mean"),
            first_year=("test_year", "min"),
            last_year=("test_year", "max"),
        )
        .sort_values("mean_macro_f1", ascending=False)
        .reset_index(drop=True)
    )
    pivot = yearly.pivot_table(index="test_year", columns="display_name", values="test_macro_f1", aggfunc="first").reset_index()
    is_full_year_set = list(map(int, selected_years)) == list(map(int, full_years))
    if is_full_year_set:
        yearly_path = RESULT_ROOT / "figure4_yearly.csv"
        summary_path = RESULT_ROOT / "figure4_summary.csv"
        pivot_path = RESULT_ROOT / "figure4_pivot_macro_f1.csv"
        result_md = RESULT_ROOT / "RESULTS.md"
    else:
        tag = f"subset_{min(selected_years)}_{max(selected_years)}_{len(selected_years)}y"
        yearly_path = RESULT_ROOT / f"figure4_yearly_{tag}.csv"
        summary_path = RESULT_ROOT / f"figure4_summary_{tag}.csv"
        pivot_path = RESULT_ROOT / f"figure4_pivot_macro_f1_{tag}.csv"
        result_md = RESULT_ROOT / f"RESULTS_{tag}.md"
    yearly.to_csv(yearly_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    pivot.to_csv(pivot_path, index=False, encoding="utf-8-sig")
    lines = [
        "# Figure 4 baseline comparison",
        "",
        f"{pf.MARKET.upper()} `(19,1)`, top20 by pretrain average trading amount, `{pf.FEATURE_SET_NAME}` features.",
        f"Years: {min(selected_years)}--{max(selected_years)} ({len(selected_years)} windows).",
        "" if is_full_year_set else "Partial-year runs write subset files and do not overwrite the final full-window Figure 4 CSVs.",
        "",
        "```",
        summary.to_string(index=False),
        "```",
        "",
    ]
    result_md.write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--market", choices=sorted(pf.MARKET_CONFIGS), default=pf.MARKET)
    p.add_argument("--years", nargs="+", type=int, default=None)
    p.add_argument("--models", nargs="+", choices=BASELINE_MODELS, default=BASELINE_MODELS)
    p.add_argument("--mart_variants", nargs="+", choices=MART_VARIANT_CHOICES, default=DEFAULT_MART_VARIANTS)
    p.add_argument("--epochs", type=int, default=40, help="MART epoch budget to read.")
    p.add_argument("--baseline_epochs", type=int, default=16)
    p.add_argument("--baseline_batch_size", type=int, default=256)
    p.add_argument("--baseline_patience", type=int, default=4)
    p.add_argument("--baseline_lr", type=float, default=1.0e-3)
    p.add_argument("--panel_batch_size", type=int, default=16, help="Dates per batch for panel models (PatchTST).")
    p.add_argument("--min_rows", type=int, default=1200)
    p.add_argument("--min_pretrain_rows", type=int, default=300)
    p.add_argument("--rebuild_data", action="store_true")
    p.add_argument("--seed", type=int, default=pf.SEED)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--seed_dir", action="store_true", help="Write to (and read MART from) *_seed<seed> result roots.")
    p.add_argument("--tag", default="", help="Extra result-root suffix, e.g. a training-budget tag.")
    p.add_argument("--baselines_only", action="store_true", help="Do not require or merge MART results.")
    p.add_argument("--cpu", action="store_true", help="Allow running without CUDA.")
    p.add_argument("--no_asset_id", action="store_true", help="Zero the asset one-hot input channels.")
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--label_mode", choices=["per_asset", "pooled"], default="per_asset")
    p.add_argument("--std_mode", choices=["pooled", "per_asset"], default="pooled")
    p.add_argument("--exclude_codes", nargs="+", default=[], help="codes dropped before top-20 selection")
    p.add_argument("--top_n", type=int, default=20, help="universe size")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    pf.set_protocol(args.horizon, args.label_mode, args.std_mode, args.exclude_codes, args.top_n)
    if args.label_mode != "per_asset" and {"persistence", "markov"} & set(args.models):
        raise ValueError("persistence/markov use per-asset current-regime thresholds; not valid with pooled labels")
    configure_market(args.market, int(args.seed) if args.seed_dir else None, args.tag)
    import torch
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing silent CPU fallback (pass --cpu to allow).")
    print(f"[figure4] market={pf.MARKET} seed={args.seed} device={'cuda' if torch.cuda.is_available() else 'cpu'} "
          f"budget=epochs{args.baseline_epochs}/patience{args.baseline_patience}/batch{args.baseline_batch_size} out={RESULT_ROOT}", flush=True)
    full_years = list(range(pf.FIRST_YEAR, pf.LAST_YEAR + 1))
    years = args.years or full_years
    planned_baselines = []
    for year in years:
        for model in args.models:
            if args.force or not (RESULT_ROOT / "baselines" / f"test_{int(year)}" / model / f"{model}_summary.json").exists():
                planned_baselines.append((int(year), model))
    mart_variants = [] if args.baselines_only else args.mart_variants
    missing_mart = [str(mart_summary_path(v, y, int(args.epochs))) for y in years for v in mart_variants if not mart_summary_path(v, y, int(args.epochs)).exists()]
    save_json({"args": vars(args), "years": years, "planned_baselines": planned_baselines, "missing_mart": missing_mart}, RESULT_ROOT / "latest_run_config.json")
    print(f"[PLAN] years={len(years)} baseline_runs={len(planned_baselines)} missing_mart={len(missing_mart)}", flush=True)
    if args.dry_run:
        for item in planned_baselines[:80]:
            print(" ", item, flush=True)
        if len(planned_baselines) > 80:
            print(f"  ... {len(planned_baselines) - 80} more", flush=True)
        return 0
    if missing_mart:
        raise FileNotFoundError("MART results are incomplete. Run run_top20_amount_ablation.py first.\n" + "\n".join(missing_mart[:10]))
    rows: List[Dict] = []
    for year in years:
        rows.extend(run_baselines_for_year(int(year), args))
        rows.extend(load_mart_rows([int(year)], int(args.epochs), mart_variants))
    write_outputs(rows, list(map(int, years)), full_years)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
