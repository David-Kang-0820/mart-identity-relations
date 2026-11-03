# Disentangling Asset Identity from Cross-Asset Relations: code and results

Code, experiment definitions and per-year results for the paper

> B. Kang, J. Soh, D. Kim. *Disentangling Asset Identity from Cross-Asset Relations in Multi-Asset
> Transformers.* (under review)

The paper asks whether the gains of cross-asset modules in a multi-asset Transformer (MART) come from relations
between assets or from knowing which asset is predicted. It compares identity-matched and identity-free variants
on an asset-level regime task and on cross-sectional return ranking, over 52 rolling yearly test windows on
KOSPI, KOSDAQ and Nasdaq stocks, with paired Wilcoxon tests, Holm correction, TOST equivalence tests and
bootstrap intervals.

The raw market data cannot be redistributed (see [Data](#data)). Everything else is here: the models, the data
pipeline, universe selection, every experiment run for the paper, the statistical analysis, and the seed-level
yearly results from which all tables are computed.

## Contents

```
code/paper_final2/run_top20_amount_ablation.py   MART variants, MASTER re-implementation, data pipeline, universes
code/paper_final2/run_figure4_baseline_comparison.py   baselines on the same panels
code/compare_algo/run_compare_algos.py           baseline models (persistence, Markov, logistic regression, LSTM,
                                                  XGBoost, LightGBM, TCN, DLinear, PatchTST wrapper)
experiments/jobs_mart.txt, jobs_baselines.txt    every run of the paper, grouped by paper section
experiments/run_jobs.sh                          runs a job list in parallel
analysis/*.py                                    aggregation, statistics, tables and figures
analysis/out/*.csv                               results: per seed and test year, contrasts, tables
universes/<setting>/test_<year>.csv              the stocks in every universe and test window
```

## Installation

```bash
pip install -r requirements.txt
```

A CUDA GPU is recommended; the runners fall back to the CPU with `--cpu`. PatchTST uses the official
implementation: clone <https://github.com/yuqinie98/PatchTST> (Apache-2.0) into
`code/compare_algo/official_baselines/PatchTST`. It is only needed for the PatchTST baseline.

## Data

Daily OHLCV data were obtained through the CREON Plus API of Daishin Securities. The provider's terms do not
allow redistribution, so the raw files are not included. To
rerun the experiments, provide one CSV per stock with the columns

```
day (YYYYMMDD), open, high, low, close, vol, amount        # optional: time
```

in two folders, set through environment variables (defaults in parentheses):

```bash
export MART_RAW_KRX_DIR=/path/to/krx_daily        # code/data/inc_all_1d; KOSPI and KOSDAQ, files named <6-digit code>.csv
export MART_RAW_NASDAQ_DIR=/path/to/nasdaq_daily  # code/data/nasdaq_data; files named <ticker>.csv
```

Korean stocks also need a listing file `code/paper_final2/stock_codes.csv` with the columns `code, name, market`
(`market` is `KOSPI` or `KOSDAQ`). ETFs, ETNs, SPACs, REITs and preferred shares are removed by name.

The archives contain only stocks still trading in 2026, so all universes are survivor-only (paper, Section 3).
`universes/` lists the stocks of every universe and window, so that a reimplementation with other data can check
that it selects the same stocks.

## Running the experiments

Each line of a job list is `<market> <seed> <runner arguments>`:

```bash
bash experiments/run_jobs.sh mart experiments/jobs_mart.txt 4
bash experiments/run_jobs.sh baselines experiments/jobs_baselines.txt 4
```

Runs that are already complete are skipped. The runners build a data cache for each market and data setting on
first use. Before running jobs in parallel, build the caches once, for example:

```bash
for m in kospi kosdaq nasdaq; do
  python code/paper_final2/run_top20_amount_ablation.py --market $m --data_only
  for u in corr leadlag random; do python code/paper_final2/run_top20_amount_ablation.py --market $m --data_only --universe $u; done
  python code/paper_final2/run_top20_amount_ablation.py --market $m --data_only --top_n 50
  python code/paper_final2/run_top20_amount_ablation.py --market $m --data_only --transfer
done
```

Main runner options: `--label_mode per_asset|pooled|xsret` (regime labels with per-asset or pooled tertile
thresholds, or the cross-sectional ranking task), `--std_mode pooled|per_asset`, `--horizon H` (task `(20-H, H)`,
or the ranking horizon), `--top_n`, `--universe amount|corr|leadlag|random`, `--transfer`, `--variants ...`.
See `--help` of each runner.

Model names in the code and in the paper:

| code | paper |
|---|---|
| `temporal_only` | MART-Temporal |
| `temporal_asset_emb` | Temporal+AE |
| `temporal_asset_bias` / `temporal_asset_emb_perm` | per-asset bias / permuted identity |
| `temporal_cross_asset` / `temporal_cross_asset_no_ae` | MART-Cross / Cross w/o AE |
| `temporal_graph_feature_corr_classwise_r3` / `corrvote_no_asset_emb` | MART-CorrVote / CorrVote w/o AE |
| `corrvote_perm_p` / `corrvote_uniform_p` | CorrVote with permuted / uniform relation matrix |
| `temporal_asset_emb_ctxmean` / `temporal_ctxmean` | Temporal+AE+PanelMean / PanelMean w/o AE |
| `master`, `master_asset_emb`, `master_no_inter`, `master_no_inter_asset_emb` | MASTER (re-implementation) with/without inter-stock attention and asset embedding |

Results are written to `code/paper_final2/results/`. Exact numbers can differ slightly on other hardware and
library versions (GPU nondeterminism); the paper's conclusions rest on paired contrasts over test years and seeds.

## Analysis

The scripts in `analysis/` read the run results and write `analysis/out/`. Their outputs are included, so the
statistics, tables and figures can be recomputed without the data:

| script | paper | needs run results |
|---|---|---|
| `aggregate_e1_e2.py` | main results and contrasts (Tables 2-3), Holm correction | yes |
| `aggregate_followups.py`, `aggregate_r.py`, `aggregate_q.py` | interventions, unseen assets, horizons, robustness (Sections 5.4-5.7) | yes |
| `equivalence.py`, `equivalence_sensitivity.py` | TOST equivalence, margin and block-bootstrap sensitivity (Table 7) | no |
| `residual_identity.py`, `cross_noae_assets.py` | label-location mechanism, asset-level regressions (Sections 5.4, 5.8, 5.9) | yes |
| `e4_change_calibration.py`, `e4_days_since_change.py` | where identity helps (Section 5.3) | yes |
| `r3_backtest.py` | economic value (Section 5.10) | yes |
| `aggregate_u.py` | relation-rich and placebo universes, identity-free comparisons (Sections 5.8-5.9) | yes |
| `aggregate_xs.py` | cross-sectional ranking task (Section 6) | yes |
| `aggregate_exgoog.py`, `e6_data_checks.py` | data caveats (duplicate share classes, survivorship) | yes (raw data for `e6`) |
| `make_tables.py`, `make_figure_decomp.py` | LaTeX tables and figure, written to `paper/` | no |

Key result files: `yearly_long.csv`, `followup_yearly.csv`, `r_yearly.csv`, `q_yearly.csv`, `u_yearly.csv` and
`xs_yearly.csv` hold the score of every model, seed and test year; the `*_contrasts.csv` files hold the paired
differences, Wilcoxon and Holm-adjusted p-values, TOST results and bootstrap intervals.

## License

Code: MIT (see `LICENSE`). The universe lists and result files are derived from third-party market data and are
provided for research use.
