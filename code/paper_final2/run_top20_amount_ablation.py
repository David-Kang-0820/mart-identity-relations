#!/usr/bin/env python3
"""Paper-final top20 amount ablation for task `(19,1)`.

This is intentionally independent from `paper_clean`, `attention_analysis`, and
`graph_analysis` stage caches. It reads raw OHLCV CSVs, selects the top 20
stocks by pretrain-period average trading amount for each test year, builds a
local feature cache under
`code/paper_final2/data`, and trains the three main paper ablation structures:

Final paper feature policy: the 44 engineered channels use the original
legacy60 feature schema, including `ret60`, `ma10_ma60`, and `vol60`, plus one
availability mask for 45 input channels.

1. `temporal_only`
   45->32 + time embedding -> temporal attn_ffn -> last-5 mean -> head.
   No asset embedding and no cross-asset module.

2. `temporal_cross_asset`
   45->32 + time embedding + asset embedding -> cross attn_ffn
   -> temporal attn_ffn -> last-5 mean -> head.
   All modules are trained end-to-end for the full epoch count.

3. `temporal_graph_feature_corr_classwise_r3`
   MART-CorrVote: keep the same temporal backbone, but build voting weights from
   non-market temporal feature correlations, then update logits with class-wise
   residual scales for three correlation-weighted voting iterations.

Legacy graph diagnostic variants remain in GRAPH_VARIANT_SPECS for comparison,
but they are not part of the main paper Figure 4 / Table 1 lineup.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
RAW_KOSPI_DIR = Path(__import__("os").environ.get("MART_RAW_KRX_DIR", str(HERE.parents[1] / "data" / "inc_all_1d")))
RAW_NASDAQ_DIR = Path(__import__("os").environ.get("MART_RAW_NASDAQ_DIR", str(HERE.parents[1] / "data" / "nasdaq_data")))
FEATURE_SET_NAME = "legacy60feat_v1"
ELIGIBILITY_POLICY = "quality_gates_use_start_to_finetune_end_no_test_year"
MODEL_SELECTION_POLICY = "best_validation_macro_f1_with_patience"
STOCK_CODES_CSV = HERE / "stock_codes.csv"

MARKET_CONFIGS = {
    "kospi": {
        "raw_dir": RAW_KOSPI_DIR,
        "data_root": HERE / "data" / "kospi_top20_amount",
        "result_root": HERE / "results" / "top20_amount_ablation_kospi_19_1",
        "first_year": 2006,
        "last_year": 2025,
    },
    "nasdaq": {
        "raw_dir": RAW_NASDAQ_DIR,
        "data_root": HERE / "data" / "nasdaq_top20_amount",
        "result_root": HERE / "results" / "top20_amount_ablation_nasdaq_19_1",
        "first_year": 2014,
        "last_year": 2025,
    },
    # Paper 2 addition: KOSDAQ codes live in the same raw archive as KOSPI.
    "kosdaq": {
        "raw_dir": RAW_KOSPI_DIR,
        "data_root": HERE / "data" / "kosdaq_top20_amount",
        "result_root": HERE / "results" / "top20_amount_ablation_kosdaq_19_1",
        "first_year": 2006,
        "last_year": 2025,
    },
}
MARKET = "kospi"
RAW_DATA_DIR = MARKET_CONFIGS[MARKET]["raw_dir"]
DATA_ROOT = MARKET_CONFIGS[MARKET]["data_root"]
RESULT_ROOT = MARKET_CONFIGS[MARKET]["result_root"]
FIRST_YEAR = MARKET_CONFIGS[MARKET]["first_year"]
LAST_YEAR = MARKET_CONFIGS[MARKET]["last_year"]
LOOKBACK = 60
REGIME_WINDOW = 20
HORIZON = 1
# Paper-2 protocol switches (defaults reproduce the original protocol).
# LABEL_MODE: "per_asset" tertile thresholds from each asset's pretrain scores, or "pooled" over all assets;
# "xsret" is the standard stock-prediction task instead: regress the H-day forward return after cross-sectional
# rank normalization per date (CSRankNorm), MSE loss, evaluated by IC / RankIC.
# STD_MODE: "pooled" feature standardization over all assets, or "per_asset" with each asset's own pretrain stats.
LABEL_MODE = "per_asset"
STD_MODE = "pooled"
# EXCLUDE_CODES: codes removed from the candidate list before top-20 selection (e.g. a duplicate share class);
# such runs use their own data cache because the selected universe changes.
EXCLUDE_CODES: Tuple[str, ...] = ()
# TOP_N: universe size (assets ranked 1..TOP_N by pretrain trading value).
# TRANSFER: also evaluate each trained model on the next TOP_N assets (ranks TOP_N+1..2*TOP_N), which it never saw.
TOP_N = 20
TRANSFER = False
# UNIVERSE: "amount" takes the TOP_N most traded assets; "corr" / "leadlag" take, from the RELATION_POOL most traded,
# the TOP_N assets chosen greedily to maximize the mean pairwise same-day return correlation / the mean pairwise
# lag-1 cross-correlation max(|corr(r_i,t-1, r_j,t)|, |corr(r_j,t-1, r_i,t)|), both over the pretrain period;
# "random" draws TOP_N of the same pool at random (placebo for the selection step).
UNIVERSE = "amount"
RELATION_POOL = 100


def set_protocol(horizon: int = 1, label_mode: str = "per_asset", std_mode: str = "pooled",
                 exclude_codes=(), top_n: int = 20, transfer: bool = False, universe: str = "amount") -> None:
    global HORIZON, LABEL_MODE, STD_MODE, EXCLUDE_CODES, DATA_ROOT, TOP_N, TRANSFER, UNIVERSE
    HORIZON, LABEL_MODE, STD_MODE = int(horizon), str(label_mode), str(std_mode)
    EXCLUDE_CODES = tuple(sorted(str(c) for c in exclude_codes))
    TOP_N, TRANSFER, UNIVERSE = int(top_n), bool(transfer), str(universe)
    DATA_ROOT = data_root_for(MARKET)


def is_regression() -> bool:
    return LABEL_MODE == "xsret"


def num_outputs() -> int:
    return 1 if is_regression() else 3


def exclude_suffix() -> str:
    s = "_ex" + "-".join(EXCLUDE_CODES) if EXCLUDE_CODES else ""
    return s + (f"_top{TOP_N}" if TOP_N != 20 else "") + (f"_u{UNIVERSE}" if UNIVERSE != "amount" else "")


def data_root_for(market: str) -> Path:
    root = MARKET_CONFIGS[market]["data_root"]
    return root.parent / f"{root.name}{exclude_suffix()}"


def protocol_suffix() -> str:
    """Result-root suffix replacing the original `_19_1`; non-default switches are appended."""
    s = f"_{int(REGIME_WINDOW) - int(HORIZON)}_{int(HORIZON)}"
    if LABEL_MODE != "per_asset":
        s += f"_lab{LABEL_MODE}"
    if STD_MODE != "pooled":
        s += f"_std{STD_MODE}"
    return s + exclude_suffix() + ("_xfer" if TRANSFER else "")
SEED = 42

GRAPH_VARIANT_SPECS = {
    # Legacy return-correlation diagnostics. Kept for reproducibility, not used
    # by the main paper Figure 4 / Table 1 defaults.
    "temporal_graph_lookback_corr": {
        "graph_block": "lookback_corr_alpha_scalar",
        "alpha_mode": "scalar",
        "iterations": 1,
        "learned_parameters": "one scalar alpha",
    },
    "temporal_graph_lookback_corr_r2": {
        "graph_block": "lookback_corr_alpha_scalar_r2",
        "alpha_mode": "scalar",
        "iterations": 2,
        "learned_parameters": "one scalar alpha",
    },
    "temporal_graph_lookback_corr_r3": {
        "graph_block": "lookback_corr_alpha_scalar_r3",
        "alpha_mode": "scalar",
        "iterations": 3,
        "learned_parameters": "one scalar alpha",
    },
    "temporal_graph_lookback_corr_classwise": {
        "graph_block": "lookback_corr_alpha_classwise",
        "alpha_mode": "classwise",
        "iterations": 1,
        "learned_parameters": "three class-wise alpha values",
    },
    "temporal_graph_lookback_corr_classwise_r2": {
        "graph_block": "lookback_corr_alpha_classwise_r2",
        "alpha_mode": "classwise",
        "iterations": 2,
        "learned_parameters": "three class-wise alpha values",
    },
    "temporal_graph_lookback_corr_classwise_r3": {
        "graph_block": "lookback_corr_alpha_classwise_r3",
        "alpha_mode": "classwise",
        "iterations": 3,
        "learned_parameters": "three class-wise alpha values",
    },
    "temporal_graph_lookback_corr_classwise_r4": {
        "graph_block": "lookback_corr_alpha_classwise_r4",
        "alpha_mode": "classwise",
        "iterations": 4,
        "learned_parameters": "three class-wise alpha values",
    },
    "temporal_graph_lookback_corr_classwise_r5": {
        "graph_block": "lookback_corr_alpha_classwise_r5",
        "alpha_mode": "classwise",
        "iterations": 5,
        "learned_parameters": "three class-wise alpha values",
    },
    "temporal_graph_lookback_corr_classwise_r6": {
        "graph_block": "lookback_corr_alpha_classwise_r6",
        "alpha_mode": "classwise",
        "iterations": 6,
        "learned_parameters": "three class-wise alpha values",
    },
    # CorrVote diagnostics. The representative paper model is r3.
    "temporal_graph_feature_corr_classwise_r2": {
        "graph_block": "lookback_feature_corr_alpha_classwise_r2",
        "alpha_mode": "classwise",
        "iterations": 2,
        "corr_source": "feature_nonmarket",
        "learned_parameters": "three class-wise alpha values",
    },
    "temporal_graph_feature_corr_classwise_r3": {
        "graph_block": "lookback_feature_corr_alpha_classwise_r3",
        "alpha_mode": "classwise",
        "iterations": 3,
        "corr_source": "feature_nonmarket",
        "learned_parameters": "three class-wise alpha values",
    },
    "temporal_graph_feature_corr_classwise_r4": {
        "graph_block": "lookback_feature_corr_alpha_classwise_r4",
        "alpha_mode": "classwise",
        "iterations": 4,
        "corr_source": "feature_nonmarket",
        "learned_parameters": "three class-wise alpha values",
    },
    "temporal_graph_feature_corr_classwise_r5": {
        "graph_block": "lookback_feature_corr_alpha_classwise_r5",
        "alpha_mode": "classwise",
        "iterations": 5,
        "corr_source": "feature_nonmarket",
        "learned_parameters": "three class-wise alpha values",
    },
    # Paper 2 decomposition controls for CorrVote (r3). p_mode "perm" relabels
    # the correlation graph with a random asset permutation per forward pass;
    # "uniform" replaces P by equal weights 1/(A-1) (a panel-mean logit vote).
    "corrvote_no_asset_emb": {
        "graph_block": "lookback_feature_corr_alpha_classwise_r3",
        "alpha_mode": "classwise",
        "iterations": 3,
        "corr_source": "feature_nonmarket",
        "learned_parameters": "three class-wise alpha values",
        "asset_emb": False,
    },
    "corrvote_perm_p": {
        "graph_block": "permuted_feature_corr_alpha_classwise_r3",
        "alpha_mode": "classwise",
        "iterations": 3,
        "corr_source": "feature_nonmarket",
        "p_mode": "perm",
        "learned_parameters": "three class-wise alpha values",
    },
    "corrvote_uniform_p": {
        "graph_block": "uniform_alpha_classwise_r3",
        "alpha_mode": "classwise",
        "iterations": 3,
        "corr_source": "feature_nonmarket",
        "p_mode": "uniform",
        "learned_parameters": "three class-wise alpha values",
    },
}

# Paper 2: temporal backbones that differ from temporal_only only by the asset
# slot embedding and, optionally, a projected cross-sectional mean context.
EXTRA_TEMPORAL_VARIANTS = ["temporal_asset_emb", "temporal_asset_emb_ctxmean"]
# Identity mechanism controls on the temporal backbone:
#   temporal_asset_bias      no embedding; a learned per-asset class bias (3 numbers per asset) added to the logits
#   temporal_asset_emb_perm  same embedding table, but assets are assigned to embedding rows by a fresh random
#                            permutation at every forward pass (same capacity, no identity)
IDENTITY_CONTROL_VARIANTS = ["temporal_asset_bias", "temporal_asset_emb_perm"]

MASTER_VARIANT_NAMES = ["master", "master_asset_emb", "master_no_inter", "master_no_inter_asset_emb"]
# Same-date cross attention without the asset embedding: a relational module with no identity (revision check
# of whether Cross's ranking-task gain over Temporal+AE comes from relations or from offsetting the embedding).
CROSS_VARIANTS = {"temporal_cross_asset", "temporal_cross_asset_no_ae"}
# Cross-sectional mean context (PanelMean) with or without the embedding; without it, a relation-free and
# identity-free market context for comparison with Cross w/o AE.
CTXMEAN_VARIANTS = {"temporal_asset_emb_ctxmean", "temporal_ctxmean"}
VALID_VARIANTS = ["temporal_only", *sorted(CROSS_VARIANTS), *EXTRA_TEMPORAL_VARIANTS, "temporal_ctxmean",
                  *GRAPH_VARIANT_SPECS.keys(),
                  *MASTER_VARIANT_NAMES, *IDENTITY_CONTROL_VARIANTS]
PAPER2_E1_VARIANTS = [
    "temporal_only",
    "temporal_asset_emb",
    "temporal_asset_emb_ctxmean",
    "temporal_cross_asset",
    "temporal_graph_feature_corr_classwise_r3",
    "corrvote_no_asset_emb",
    "corrvote_perm_p",
    "corrvote_uniform_p",
]


def uses_asset_embedding(variant: str) -> bool:
    spec = GRAPH_VARIANT_SPECS.get(variant)
    if spec is not None:
        return bool(spec.get("asset_emb", True))
    return variant in {"temporal_cross_asset", *EXTRA_TEMPORAL_VARIANTS, "master_asset_emb", "master_no_inter_asset_emb",
                       "temporal_asset_emb_perm"}
MAIN_PAPER_VARIANTS = ["temporal_only", "temporal_cross_asset", "temporal_graph_feature_corr_classwise_r3"]


def current_task_name() -> str:
    return f"{int(REGIME_WINDOW) - int(HORIZON)},{int(HORIZON)}"

FEATURE_COLS = [
    "ret1", "ret2", "ret3", "ret5", "ret10", "ret20", "ret60",
    "logret1", "range_oc", "range_hl",
    "close_ma5", "close_ma20", "ma5_ma20", "ma10_ma60",
    "vol5", "vol10", "vol20", "vol60",
    "vol_log", "amount_log", "vol_z20",
    "atr14", "rsi14",
    "mkt_ret1", "mkt_ret5", "mkt_ret20",
    "mkt_vol20", "mkt_close_ma20", "mkt_ma5_ma20",
    "breadth_up1", "breadth_up5",
    "rank_ret1", "rank_ret5", "rank_ret20",
    "rank_vol20", "rank_close_ma20", "rank_ma5_ma20",
    "rank_vol_z20", "rank_rsi14",
    "excess_ret1", "excess_ret5", "excess_ret20",
    "excess_close_ma20", "excess_ma5_ma20",
]
FEATURE_POLICY = "legacy_60_day_features_ret60_ma10_ma60_vol60"
FEATURE_POLICY_MANIFEST = "legacy feature set; includes ret60, ma10_ma60, and vol60"
# Final paper uses the legacy60 feature set as-is. This empty manifest field is
# kept only to make it explicit that no max20 feature replacement is active.
REPLACED_OLD_FEATURES: Dict[str, str] = {}


def configure_market(market: str) -> None:
    global MARKET, RAW_DATA_DIR, DATA_ROOT, RESULT_ROOT, FIRST_YEAR, LAST_YEAR
    key = str(market).lower()
    if key not in MARKET_CONFIGS:
        raise ValueError(f"unknown market: {market}")
    cfg = MARKET_CONFIGS[key]
    MARKET = key
    RAW_DATA_DIR = cfg["raw_dir"]
    DATA_ROOT = data_root_for(key)
    RESULT_ROOT = cfg["result_root"]
    FIRST_YEAR = int(cfg["first_year"])
    LAST_YEAR = int(cfg["last_year"])


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def zfill6(x) -> str:
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    # Paper 2 fix: post-2024 KRX codes are alphanumeric (e.g. 0010V0). Stripping
    # letters collided them with old numeric codes and loaded the wrong raw file.
    if re.search(r"[A-Za-z]", s):
        return s.upper()
    digits = re.sub(r"[^0-9]", "", s)
    return digits.zfill(6) if digits else s


def split_bounds(test_year: int) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    year = int(test_year)
    start = pd.Timestamp(f"{year - 6}-01-01")
    pretrain_end = pd.Timestamp(f"{year - 2}-12-31")
    finetune_end = pd.Timestamp(f"{year - 1}-12-31")
    test_end = pd.Timestamp(f"{year}-12-31")
    return start, pretrain_end, finetune_end, test_end


def is_excluded_name(name: str) -> bool:
    s = str(name).strip()
    u = s.upper()
    if any(k in u for k in ["KODEX", "TIGER", "KOSEF", "KBSTAR", "HANARO", "ARIRANG", "ACE", "RISE", "PLUS", "SOL", "ETF", "ETN"]):
        return True
    if any(k in u for k in ["SPAC", "REIT", "2X", "LEVERAGE", "INVERSE"]):
        return True
    if s.endswith("우") or s.endswith("우B") or s.endswith("우C") or "우선주" in s:
        return True
    return False


def load_market_metadata() -> pd.DataFrame:
    if MARKET == "nasdaq":
        rows = []
        for path in sorted(RAW_DATA_DIR.glob("*.csv")):
            symbol = path.stem.strip()
            if not symbol or symbol.startswith("_"):
                continue
            rows.append({"code": symbol, "name": symbol, "market": "NASDAQ_RAW"})
        df = pd.DataFrame(rows).drop_duplicates("code")
        return df[~df["code"].isin(EXCLUDE_CODES)].reset_index(drop=True)

    if not STOCK_CODES_CSV.exists():
        raise FileNotFoundError(f"missing metadata csv: {STOCK_CODES_CSV}")

    df = pd.read_csv(STOCK_CODES_CSV, dtype={"code": str})
    df["code"] = df["code"].apply(zfill6)
    df["name"] = df["name"].astype(str)
    df["market"] = df["market"].astype(str).str.upper()
    df = df[df["market"].eq(MARKET.upper())].copy()
    df = df[~df["name"].map(is_excluded_name)].copy()
    if MARKET == "kosdaq":
        # KOSDAQ lists SPACs and REIT-like names in Korean; KOSPI keeps the
        # original rule unchanged for reproducibility.
        df = df[~df["name"].str.contains("스팩|기업인수목적|리츠", regex=True)].copy()
    df = df[df["code"].map(lambda c: (RAW_DATA_DIR / f"{c}.csv").exists())].copy()
    df = df[~df["code"].isin(EXCLUDE_CODES)]
    return df.drop_duplicates("code").reset_index(drop=True)


def load_one_raw(code: str) -> pd.DataFrame:
    path = RAW_DATA_DIR / f"{code}.csv"
    df = pd.read_csv(path)
    required = ["day", "open", "high", "low", "close", "vol", "amount"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{code} missing columns: {missing}")

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(pd.to_numeric(df["day"], errors="coerce").astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    out["time"] = pd.to_numeric(df["time"], errors="coerce") if "time" in df.columns else 0
    for c in ["open", "high", "low", "close", "vol", "amount"]:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    out["code"] = code
    out = out.dropna(subset=["date", "open", "high", "low", "close", "vol", "amount"]).copy()
    out = out.sort_values(["date", "time"]).drop_duplicates("date", keep="last").reset_index(drop=True)
    out = out[(out["open"] > 0) & (out["high"] > 0) & (out["low"] > 0) & (out["close"] > 0)].copy()
    out = out[(out["high"] >= out["low"]) & (out["vol"] >= 0) & (out["amount"] >= 0)].copy()
    return out.reset_index(drop=True)


def compute_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["ret1"] = x["close"].pct_change(1)
    x["ret2"] = x["close"].pct_change(2)
    x["ret3"] = x["close"].pct_change(3)
    x["ret5"] = x["close"].pct_change(5)
    x["ret10"] = x["close"].pct_change(10)
    x["ret20"] = x["close"].pct_change(20)
    x["ret60"] = x["close"].pct_change(60)
    x["logret1"] = np.log(x["close"]).diff(1)
    x["range_oc"] = (x["close"] - x["open"]) / x["open"].replace(0, np.nan)
    x["range_hl"] = (x["high"] - x["low"]) / x["open"].replace(0, np.nan)
    x["ma5"] = x["close"].rolling(5).mean()
    x["ma10"] = x["close"].rolling(10).mean()
    x["ma20"] = x["close"].rolling(20).mean()
    x["ma60"] = x["close"].rolling(60).mean()
    x["close_ma5"] = x["close"] / x["ma5"] - 1.0
    x["close_ma20"] = x["close"] / x["ma20"] - 1.0
    x["ma5_ma20"] = x["ma5"] / x["ma20"] - 1.0
    x["ma10_ma60"] = x["ma10"] / x["ma60"] - 1.0
    x["vol5"] = x["ret1"].rolling(5).std()
    x["vol10"] = x["ret1"].rolling(10).std()
    x["vol20"] = x["ret1"].rolling(20).std()
    x["vol60"] = x["ret1"].rolling(60).std()
    x["vol_log"] = np.log1p(x["vol"].clip(lower=0))
    x["amount_log"] = np.log1p(x["amount"].clip(lower=0))
    x["vol_z20"] = (x["vol_log"] - x["vol_log"].rolling(20).mean()) / (x["vol_log"].rolling(20).std() + 1e-8)
    prev_close = x["close"].shift(1)
    tr = pd.concat([(x["high"] - x["low"]).abs(), (x["high"] - prev_close).abs(), (x["low"] - prev_close).abs()], axis=1).max(axis=1)
    x["atr14"] = tr.rolling(14).mean() / x["close"]
    diff = x["close"].diff()
    up = diff.clip(lower=0.0)
    down = (-diff).clip(lower=0.0)
    rs = up.rolling(14).mean() / (down.rolling(14).mean() + 1e-8)
    x["rsi14"] = (100.0 - 100.0 / (1.0 + rs)) / 100.0
    return x


def assign_split(df: pd.DataFrame, pretrain_end: pd.Timestamp, finetune_end: pd.Timestamp, test_end: pd.Timestamp) -> pd.DataFrame:
    x = df.copy()
    x["split"] = None
    x.loc[x["date"] <= pretrain_end, "split"] = "pretrain"
    x.loc[(x["date"] > pretrain_end) & (x["date"] <= finetune_end), "split"] = "finetune"
    x.loc[(x["date"] > finetune_end) & (x["date"] <= test_end), "split"] = "test"
    return x


def add_market_rank_features(frames: List[pd.DataFrame]) -> pd.DataFrame:
    cols = ["date", "code", "ret1", "ret5", "ret20", "vol20", "close_ma20", "ma5_ma20", "vol_z20", "rsi14"]
    panel = pd.concat([df[cols].copy() for df in frames], axis=0, ignore_index=True).sort_values(["date", "code"])
    mkt = panel.groupby("date", as_index=False).agg(
        mkt_ret1=("ret1", "mean"),
        mkt_ret5=("ret5", "mean"),
        mkt_ret20=("ret20", "mean"),
        mkt_vol20=("vol20", "mean"),
        mkt_close_ma20=("close_ma20", "mean"),
        mkt_ma5_ma20=("ma5_ma20", "mean"),
        breadth_up1=("ret1", lambda s: float((s > 0).mean())),
        breadth_up5=("ret5", lambda s: float((s > 0).mean())),
    )
    for c in ["ret1", "ret5", "ret20", "vol20", "close_ma20", "ma5_ma20", "vol_z20", "rsi14"]:
        panel[f"rank_{c}"] = panel.groupby("date")[c].rank(method="average", pct=True)
    rank_cols = ["date", "code"] + [f"rank_{c}" for c in ["ret1", "ret5", "ret20", "vol20", "close_ma20", "ma5_ma20", "vol_z20", "rsi14"]]
    return panel[rank_cols].merge(mkt, on="date", how="left")


def add_excess_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["excess_ret1"] = x["ret1"] - x["mkt_ret1"]
    x["excess_ret5"] = x["ret5"] - x["mkt_ret5"]
    x["excess_ret20"] = x["ret20"] - x["mkt_ret20"]
    x["excess_close_ma20"] = x["close_ma20"] - x["mkt_close_ma20"]
    x["excess_ma5_ma20"] = x["ma5_ma20"] - x["mkt_ma5_ma20"]
    return x


def eligible_raw(
    df: pd.DataFrame,
    start: pd.Timestamp,
    pretrain_end: pd.Timestamp,
    eligibility_end: pd.Timestamp,
    min_rows: int,
    min_pretrain_rows: int,
) -> bool:
    x = df[(df["date"] >= start) & (df["date"] <= eligibility_end)]
    if len(x) < min_rows:
        return False
    pre = x[x["date"] <= pretrain_end]
    if len(pre) < min_pretrain_rows:
        return False
    if x["close"].notna().mean() < 0.98:
        return False
    if x["close"].nunique() < 50:
        return False
    if (x["vol"] > 0).mean() < 0.95:
        return False
    return True


def stage1_dir_for(year: int) -> Path:
    return DATA_ROOT / f"test_{int(year)}" / "stage1"


def cache_matches_feature_schema(stage1: Path, codes: List[str]) -> bool:
    if not codes:
        return False
    sample = stage1 / codes[0] / "asset_tables" / f"{codes[0]}.csv"
    if not sample.exists():
        return False
    try:
        cols = set(pd.read_csv(sample, nrows=1).columns)
    except Exception:
        return False
    return set(FEATURE_COLS).issubset(cols)


def manifest_matches_current_policy(manifest_path: Path, year: int) -> bool:
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    _start, _pretrain_end, finetune_end, _test_end = split_bounds(year)
    return (
        manifest.get("feature_set_name") == FEATURE_SET_NAME
        and manifest.get("eligibility_policy") == ELIGIBILITY_POLICY
        and manifest.get("eligibility_end") == str(finetune_end.date())
    )


def build_or_load_stage1(year: int, args) -> Tuple[List[str], Dict[str, pd.DataFrame], Path]:
    stage1 = stage1_dir_for(year)
    selected_path = DATA_ROOT / "selected_top20" / f"test_{int(year)}_top20_amount.csv"
    manifest_path = DATA_ROOT / "manifests" / f"test_{int(year)}_manifest.json"
    if stage1.exists() and selected_path.exists() and not args.rebuild_data:
        selected = pd.read_csv(selected_path, dtype={"code": str})
        codes = selected["code"].astype(str).map(zfill6).tolist()
        if cache_matches_feature_schema(stage1, codes) and manifest_matches_current_policy(manifest_path, year):
            asset_map = {}
            for code in codes:
                path = stage1 / code / "asset_tables" / f"{code}.csv"
                df = pd.read_csv(path)
                df["date"] = df["date"].astype(str)
                if "feature_ready" in df.columns:
                    df = df[df["feature_ready"] == 1].copy()
                asset_map[code] = df.sort_values("date").reset_index(drop=True)
            return codes, asset_map, stage1
        print(f"[CACHE] test_{year} cache schema/policy mismatch; rebuilding {stage1}", flush=True)

    start, pretrain_end, finetune_end, test_end = split_bounds(year)
    meta = load_market_metadata()
    rows = []
    raw_map: Dict[str, pd.DataFrame] = {}
    for _, row in meta.iterrows():
        code = str(row["code"])
        try:
            raw = load_one_raw(code)
        except Exception:
            continue
        if not eligible_raw(raw, start, pretrain_end, finetune_end, int(args.min_rows), int(args.min_pretrain_rows)):
            continue
        pre = raw[(raw["date"] >= start) & (raw["date"] <= pretrain_end)]
        avg_amount = pre["amount"].replace([np.inf, -np.inf], np.nan).dropna().mean()
        if not np.isfinite(avg_amount) or avg_amount <= 0:
            continue
        eligible_window_rows = int(len(raw[(raw["date"] >= start) & (raw["date"] <= finetune_end)]))
        rows.append({
            "code": code,
            "name": row["name"],
            "avg_amount_pretrain": float(avg_amount),
            "eligible_window_rows": eligible_window_rows,
            "raw_rows_total": int(len(raw)),
        })
        raw_map[code] = raw[(raw["date"] >= start) & (raw["date"] <= test_end)].copy()

    ranking = pd.DataFrame(rows).sort_values("avg_amount_pretrain", ascending=False).reset_index(drop=True)
    if len(ranking) < TOP_N:
        raise RuntimeError(f"test_{year}: only {len(ranking)} eligible assets, need {TOP_N}")
    relation_info = {}
    if UNIVERSE == "amount":
        selected = ranking.head(TOP_N).copy()
    else:
        selected, relation_info = select_by_relation(ranking, raw_map, start, pretrain_end)
    codes = selected["code"].astype(str).tolist()

    ensure_dir(stage1)
    ensure_dir(selected_path.parent)
    ensure_dir(manifest_path.parent)
    selected.to_csv(selected_path, index=False, encoding="utf-8-sig")
    ranking.to_csv(DATA_ROOT / "selected_top20" / f"test_{int(year)}_eligible_amount_ranking.csv", index=False, encoding="utf-8-sig")

    asset_map, meta_rows = build_asset_tables(codes, raw_map, stage1, pretrain_end, finetune_end, test_end)
    pd.DataFrame(meta_rows).to_csv(DATA_ROOT / "manifests" / f"test_{int(year)}_asset_summary.csv", index=False, encoding="utf-8-sig")
    save_json({
        "test_year": int(year),
        "source_raw_dir": str(RAW_DATA_DIR),
        "market": MARKET,
        "stage1_dir": str(stage1),
        "selection": f"pretrain average amount top{TOP_N}" if UNIVERSE == "amount"
        else f"{UNIVERSE}-greedy {TOP_N} of the top{RELATION_POOL} by pretrain average amount",
        **relation_info,
        "eligibility_policy": ELIGIBILITY_POLICY,
        "eligibility_end": str(finetune_end.date()),
        "eligibility_note": "Quality gates and row-count filters are computed only through finetune_end; test-year data is not used for universe eligibility.",
        "feature_set_name": FEATURE_SET_NAME,
        "feature_policy": FEATURE_POLICY_MANIFEST,
        "replaced_old_features": REPLACED_OLD_FEATURES,
        "start": str(start.date()),
        "pretrain_end": str(pretrain_end.date()),
        "finetune_end": str(finetune_end.date()),
        "test_end": str(test_end.date()),
        "num_assets": TOP_N,
        "feature_cols": FEATURE_COLS,
        "old_stage1_cache_used": False,
        "raw_universe_all": False,
    }, manifest_path)
    return codes, asset_map, stage1


def relation_matrix(rets: pd.DataFrame, mode: str) -> np.ndarray:
    """Pairwise relation strength between assets over the pretrain period (dates x assets of daily returns)."""
    if mode == "corr":
        s = rets.corr(min_periods=250).to_numpy()
    else:
        lagged = rets.shift(1)
        valid = rets.notna()
        rows = []
        for col in rets.columns:  # c[i, j]: asset i leads asset j by one day
            n = (valid & lagged[col].notna().to_numpy()[:, None]).sum().to_numpy()
            rows.append(np.where(n >= 250, rets.corrwith(lagged[col]).to_numpy(), np.nan))
        c = np.vstack(rows)
        s = np.maximum(np.abs(c), np.abs(c.T))
    s = np.nan_to_num(s, nan=0.0)
    np.fill_diagonal(s, 0.0)
    return s


def select_by_relation(ranking: pd.DataFrame, raw_map, start, pretrain_end) -> Tuple[pd.DataFrame, Dict]:
    """Greedy TOP_N subset of the RELATION_POOL most traded assets with the highest mean pairwise relation."""
    pool = ranking.head(RELATION_POOL).reset_index(drop=True)
    rets = {}
    for code in pool["code"].astype(str):
        raw = raw_map[code]
        pre = raw[(raw["date"] >= start) & (raw["date"] <= pretrain_end)].set_index("date")["close"]
        rets[code] = pre.pct_change()
    rets = pd.DataFrame(rets).sort_index()
    if UNIVERSE == "random":  # placebo: same pool, no relational selection; fixed per test year across seeds
        s = relation_matrix(rets, "corr")
        rng = np.random.default_rng(int(pd.Timestamp(pretrain_end).year))
        chosen = [int(k) for k in rng.choice(len(pool), size=TOP_N, replace=False)]
    else:
        s = relation_matrix(rets, UNIVERSE)
        i, j = np.unravel_index(np.argmax(s), s.shape)
        chosen = [int(i), int(j)]
        while len(chosen) < TOP_N:
            rest = [k for k in range(len(pool)) if k not in chosen]
            chosen.append(max(rest, key=lambda k: s[k, chosen].mean()))
    chosen = sorted(chosen)  # keep the trading-value order for the asset slots
    top = list(range(TOP_N))
    mean_rel = lambda idx: float(s[np.ix_(idx, idx)].sum() / (len(idx) * (len(idx) - 1)))
    corr = relation_matrix(rets, "corr")
    lead = relation_matrix(rets, "leadlag")
    info = {
        "relation_mode": UNIVERSE,
        "mean_corr_selected": float(corr[np.ix_(chosen, chosen)].sum() / (TOP_N * (TOP_N - 1))),
        "mean_corr_amount_top": float(corr[np.ix_(top, top)].sum() / (TOP_N * (TOP_N - 1))),
        "mean_leadlag_selected": float(lead[np.ix_(chosen, chosen)].sum() / (TOP_N * (TOP_N - 1))),
        "mean_leadlag_amount_top": float(lead[np.ix_(top, top)].sum() / (TOP_N * (TOP_N - 1))),
        "mean_relation_selected": mean_rel(chosen),
    }
    return pool.iloc[chosen].copy(), info


def build_asset_tables(codes, raw_map, stage1, pretrain_end, finetune_end, test_end):
    """Daily, cross-sectional (within `codes`) and excess features for one panel; writes per-asset tables."""
    base_frames = []
    base_map = {}
    for code in codes:
        df = compute_daily_features(raw_map[code])
        base_frames.append(df.copy())
        base_map[code] = df.copy()
    market_rank = add_market_rank_features(base_frames)

    asset_map = {}
    meta_rows = []
    for code in codes:
        df = base_map[code].merge(market_rank[market_rank["code"].eq(code)].drop(columns=["code"]), on="date", how="left")
        df = add_excess_features(df)
        df = assign_split(df, pretrain_end, finetune_end, test_end)
        df["feature_ready"] = df[FEATURE_COLS].notna().all(axis=1).astype(int)
        save_cols = ["date", "code", "open", "high", "low", "close", "vol", "amount", *FEATURE_COLS, "split", "feature_ready"]
        out_dir = ensure_dir(stage1 / code / "asset_tables")
        df[save_cols].to_csv(out_dir / f"{code}.csv", index=False, encoding="utf-8-sig")
        fdf = df[df["feature_ready"] == 1].copy()
        fdf["date"] = fdf["date"].dt.strftime("%Y-%m-%d")
        asset_map[code] = fdf.sort_values("date").reset_index(drop=True)
        meta_rows.append({
            "code": code,
            "feature_ready_rows": int(df["feature_ready"].sum()),
            "pretrain_rows": int((df["split"] == "pretrain").sum()),
            "finetune_rows": int((df["split"] == "finetune").sum()),
            "test_rows": int((df["split"] == "test").sum()),
        })
    return asset_map, meta_rows


def build_or_load_transfer(year: int, args) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
    """Assets ranked TOP_N+1..2*TOP_N in the same eligibility ranking; features computed within that group."""
    ranking_path = DATA_ROOT / "selected_top20" / f"test_{int(year)}_eligible_amount_ranking.csv"
    ranking = pd.read_csv(ranking_path, dtype={"code": str})
    codes = ranking["code"].astype(str).map(lambda c: c if MARKET == "nasdaq" else zfill6(c)).tolist()[TOP_N:2 * TOP_N]
    if len(codes) < TOP_N:
        raise RuntimeError(f"test_{year}: only {len(codes)} transfer assets")
    stage1 = DATA_ROOT / f"test_{int(year)}" / "stage1_transfer"
    if all(cache_matches_feature_schema(stage1, [c]) for c in codes):
        asset_map = {}
        for code in codes:
            df = pd.read_csv(stage1 / code / "asset_tables" / f"{code}.csv")
            df["date"] = df["date"].astype(str)
            asset_map[code] = df[df["feature_ready"] == 1].sort_values("date").reset_index(drop=True)
        return codes, asset_map
    start, pretrain_end, finetune_end, test_end = split_bounds(year)
    raw_map = {}
    for code in codes:
        raw = load_one_raw(code)
        raw_map[code] = raw[(raw["date"] >= start) & (raw["date"] <= test_end)].copy()
    asset_map, _ = build_asset_tables(codes, raw_map, ensure_dir(stage1), pretrain_end, finetune_end, test_end)
    return codes, asset_map


def pooled_feature_stats(x_panel: np.ndarray, split_arr: np.ndarray) -> List[Tuple[float, float]]:
    """Pretrain mean/std per feature pooled over assets (the default standardization statistics)."""
    train_mask = split_arr == "pretrain"
    avail = x_panel[..., -1] > 0.5
    stats = []
    for f in range(x_panel.shape[-1] - 1):
        vals = x_panel[train_mask, :, f][avail[train_mask]]
        mu, sd = (float(vals.mean()), float(vals.std())) if vals.size else (0.0, 1.0)
        stats.append((mu, sd if sd >= 1e-8 else 1.0))
    return stats


def standardize_panel_with_availability(x_panel: np.ndarray, split_arr: np.ndarray, pooled_stats=None) -> np.ndarray:
    """`pooled_stats` (pooled mode only) applies another panel's statistics, e.g. the training universe's."""
    out = x_panel.copy()
    if STD_MODE == "pooled" and pooled_stats is not None:
        avail = out[..., -1] > 0.5
        for f, (mu, sd) in enumerate(pooled_stats):
            out[..., f] = np.where(avail, (out[..., f] - mu) / sd, 0.0)
        return out
    train_mask = split_arr == "pretrain"
    feat_dim = out.shape[-1] - 1
    avail = out[..., -1] > 0.5
    if STD_MODE == "per_asset":
        for a in range(out.shape[1]):
            rows = train_mask & avail[:, a]
            for f in range(feat_dim):
                vals = out[rows, a, f]
                mu, sd = (float(vals.mean()), float(vals.std())) if vals.size else (0.0, 1.0)
                if sd < 1e-8:
                    sd = 1.0
                out[:, a, f] = np.where(avail[:, a], (out[:, a, f] - mu) / sd, 0.0)
        return out
    for f in range(feat_dim):
        vals = out[train_mask, :, f][avail[train_mask]]
        if vals.size == 0:
            mu, sd = 0.0, 1.0
        else:
            mu, sd = float(vals.mean()), float(vals.std())
            if sd < 1e-8:
                sd = 1.0
        out[..., f] = np.where(avail, (out[..., f] - mu) / sd, 0.0)
    return out


def panel_from_asset_map(asset_map: Dict[str, pd.DataFrame], codes: List[str], pooled_stats=None,
                         return_stats: bool = False):
    all_dates = set()
    split_map = {}
    for code in codes:
        df = asset_map[code]
        all_dates.update(df["date"].astype(str).tolist())
        for _, row in df[["date", "split"]].dropna().iterrows():
            date = str(row["date"])
            split = row["split"]
            prev = split_map.get(date)
            if prev is None:
                split_map[date] = split
            elif prev != split:
                raise RuntimeError(f"split mismatch date={date} prev={prev} current={split} code={code}")

    dates = sorted(all_dates)
    date_index = {d: i for i, d in enumerate(dates)}
    x_panel = np.zeros((len(dates), len(codes), len(FEATURE_COLS) + 1), dtype=np.float32)
    ret_panel = np.zeros((len(dates), len(codes)), dtype=np.float32)
    if is_regression():
        # [..., 0] rank-normalized target, [..., 1] raw forward return; NaN where unavailable
        y_panel = np.full((len(dates), len(codes), 2), np.nan, dtype=np.float32)
    else:
        y_panel = np.full((len(dates), len(codes)), -1, dtype=np.int64)
    split_arr = np.array([split_map.get(d, "unknown") for d in dates], dtype=object)
    base_dates = pd.DataFrame({"date": dates})

    pooled_q = None
    if LABEL_MODE == "pooled":
        pooled_scores = []
        for code in codes:
            df = asset_map[code].sort_values("date")
            s = pd.to_numeric(df["ret1"], errors="coerce").rolling(REGIME_WINDOW).mean()
            pooled_scores.append(s[(df["split"] == "pretrain").values].dropna())
        allp = pd.concat(pooled_scores)
        pooled_q = (float(allp.quantile(0.3333)), float(allp.quantile(0.6667)))

    for a, code in enumerate(codes):
        df = asset_map[code].copy().sort_values("date").reset_index(drop=True)
        if is_regression():
            log_cum = np.log1p(pd.to_numeric(df["ret1"], errors="coerce")).cumsum()
            fwd = np.expm1(log_cum.shift(-HORIZON) - log_cum)
            future_split = df["split"].shift(-HORIZON)
            fwd[(future_split.isna()) | (future_split != df["split"])] = np.nan
            pos = df["date"].astype(str).map(date_index)
            ok = pos.notna() & fwd.notna()
            y_panel[pos[ok].astype(int).values, a, 1] = fwd[ok].values
        else:
            fill_regime_labels(df, y_panel, a, date_index, pooled_q)
        fdf = base_dates.merge(df[["date"] + FEATURE_COLS], on="date", how="left")
        avail = (~fdf[FEATURE_COLS].isna().all(axis=1)).astype(float).values.reshape(-1, 1)
        x = fdf[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float32)
        x_panel[:, a, :] = np.concatenate([x, avail.astype(np.float32)], axis=1)
        ret_panel[:, a] = pd.to_numeric(fdf["ret1"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float32)

    if is_regression():
        # CSRankNorm: per-date rank percentile of the forward return, centred and scaled to unit variance
        raw = pd.DataFrame(y_panel[..., 1])
        enough = raw.notna().sum(axis=1) >= 5
        y_panel[..., 0] = ((raw.rank(axis=1, pct=True) - 0.5) * 3.46).where(enough, np.nan).values
        y_panel[~enough.values, :, 1] = np.nan

    own_stats = pooled_feature_stats(x_panel, split_arr) if return_stats else None
    x_panel = standardize_panel_with_availability(x_panel, split_arr, pooled_stats)
    if return_stats:
        return dates, split_arr, x_panel, ret_panel, y_panel, own_stats
    return dates, split_arr, x_panel, ret_panel, y_panel


def fill_regime_labels(df: pd.DataFrame, y_panel: np.ndarray, a: int, date_index: Dict[str, int], pooled_q) -> None:
    """Regime target r_{t+H} (tertiles of the 20-day mean return) of one asset into column a of y_panel."""
    score = pd.to_numeric(df["ret1"], errors="coerce").rolling(REGIME_WINDOW).mean()
    pretrain_score = score[df["split"] == "pretrain"].dropna()
    if pretrain_score.empty:
        raise RuntimeError(f"empty pretrain regime score: asset {a}")
    if pooled_q is not None:
        q33, q66 = pooled_q
    else:
        q33 = float(pretrain_score.quantile(0.3333))
        q66 = float(pretrain_score.quantile(0.6667))
    regime = pd.Series(np.nan, index=df.index, dtype=float)
    valid = score.notna()
    regime.loc[valid & (score <= q33)] = 0
    regime.loc[valid & (score > q33) & (score <= q66)] = 1
    regime.loc[valid & (score > q66)] = 2
    target = regime.shift(-HORIZON)
    future_split = df["split"].shift(-HORIZON)
    target[(future_split.isna()) | (future_split != df["split"])] = np.nan
    for idx, row in df.iterrows():
        pos = date_index.get(str(row["date"]))
        if pos is not None and not pd.isna(target.iloc[idx]):
            y_panel[pos, a] = int(target.iloc[idx])


class PanelDataset(Dataset):
    def __init__(self, dates, split_arr, x_panel, ret_panel, y_panel, lookback: int, split_name: str):
        self.dates = dates
        self.split_arr = split_arr
        self.x_panel = x_panel
        self.ret_panel = ret_panel
        self.y_panel = y_panel
        self.lookback = int(lookback)
        self.regression = y_panel.dtype.kind == "f"
        has_label = np.isfinite(y_panel[..., 0]).any(axis=1) if self.regression else (y_panel >= 0).any(axis=1)
        self.indices = [
            t for t in range(self.lookback - 1, len(dates))
            if split_arr[t] == split_name and has_label[t]
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict:
        t = self.indices[idx]
        s = t - self.lookback + 1
        return {
            "x": torch.tensor(self.x_panel[s:t + 1], dtype=torch.float32),
            "ret": torch.tensor(self.ret_panel[s:t + 1], dtype=torch.float32),
            "y": torch.tensor(self.y_panel[t], dtype=torch.float32 if self.regression else torch.long),
            "date": self.dates[t],
        }


class SingleHeadAttention(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(float(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        return self.out_proj(torch.matmul(self.dropout(weights), v))


class AttnFfnBlock(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = SingleHeadAttention(d_model, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout1(self.attn(self.norm1(x)))
        ff = self.linear2(self.dropout(F.gelu(self.linear1(self.norm2(x)))))
        return x + self.dropout2(ff)


def lookback_corr_from_returns(ret: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Signed asset correlation over the same lookback window as the temporal encoder."""
    x = torch.nan_to_num(ret.float(), nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.mean(dim=1, keepdim=True)
    cov = torch.matmul(x.transpose(1, 2), x)
    var = torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(eps)
    denom = torch.sqrt(var.unsqueeze(-1) * var.unsqueeze(-2)).clamp_min(eps)
    corr = (cov / denom).clamp(-1.0, 1.0)
    assets = corr.shape[-1]
    eye = torch.eye(assets, device=corr.device, dtype=torch.bool).view(1, assets, assets)
    return corr.masked_fill(eye, 0.0)


def graph_feature_indices() -> List[int]:
    """Feature channels used for graph edge corr; excludes market-common channels and availability mask."""
    excluded_prefixes = ("mkt_", "breadth_")
    return [i for i, name in enumerate(FEATURE_COLS) if not name.startswith(excluded_prefixes)]


def lookback_corr_from_features(x: torch.Tensor, feature_idx: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Mean signed asset correlation over selected temporal input feature sequences."""
    feat = torch.index_select(x.float(), dim=-1, index=feature_idx.to(x.device))
    feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    feat = feat.permute(0, 3, 1, 2)  # [B,F,T,A]
    feat = feat - feat.mean(dim=2, keepdim=True)
    cov = torch.einsum("bfta,bftc->bfac", feat, feat)
    var = torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(eps)
    denom = torch.sqrt(var.unsqueeze(-1) * var.unsqueeze(-2)).clamp_min(eps)
    corr = (cov / denom).clamp(-1.0, 1.0).mean(dim=1)
    assets = corr.shape[-1]
    eye = torch.eye(assets, device=corr.device, dtype=torch.bool).view(1, assets, assets)
    return corr.masked_fill(eye, 0.0)


class LookbackCorrLogitPropagation(nn.Module):
    """Lookback-corr logit propagation with signed abs-row-normalized weights."""

    def __init__(self, alpha_mode: str = "scalar", iterations: int = 1, corr_source: str = "ret1", p_mode: str = "corr"):
        super().__init__()
        if alpha_mode not in {"scalar", "classwise"}:
            raise ValueError(f"unknown alpha_mode: {alpha_mode}")
        if corr_source not in {"ret1", "feature_nonmarket"}:
            raise ValueError(f"unknown corr_source: {corr_source}")
        if p_mode not in {"corr", "perm", "uniform"}:
            raise ValueError(f"unknown p_mode: {p_mode}")
        self.alpha_mode = str(alpha_mode)
        self.iterations = int(iterations)
        self.corr_source = str(corr_source)
        self.p_mode = str(p_mode)
        self.n_out = num_outputs()
        self.alpha = nn.Parameter(torch.zeros(1 if self.alpha_mode == "scalar" else self.n_out))
        if self.corr_source == "feature_nonmarket":
            indices = graph_feature_indices()
            if not indices:
                raise RuntimeError("feature_nonmarket graph corr has no selected feature channels")
            self.register_buffer("feature_idx", torch.tensor(indices, dtype=torch.long), persistent=False)
        else:
            self.feature_idx = None

    def forward(self, logits: torch.Tensor, ret: torch.Tensor, x: torch.Tensor = None) -> torch.Tensor:
        if self.corr_source == "feature_nonmarket":
            if x is None:
                raise ValueError("feature_nonmarket graph corr requires x lookback tensor")
            corr = lookback_corr_from_features(x, self.feature_idx)
        else:
            corr = lookback_corr_from_returns(ret)
        weights = corr / corr.abs().sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        assets = weights.shape[-1]
        if self.p_mode == "perm":
            perm = torch.randperm(assets, device=weights.device)
            weights = weights[:, perm][:, :, perm]
        elif self.p_mode == "uniform":
            eye = torch.eye(assets, device=weights.device, dtype=weights.dtype)
            weights = ((1.0 - eye) / max(assets - 1, 1)).unsqueeze(0).expand_as(weights)
        scale = torch.tanh(self.alpha)
        scale = scale.view(1, 1, 1) if self.alpha_mode == "scalar" else scale.view(1, 1, self.n_out)
        out = logits
        for _ in range(max(self.iterations, 1)):
            msg = torch.bmm(weights, out)
            out = out + scale * msg
        return out


class PaperAblationModel(nn.Module):
    def __init__(self, variant: str, num_features: int, num_assets: int, d_model: int, dropout: float, recent_k: int):
        super().__init__()
        if variant not in VALID_VARIANTS:
            raise ValueError(f"unknown variant: {variant}")
        self.variant = str(variant)
        self.graph_spec = GRAPH_VARIANT_SPECS.get(self.variant)
        self.d_model = int(d_model)
        self.recent_k = int(recent_k)
        self.input_proj = nn.Linear(num_features, d_model)
        self.time_emb = nn.Embedding(LOOKBACK, d_model)
        self.asset_emb = nn.Embedding(num_assets, d_model) if uses_asset_embedding(self.variant) else None
        n_out = num_outputs()
        self.asset_bias = nn.Parameter(torch.zeros(num_assets, n_out)) if self.variant == "temporal_asset_bias" else None
        # Cross-sectional mean of the available assets' features, projected and
        # added to every asset token (matches the context the baselines receive).
        self.ctx_proj = nn.Linear(num_features - 1, d_model) if self.variant in CTXMEAN_VARIANTS else None
        self.in_dropout = nn.Dropout(dropout)
        self.cross = AttnFfnBlock(d_model, d_model * 2, dropout) if self.variant in CROSS_VARIANTS else None
        self.temporal = AttnFfnBlock(d_model, d_model * 2, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, n_out))
        self.logit_prop = (
            LookbackCorrLogitPropagation(
                self.graph_spec["alpha_mode"],
                self.graph_spec["iterations"],
                self.graph_spec.get("corr_source", "ret1"),
                self.graph_spec.get("p_mode", "corr"),
            )
            if self.graph_spec is not None
            else None
        )

    def forward(self, x: torch.Tensor, ret: torch.Tensor = None) -> torch.Tensor:
        bsz, length, assets, _features = x.shape
        z = self.input_proj(x)
        t_idx = torch.arange(length, device=x.device)
        z = z + self.time_emb(t_idx).view(1, length, 1, self.d_model)
        if self.asset_emb is not None:
            if self.variant == "temporal_asset_emb_perm":
                a_idx = torch.randperm(assets, device=x.device)
            else:
                a_idx = torch.arange(assets, device=x.device)
            z = z + self.asset_emb(a_idx).view(1, 1, assets, self.d_model)
        if self.ctx_proj is not None:
            avail = (x[..., -1:] > 0.5).float()
            ctx = (x[..., :-1] * avail).sum(dim=2) / avail.sum(dim=2).clamp_min(1.0)
            z = z + self.ctx_proj(ctx).unsqueeze(2)
        z = self.in_dropout(z)

        if self.cross is not None:
            cross_in = z.reshape(bsz * length, assets, self.d_model)
            z = self.cross.forward_tokens(cross_in).reshape(bsz, length, assets, self.d_model)

        temp_in = z.permute(0, 2, 1, 3).reshape(bsz * assets, length, self.d_model)
        z = self.temporal.forward_tokens(temp_in).reshape(bsz, assets, length, self.d_model).permute(0, 2, 1, 3)
        pooled = z[:, -self.recent_k:, :, :].mean(dim=1)
        logits = self.head(self.norm(pooled))
        if self.asset_bias is not None:
            logits = logits + self.asset_bias.unsqueeze(0)
        if self.logit_prop is not None:
            if ret is None:
                raise ValueError(f"{self.variant} requires ret lookback tensor")
            logits = self.logit_prop(logits, ret, x)
        return logits


class MasterModel(nn.Module):
    """Re-implementation of the MASTER architecture (Li et al., AAAI 2024) for the regime task.

    Market-guided gating rescales the stock features with softmax(W m / beta) * F, where m are the market
    features (mkt_*, breadth_*) on the last day; then intra-stock (temporal) attention per stock, inter-stock
    attention across assets at every time step, and attention pooling over time with the last step as query.
    MASTER has no asset embedding; `asset_emb` adds one, `inter` toggles the inter-stock block.
    """

    def __init__(self, num_features: int, num_assets: int, d_model: int, dropout: float,
                 asset_emb: bool, inter: bool, beta: float = 5.0, n_heads: int = 4):
        super().__init__()
        market = [i for i, n in enumerate(FEATURE_COLS) if n.startswith(("mkt_", "breadth_"))]
        stock = [i for i in range(num_features) if i not in market]  # includes the availability mask
        self.register_buffer("market_idx", torch.tensor(market, dtype=torch.long), persistent=False)
        self.register_buffer("stock_idx", torch.tensor(stock, dtype=torch.long), persistent=False)
        self.beta = float(beta)
        self.gate = nn.Linear(len(market), len(stock))
        self.input_proj = nn.Linear(len(stock), d_model)
        self.time_emb = nn.Embedding(LOOKBACK, d_model)
        self.asset_emb = nn.Embedding(num_assets, d_model) if asset_emb else None
        self.intra = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 2, dropout, batch_first=True, norm_first=True)
        self.inter = (nn.TransformerEncoderLayer(d_model, n_heads // 2, d_model * 2, dropout, batch_first=True, norm_first=True)
                      if inter else None)
        self.pool_q = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_outputs())

    def forward(self, x: torch.Tensor, ret: torch.Tensor = None) -> torch.Tensor:
        bsz, length, assets, _ = x.shape
        # market features are the same for every available asset; average over the available ones
        avail = (x[:, -1, :, -1:] > 0.5).float()
        m = (x[:, -1].index_select(-1, self.market_idx) * avail).sum(1) / avail.sum(1).clamp_min(1.0)
        alpha = len(self.stock_idx) * torch.softmax(self.gate(m) / self.beta, dim=-1)
        z = self.input_proj(x.index_select(-1, self.stock_idx) * alpha.view(bsz, 1, 1, -1))
        z = z + self.time_emb(torch.arange(length, device=x.device)).view(1, length, 1, -1)
        if self.asset_emb is not None:
            z = z + self.asset_emb(torch.arange(assets, device=x.device)).view(1, 1, assets, -1)
        d = z.shape[-1]
        z = self.intra(z.permute(0, 2, 1, 3).reshape(bsz * assets, length, d)).reshape(bsz, assets, length, d)
        if self.inter is not None:
            z = self.inter(z.permute(0, 2, 1, 3).reshape(bsz * length, assets, d)).reshape(bsz, length, assets, d)
            z = z.permute(0, 2, 1, 3)
        z = self.norm(z)  # [B, A, T, d]
        score = torch.einsum("batd,bad->bat", z, self.pool_q(z[:, :, -1, :])) / math.sqrt(d)
        pooled = torch.einsum("bat,batd->bad", torch.softmax(score, dim=-1), z)
        return self.head(pooled)


MASTER_VARIANTS = {
    "master": dict(asset_emb=False, inter=True),
    "master_asset_emb": dict(asset_emb=True, inter=True),
    "master_no_inter": dict(asset_emb=False, inter=False),
    "master_no_inter_asset_emb": dict(asset_emb=True, inter=False),
}


def build_model(variant: str, num_features: int, num_assets: int, args) -> nn.Module:
    if variant in MASTER_VARIANTS:
        return MasterModel(num_features, num_assets, int(args.master_d_model), float(args.dropout), **MASTER_VARIANTS[variant])
    return PaperAblationModel(variant, num_features, num_assets, int(args.d_model), float(args.dropout), int(args.recent_k))


def class_weights_from_train(train_ds: PanelDataset, device: torch.device) -> torch.Tensor:
    if train_ds.regression:
        return None
    y_parts = [train_ds[i]["y"].numpy() for i in range(len(train_ds))]
    y = np.concatenate(y_parts) if y_parts else np.array([], dtype=np.int64)
    y = y[y >= 0]
    counts = np.bincount(y, minlength=3).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def masked_loss(logits: torch.Tensor, y: torch.Tensor, weights: torch.Tensor, label_smoothing: float) -> torch.Tensor:
    if y.is_floating_point():  # xsret: MSE on the rank-normalized target
        valid = torch.isfinite(y[..., 0])
        if not torch.any(valid):
            return logits.sum() * 0.0
        return F.mse_loss(logits[..., 0][valid], y[..., 0][valid])
    valid = y >= 0
    if not torch.any(valid):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], y[valid], weight=weights, label_smoothing=float(label_smoothing))


def metrics_from_flat(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    return {
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "overall_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "overall_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "overall_weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def ranking_metrics(scores: List[np.ndarray], targets: List[np.ndarray]) -> Dict:
    """Daily cross-sectional IC (Pearson vs. the forward return clipped to +/-50%), RankIC (Spearman), their
    information ratios, and the mean top-minus-bottom-quintile forward return; dates with < 5 assets skipped."""
    ic, ric, spread = [], [], []
    for s, r in zip(scores, targets):
        ok = np.isfinite(s) & np.isfinite(r)
        if ok.sum() < 5:
            continue
        s, r = s[ok], r[ok]
        rc = np.clip(r, -0.5, 0.5)
        if np.std(s) > 0 and np.std(rc) > 0:
            ic.append(float(np.corrcoef(s, rc)[0, 1]))
            ric.append(float(pd.Series(s).corr(pd.Series(r), method="spearman")))
        k = max(1, len(s) // 5)
        order = np.argsort(s)
        spread.append(float(r[order[-k:]].mean() - r[order[:k]].mean()))
    ic, ric = np.array(ic), np.array(ric)
    return {
        "ic": float(ic.mean()) if len(ic) else np.nan,
        "rank_ic": float(ric.mean()) if len(ric) else np.nan,
        "icir": float(ic.mean() / ic.std()) if len(ic) > 1 and ic.std() > 0 else np.nan,
        "rank_icir": float(ric.mean() / ric.std()) if len(ric) > 1 and ric.std() > 0 else np.nan,
        "ls_spread": float(np.mean(spread)) if spread else np.nan,
        "n_days": int(len(ric)),
        "selection_score": float(ric.mean()) if len(ric) else -np.inf,
    }


@torch.no_grad()
def evaluate_regression(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[Dict, pd.DataFrame]:
    model.eval()
    scores, targets, rows = [], [], []
    for batch in loader:
        out = model(batch["x"].to(device), batch["ret"].to(device))[..., 0].float().cpu().numpy()
        y = batch["y"].numpy()
        for i, date in enumerate(batch["date"]):
            valid = np.isfinite(y[i, :, 0])
            s = np.where(valid, out[i], np.nan)
            scores.append(s)
            targets.append(y[i, :, 1])
            rows.append({"date": date, "y_true": json.dumps(np.round(y[i, :, 1], 6).tolist()),
                         "y_pred": json.dumps(np.round(s, 5).tolist())})
    return ranking_metrics(scores, targets), pd.DataFrame(rows)


def factor_baselines(train_ds: PanelDataset, test_ds: PanelDataset) -> Dict:
    """Model-free references on the test windows: 1-day reversal, 20-day momentum, and each asset's mean
    training-period target (an identity-only predictor)."""
    asset_mean = np.nanmean(np.stack([train_ds[i]["y"].numpy()[:, 0] for i in range(len(train_ds))]), axis=0)
    out = {"reversal_1d": ([], []), "momentum_20d": ([], []), "asset_mean_train": ([], [])}
    for i in range(len(test_ds)):
        item = test_ds[i]
        y = item["y"].numpy()
        valid = np.isfinite(y[:, 0])
        ret = item["ret"].numpy()
        for name, s in (("reversal_1d", -ret[-1]), ("momentum_20d", ret[-REGIME_WINDOW:].mean(axis=0)),
                        ("asset_mean_train", asset_mean)):
            out[name][0].append(np.where(valid, s, np.nan))
            out[name][1].append(y[:, 1])
    return {name: ranking_metrics(*v) for name, v in out.items()}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[Dict, pd.DataFrame]:
    if loader.dataset.regression:
        return evaluate_regression(model, loader, device)
    model.eval()
    y_true_all, y_pred_all, rows = [], [], []
    for batch in loader:
        x = batch["x"].to(device)
        ret = batch["ret"].to(device)
        y = batch["y"].to(device)
        logits = model(x, ret)
        pred = logits.argmax(dim=-1)
        prob = torch.softmax(logits.float(), dim=-1)
        valid = y >= 0
        if torch.any(valid):
            y_true_all.append(y[valid].cpu().numpy())
            y_pred_all.append(pred[valid].cpu().numpy())
        for i, date in enumerate(batch["date"]):
            rows.append({
                "date": date,
                "y_true": json.dumps(y[i].cpu().tolist()),
                "y_pred": json.dumps(pred[i].cpu().tolist()),
                "y_prob": json.dumps(np.round(prob[i].cpu().numpy(), 5).tolist()),
            })
    if y_true_all:
        metrics = metrics_from_flat(np.concatenate(y_true_all), np.concatenate(y_pred_all))
    else:
        metrics = {"overall_accuracy": np.nan, "overall_balanced_accuracy": np.nan, "overall_macro_f1": np.nan, "overall_weighted_f1": np.nan, "confusion_matrix": []}
    return metrics, pd.DataFrame(rows)


def train_one_epoch(model, loader, optimizer, weights, device, label_smoothing: float) -> float:
    model.train()
    total_loss, n_obs = 0.0, 0
    for batch in loader:
        x = batch["x"].to(device)
        ret = batch["ret"].to(device)
        y = batch["y"].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = masked_loss(model(x, ret), y, weights, label_smoothing)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        bsz = int(x.shape[0])
        total_loss += float(loss.detach().cpu()) * bsz
        n_obs += bsz
    return total_loss / max(n_obs, 1)


def summary_path_for(variant: str, year: int, epochs: int) -> Path:
    return RESULT_ROOT / variant / "projected_d32" / f"epoch_{int(epochs)}" / f"test_{int(year)}" / "summary.json"


def prepare_year(year: int, args):
    codes, asset_map, stage1 = build_or_load_stage1(year, args)
    dates, split_arr, x_panel, ret_panel, y_panel, stats = panel_from_asset_map(asset_map, codes, return_stats=True)
    train_ds = PanelDataset(dates, split_arr, x_panel, ret_panel, y_panel, LOOKBACK, "pretrain")
    val_ds = PanelDataset(dates, split_arr, x_panel, ret_panel, y_panel, LOOKBACK, "finetune")
    test_ds = PanelDataset(dates, split_arr, x_panel, ret_panel, y_panel, LOOKBACK, "test")
    if min(len(train_ds), len(val_ds), len(test_ds)) == 0:
        raise RuntimeError(f"empty split for test_{year}")
    xfer = None
    if TRANSFER:
        # unseen assets: labels use their own pretrain thresholds; pooled standardization reuses the
        # training universe's statistics, per-asset standardization uses each asset's own pretrain history
        x_codes, x_map = build_or_load_transfer(year, args)
        xd = panel_from_asset_map(x_map, x_codes, pooled_stats=stats)
        xfer = (x_codes, PanelDataset(*xd, LOOKBACK, "test"))
    return codes, stage1, x_panel, train_ds, val_ds, test_ds, xfer


def run_one(prepared, year: int, variant: str, args) -> Dict:
    path = summary_path_for(variant, year, int(args.epochs))
    if path.exists() and not args.force:
        return json.loads(path.read_text(encoding="utf-8"))

    seed_offsets = {
        "temporal_only": 0,
        "temporal_cross_asset": 17,
        "temporal_cross_asset_no_ae": 17,
        "temporal_graph_lookback_corr": 29,
        "temporal_graph_lookback_corr_r2": 31,
        "temporal_graph_lookback_corr_r3": 37,
        "temporal_graph_lookback_corr_classwise": 41,
        "temporal_graph_lookback_corr_classwise_r2": 43,
        "temporal_graph_lookback_corr_classwise_r3": 47,
        "temporal_graph_lookback_corr_classwise_r4": 53,
        "temporal_graph_lookback_corr_classwise_r5": 59,
        "temporal_graph_lookback_corr_classwise_r6": 61,
        "temporal_graph_feature_corr_classwise_r2": 67,
        "temporal_graph_feature_corr_classwise_r3": 71,
        "temporal_graph_feature_corr_classwise_r4": 73,
        "temporal_graph_feature_corr_classwise_r5": 79,
        "temporal_asset_emb": 83,
        "temporal_asset_emb_ctxmean": 89,
        "temporal_ctxmean": 89,
        "corrvote_no_asset_emb": 97,
        "corrvote_perm_p": 101,
        "corrvote_uniform_p": 103,
        "master": 107,
        "master_asset_emb": 109,
        "master_no_inter": 113,
        "master_no_inter_asset_emb": 127,
        "temporal_asset_bias": 131,
        "temporal_asset_emb_perm": 137,
    }
    seed_offset = seed_offsets[variant]
    seed_everything(int(args.seed) + int(year) * 1009 + seed_offset)
    codes, stage1, x_panel, train_ds, val_ds, test_ds, xfer = prepared
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing silent CPU fallback (pass --cpu to allow)")
    device = torch.device("cpu" if args.cpu else "cuda")
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=int(args.batch_size), shuffle=False, drop_last=False)
    model = build_model(variant, x_panel.shape[-1], len(codes), args).to(device)
    weights = class_weights_from_train(train_ds, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    history = []
    best_state = None
    best_epoch = 0
    best_val_macro = -float("inf")
    stale_epochs = 0
    use_val_selection = not bool(args.no_val_selection)
    for epoch in range(1, int(args.epochs) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, weights, device, float(args.label_smoothing))
        val_metrics, _ = evaluate(model, val_loader, device)
        val_macro = float(val_metrics.get("selection_score", val_metrics.get("overall_macro_f1")))
        improved = val_macro > best_val_macro + float(args.min_delta)
        if improved:
            best_val_macro = val_macro
            best_epoch = int(epoch)
            stale_epochs = 0
            if use_val_selection:
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale_epochs += 1
        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "is_best_val_macro_f1": bool(improved),
            **{f"val_{k}": v for k, v in val_metrics.items() if k != "confusion_matrix"},
        })
        if use_val_selection and stale_epochs >= int(args.patience):
            break

    if use_val_selection and best_state is not None:
        model.load_state_dict(best_state)
        selected_epoch = int(best_epoch)
        model_selection = MODEL_SELECTION_POLICY
    else:
        selected_epoch = int(history[-1]["epoch"]) if history else int(args.epochs)
        model_selection = "final_epoch_no_validation_selection"

    val_metrics, val_pred = evaluate(model, val_loader, device)
    test_metrics, test_pred = evaluate(model, test_loader, device)
    out_dir = ensure_dir(path.parent)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False, encoding="utf-8-sig")
    if args.save_predictions:
        val_pred.to_csv(out_dir / "val_predictions.csv", index=False, encoding="utf-8-sig")
        test_pred.to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    transfer = None
    if xfer is not None:
        x_codes, x_ds = xfer
        x_metrics, x_pred = evaluate(model, DataLoader(x_ds, batch_size=int(args.batch_size), shuffle=False), device)
        transfer = {"codes": list(x_codes), "test_metrics": x_metrics}
        if args.save_predictions:
            x_pred.to_csv(out_dir / "transfer_test_predictions.csv", index=False, encoding="utf-8-sig")

    graph_spec = GRAPH_VARIANT_SPECS.get(variant)
    graph_corr_source = graph_spec.get("corr_source", "ret1") if graph_spec is not None else None
    graph_feature_idx = graph_feature_indices() if graph_corr_source == "feature_nonmarket" else []
    graph_formula = None
    if graph_spec is not None:
        if int(graph_spec["iterations"]) == 1:
            if graph_corr_source == "feature_nonmarket":
                graph_formula = "R=mean_f corr(X_f over lookback), R_aa=0; P_aj=R_aj/(sum_k |R_ak| + eps); M=P@L0; L=L0+tanh(alpha)*M"
            else:
                graph_formula = "R_aa=0; P_aj=R_aj/(sum_k |R_ak| + eps); M=P@L0; L=L0+tanh(alpha)*M"
        else:
            corr_formula = "R=mean_f corr(X_f over lookback)" if graph_corr_source == "feature_nonmarket" else "R=corr(ret1 over lookback)"
            graph_formula = (
                f"{corr_formula}; R_aa=0; P_aj=R_aj/(sum_k |R_ak| + eps); "
                f"for r=1..{int(graph_spec['iterations'])}: M^r=P@L^(r-1), "
                "L^r=L^(r-1)+tanh(alpha)*M^r"
            )

    summary = {
        "market": MARKET,
        "test_year": int(year),
        "task": current_task_name(),
        "variant": variant,
        "config_name": "projected_d32",
        "fixed_epochs": int(args.epochs),
        "max_epochs": int(args.epochs),
        "trained_epochs": int(history[-1]["epoch"]) if history else 0,
        "selected_epoch": int(selected_epoch),
        "model_selection": model_selection,
        "patience": int(args.patience) if use_val_selection else None,
        "min_delta": float(args.min_delta) if use_val_selection else None,
        "description": f"Paper-final independent raw-data {MARKET.upper()} top20 amount ablation. No soft-sector universe and no old stage1 cache.",
        "input_shape": {"lookback": LOOKBACK, "assets": int(len(codes)), "features": int(x_panel.shape[-1])},
        "selected_codes": list(codes),
        "dataset_sizes": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "transfer": transfer,
        "factor_baselines": factor_baselines(train_ds, test_ds) if train_ds.regression else None,
        "model_family": "MASTER" if variant in MASTER_VARIANTS else "MART",
        "master_d_model": int(args.master_d_model) if variant in MASTER_VARIANTS else None,
        "spec": {
            "d_model": int(args.d_model),
            "nhead": 1,
            "dim_feedforward": int(args.d_model) * 2,
            "input_projection": True,
            "time_embedding": True,
            "asset_embedding": uses_asset_embedding(variant),
            "ctx_mean_context": bool(variant in CTXMEAN_VARIANTS),
            "graph_p_mode": graph_spec.get("p_mode", "corr") if graph_spec is not None else None,
            "device": str(device),
            "cross_block": "attn_ffn" if variant in CROSS_VARIANTS else None,
            "graph_block": graph_spec["graph_block"] if graph_spec is not None else None,
            "graph_description": (
                "Same-LOOKBACK signed non-market feature-correlation logit propagation over temporal logits"
                if graph_corr_source == "feature_nonmarket"
                else (
                    "Same-LOOKBACK signed return correlation logit propagation over temporal logits"
                    if graph_spec is not None
                    else None
                )
            ),
            "graph_formula": graph_formula,
            "graph_window": LOOKBACK if graph_spec is not None else None,
            "graph_recent_k": 1 if graph_spec is not None else None,
            "graph_corr_source": graph_corr_source,
            "graph_feature_corr_policy": (
                "mean signed corr over selected temporal input features; excludes availability mask, mkt_* and breadth_*"
                if graph_corr_source == "feature_nonmarket"
                else None
            ),
            "graph_feature_count": len(graph_feature_idx) if graph_corr_source == "feature_nonmarket" else None,
            "graph_feature_cols": [FEATURE_COLS[i] for i in graph_feature_idx] if graph_corr_source == "feature_nonmarket" else None,
            "graph_alpha_mode": graph_spec["alpha_mode"] if graph_spec is not None else None,
            "graph_iterations": int(graph_spec["iterations"]) if graph_spec is not None else None,
            "graph_learned_parameters": graph_spec["learned_parameters"] if graph_spec is not None else None,
            "graph_training_note": (
                "Graph-side learnable parameters are alpha only, but the temporal encoder/head and alpha are trained end-to-end."
                if graph_spec is not None
                else None
            ),
            "graph_no_cluster_topk_gate": bool(graph_spec is not None),
            "temporal_variant": "attn_ffn",
            "pooling": "last5_mean",
            "selection": "pretrain_avg_amount_top20",
            "selection_metric": "amount",
            "eligibility_policy": ELIGIBILITY_POLICY,
            "eligibility_window_end": "finetune_end",
            "model_selection": model_selection,
            "max_epochs": int(args.epochs),
            "trained_epochs": int(history[-1]["epoch"]) if history else 0,
            "selected_epoch": int(selected_epoch),
            "feature_set_name": FEATURE_SET_NAME,
            "feature_policy": FEATURE_POLICY,
            "replaced_old_features": REPLACED_OLD_FEATURES,
            "raw_data_dir": str(RAW_DATA_DIR),
            "local_stage1_dir": str(stage1),
            "old_stage1_cache_used": False,
            "soft_sector_universe_used": False,
            "raw_universe_all": False,
            "feature_cols": FEATURE_COLS,
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "label_smoothing": float(args.label_smoothing),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
        },
    }
    save_json(summary, path)
    score = (f"rank_ic={test_metrics['rank_ic']:.6f} ic={test_metrics['ic']:.6f}" if train_ds.regression else
             f"macro={test_metrics['overall_macro_f1']:.6f} acc={test_metrics['overall_accuracy']:.6f}")
    print(
        f"[paper-final-top20-amount] {MARKET} seed={args.seed} device={device} {variant} test_{year} {score} "
        f"selected_epoch={selected_epoch} trained_epochs={history[-1]['epoch'] if history else 0}",
        flush=True,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def flatten(row: Dict) -> Dict:
    graph_variant_spec = GRAPH_VARIANT_SPECS.get(row["variant"])
    graph_alpha_mode = row["spec"].get("graph_alpha_mode")
    graph_iterations = row["spec"].get("graph_iterations")
    graph_corr_source = row["spec"].get("graph_corr_source")
    if graph_variant_spec is not None:
        graph_alpha_mode = graph_alpha_mode or graph_variant_spec["alpha_mode"]
        graph_iterations = graph_iterations or graph_variant_spec["iterations"]
        graph_corr_source = graph_corr_source or graph_variant_spec.get("corr_source", "ret1")
    return {
        "market": row["market"],
        "test_year": int(row["test_year"]),
        "task": row["task"],
        "variant": row["variant"],
        "config_name": row["config_name"],
        "fixed_epochs": int(row["fixed_epochs"]),
        "max_epochs": int(row.get("max_epochs", row["fixed_epochs"])),
        "trained_epochs": int(row.get("trained_epochs", row["fixed_epochs"])),
        "selected_epoch": int(row.get("selected_epoch", row.get("trained_epochs", row["fixed_epochs"]))),
        "model_selection": row.get("model_selection", "unknown"),
        "n_assets": int(row["input_shape"]["assets"]),
        "d_model": int(row["spec"]["d_model"]),
        "asset_embedding": bool(row["spec"]["asset_embedding"]),
        "cross_block": row["spec"]["cross_block"] or "",
        "graph_block": row["spec"].get("graph_block") or "",
        "graph_corr_source": graph_corr_source or "",
        "graph_feature_count": row["spec"].get("graph_feature_count") or "",
        "graph_alpha_mode": graph_alpha_mode or "",
        "graph_iterations": graph_iterations or "",
        "test_macro_f1": float(row["test_metrics"]["overall_macro_f1"]),
        "test_accuracy": float(row["test_metrics"]["overall_accuracy"]),
        "test_balanced_accuracy": float(row["test_metrics"]["overall_balanced_accuracy"]),
        "test_weighted_f1": float(row["test_metrics"]["overall_weighted_f1"]),
        "val_macro_f1": float(row["val_metrics"]["overall_macro_f1"]),
        "train_windows": int(row["dataset_sizes"]["train"]),
        "val_windows": int(row["dataset_sizes"]["val"]),
        "test_windows": int(row["dataset_sizes"]["test"]),
        "selected_codes": ",".join(row["selected_codes"]),
    }


def collect_rows() -> List[Dict]:
    rows = []
    for path in RESULT_ROOT.glob("*/*/epoch_*/test_*/summary.json"):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def write_outputs(print_table: bool = True) -> None:
    if is_regression():  # the classification summary tables do not apply; analysis reads summary.json directly
        return
    rows = collect_rows()
    if not rows:
        return
    yearly = pd.DataFrame([flatten(r) for r in rows]).drop_duplicates(["variant", "fixed_epochs", "test_year"], keep="last")
    yearly = yearly.sort_values(["variant", "fixed_epochs", "test_year"]).reset_index(drop=True)
    summary = (
        yearly.groupby(
            [
                "variant",
                "config_name",
                "fixed_epochs",
                "n_assets",
                "d_model",
                "asset_embedding",
                "cross_block",
                "graph_block",
                "graph_corr_source",
                "graph_feature_count",
                "graph_alpha_mode",
                "graph_iterations",
            ],
            as_index=False,
        )
        .agg(
            n_windows=("test_year", "nunique"),
            mean_macro_f1=("test_macro_f1", "mean"),
            mean_accuracy=("test_accuracy", "mean"),
            mean_balanced_accuracy=("test_balanced_accuracy", "mean"),
            mean_weighted_f1=("test_weighted_f1", "mean"),
            mean_val_macro_f1=("val_macro_f1", "mean"),
            mean_selected_epoch=("selected_epoch", "mean"),
            mean_trained_epochs=("trained_epochs", "mean"),
            first_year=("test_year", "min"),
            last_year=("test_year", "max"),
        )
        .sort_values("variant")
        .reset_index(drop=True)
    )
    pivot = yearly.pivot_table(index=["fixed_epochs", "test_year"], columns="variant", values="test_macro_f1", aggfunc="first").reset_index()
    if {"temporal_cross_asset", "temporal_only"}.issubset(pivot.columns):
        pivot["delta_cross_minus_temporal"] = pivot["temporal_cross_asset"] - pivot["temporal_only"]
    if {"temporal_graph_lookback_corr", "temporal_only"}.issubset(pivot.columns):
        pivot["delta_graph_minus_temporal"] = pivot["temporal_graph_lookback_corr"] - pivot["temporal_only"]
    for variant in GRAPH_VARIANT_SPECS:
        if {variant, "temporal_only"}.issubset(pivot.columns):
            pivot[f"delta_{variant}_minus_temporal"] = pivot[variant] - pivot["temporal_only"]

    ensure_dir(RESULT_ROOT)
    yearly.to_csv(RESULT_ROOT / "top20_amount_ablation_yearly.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(RESULT_ROOT / "top20_amount_ablation_summary.csv", index=False, encoding="utf-8-sig")
    pivot.to_csv(RESULT_ROOT / "top20_amount_ablation_comparison.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# Paper-final top20 amount ablation",
        "",
        f"Independent run from raw {MARKET.upper()} OHLCV. Soft-sector universe and old stage1 cache are not used.",
        "",
        "```",
        summary.to_string(index=False),
        "```",
        "",
    ]
    (RESULT_ROOT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    if print_table:
        print("")
        print(summary.to_string(index=False))
        if len(pivot) > 0:
            print("")
            print(pivot.tail().to_string(index=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--market", choices=sorted(MARKET_CONFIGS), default=MARKET)
    p.add_argument("--years", nargs="+", type=int, default=None)
    p.add_argument(
        "--variants",
        nargs="+",
        default=MAIN_PAPER_VARIANTS,
        choices=VALID_VARIANTS,
    )
    p.add_argument("--d_model", type=int, default=32)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--recent_k", type=int, default=5)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight_decay", type=float, default=1.0e-4)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--min_delta", type=float, default=0.0)
    p.add_argument("--no_val_selection", action="store_true")
    p.add_argument("--min_rows", type=int, default=1200)
    p.add_argument("--min_pretrain_rows", type=int, default=300)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--force", action="store_true")
    p.add_argument("--rebuild_data", action="store_true")
    p.add_argument("--save_predictions", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--data_only", action="store_true")
    p.add_argument("--seed_dir", action="store_true", help="write results under <result_root>_seed<seed>")
    p.add_argument("--e1", action="store_true", help="run the paper-2 E1 decomposition variant set")
    p.add_argument("--horizon", type=int, default=1, help="label horizon H; task is (20-H, H)")
    p.add_argument("--label_mode", choices=["per_asset", "pooled", "xsret"], default="per_asset")
    p.add_argument("--std_mode", choices=["pooled", "per_asset"], default="pooled")
    p.add_argument("--exclude_codes", nargs="+", default=[], help="codes dropped before top-20 selection")
    p.add_argument("--top_n", type=int, default=20, help="universe size")
    p.add_argument("--transfer", action="store_true", help="also test on the next top_n (unseen) assets")
    p.add_argument("--universe", choices=["amount", "corr", "leadlag", "random"], default="amount",
                   help="asset selection: most traded, or a relation-rich subset of the 100 most traded")
    p.add_argument("--lookback", type=int, default=60, help="input sequence length in days")
    p.add_argument("--master_d_model", type=int, default=64)
    p.add_argument("--tag", default="", help="extra result-root suffix, e.g. a hyperparameter setting")
    args = p.parse_args()
    if args.e1:
        args.variants = list(PAPER2_E1_VARIANTS)
    return args


def main() -> int:
    args = parse_args()
    configure_market(args.market)
    set_protocol(args.horizon, args.label_mode, args.std_mode, args.exclude_codes, args.top_n, args.transfer,
                 args.universe)
    global RESULT_ROOT, LOOKBACK
    LOOKBACK = int(args.lookback)
    RESULT_ROOT = RESULT_ROOT.parent / RESULT_ROOT.name.replace("_19_1", protocol_suffix())
    if args.tag:
        RESULT_ROOT = RESULT_ROOT.parent / f"{RESULT_ROOT.name}_{args.tag}"
    if args.seed_dir:
        # Paper 2 multi-seed runs: keep each seed's summaries apart.
        RESULT_ROOT = RESULT_ROOT.parent / f"{RESULT_ROOT.name}_seed{int(args.seed)}"
    years = args.years or list(range(FIRST_YEAR, LAST_YEAR + 1))
    planned = []
    for year in years:
        for variant in args.variants:
            if args.force or not summary_path_for(variant, int(year), int(args.epochs)).exists():
                planned.append((variant, int(year)))
    ensure_dir(RESULT_ROOT)
    save_json({"args": vars(args), "planned_runs": planned, "raw_data_dir": str(RAW_DATA_DIR), "data_root": str(DATA_ROOT)}, RESULT_ROOT / "latest_run_config.json")
    print(f"[PLAN] market={MARKET} raw={RAW_DATA_DIR} local_data={DATA_ROOT} runs={len(planned)}", flush=True)
    if args.dry_run:
        for item in planned:
            print(" ", item, flush=True)
        return 0

    prepared_by_year = {}
    for year in years:
        if year not in prepared_by_year:
            prepared_by_year[year] = prepare_year(int(year), args)
        if args.data_only:
            codes = prepared_by_year[year][0]
            print(f"[DATA] test_{year} selected_top20={','.join(codes)}", flush=True)
            continue
        for variant in args.variants:
            if args.force or not summary_path_for(variant, int(year), int(args.epochs)).exists():
                run_one(prepared_by_year[year], int(year), variant, args)
                write_outputs(print_table=False)
    if not args.data_only:
        write_outputs(print_table=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
