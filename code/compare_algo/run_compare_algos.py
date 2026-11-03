#!/usr/bin/env python3
"""Run paper comparison algorithms on the paper_clean soft-sector panel.

This script is deliberately self-contained.  It reads only stage1 asset tables
and selected-code CSVs produced by paper_clean, then writes fresh comparison
outputs under compare_algo/results by default.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


HERE = Path(__file__).resolve().parent
OFFICIAL_ROOT = HERE / "official_baselines"
RESULT_ROOT = HERE / "results" / "model_comparison"

SEED = 42
LOOKBACK = 60
DEFAULT_EPOCHS = 16
DEFAULT_BATCH_SIZE = 256
DEFAULT_PANEL_BATCH_SIZE = 16
DEFAULT_LR = 1e-3
DEFAULT_PATIENCE = 4

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

MARKET_YEARS = {
    "kospi": (2006, 2025),
    "nasdaq": (2014, 2025),
}

AVAILABLE_MODELS = [
    "logreg",
    "lstm",
    "xgboost",
    "lightgbm",
    "tcn",
    "dlinear",
    "markov",
    "itransformer",
    "patchtst",
    "crossformer",
    "master",
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    if len(y_true) == 0:
        return {
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "macro_f1": float("nan"),
            "weighted_f1": float("nan"),
            "confusion_matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        }
    labels = [0, 1, 2]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def parse_code(x) -> str:
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if s.isdigit() and len(s) <= 6:
        return s.zfill(6)
    return s


def market_roots(paper_clean_root: Path, market: str, args: argparse.Namespace) -> Tuple[Path, Path]:
    data_root = Path(args.data_root) if args.data_root else paper_clean_root
    if market == "kospi":
        rep_root = Path(args.kospi_reps_root) if args.kospi_reps_root else (
            data_root / "exp_after_defense" / "pca30_gmm_kospi_only_soft_sector_universe_mart"
        )
        stage_root = Path(args.kospi_stage_root) if args.kospi_stage_root else rep_root
        return rep_root, stage_root
    if market == "nasdaq":
        rep_root = Path(args.nasdaq_reps_root) if args.nasdaq_reps_root else (
            data_root / "exp_after_defense" / "pca30_gmm_nasdaq_common_stock_soft_sector_universe_mart"
        )
        stage_root = Path(args.nasdaq_stage_root) if args.nasdaq_stage_root else rep_root
        return rep_root, stage_root
    raise ValueError(market)


def load_selected_codes(rep_root: Path, market: str, year: int) -> List[str]:
    path = rep_root / market / f"test_{int(year)}" / "selected_pca30_gmm20_representatives.csv"
    if not path.exists():
        top_amount_path = rep_root / "selected_top20" / f"test_{int(year)}_top20_amount.csv"
        if top_amount_path.exists():
            path = top_amount_path
    if not path.exists():
        raise FileNotFoundError(f"Missing selected-code CSV: {path}")
    df = pd.read_csv(path)
    code_col = "code" if "code" in df.columns else df.columns[0]
    return [parse_code(x) for x in df[code_col].tolist()]


def load_asset_table(stage_root: Path, market: str, year: int, code: str) -> pd.DataFrame:
    base = stage_root / market / f"test_{int(year)}" / "stage1"
    path = base / code / "asset_tables" / f"{code}.csv"
    if not path.exists():
        final2_path = stage_root / f"test_{int(year)}" / "stage1" / code / "asset_tables" / f"{code}.csv"
        if final2_path.exists():
            path = final2_path
    if not path.exists():
        # Some older caches store Nasdaq stage1 under a shared stage1_cache.
        alt = stage_root / market / f"test_{int(year)}" / code / "asset_tables" / f"{code}.csv"
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"Missing stage1 asset table: {path}")
    df = pd.read_csv(path)
    df["date"] = df["date"].astype(str)
    return df


def load_soft_assets(market: str, year: int, args: argparse.Namespace) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
    paper_clean_root = Path(args.paper_clean_root).resolve()
    rep_root, stage_root = market_roots(paper_clean_root, market, args)
    codes = load_selected_codes(rep_root, market, year)
    asset_map = {code: load_asset_table(stage_root, market, year, code) for code in codes}
    return codes, asset_map


def standardize_panel_with_availability(x_panel: np.ndarray, split_arr: np.ndarray) -> np.ndarray:
    out = x_panel.copy()
    train_mask = np.asarray(split_arr).astype(str) == "pretrain"
    if train_mask.sum() == 0:
        raise ValueError("No pretrain rows for standardization.")
    feat_dim = out.shape[-1] - 1
    avail = out[..., -1] > 0.5
    for f_idx in range(feat_dim):
        vals = out[train_mask, :, f_idx][avail[train_mask]]
        if vals.size == 0:
            mu, sd = 0.0, 1.0
        else:
            mu = float(vals.mean())
            sd = float(vals.std())
            if sd < 1e-8:
                sd = 1.0
        out[..., f_idx] = np.where(avail, (out[..., f_idx] - mu) / sd, 0.0)
    return out.astype(np.float32)


def panel_for_task(
    asset_map: Dict[str, pd.DataFrame],
    codes: Sequence[str],
    regime_window: int,
    horizon: int,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    all_dates = set()
    split_map: Dict[str, str] = {}
    for code in codes:
        df = asset_map[code]
        all_dates.update(df["date"].astype(str).tolist())
        for _, row in df[["date", "split"]].dropna().iterrows():
            date = str(row["date"])
            split = str(row["split"])
            prev = split_map.get(date)
            if prev is None:
                split_map[date] = split
            elif prev != split:
                raise RuntimeError(f"split mismatch date={date} prev={prev} current={split} code={code}")

    dates = sorted(all_dates)
    date_index = {d: i for i, d in enumerate(dates)}
    base_dates = pd.DataFrame({"date": dates})
    t_size, a_size = len(dates), len(codes)
    x_panel = np.zeros((t_size, a_size, len(FEATURE_COLS) + 1), dtype=np.float32)
    y_panel = np.full((t_size, a_size), -1, dtype=np.int64)
    y_cur_panel = np.full((t_size, a_size), -1, dtype=np.int64)
    split_arr = np.array([split_map.get(d, "unknown") for d in dates], dtype=object)

    for a_idx, code in enumerate(codes):
        df = asset_map[code].copy().sort_values("date").reset_index(drop=True)
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{code} missing feature columns: {missing[:8]}")
        score = pd.to_numeric(df["ret1"], errors="coerce").rolling(int(regime_window)).mean()
        pretrain_score = score[df["split"].astype(str) == "pretrain"].dropna()
        if pretrain_score.empty:
            raise RuntimeError(f"empty pretrain regime score: {code}")
        q33 = float(pretrain_score.quantile(0.3333))
        q66 = float(pretrain_score.quantile(0.6667))

        regime = pd.Series(np.nan, index=df.index, dtype=float)
        valid_score = score.notna()
        regime.loc[valid_score & (score <= q33)] = 0
        regime.loc[valid_score & (score > q33) & (score <= q66)] = 1
        regime.loc[valid_score & (score > q66)] = 2
        target = regime.shift(-int(horizon))
        future_split = df["split"].shift(-int(horizon))
        target[(future_split.isna()) | (future_split != df["split"])] = np.nan

        for idx, row in df.iterrows():
            pos = date_index.get(str(row["date"]))
            if pos is None:
                continue
            if not pd.isna(regime.iloc[idx]):
                y_cur_panel[pos, a_idx] = int(regime.iloc[idx])
            if not pd.isna(target.iloc[idx]):
                y_panel[pos, a_idx] = int(target.iloc[idx])

        fdf = base_dates.merge(df[["date"] + FEATURE_COLS], on="date", how="left")
        avail = (~fdf[FEATURE_COLS].isna().all(axis=1)).astype(float).values.reshape(-1, 1)
        x = fdf[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float32)
        x_panel[:, a_idx, :] = np.concatenate([x, avail.astype(np.float32)], axis=1)

    x_panel = standardize_panel_with_availability(x_panel, split_arr)
    return dates, split_arr, x_panel, y_panel, y_cur_panel, FEATURE_COLS + ["avail_mask"]


class SeqDataset(Dataset):
    def __init__(self, samples: List[Tuple[np.ndarray, int]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


class PanelSeqDataset(Dataset):
    def __init__(
        self,
        dates: Sequence[str],
        split_arr: np.ndarray,
        x_panel: np.ndarray,
        y_panel: np.ndarray,
        lookback: int,
        split_name: str,
    ):
        self.dates = list(dates)
        self.x_panel = x_panel
        self.y_panel = y_panel
        self.lookback = int(lookback)
        self.indices = []
        for t in range(self.lookback - 1, len(self.dates)):
            if str(split_arr[t]) == split_name and (y_panel[t] >= 0).any():
                self.indices.append(t)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        t = self.indices[idx]
        s = t - self.lookback + 1
        return {
            "x": torch.tensor(self.x_panel[s:t + 1], dtype=torch.float32),
            "y": torch.tensor(self.y_panel[t], dtype=torch.long),
            "date": self.dates[t],
        }


def build_daily_context(x_panel: np.ndarray) -> np.ndarray:
    feat = x_panel[..., :-1]
    avail = x_panel[..., -1:] > 0.5
    cnt = np.maximum(avail.sum(axis=1), 1)
    feat_mean = (feat * avail).sum(axis=1) / cnt
    avail_rate = x_panel[..., -1].mean(axis=1, keepdims=True)
    return np.concatenate([feat_mean, avail_rate.astype(np.float32)], axis=1).astype(np.float32)


def build_fair_samples(
    x_panel: np.ndarray,
    y_panel: np.ndarray,
    split_arr: np.ndarray,
    dates: Sequence[str],
    codes: Sequence[str],
    lookback: int,
    asset_onehot: bool = True,
) -> Dict:
    t_size, a_size, feat_dim = x_panel.shape
    context = build_daily_context(x_panel)
    # asset_onehot=False zeroes the identity channels (same input width, no asset identity).
    eye = np.eye(a_size, dtype=np.float32) if asset_onehot else np.zeros((a_size, a_size), dtype=np.float32)
    out = {
        "train": {"flat": [], "seq": []},
        "val": {"flat": [], "seq": []},
        "test": {"flat": [], "seq": [], "flat_meta": [], "seq_meta": []},
        "flat_dim": (feat_dim * 6) + a_size,
        "seq_input_dim": (feat_dim * 2) + a_size,
        "feature_dim": feat_dim,
    }
    split_name_map = {"pretrain": "train", "finetune": "val", "test": "test"}
    for t in range(lookback - 1, t_size):
        split_name = split_name_map.get(str(split_arr[t]))
        if split_name is None:
            continue
        s = t - lookback + 1
        ctx_seq = context[s:t + 1]
        for a_idx in range(a_size):
            y = int(y_panel[t, a_idx])
            if y < 0:
                continue
            own_seq = x_panel[s:t + 1, a_idx, :]
            onehot = eye[a_idx]
            repeated_onehot = np.repeat(onehot[None, :], lookback, axis=0)
            seq_x = np.concatenate([own_seq, ctx_seq, repeated_onehot], axis=1).astype(np.float32)
            flat_cont = np.concatenate(
                [own_seq[-1], own_seq.mean(axis=0), own_seq.std(axis=0), ctx_seq[-1], ctx_seq.mean(axis=0), ctx_seq.std(axis=0)],
                axis=0,
            ).astype(np.float32)
            flat_x = np.concatenate([flat_cont, onehot], axis=0).astype(np.float32)
            if split_name == "test":
                meta = (str(dates[t]), str(codes[a_idx]), y)
                out["test"]["flat_meta"].append(meta)
                out["test"]["seq_meta"].append(meta)
            out[split_name]["flat"].append((flat_x, y))
            out[split_name]["seq"].append((seq_x, y))
    return out


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 3))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.norm(out[:, -1, :]))


class CausalConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.ConstantPad1d((pad, 0), 0.0),
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.ConstantPad1d((pad, 0), 0.0),
            nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.residual(x)


class TCNClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96, num_layers: int = 3, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_ch = input_dim
        for idx in range(num_layers):
            layers.append(CausalConvBlock(in_ch, hidden_dim, kernel_size, 2 ** idx, dropout))
            in_ch = hidden_dim
        self.tcn = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 3))

    def forward(self, x):
        h = self.tcn(x.transpose(1, 2))
        return self.head(self.norm(h[:, :, -1]))


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, x):
        pad = self.kernel_size // 2
        x_pad = torch.cat([x[:, :1, :].repeat(1, pad, 1), x, x[:, -1:, :].repeat(1, pad, 1)], dim=1)
        return self.avg(x_pad.transpose(1, 2)).transpose(1, 2)


class DLinearClassifier(nn.Module):
    def __init__(self, seq_len: int, input_dim: int, proj_len: int = 8, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        self.avg = MovingAverage(kernel_size)
        self.seasonal_linear = nn.Linear(seq_len, proj_len)
        self.trend_linear = nn.Linear(seq_len, proj_len)
        flat_dim = input_dim * proj_len
        self.norm = nn.LayerNorm(flat_dim)
        self.head = nn.Sequential(nn.Linear(flat_dim, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 3))

    def forward(self, x):
        trend = self.avg(x)
        seasonal = x - trend
        mixed = self.seasonal_linear(seasonal.transpose(1, 2)) + self.trend_linear(trend.transpose(1, 2))
        return self.head(self.norm(mixed.flatten(start_dim=1)))


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = criterion(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.item()) * yb.numel()
        total_n += int(yb.numel())
    return total_loss / max(total_n, 1)


@torch.no_grad()
def eval_sequence_model(model, loader, device, return_prob: bool = False):
    model.eval()
    y_true_all, y_pred_all, y_prob_all = [], [], []
    for xb, yb in loader:
        prob = torch.softmax(model(xb.to(device)), dim=-1).cpu().numpy()
        y_true_all.append(yb.numpy())
        y_pred_all.append(prob.argmax(axis=-1))
        y_prob_all.append(prob)
    if not y_true_all:
        empty = (metric_dict(np.array([]), np.array([])), np.array([], dtype=np.int64))
        return (*empty, np.zeros((0, 3), dtype=np.float32)) if return_prob else empty
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    if return_prob:
        return metric_dict(y_true, y_pred), y_pred, np.concatenate(y_prob_all)
    return metric_dict(y_true, y_pred), y_pred


def class_weights(labels: Sequence[int], device) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=3).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_logreg(samples: Dict, out_dir: Path, force: bool) -> Dict:
    summary_path = out_dir / "logreg_summary.json"
    if summary_path.exists() and not force:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    ensure_dir(out_dir)
    train_items, val_items, test_items = samples["train"]["flat"], samples["val"]["flat"], samples["test"]["flat"]
    cont_dim = samples["feature_dim"] * 6
    x_train = np.stack([x for x, _ in train_items])
    y_train = np.asarray([y for _, y in train_items], dtype=np.int64)
    x_val = np.stack([x for x, _ in val_items])
    y_val = np.asarray([y for _, y in val_items], dtype=np.int64)
    x_test = np.stack([x for x, _ in test_items])
    y_test = np.asarray([y for _, y in test_items], dtype=np.int64)
    scaler = StandardScaler().fit(x_train[:, :cont_dim])
    for arr in (x_train, x_val, x_test):
        arr[:, :cont_dim] = scaler.transform(arr[:, :cont_dim])
    # lbfgs is multinomial by default; the multi_class argument was removed in sklearn 1.8
    clf = LogisticRegression(max_iter=3000, solver="lbfgs", n_jobs=1)
    clf.fit(x_train, y_train)
    val_pred = clf.predict(x_val)
    test_pred = clf.predict(x_test)
    pred_df = pd.DataFrame(samples["test"]["flat_meta"], columns=["date", "code", "y_true"])
    pred_df["pred"] = test_pred.astype(int)
    summary = {
        "model": "logreg",
        "input": "own_60d_summary + panel_context_60d_summary + asset_onehot",
        "train_size": len(train_items),
        "val_size": len(val_items),
        "test_size": len(test_items),
        "val_metrics": metric_dict(y_val, val_pred),
        "test_metrics": metric_dict(y_test, test_pred),
    }
    save_json(summary, summary_path)
    pred_df.to_csv(out_dir / "logreg_test_predictions.csv", index=False, encoding="utf-8-sig")
    return summary


def run_tree(samples: Dict, out_dir: Path, model_name: str, builder: Callable, force: bool) -> Dict:
    summary_path = out_dir / f"{model_name}_summary.json"
    if summary_path.exists() and not force:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    ensure_dir(out_dir)
    train_items, val_items, test_items = samples["train"]["flat"], samples["val"]["flat"], samples["test"]["flat"]
    x_train = np.stack([x for x, _ in train_items])
    y_train = np.asarray([y for _, y in train_items], dtype=np.int64)
    x_val = np.stack([x for x, _ in val_items])
    y_val = np.asarray([y for _, y in val_items], dtype=np.int64)
    x_test = np.stack([x for x, _ in test_items])
    y_test = np.asarray([y for _, y in test_items], dtype=np.int64)
    model = builder()
    fit_kwargs = {}
    if model_name == "xgboost":
        fit_kwargs = {"eval_set": [(x_val, y_val)], "verbose": False}
    if model_name == "lightgbm":
        try:
            import lightgbm as lgb
            fit_kwargs = {"eval_set": [(x_val, y_val)], "callbacks": [lgb.early_stopping(30, verbose=False)]}
        except Exception:
            fit_kwargs = {}
    model.fit(x_train, y_train, **fit_kwargs)
    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)
    pred_df = pd.DataFrame(samples["test"]["flat_meta"], columns=["date", "code", "y_true"])
    pred_df["pred"] = np.asarray(test_pred, dtype=np.int64)
    summary = {
        "model": model_name,
        "input": "own_60d_summary + panel_context_60d_summary + asset_onehot",
        "train_size": len(train_items),
        "val_size": len(val_items),
        "test_size": len(test_items),
        "val_metrics": metric_dict(y_val, val_pred),
        "test_metrics": metric_dict(y_test, test_pred),
    }
    save_json(summary, summary_path)
    pred_df.to_csv(out_dir / f"{model_name}_test_predictions.csv", index=False, encoding="utf-8-sig")
    return summary


def run_sequence(samples: Dict, out_dir: Path, model_name: str, builder: Callable[[], nn.Module], args: argparse.Namespace) -> Dict:
    summary_path = out_dir / f"{model_name}_summary.json"
    if summary_path.exists() and not args.force:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    ensure_dir(out_dir)
    seed_everything(args.seed)
    train_items, val_items, test_items = samples["train"]["seq"], samples["val"]["seq"], samples["test"]["seq"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(SeqDataset(train_items), batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(SeqDataset(val_items), batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(SeqDataset(test_items), batch_size=args.batch_size, shuffle=False, drop_last=False)
    model = builder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights([y for _, y in train_items], device), label_smoothing=0.05)
    history, best_state = [], None
    best_epoch, best_val, patience = -1, -1.0, 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _ = eval_sequence_model(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_macro_f1": val_metrics["macro_f1"]})
        if val_metrics["macro_f1"] > best_val:
            best_val = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    val_metrics, _ = eval_sequence_model(model, val_loader, device)
    test_metrics, test_pred, test_prob = eval_sequence_model(model, test_loader, device, return_prob=True)
    pred_df = pd.DataFrame(samples["test"]["seq_meta"], columns=["date", "code", "y_true"])
    pred_df["pred"] = np.asarray(test_pred, dtype=np.int64)
    for k in range(test_prob.shape[1]):
        pred_df[f"prob_{k}"] = np.round(test_prob[:, k], 5)
    summary = {
        "model": model_name,
        "input": "own_60d_sequence + panel_context_60d_sequence + asset_onehot",
        "lookback": args.lookback,
        "train_size": len(train_items),
        "val_size": len(val_items),
        "test_size": len(test_items),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    save_json(history, out_dir / f"{model_name}_history.json")
    save_json(summary, summary_path)
    pred_df.to_csv(out_dir / f"{model_name}_test_predictions.csv", index=False, encoding="utf-8-sig")
    torch.save(model.state_dict(), out_dir / f"{model_name}_best_model.pt")
    return summary


def run_markov(dates, split_arr, y_panel, y_cur_panel, codes: Sequence[str], out_dir: Path, force: bool) -> Dict:
    summary_path = out_dir / "markov_summary.json"
    if summary_path.exists() and not force:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    ensure_dir(out_dir)
    counts = np.ones((3, 3), dtype=np.float64)
    train_mask = np.asarray(split_arr).astype(str) == "pretrain"
    for t in np.where(train_mask)[0]:
        valid = (y_cur_panel[t] >= 0) & (y_panel[t] >= 0)
        for cur, nxt in zip(y_cur_panel[t][valid], y_panel[t][valid]):
            counts[int(cur), int(nxt)] += 1.0
    transition = counts / counts.sum(axis=1, keepdims=True)
    pred_state = transition.argmax(axis=1)
    y_true, y_pred, rows = [], [], []
    test_mask = np.asarray(split_arr).astype(str) == "test"
    for t in np.where(test_mask)[0]:
        valid = (y_cur_panel[t] >= 0) & (y_panel[t] >= 0)
        for a_idx in np.where(valid)[0]:
            cur = int(y_cur_panel[t, a_idx])
            true = int(y_panel[t, a_idx])
            pred = int(pred_state[cur])
            y_true.append(true)
            y_pred.append(pred)
            rows.append({"date": str(dates[t]), "code": str(codes[a_idx]), "y_true": true, "pred": pred})
    summary = {
        "model": "markov",
        "input": "current regime + pooled train transition matrix",
        "train_transition_matrix": transition.tolist(),
        "test_metrics": metric_dict(np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)),
        "test_size": len(y_true),
    }
    save_json(summary, summary_path)
    pd.DataFrame(rows).to_csv(out_dir / "markov_test_predictions.csv", index=False, encoding="utf-8-sig")
    return summary


def prepend_sys_path(path: Path) -> None:
    p = str(path)
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)


def clear_modules(prefixes: Sequence[str]) -> None:
    for name in list(sys.modules.keys()):
        if any(name == p or name.startswith(p + ".") for p in prefixes):
            sys.modules.pop(name, None)


def ensure_reformer_stub() -> None:
    if "reformer_pytorch" in sys.modules:
        return
    stub = types.ModuleType("reformer_pytorch")

    class LSHSelfAttention(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise RuntimeError("reformer_pytorch is not bundled; default iTransformer path should not request it.")

    stub.LSHSelfAttention = LSHSelfAttention
    sys.modules["reformer_pytorch"] = stub


def load_official_model(name: str):
    if name == "itransformer":
        repo = OFFICIAL_ROOT / "iTransformer"
        clear_modules(["model", "layers", "utils"])
        prepend_sys_path(repo)
        ensure_reformer_stub()
        from model.iTransformer import Model
        return Model
    if name == "patchtst":
        repo = OFFICIAL_ROOT / "PatchTST" / "PatchTST_supervised"
        clear_modules(["models", "layers", "utils"])
        prepend_sys_path(repo)
        from models.PatchTST import Model
        return Model
    if name == "crossformer":
        repo = OFFICIAL_ROOT / "Crossformer"
        clear_modules(["cross_models", "cross_exp", "utils", "data"])
        prepend_sys_path(repo)
        from cross_models.cross_former import Crossformer
        return Crossformer
    if name == "master":
        repo = OFFICIAL_ROOT / "MASTER"
        clear_modules(["master", "base_model"])
        prepend_sys_path(repo)
        from master import MASTER
        return MASTER
    raise ValueError(name)


class AssetFeatureHead(nn.Module):
    def __init__(self, n_feat: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(n_feat), nn.Linear(n_feat, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 3))

    def forward(self, x):
        return self.net(x)


class OfficialITransformerRegime(nn.Module):
    def __init__(self, asset_count: int, n_feat: int, args):
        super().__init__()
        model_cls = load_official_model("itransformer")
        cfg = SimpleNamespace(
            seq_len=args.lookback, pred_len=1, output_attention=False, use_norm=True, embed="fixed", freq="d",
            dropout=args.dropout, factor=1, d_model=args.d_model, n_heads=args.n_heads,
            e_layers=args.e_layers, d_ff=args.d_ff, activation="gelu", class_strategy="projection",
        )
        self.asset_count = asset_count
        self.n_feat = n_feat
        self.backbone = model_cls(cfg)
        self.head = AssetFeatureHead(n_feat, args.d_model, args.dropout)

    def forward(self, x):
        bsz, seq_len, assets, feats = x.shape
        out = self.backbone(x.reshape(bsz, seq_len, assets * feats), None, None, None)
        return self.head(out[:, -1, :].reshape(bsz, assets, feats))


class OfficialPatchTSTRegime(nn.Module):
    def __init__(self, asset_count: int, n_feat: int, args):
        super().__init__()
        model_cls = load_official_model("patchtst")
        cfg = SimpleNamespace(
            enc_in=asset_count * n_feat, seq_len=args.lookback, pred_len=1, e_layers=args.e_layers,
            n_heads=args.n_heads, d_model=args.d_model, d_ff=args.d_ff, dropout=args.dropout,
            fc_dropout=args.dropout, head_dropout=args.dropout, individual=False, patch_len=args.patch_len,
            stride=args.patch_stride, padding_patch="end", revin=True, affine=True, subtract_last=False,
            decomposition=False, kernel_size=25,
        )
        self.asset_count = asset_count
        self.n_feat = n_feat
        self.backbone = model_cls(cfg)
        self.head = AssetFeatureHead(n_feat, args.d_model, args.dropout)

    def forward(self, x):
        bsz, seq_len, assets, feats = x.shape
        out = self.backbone(x.reshape(bsz, seq_len, assets * feats))
        return self.head(out[:, -1, :].reshape(bsz, assets, feats))


class OfficialCrossformerRegime(nn.Module):
    def __init__(self, asset_count: int, n_feat: int, args):
        super().__init__()
        model_cls = load_official_model("crossformer")
        self.asset_count = asset_count
        self.n_feat = n_feat
        self.backbone = model_cls(
            data_dim=asset_count * n_feat, in_len=args.lookback, out_len=1, seg_len=args.seg_len,
            win_size=args.win_size, factor=args.router_factor, d_model=args.d_model, d_ff=args.d_ff,
            n_heads=args.n_heads, e_layers=args.e_layers, dropout=args.dropout, baseline=False,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        self.head = AssetFeatureHead(n_feat, args.d_model, args.dropout)

    def forward(self, x):
        bsz, seq_len, assets, feats = x.shape
        out = self.backbone(x.reshape(bsz, seq_len, assets * feats))
        return self.head(out[:, -1, :].reshape(bsz, assets, feats))


class OfficialMASTERRegime(nn.Module):
    def __init__(self, asset_count: int, n_feat: int, feature_cols: Sequence[str], args):
        super().__init__()
        model_cls = load_official_model("master")
        gate_names = {"mkt_ret1", "mkt_ret5", "mkt_ret20", "mkt_vol20", "mkt_close_ma20", "mkt_ma5_ma20", "breadth_up1", "breadth_up5"}
        self.gate_idx = [i for i, c in enumerate(feature_cols) if c in gate_names]
        self.src_idx = [i for i, c in enumerate(feature_cols) if c not in gate_names]
        if not self.gate_idx:
            raise ValueError("MASTER wrapper needs market-guided feature columns.")
        self.backbone = model_cls(
            d_feat=len(self.src_idx),
            d_model=args.d_model,
            t_nhead=args.master_t_nhead,
            s_nhead=args.master_s_nhead,
            T_dropout_rate=args.master_dropout,
            S_dropout_rate=args.master_dropout,
            gate_input_start_index=len(self.src_idx),
            gate_input_end_index=len(self.src_idx) + len(self.gate_idx),
            beta=args.master_beta,
        )
        self.backbone.layers[-1] = nn.Identity()
        self.head = nn.Linear(args.d_model, 3)
        self.asset_count = asset_count
        self.n_feat = n_feat

    def forward(self, x):
        bsz, _seq_len, assets, _feats = x.shape
        x_ordered = torch.cat([x[..., self.src_idx], x[..., self.gate_idx]], dim=-1)
        outs = []
        for b in range(bsz):
            feat = self.backbone(x_ordered[b].permute(1, 0, 2).contiguous())
            outs.append(self.head(feat))
        return torch.stack(outs, dim=0)


@torch.no_grad()
def eval_panel_model(model, loader, device) -> Tuple[Dict, pd.DataFrame]:
    model.eval()
    true_all, pred_all, rows = [], [], []
    dataset = loader.dataset
    codes = getattr(dataset, "codes", None)
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].numpy()
        prob = torch.softmax(model(x), dim=-1).cpu().numpy()
        pred = prob.argmax(axis=-1)
        dates = batch["date"]
        for b_idx in range(pred.shape[0]):
            date = str(dates[b_idx])
            for a_idx in range(pred.shape[1]):
                true = int(y[b_idx, a_idx])
                if true < 0:
                    continue
                code = str(codes[a_idx]) if codes is not None else str(a_idx)
                p = prob[b_idx, a_idx]
                rows.append({"date": date, "code": code, "y_true": true, "pred": int(pred[b_idx, a_idx]),
                             "prob_0": round(float(p[0]), 5), "prob_1": round(float(p[1]), 5), "prob_2": round(float(p[2]), 5)})
                true_all.append(true)
                pred_all.append(int(pred[b_idx, a_idx]))
    return metric_dict(np.asarray(true_all, dtype=np.int64), np.asarray(pred_all, dtype=np.int64)), pd.DataFrame(rows)


def train_panel_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss, total_n = 0.0, 0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        logits = model(x)
        valid = y >= 0
        if valid.sum() == 0:
            continue
        loss = criterion(logits[valid], y[valid])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.item()) * int(valid.sum().item())
        total_n += int(valid.sum().item())
    return total_loss / max(total_n, 1)


def run_official_panel(
    model_name: str,
    dates,
    split_arr,
    x_panel,
    y_panel,
    codes,
    feature_cols,
    out_dir: Path,
    args: argparse.Namespace,
) -> Dict:
    summary_path = out_dir / f"{model_name}_summary.json"
    if summary_path.exists() and not args.force:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    ensure_dir(out_dir)
    seed_everything(args.seed)
    train_ds = PanelSeqDataset(dates, split_arr, x_panel, y_panel, args.lookback, "pretrain")
    val_ds = PanelSeqDataset(dates, split_arr, x_panel, y_panel, args.lookback, "finetune")
    test_ds = PanelSeqDataset(dates, split_arr, x_panel, y_panel, args.lookback, "test")
    train_ds.codes = list(codes)
    val_ds.codes = list(codes)
    test_ds.codes = list(codes)
    train_loader = DataLoader(train_ds, batch_size=args.panel_batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.panel_batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.panel_batch_size, shuffle=False, drop_last=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_feat = x_panel.shape[-1]
    if model_name == "itransformer":
        model = OfficialITransformerRegime(len(codes), n_feat, args)
    elif model_name == "patchtst":
        model = OfficialPatchTSTRegime(len(codes), n_feat, args)
    elif model_name == "crossformer":
        model = OfficialCrossformerRegime(len(codes), n_feat, args)
    elif model_name == "master":
        model = OfficialMASTERRegime(len(codes), n_feat, feature_cols, args)
    else:
        raise ValueError(model_name)
    model = model.to(device)

    train_labels = []
    for idx in train_ds.indices:
        vals = y_panel[idx]
        train_labels.extend([int(v) for v in vals if v >= 0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_labels, device), label_smoothing=0.05)
    history, best_state = [], None
    best_epoch, best_val, patience = -1, -1.0, 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_panel_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _ = eval_panel_model(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_macro_f1": val_metrics["macro_f1"]})
        if val_metrics["macro_f1"] > best_val:
            best_val = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    val_metrics, _ = eval_panel_model(model, val_loader, device)
    test_metrics, pred_df = eval_panel_model(model, test_loader, device)
    summary = {
        "model": model_name,
        "input": "panel sequence [lookback, assets, paper_clean_features+avail_mask]",
        "lookback": args.lookback,
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "test_windows": len(test_ds),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    save_json(history, out_dir / f"{model_name}_history.json")
    save_json(summary, summary_path)
    pred_df.to_csv(out_dir / f"{model_name}_test_predictions.csv", index=False, encoding="utf-8-sig")
    torch.save(model.state_dict(), out_dir / f"{model_name}_best_model.pt")
    return summary


def run_one_window(market: str, year: int, args: argparse.Namespace) -> Dict:
    out_base = ensure_dir(Path(args.out_root) / market / f"test_{int(year)}")
    codes, asset_map = load_soft_assets(market, int(year), args)
    dates, split_arr, x_panel, y_panel, y_cur_panel, feature_cols = panel_for_task(
        asset_map, codes, int(args.regime_window), int(args.horizon)
    )
    samples = build_fair_samples(x_panel, y_panel, split_arr, dates, codes, args.lookback)
    row = {
        "market": market,
        "test_year": int(year),
        "task": f"{int(args.regime_window) - int(args.horizon)},{int(args.horizon)}",
        "regime_window": int(args.regime_window),
        "horizon": int(args.horizon),
        "selected_codes": ",".join(codes),
    }

    for model_name in args.models:
        model_out = out_base / model_name
        if model_name == "logreg":
            summary = run_logreg(samples, model_out, args.force)
        elif model_name == "lstm":
            summary = run_sequence(samples, model_out, "lstm", lambda: LSTMClassifier(samples["seq_input_dim"]), args)
        elif model_name == "xgboost":
            def builder():
                from xgboost import XGBClassifier
                return XGBClassifier(
                    objective="multi:softmax", num_class=3, n_estimators=300, max_depth=4,
                    learning_rate=0.05, subsample=0.85, colsample_bytree=0.85,
                    reg_lambda=1.0, min_child_weight=1.0, tree_method="hist",
                    random_state=args.seed, n_jobs=4,
                )
            summary = run_tree(samples, model_out, "xgboost", builder, args.force)
        elif model_name == "lightgbm":
            def builder():
                from lightgbm import LGBMClassifier
                return LGBMClassifier(
                    objective="multiclass", num_class=3, n_estimators=400, learning_rate=0.05,
                    num_leaves=31, max_depth=-1, min_child_samples=20, subsample=0.85,
                    colsample_bytree=0.85, reg_lambda=1.0, random_state=args.seed,
                    n_jobs=4, verbosity=-1,
                )
            summary = run_tree(samples, model_out, "lightgbm", builder, args.force)
        elif model_name == "tcn":
            summary = run_sequence(samples, model_out, "tcn", lambda: TCNClassifier(samples["seq_input_dim"]), args)
        elif model_name == "dlinear":
            summary = run_sequence(
                samples,
                model_out,
                "dlinear",
                lambda: DLinearClassifier(args.lookback, samples["seq_input_dim"]),
                args,
            )
        elif model_name == "markov":
            summary = run_markov(dates, split_arr, y_panel, y_cur_panel, codes, model_out, args.force)
        elif model_name in {"itransformer", "patchtst", "crossformer", "master"}:
            summary = run_official_panel(model_name, dates, split_arr, x_panel, y_panel, codes, feature_cols, model_out, args)
        else:
            raise ValueError(model_name)
        metrics = summary.get("test_metrics", {})
        row[f"{model_name}_test_accuracy"] = metrics.get("accuracy", np.nan)
        row[f"{model_name}_test_macro_f1"] = metrics.get("macro_f1", np.nan)
        print(f"[DONE] {market} {year} {model_name} macro_f1={row[f'{model_name}_test_macro_f1']}", flush=True)
    save_json(row, out_base / "window_summary.json")
    return row


def write_outputs(rows: List[Dict], out_root: Path) -> None:
    ensure_dir(out_root)
    if not rows:
        save_json([], out_root / "comparison_rows.json")
        return
    df = pd.DataFrame(rows).sort_values(["market", "test_year"]).reset_index(drop=True)
    df.to_csv(out_root / "comparison_yearly.csv", index=False, encoding="utf-8-sig")
    agg_rows = []
    for market, sub in df.groupby("market"):
        row = {"market": market, "n_windows": int(len(sub))}
        for col in df.columns:
            if col.endswith("_test_macro_f1"):
                row[f"mean_{col}"] = float(sub[col].mean())
        agg_rows.append(row)
    pd.DataFrame(agg_rows).to_csv(out_root / "comparison_aggregate.csv", index=False, encoding="utf-8-sig")
    save_json(rows, out_root / "comparison_rows.json")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paper_clean_root", type=str, default=str((HERE.parent / "code" / "paper_clean").resolve()))
    p.add_argument("--data_root", type=str, default=None, help="Override PAPER_CLEAN_DATA_ROOT-style cache root.")
    p.add_argument("--kospi_reps_root", type=str)
    p.add_argument("--kospi_stage_root", type=str)
    p.add_argument("--nasdaq_reps_root", type=str)
    p.add_argument("--nasdaq_stage_root", type=str)
    p.add_argument("--out_root", type=str, default=str(RESULT_ROOT))
    p.add_argument("--markets", nargs="+", choices=sorted(MARKET_YEARS), default=["kospi", "nasdaq"])
    p.add_argument("--years", nargs="+", type=int)
    p.add_argument("--models", nargs="+", choices=AVAILABLE_MODELS, default=["logreg", "lstm", "xgboost", "lightgbm", "tcn", "dlinear", "markov"])
    p.add_argument("--regime_window", type=int, default=20)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--lookback", type=int, default=LOOKBACK)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--panel_batch_size", type=int, default=DEFAULT_PANEL_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--e_layers", type=int, default=2)
    p.add_argument("--d_ff", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patch_len", type=int, default=12)
    p.add_argument("--patch_stride", type=int, default=6)
    p.add_argument("--seg_len", type=int, default=6)
    p.add_argument("--win_size", type=int, default=2)
    p.add_argument("--router_factor", type=int, default=10)
    p.add_argument("--master_t_nhead", type=int, default=4)
    p.add_argument("--master_s_nhead", type=int, default=2)
    p.add_argument("--master_beta", type=float, default=5.0)
    p.add_argument("--master_dropout", type=float, default=0.5)
    p.add_argument("--max_runs", type=int)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    seed_everything(args.seed)
    out_root = Path(args.out_root)
    ensure_dir(out_root)
    rows_path = out_root / "comparison_rows.json"
    rows: List[Dict] = []
    if rows_path.exists() and not args.force:
        rows = json.loads(rows_path.read_text(encoding="utf-8"))
    by_key = {(r["market"], int(r["test_year"])): r for r in rows}
    planned = []
    for market in args.markets:
        first, last = MARKET_YEARS[market]
        years = args.years if args.years else list(range(first, last + 1))
        for year in years:
            key = (market, int(year))
            if key not in by_key or args.force:
                planned.append(key)
    if args.max_runs is not None:
        planned = planned[: int(args.max_runs)]
    save_json({"args": vars(args), "planned": planned}, out_root / "latest_run_config.json")
    print(f"[PLAN] windows={len(planned)} models={args.models}", flush=True)
    if args.dry_run:
        for market, year in planned:
            rep_root, stage_root = market_roots(Path(args.paper_clean_root).resolve(), market, args)
            print(f"  {market} {year} reps={rep_root} stage={stage_root}", flush=True)
        write_outputs([by_key[k] for k in sorted(by_key)], out_root)
        return 0
    for market, year in planned:
        row = run_one_window(market, year, args)
        by_key[(market, year)] = row
        write_outputs([by_key[k] for k in sorted(by_key)], out_root)
    write_outputs([by_key[k] for k in sorted(by_key)], out_root)
    print(f"[DONE] {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
