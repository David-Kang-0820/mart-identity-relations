"""Revision: trainable parameter counts of every compared variant (20 assets; the regime task has 3 outputs, the
ranking task 1). Output: analysis/out/param_counts.csv"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

CODE = Path(__file__).resolve().parents[1] / "code" / "paper_final2"
sys.path.insert(0, str(CODE))
import run_top20_amount_ablation as R  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
VARIANTS = ["temporal_only", "temporal_asset_emb", "temporal_asset_bias", "temporal_cross_asset",
            "temporal_graph_feature_corr_classwise_r3", "corrvote_no_asset_emb", "corrvote_perm_p",
            "master", "master_asset_emb", "master_no_inter", "master_no_inter_asset_emb"]


def count(variant: str, label_mode: str, master_d: int = 64, lookback: int = 60, assets: int = 20) -> int:
    R.set_protocol(1, label_mode, "pooled", (), assets, False)
    R.LOOKBACK = lookback
    args = SimpleNamespace(d_model=32, dropout=0.1, recent_k=5, master_d_model=master_d)
    model = R.build_model(variant, len(R.FEATURE_COLS) + 1, assets, args)  # + availability mask
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    rows = []
    for v in VARIANTS:
        rows.append(dict(variant=v, regime_20=count(v, "per_asset"), rank_50=count(v, "xsret", assets=50)))
    for v in ("master", "master_asset_emb", "master_no_inter"):
        rows.append(dict(variant=f"{v} (d=256, lookback 8)", regime_20=None,
                         rank_50=count(v, "xsret", master_d=256, lookback=8, assets=50)))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "param_counts.csv", index=False)
    print("input features + availability mask:", len(R.FEATURE_COLS) + 1)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
