"""Paper 2 E6: data checks on the top-20 universes.

1. Survivorship (KOSPI/KOSDAQ). The universe is drawn from stock_codes.csv, a current listing snapshot, so
   stocks delisted before the snapshot are never candidates. For every test year we rank ALL raw files that pass
   the runner's eligibility rule by training-period average trading value and count the raw files missing from
   the snapshot that would rank above the 20th selected stock. Files whose data end before 2026 are treated as
   delisted; files that continue but are missing from the snapshot are other instruments (ETFs, ETNs, ...).
   Current market labels are also compared with the market a stock traded on (not available in the data), so we
   only report the number of selected stocks whose label could have changed (listed in both rankings).
2. Nasdaq: share of raw files whose data end before 2025 (does the archive contain delisted names?).
3. Duplicate share classes: pairs within a selected top-20 whose training-period daily returns correlate above 0.95.

Outputs: analysis/out/e6_survivorship.csv, e6_duplicates.csv; summary on stdout.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "paper2_mart" / "code" / "paper_final2"
os.environ.setdefault("MART_RAW_KRX_DIR", str(ROOT / "code" / "data" / "inc_all_1d"))
os.environ.setdefault("MART_RAW_NASDAQ_DIR", str(ROOT / "code" / "data" / "nasdaq_data"))
sys.path.insert(0, str(CODE))
import run_top20_amount_ablation as R  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
YEARS = {"kospi": range(2006, 2027), "kosdaq": range(2006, 2027), "nasdaq": range(2014, 2027)}
MIN_ROWS, MIN_PRE = 1200, 300
DELISTED_BEFORE = pd.Timestamp("2026-01-01")


def load_all(raw_dir: Path) -> dict[str, pd.DataFrame]:
    R.RAW_DATA_DIR = raw_dir
    out = {}
    for p in sorted(raw_dir.glob("*.csv")):
        try:
            out[p.stem] = R.load_one_raw(p.stem)
        except Exception:
            continue
    return out


def ranking(raw: dict[str, pd.DataFrame], year: int) -> pd.DataFrame:
    start, pre_end, ft_end, _ = R.split_bounds(year)
    rows = []
    for code, df in raw.items():
        if not R.eligible_raw(df, start, pre_end, ft_end, MIN_ROWS, MIN_PRE):
            continue
        amt = df.loc[(df.date >= start) & (df.date <= pre_end), "amount"].replace([np.inf, -np.inf], np.nan).mean()
        if np.isfinite(amt) and amt > 0:
            rows.append(dict(code=code, amount=amt, last_date=df.date.max()))
    return pd.DataFrame(rows).sort_values("amount", ascending=False).reset_index(drop=True)


def selected(market: str, year: int) -> list[str]:
    f = CODE / "data" / f"{market}_top20_amount" / "selected_top20" / f"test_{year}_top20_amount.csv"
    s = pd.read_csv(f, dtype={"code": str}).code
    return [R.zfill6(c) for c in s] if market != "nasdaq" else s.tolist()


def duplicates(market: str, year: int, raw: dict[str, pd.DataFrame]) -> list[dict]:
    start, pre_end, _, _ = R.split_bounds(year)
    codes = selected(market, year)
    ret = pd.DataFrame({c: raw[c].set_index("date").close.loc[start:pre_end].pct_change() for c in codes})
    corr = ret.corr(min_periods=250)
    rows = []
    for i, a in enumerate(codes):
        for b in codes[i + 1:]:
            if corr.loc[a, b] > 0.95:
                rows.append(dict(market=market, year=year, a=a, b=b, corr=corr.loc[a, b]))
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    snap = pd.read_csv(CODE / "stock_codes.csv", dtype={"code": str})
    snap["code"] = snap.code.map(R.zfill6)
    snap_market = dict(zip(snap.code, snap.market.str.upper()))

    kr = load_all(Path(os.environ["MART_RAW_KRX_DIR"]))
    us = load_all(Path(os.environ["MART_RAW_NASDAQ_DIR"]))
    print(f"raw files: korea {len(kr)} (in snapshot {sum(c in snap_market for c in kr)}), nasdaq {len(us)}")
    last_us = pd.Series({c: d.date.max() for c, d in us.items()})
    print(f"nasdaq files ending before 2025: {(last_us < pd.Timestamp('2025-01-01')).sum()} of {len(last_us)}")

    surv, dup = [], []
    for year in YEARS["kospi"]:
        rk = ranking(kr, year)
        rk["snapshot"] = rk.code.map(snap_market).fillna("ABSENT")
        rk["delisted"] = rk.last_date < DELISTED_BEFORE
        for market in ("kospi", "kosdaq"):
            sel = selected(market, year)
            cutoff = rk.loc[rk.code.isin(sel), "amount"].min()
            above = rk[(rk.amount >= cutoff) & ~rk.code.isin(sel)]
            absent = above[above.snapshot == "ABSENT"]
            surv.append(dict(market=market, year=year, cutoff_amount=cutoff,
                             absent_delisted_above=int(absent.delisted.sum()),
                             absent_listed_above=int((~absent.delisted).sum()),
                             absent_delisted_codes=" ".join(absent[absent.delisted].code),
                             selected_delisted_later=int(rk[rk.code.isin(sel)].delisted.sum())))
            dup += duplicates(market, year, kr)
    for year in YEARS["nasdaq"]:
        dup += duplicates("nasdaq", year, us)

    s = pd.DataFrame(surv)
    s.to_csv(OUT / "e6_survivorship.csv", index=False)
    d = pd.DataFrame(dup, columns=["market", "year", "a", "b", "corr"])
    d.to_csv(OUT / "e6_duplicates.csv", index=False)
    pd.set_option("display.width", 220)
    print("\nSURVIVORSHIP (delisted raw files missing from the snapshot that out-trade the 20th selected stock)")
    print(s.drop(columns="cutoff_amount").to_string(index=False))
    print("\nDUPLICATE PAIRS (training-period return corr > 0.95)")
    print(d.to_string(index=False) if len(d) else "none")


if __name__ == "__main__":
    main()
