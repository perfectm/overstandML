"""TradeSteward export -> OO contract translator (spec section 9, Phase 0).

The only TS-specific code in the whole system. Mapped against the first
real export (RUN_G-trades-20260826-171604.csv: 12,142 SPX trades,
2022-05-16 -> 2026-08-10, 79 strategies, 103 columns).

Verified identities from that file:
  Trade P/L = Total Entry Value + Exit Price*100*Quantity - Total Exit Fee
  (i.e. Trade P/L is NET of all fees; Total Entry Value is net of entry fee)
  Entry/Exit Price are per-share net position prices (negative = debit).

Sign/format conventions marked `verify` should be re-checked against a real
Option Omega CSV when Mauro shares one (spec section 9 sanity anchor):
  - Premium sign convention (here: TS Entry Price passthrough, neg = debit)
  - Gap / Movement units (here: percent)
  - Legs string format (here: "QTY YYYY-MM-DD STRIKE C|P STO|BTO PRICE")
  - "Timed Exit" status label passthrough
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .oo_contract import (
    OO_COLUMNS,
    STATUS_BACKTEST_COMPLETED,
    STATUS_EXPIRED,
    STATUS_PROFIT_TARGET,
    STATUS_STOP_LOSS,
    validate_oo,
)

# TS column name -> OO column name (straight renames)
COLUMN_MAP: dict[str, str] = {
    "Entry Date": "Date Opened",
    "Entry Time": "Time Opened",
    "Exit Date": "Date Closed",
    "Exit Time": "Time Closed",
    "Trade P/L": "P/L",
    "Quantity": "No. of Contracts",
    "Buying Power": "Margin Req.",
    "Position Name": "Strategy",
    "Entry Price": "Opening Price",
    "Exit Price": "Closing Price",
    "Total Entry Fee": "Opening Commissions + Fees",
    "Total Exit Fee": "Closing Commissions + Fees",
    "VIX Entry": "Opening VIX",
    "VIX Exit": "Closing VIX",
}

# TS "Exit Reason" -> OO "Reason For Close". SPX is cash-settled: Assigned /
# Exercised only happen at settlement, so they fold into Expired.
STATUS_MAP: dict[str, str] = {
    "Expired": STATUS_EXPIRED,
    "Assigned": STATUS_EXPIRED,
    "Exercised": STATUS_EXPIRED,
    "Profit Target": STATUS_PROFIT_TARGET,
    "Stop Loss (Algo)": STATUS_STOP_LOSS,
    "Timed Exit": "Timed Exit",  # verify against a real OO log
}

_LEG_TYPE = {
    "Long Call": ("BTO", "C"),
    "Short Call": ("STO", "C"),
    "Long Put": ("BTO", "P"),
    "Short Put": ("STO", "P"),
}


def _build_legs(f: pd.DataFrame) -> pd.Series:
    """Assemble an OO-style Legs string from the OptN-* column groups."""
    groups = [
        (f"Opt{i}-Type", f"Opt{i}-Quantity", f"Opt{i}-Strike", f"Opt{i}-Expiration", f"Opt{i}-Entry Price")
        for i in range(1, 5)
        if f"Opt{i}-Type" in f.columns
    ]

    def legs_for(row) -> str:
        legs = []
        for t_col, q_col, s_col, e_col, p_col in groups:
            t = row[t_col]
            if pd.isna(t):
                continue
            side, cp = _LEG_TYPE.get(t, ("?", "?"))
            legs.append(
                f"{int(row[q_col])} {row[e_col]} {row[s_col]:g} {cp} {side} {abs(row[p_col]):g}"
            )
        return " | ".join(legs)

    return f.apply(legs_for, axis=1)


# OO columns derived from (renamed) TS columns
DERIVED = {
    # per-share net entry price; negative = debit. verify sign vs real OO log
    "Premium": lambda f: f["Opening Price"],
    "Avg. Closing Cost": lambda f: f["Closing Price"],
    "Reason For Close": lambda f: (
        f["Exit Reason"].map(STATUS_MAP).fillna(f["Exit Reason"])
        .fillna(STATUS_BACKTEST_COMPLETED)  # open at export time
    ),
    "Legs": _build_legs,
    # percent units. verify vs real OO log
    "Gap": lambda f: (f["Underlying Open"] / f["Underlying Previous Close"] - 1.0) * 100.0,
    "Movement": lambda f: (f["Underlying Close"] / f["Underlying Open"] - 1.0) * 100.0,
    # relative funds (cumulative net P/L); OO's is absolute account equity
    "Funds at Close": lambda f: f["Total Running P/L"],
}


def convert(ts_df: pd.DataFrame) -> pd.DataFrame:
    """Translate one TS export frame into a contract-valid OO frame.

    Drops non-trade rows (TS appends an end-of-file marker row). TS source
    columns not consumed by the mapping ride along after the OO columns.
    """
    ts_df = ts_df[ts_df["Entry Date"].notna()]  # end-of-file marker row
    out = ts_df.rename(columns=COLUMN_MAP)

    for oo_col, fn in DERIVED.items():
        out[oo_col] = fn(out)

    for col in OO_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[OO_COLUMNS + [c for c in out.columns if c not in OO_COLUMNS]]

    return validate_oo(out)


def convert_file(ts_path: str | Path, out_path: str | Path | None = None) -> pd.DataFrame:
    """Convert a TS export CSV; optionally write the OO-format CSV."""
    oo = convert(pd.read_csv(ts_path))
    if out_path is not None:
        oo.to_csv(out_path, index=False)
    return oo


def split_by_strategy(oo_df: pd.DataFrame, directory: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """One OO frame per strategy (spec Phase 0: 'one CSV per strategy');
    writes <slug>.csv files when a directory is given."""
    out = {name: g.reset_index(drop=True) for name, g in oo_df.groupby("Strategy")}
    if directory is not None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, g in out.items():
            slug = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name).strip()
            g.to_csv(directory / f"{slug}.csv", index=False)
    return out


def inspect_headers(ts_path: str | Path) -> None:
    """Print TS headers vs unmapped OO columns — the COLUMN_MAP worksheet."""
    ts_cols = list(pd.read_csv(ts_path, nrows=0).columns)
    mapped_oo = set(COLUMN_MAP.values()) | set(DERIVED)
    print(f"TS columns ({len(ts_cols)}):")
    for c in ts_cols:
        arrow = f" -> {COLUMN_MAP[c]}" if c in COLUMN_MAP else "   (ride-along)"
        print(f"  {c}{arrow}")
    print("\nOO columns not produced:")
    for c in OO_COLUMNS:
        if c not in mapped_oo:
            print(f"  {c}")


if __name__ == "__main__":
    import sys

    inspect_headers(sys.argv[1])
