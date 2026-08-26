"""TradeSteward export -> OO contract translator (spec section 9, Phase 0).

The only TS-specific code in the whole system. Cotton Mike: "just ordering
and labels" — TS exports carry the same content as OO logs under different
column names/order, so the adapter is a rename + reorder + status-label
mapping, plus small derivations for anything OO carries that TS doesn't
export directly.

COLUMN_MAP below is a placeholder: it cannot be filled in until the first
real TS export lands (James is uploading them to the shared Google Drive).
When it does:
  1. run `python -m mlcpo.data.ts_to_oo <ts_export.csv>` to print the TS
     headers next to the unmapped OO columns,
  2. fill COLUMN_MAP / STATUS_MAP / DERIVED,
  3. re-run — convert() validates its own output against the contract.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .oo_contract import (
    OO_COLUMNS,
    STATUS_BACKTEST_COMPLETED,
    validate_oo,
)

# TS column name -> OO column name. FILL IN from the first real TS export.
COLUMN_MAP: dict[str, str] = {
    # e.g. "OpenDate": "Date Opened",
    #      "NetPnL":   "P/L",
}

# TS trade-status value -> OO "Reason For Close" label. FILL IN likewise.
# Open non-0DTE positions must map to STATUS_BACKTEST_COMPLETED (spec s3).
STATUS_MAP: dict[str, str] = {
    # e.g. "TargetHit": "Profit Target",
    #      "Stopped":   "Stop Loss",
    #      "Open":      STATUS_BACKTEST_COMPLETED,
}

# OO columns TS has no direct counterpart for, computed from mapped columns
# after the rename. Each value is a function frame -> Series.
DERIVED: dict[str, callable] = {
    # e.g. "Premium": lambda f: f["Opening Price"] * f["No. of Contracts"] * 100,
}


def convert(ts_df: pd.DataFrame) -> pd.DataFrame:
    """Translate one TS export frame into a contract-valid OO frame."""
    if not COLUMN_MAP:
        raise NotImplementedError(
            "COLUMN_MAP is empty — fill it in from a real TS export "
            "(see module docstring)."
        )
    out = ts_df.rename(columns=COLUMN_MAP)

    ts_status_col = next(
        (ts for ts, oo in COLUMN_MAP.items() if oo == "Reason For Close"), None
    )
    if STATUS_MAP and "Reason For Close" in out.columns:
        out["Reason For Close"] = (
            out["Reason For Close"].map(STATUS_MAP).fillna(out["Reason For Close"])
        )

    for oo_col, fn in DERIVED.items():
        out[oo_col] = fn(out)

    # Full OO shape: missing ride-along columns become NA, order normalized.
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


def inspect_headers(ts_path: str | Path) -> None:
    """Print TS headers vs unmapped OO columns — the COLUMN_MAP worksheet."""
    ts_cols = list(pd.read_csv(ts_path, nrows=0).columns)
    mapped_ts = set(COLUMN_MAP)
    mapped_oo = set(COLUMN_MAP.values()) | set(DERIVED)
    print(f"TS columns ({len(ts_cols)}):")
    for c in ts_cols:
        arrow = f" -> {COLUMN_MAP[c]}" if c in mapped_ts else "   (unmapped)"
        print(f"  {c}{arrow}")
    print(f"\nOO columns still unmapped ({len(OO_COLUMNS) - len(mapped_oo & set(OO_COLUMNS))}):")
    for c in OO_COLUMNS:
        if c not in mapped_oo:
            print(f"  {c}")


if __name__ == "__main__":
    import sys

    inspect_headers(sys.argv[1])
