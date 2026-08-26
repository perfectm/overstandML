"""The Option Omega column contract (spec section 3).

The atomic input to the whole system is an Option Omega backtest/live CSV
trade log — one row per trade. Mauro: "as long as a file has that format
with all OO columns in it then it will work." This module pins that format
down so every producer (the TS adapter, or a real OO export) and every
consumer (portfolio assembly, features, diagnostics) agrees on one schema.

NOTE: OO_COLUMNS below is drafted from the standard Option Omega backtest
log export. Verify against one of Mauro's actual shared CSVs before the
acceptance test (spec section 9 sanity anchor) and adjust — the contract is
whatever HIS files contain, since his tool is the reference consumer.
"""
from __future__ import annotations

import pandas as pd

# Draft OO backtest-log columns, in export order. verify: true (spec-style
# flag — re-check against a reference CSV, names must match exactly).
OO_COLUMNS = [
    "Date Opened",
    "Time Opened",
    "Opening Price",
    "Legs",
    "Premium",
    "Closing Price",
    "Date Closed",
    "Time Closed",
    "Avg. Closing Cost",
    "Reason For Close",
    "P/L",
    "No. of Contracts",
    "Funds at Close",
    "Margin Req.",
    "Strategy",
    "Opening Commissions + Fees",
    "Closing Commissions + Fees",
    "Opening Short/Long Ratio",
    "Closing Short/Long Ratio",
    "Opening VIX",
    "Closing VIX",
    "Gap",
    "Movement",
    "Max Profit",
    "Max Loss",
]

# Columns the downstream pipeline actually depends on today. Keep this the
# minimal honest set — everything else rides along for compatibility with
# the reference tool.
REQUIRED_COLUMNS = [
    "Date Opened",
    "Time Opened",
    "Date Closed",
    "Time Closed",
    "P/L",
    "No. of Contracts",
    "Margin Req.",
    "Strategy",
    "Reason For Close",
]

DATE_COLUMNS = ["Date Opened", "Date Closed"]
NUMERIC_COLUMNS = [
    "Opening Price",
    "Premium",
    "Closing Price",
    "Avg. Closing Cost",
    "P/L",
    "No. of Contracts",
    "Funds at Close",
    "Margin Req.",
    "Opening Commissions + Fees",
    "Closing Commissions + Fees",
    "Opening VIX",
    "Closing VIX",
    "Gap",
    "Movement",
    "Max Profit",
    "Max Loss",
]

# OO status labels ("Reason For Close" values). The TS adapter must emit one
# of these per trade — including "Backtest Completed" for non-0DTE trades
# still open at export time (spec section 3 / section 9 Phase 0).
STATUS_EXPIRED = "Expired"
STATUS_PROFIT_TARGET = "Profit Target"
STATUS_STOP_LOSS = "Stop Loss"
STATUS_BACKTEST_COMPLETED = "Backtest Completed"


class ContractError(ValueError):
    """An input frame does not satisfy the OO column contract."""


def validate_oo(df: pd.DataFrame, *, strict: bool = False) -> pd.DataFrame:
    """Validate a trade-log frame against the OO contract.

    strict=False (default): require REQUIRED_COLUMNS, warn-level tolerance
    for the rest. strict=True: require the full OO_COLUMNS set (order not
    enforced — content is the contract, ordering is cosmetic).

    Returns the frame with date/numeric columns coerced to proper dtypes.
    Raises ContractError with the exact missing-column list otherwise.
    """
    needed = OO_COLUMNS if strict else REQUIRED_COLUMNS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ContractError(
            f"trade log is missing {len(missing)} OO column(s): {missing}"
        )

    out = df.copy()
    for col in DATE_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="raise").dt.normalize()
    for col in NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if out["P/L"].isna().any():
        n = int(out["P/L"].isna().sum())
        raise ContractError(f"{n} row(s) have non-numeric P/L after coercion")
    return out


def empty_oo_frame() -> pd.DataFrame:
    """An empty frame with the full OO column set — the adapter's template."""
    return pd.DataFrame(columns=OO_COLUMNS)
