"""Child descriptor features (spec section 4, open question #2).

The reference system aggregates or recalculates strategy-level data from OO
logs into child-level descriptors; the exact list is unknown. This module
implements the reasonable first candidates named in the spec, to be ablated
in Phase 3 and revised when Mauro answers Q2:

  trailing child PNL over {5, 10, 20} days
  trailing activation rate (20d) and hit rate (20d, of active days)
  drawdown state: dollars below the running equity high, days since high

Leakage rule: every descriptor for prediction date D uses PNL through D-1
only (everything is shifted by one day before joining).
"""
from __future__ import annotations

import pandas as pd

TRAILING_WINDOWS = (5, 10, 20)


def build_child_descriptors(child_daily_pnl: pd.DataFrame) -> pd.DataFrame:
    """Long-format descriptor frame from a date x child daily-PNL frame.

    Returns a DataFrame indexed by (date, child) whose row for date D holds
    descriptors computed from data through D-1 (shifted — safe to join
    directly onto prediction rows for D).
    """
    parts = {}
    for w in TRAILING_WINDOWS:
        parts[f"pnl_{w}d"] = child_daily_pnl.rolling(w, min_periods=1).sum()

    active = child_daily_pnl.ne(0)
    parts["activation_20d"] = active.rolling(20, min_periods=1).mean()
    win = child_daily_pnl.gt(0)
    parts["hit_rate_20d"] = (
        win.rolling(20, min_periods=1).sum() / active.rolling(20, min_periods=1).sum()
    )

    equity = child_daily_pnl.cumsum()
    peak = equity.cummax()
    parts["dd_dollars"] = equity - peak
    at_high = equity.ge(peak)
    # consecutive days below the running high, per child
    parts["days_since_high"] = pd.DataFrame(
        {c: (~at_high[c]).groupby(at_high[c].cumsum()).cumsum() for c in child_daily_pnl}
    )

    # shift everything: row D describes the child as of D-1's close
    long = pd.concat(
        {name: frame.shift(1).stack() for name, frame in parts.items()}, axis=1
    )
    long.index.names = ["date", "child"]
    return long
