"""Market features (spec section 4) — leakage timing is the core design rule.

Day-of (D) features, captured just after market open (~9:31 ET), just before
prediction:
    opening_vix, opening_spx, opening_gap

Prior-close (D-1) features — everything else, including VIX term structure:
    vix, vix9d, vix3m, vix6m  (VIX1D deliberately excluded so far)

TODO(Phase 3): implement fetch + as-of alignment. Symbols: ^VIX, ^VIX9D,
^VIX3M, ^VIX6M, ^GSPC (or an equivalent data source).
"""
from __future__ import annotations

DAY_OF_OPEN = ["opening_vix", "opening_spx", "opening_gap"]
PRIOR_CLOSE = ["vix", "vix9d", "vix3m", "vix6m"]


def build_market_features(dates):
    """Return a DataFrame indexed by trading date with the columns above,
    each value timestamped per the leakage rules."""
    raise NotImplementedError("Phase 3")
