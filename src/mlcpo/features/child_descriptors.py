"""Child descriptor features (spec section 4, open question #2).

The reference system aggregates or recalculates strategy-level data from OO
logs into child-level descriptors. The exact list is unknown. Reasonable
first candidates (to be validated, and ablated in Phase 3):
  - trailing child PNL over {5, 10, 20} days
  - trailing hit rate / activation count
  - drawdown state (distance from equity high)
  - composition summaries (credit/debit mix, avg DTE)
A parent named "Beta_features_1" in the reference UI suggests market-beta
descriptors have been tried.
"""
from __future__ import annotations


def build_child_descriptors(child_daily_pnl):
    raise NotImplementedError("Phase 3")
