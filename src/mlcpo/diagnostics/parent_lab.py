"""Parent Lab diagnostics (spec section 7) — 'reverse-engineer ML without
running it'. Quantifies, before any training, whether a candidate set has:

  exploitable dispersion — oracle gap (perfect-foresight picks vs best
      static), rolling oracle edge, cross-child outcome dispersion,
      best-minus-second-best daily gap distribution
  rotation — realized winner frequency per child, rotation entropy,
      decisive-day percentage
  independence — pairwise strategy-overlap Jaccard, child PNL correlation,
      co-activation rates, joint losing days

Cohort semantics matter: a mature cutoff excludes provisional (too-recent)
child-date cohorts from outcome comparisons; equity is viewed both by Date
Closed (realized) and Date Opened (entry cohorts).

TODO(Phase 2): implement each metric against daily child PNL frames; these
double as validation of the Phase 0 data adapter.
"""
from __future__ import annotations


def oracle_gap(child_daily_pnl) -> float:
    """Sum of daily max-across-children PNL minus best single child's total
    (perfect-foresight upper bound vs best static)."""
    raise NotImplementedError("Phase 2")


def rotation_entropy(daily_winner_series) -> float:
    """Normalized entropy of the realized daily-winner distribution."""
    raise NotImplementedError("Phase 2")


def decisive_days(child_daily_pnl, min_gap: float = 0.0) -> float:
    """Share of days where the best child's PNL exceeds the second-best by a
    meaningful margin."""
    raise NotImplementedError("Phase 2")


def strategy_jaccard(children) -> "pd.DataFrame":
    """Pairwise Jaccard similarity of child strategy memberships."""
    raise NotImplementedError("Phase 2")
