"""Walk-forward CPO engine (spec section 5).

Loop: train on IS window (default 3 months) -> predict next OoS window
(default 1 day) -> advance 1 day -> repeat. Anchored (expanding) or
unanchored (rolling) window. One row per (date, child) with market features
+ child descriptors; regressor scores each child; rank; take top-k.

Open question #1 (spec section 10): exact target definition, and the "D-Day
threshold 0.65" mechanism (likely a trade/no-trade probability gate layered
on the ranker). Until answered, default target = next-day child PNL.

Baselines (spec section 5, non-negotiable honesty rules):
  BASELINE   — the candidate pool reference (definition to verify, Q4)
  BEST CHILD — the child that looked best using ONLY information available
               at comparison start (the as-of-start pick)
  ML         — the walk-forward selection
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class WalkForwardConfig:
    is_months: int = 3
    oos_days: int = 1
    anchored: bool = True
    top_k: int = 1
    d_day_threshold: float = 0.65
    initial_equity: float = 100_000.0


def run_walkforward(dataset, hp_set: dict, cfg: WalkForwardConfig):
    """Execute the daily walk-forward. Returns per-cycle picks and the
    Baseline / Best-Child / ML equity curves + metrics tables."""
    raise NotImplementedError("Phase 3")
