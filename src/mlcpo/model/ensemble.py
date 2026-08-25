"""Ensemble of HP sets for live operation (spec section 5, open question #3).

The reference system trades an ensemble of HP sets, reviewed every 1-2
months. Aggregation method unknown — candidate designs to test: mean rank,
majority vote on top-k, mean predicted score.
"""
from __future__ import annotations


def ensemble_pick(per_set_scores, top_k: int = 1):
    raise NotImplementedError("Phase 4")
