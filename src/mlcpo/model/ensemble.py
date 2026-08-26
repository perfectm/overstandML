"""Ensemble of HP sets for live operation (spec section 5, open question #3).

The reference system trades an ensemble of HP sets, reviewed every 1-2
months; its aggregation method is unknown. Implemented candidates:

  mean_rank  — each set ranks the children per day (1 = best); average the
               ranks; lowest mean rank wins. Scale-free, the default.
  vote       — each set votes for its top-k; most votes wins, mean rank
               breaks ties.
  mean_score — average the raw predicted scores. Scale-sensitive; only
               sensible when all sets predict the same target.

Motivation measured on the style parent (2026-08-26): single sets agree on
as little as 18% of daily picks while landing within 2.5% of each other's
total PNL — pick sequences are fragile, aggregate outcomes are not.
"""
from __future__ import annotations

import pandas as pd

from .walkforward import TARGET, WalkForwardConfig, WalkForwardResult, run_walkforward


def combine_scores(
    per_set_scores: dict[str, pd.DataFrame],
    method: str = "mean_rank",
    top_k: int = 1,
) -> pd.DataFrame:
    """Combine {set name: date x child score frame} into one date x child
    frame where higher = better, per the chosen method."""
    frames = list(per_set_scores.values())
    if method == "mean_rank":
        # rank 1 = best; negate mean rank so higher = better
        ranks = [f.rank(axis=1, ascending=False) for f in frames]
        return -sum(ranks) / len(ranks)
    if method == "vote":
        votes = sum(
            f.rank(axis=1, ascending=False).le(top_k).astype(float) for f in frames
        )
        # fractional mean-rank tie-break, scaled to never outweigh a vote
        mean_rank = sum(f.rank(axis=1, ascending=False) for f in frames) / len(frames)
        n = frames[0].shape[1]
        return votes + (n - mean_rank) / (n * len(frames) * 10)
    if method == "mean_score":
        return sum(frames) / len(frames)
    raise ValueError(f"unknown method {method!r}")


def ensemble_pick(
    per_set_scores: dict[str, pd.DataFrame],
    top_k: int = 1,
    method: str = "mean_rank",
) -> pd.Series:
    """Per-date list of the top-k children under the combined score."""
    combined = combine_scores(per_set_scores, method, top_k)
    return combined.apply(lambda r: list(r.nlargest(top_k).index), axis=1)


def run_ensemble(
    dataset: pd.DataFrame,
    hp_sets: dict[str, dict],
    cfg: WalkForwardConfig | None = None,
    method: str = "mean_rank",
    per_set_results: dict[str, WalkForwardResult] | None = None,
) -> tuple[WalkForwardResult, dict[str, WalkForwardResult]]:
    """Run the walk-forward once per HP set, combine daily scores, realize
    the ensemble picks. Returns (ensemble result, per-set results).

    per_set_results lets callers pass already-computed runs (they are keyed
    by HP-set name and reused instead of re-running).
    """
    cfg = cfg or WalkForwardConfig()
    per_set_results = dict(per_set_results or {})
    for name, hp in hp_sets.items():
        if name not in per_set_results:
            per_set_results[name] = run_walkforward(dataset, hp, cfg)

    scores = {name: r.scores for name, r in per_set_results.items()}
    picks = ensemble_pick(scores, cfg.top_k, method)

    first = next(iter(per_set_results.values()))
    realized = dataset[TARGET].unstack("child").reindex(picks.index)
    ml_pnl = pd.Series(
        [realized.loc[d, ps].sum() for d, ps in picks.items()],
        index=picks.index,
        name="ml_pnl",
    )
    cycles = first.cycles.copy()
    cycles["picks"] = picks
    cycles["ml_pnl"] = ml_pnl
    combined = combine_scores(scores, method, cfg.top_k)
    return (
        WalkForwardResult(
            cycles=cycles,
            scores=combined,
            config=cfg,
            best_child_name=first.best_child_name,
        ),
        per_set_results,
    )
