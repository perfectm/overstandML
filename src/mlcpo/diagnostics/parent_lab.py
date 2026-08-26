"""Parent Lab diagnostics (spec section 7) — 'reverse-engineer ML without
running it'. Quantifies, before any training, whether a candidate set has:

  exploitable dispersion — oracle gap (perfect-foresight picks vs best
      static), rolling oracle edge, cross-child outcome dispersion,
      best-minus-second-best daily gap distribution
  rotation — realized winner frequency per child, rotation entropy,
      decisive-day percentage
  independence — pairwise strategy-overlap Jaccard, child PNL correlation,
      co-activation rates, joint losing days

Input convention: `child_daily_pnl` is a date x child DataFrame as produced
by Parent.daily_pnl() — inactive days are 0. Rotation/dispersion stats
exclude all-child-zero days (no activity means nothing to choose between);
all-child-zero days are their own headline metric, as in the reference UI.

Cohort semantics (mature cutoff, provisional cohorts, Date Opened vs Date
Closed attribution) are the caller's responsibility: pass a frame built on
the basis and window you mean to analyze.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Mapping

import numpy as np
import pandas as pd

from . import metrics

if TYPE_CHECKING:
    from ..data.portfolio import Parent


# ---------------------------------------------------------------- activity

def active_mask(child_daily_pnl: pd.DataFrame) -> pd.DataFrame:
    """True where a child traded that day (nonzero PNL is the proxy for
    activity in a daily-PNL frame)."""
    return child_daily_pnl.ne(0)


def all_child_zero_days(child_daily_pnl: pd.DataFrame) -> int:
    """Days on which no child traded — the reference UI headline card."""
    return int((~active_mask(child_daily_pnl)).all(axis=1).sum())


def single_active_pct(child_daily_pnl: pd.DataFrame) -> pd.Series:
    """Per child: share of its active days on which it was the ONLY active
    child (days where selection is forced, not learned)."""
    act = active_mask(child_daily_pnl)
    alone = act.mul(act.sum(axis=1) == 1, axis=0)
    return alone.sum() / act.sum()


def co_activation(child_daily_pnl: pd.DataFrame) -> pd.DataFrame:
    """Pairwise P(both active) over all days."""
    act = active_mask(child_daily_pnl).astype(float)
    return act.T @ act / len(act)


# ---------------------------------------------------------------- rotation

def winner_series(child_daily_pnl: pd.DataFrame) -> pd.Series:
    """Realized daily winner (argmax across children), all-zero days
    excluded."""
    active_days = child_daily_pnl[active_mask(child_daily_pnl).any(axis=1)]
    return active_days.idxmax(axis=1)


def winner_frequency(child_daily_pnl: pd.DataFrame) -> pd.Series:
    """Share of active days each child was the realized winner."""
    w = winner_series(child_daily_pnl)
    return (
        w.value_counts(normalize=True)
        .reindex(child_daily_pnl.columns, fill_value=0.0)
    )


def rotation_entropy(daily_winner_series: pd.Series, n_children: int | None = None) -> float:
    """Normalized entropy of the realized daily-winner distribution:
    1.0 = winners uniformly rotated, 0.0 = one child always won. Normalized
    by log(n_children) (defaults to the number of distinct winners)."""
    counts = daily_winner_series.value_counts(normalize=True)
    n = n_children if n_children is not None else len(counts)
    if n <= 1:
        return 0.0
    h = float(-(counts * np.log(counts)).sum())
    return h / float(np.log(n))


def decisive_days(child_daily_pnl: pd.DataFrame, min_gap: float = 0.0) -> float:
    """Share of active days where the best child beat the second-best by
    more than min_gap — days where the pick actually mattered."""
    gaps = win_gap_series(child_daily_pnl)
    if len(gaps) == 0:
        return float("nan")
    return float((gaps > min_gap).mean())


# -------------------------------------------------------------- dispersion

def oracle_gap(child_daily_pnl: pd.DataFrame, allow_sit_out: bool = False) -> float:
    """Perfect-foresight daily picks (sum of daily max across children)
    minus the best single child's total — the unachievable upper bound on
    what selection can add over the best static choice. allow_sit_out=True
    lets the oracle take 0 on days when every child loses (a D-Day-gate
    style oracle)."""
    daily_best = child_daily_pnl.max(axis=1)
    if allow_sit_out:
        daily_best = daily_best.clip(lower=0.0)
    best_static = child_daily_pnl.sum().max()
    return float(daily_best.sum() - best_static)


def rolling_oracle_edge(child_daily_pnl: pd.DataFrame, window: int = 60) -> pd.Series:
    """Rolling oracle gap: in-window oracle total minus in-window best
    single child (reference UI shows the 60-day version)."""
    oracle = child_daily_pnl.max(axis=1).rolling(window).sum()
    best_static = child_daily_pnl.rolling(window).sum().max(axis=1)
    return (oracle - best_static).rename(f"oracle_edge_{window}d")


def win_gap_series(child_daily_pnl: pd.DataFrame) -> pd.Series:
    """Best minus second-best child PNL per active day."""
    active_days = child_daily_pnl[active_mask(child_daily_pnl).any(axis=1)]
    ranked = active_days.apply(lambda r: r.nlargest(2).values, axis=1, result_type="expand")
    return (ranked[0] - ranked[1]).rename("win_gap")


def win_gap_stats(child_daily_pnl: pd.DataFrame) -> dict:
    """Reference UI card: median win-gap, lower quartile, longest run of
    non-positive gaps (stretches where picking couldn't help)."""
    gaps = win_gap_series(child_daily_pnl)
    nonpos = gaps <= 0
    runs = nonpos.groupby((~nonpos).cumsum()).cumsum()
    return {
        "median_win_gap": float(gaps.median()),
        "q1_win_gap": float(gaps.quantile(0.25)),
        "longest_nonpositive_run": int(runs.max()) if len(runs) else 0,
    }


def daily_dispersion(child_daily_pnl: pd.DataFrame, method: str = "range") -> pd.Series:
    """Cross-child outcome dispersion per active day. method='range'
    (max - min) or 'std'. [INFERRED] The reference UI's exact definition is
    unknown; range is the default here because the win-gap stats already
    cover ranking margin."""
    active_days = child_daily_pnl[active_mask(child_daily_pnl).any(axis=1)]
    if method == "range":
        d = active_days.max(axis=1) - active_days.min(axis=1)
    elif method == "std":
        d = active_days.std(axis=1)
    else:
        raise ValueError(f"unknown method {method!r}")
    return d.rename(f"dispersion_{method}")


# ------------------------------------------------------------ independence

def strategy_jaccard(children: "Iterable | Mapping[str, Iterable[str]]") -> pd.DataFrame:
    """Pairwise Jaccard similarity of child strategy memberships. Accepts
    Child objects (uses .name/.strategies) or a {name: strategies} mapping."""
    if isinstance(children, Mapping):
        sets = {name: set(s) for name, s in children.items()}
    else:
        sets = {c.name: set(c.strategies) for c in children}
    names = list(sets)
    out = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            union = sets[a] | sets[b]
            out.loc[a, b] = len(sets[a] & sets[b]) / len(union) if union else 0.0
    return out


def child_pnl_correlation(child_daily_pnl: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Pearson correlation of daily child PNL."""
    return child_daily_pnl.corr()


def joint_losing_days(child_daily_pnl: pd.DataFrame) -> pd.DataFrame:
    """Pairwise count of days both children lost money — tail co-movement
    the correlation matrix can miss."""
    neg = child_daily_pnl.lt(0).astype(float)
    return (neg.T @ neg).astype(int)


# ----------------------------------------------------------------- report

def parent_report(parent: "Parent", basis: str = "closed") -> dict:
    """The diagnostics-dashboard headline block for one parent: is there
    enough dispersion, rotation and independence to be worth training on?

    (The reference UI's composite 'ML Opportunity Score' is omitted — its
    component formula is unknown, spec section 7.)
    """
    pnl = parent.daily_pnl(basis)
    winners = winner_series(pnl)
    jac = strategy_jaccard(parent.children)
    off_diag = jac.values[~np.eye(len(jac), dtype=bool)]
    corr = child_pnl_correlation(pnl).values[~np.eye(len(pnl.columns), dtype=bool)]

    per_child = pd.DataFrame(
        {
            "realized_pnl": pnl.sum(),
            "sharpe": pnl.apply(metrics.sharpe),
            "max_dd": {
                c: metrics.max_drawdown(metrics.equity_curve(pnl[c]))["max_dd"]
                for c in pnl.columns
            },
            "max_1d_loss": pnl.min(),
            "trade_days_pct": active_mask(pnl).mean(),
            "zero_days_pct": 1.0 - active_mask(pnl).mean(),
            "single_active_pct": single_active_pct(pnl),
            "winner_pct": winner_frequency(pnl),
        }
    )

    return {
        "oracle_gap": oracle_gap(pnl),
        "decisive_days_pct": decisive_days(pnl),
        "rotation_entropy": rotation_entropy(winners, n_children=len(pnl.columns)),
        "all_child_zero_days": all_child_zero_days(pnl),
        **win_gap_stats(pnl),
        "mean_pairwise_jaccard": float(off_diag.mean()) if len(off_diag) else 0.0,
        "max_pairwise_jaccard": float(off_diag.max()) if len(off_diag) else 0.0,
        "mean_pairwise_pnl_corr": float(np.nanmean(corr)) if len(corr) else float("nan"),
        "max_pairwise_pnl_corr": float(np.nanmax(corr)) if len(corr) else float("nan"),
        "per_child": per_child,
    }
