"""Walk-forward CPO engine (spec section 5; Mauro-confirmed loop, 26 Aug 2026).

Loop: TRAIN on the IS window (3 months ~ 62 trading days live) -> PREDICT
the next OoS day (features only) -> advance 1 day -> repeat. Unanchored
(rolling) or anchored (expanding) IS window.

Dataset: one row per (date, child). The row for date D carries only what is
knowable at ~9:31 ET on D (market features for D per the leakage rules,
child descriptors through D-1); the target is the child's realized PNL on D.
Open question #1 (spec section 10): the exact reference target definition
and the "D-Day threshold 0.65" mechanism. Until answered: target = same-day
child PNL for the day being predicted, no D-Day gate.

Baselines (spec section 5, non-negotiable honesty rules):
  BASELINE   — equal-weight mean of all children each day [INFERRED, Q4]
  BEST CHILD — the child with the best total PNL using ONLY data from
               before the first prediction date (the as-of-start pick)
  ML         — the walk-forward selection (sum of the k picked children)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..diagnostics import metrics
from ..features.child_descriptors import build_child_descriptors
from .boosters import make_model

TARGET = "target"


@dataclass
class WalkForwardConfig:
    is_months: int = 3
    oos_days: int = 1
    anchored: bool = False  # Mauro's constant-3m IS implies rolling (spec s5)
    top_k: int = 1
    d_day_threshold: float = 0.65  # unused until Q1 is answered
    initial_equity: float = 100_000.0
    min_train_rows: int = 50


@dataclass
class WalkForwardResult:
    cycles: pd.DataFrame          # per prediction date: picks, scores, PNL streams
    scores: pd.DataFrame          # date x child predicted scores
    config: WalkForwardConfig
    best_child_name: str

    @property
    def n_cycles(self) -> int:
        return len(self.cycles)

    def stream(self, name: str) -> pd.Series:
        """Daily PNL for 'ml', 'baseline' or 'best_child'."""
        return self.cycles[f"{name}_pnl"]

    def summary_table(self) -> pd.DataFrame:
        """The BASELINE / BEST CHILD / ML comparison table (spec s5)."""
        rows = {
            "BASELINE": metrics.summary(self.stream("baseline"), initial_equity=self.config.initial_equity),
            "BEST CHILD": metrics.summary(self.stream("best_child"), initial_equity=self.config.initial_equity),
            "ML": metrics.summary(self.stream("ml"), initial_equity=self.config.initial_equity),
        }
        return pd.DataFrame(rows).T


def build_dataset(
    child_daily_pnl: pd.DataFrame,
    market_features: pd.DataFrame,
    descriptors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Long-format modeling frame indexed by (date, child): market features
    (shared across children), child descriptors (child-specific, already
    D-1-shifted), and the target = that child's PNL that day.

    Restricted to dates present in BOTH the PNL frame and the feature frame.
    """
    if descriptors is None:
        descriptors = build_child_descriptors(child_daily_pnl)

    long_target = child_daily_pnl.stack().rename(TARGET)
    long_target.index.names = ["date", "child"]

    df = descriptors.join(long_target, how="inner")
    df = df.join(market_features.rename_axis("date"), on="date", how="inner")
    return df.sort_index()


def _prediction_dates(dataset: pd.DataFrame, cfg: WalkForwardConfig, start=None, end=None):
    dates = dataset.index.get_level_values("date").unique().sort_values()
    first_allowed = dates.min() + pd.DateOffset(months=cfg.is_months)
    lo = max(pd.Timestamp(start), first_allowed) if start else first_allowed
    hi = pd.Timestamp(end) if end else dates.max()
    return [d for d in dates if lo <= d <= hi]


def run_walkforward(
    dataset: pd.DataFrame,
    hp_set: dict,
    cfg: WalkForwardConfig | None = None,
    start=None,
    end=None,
) -> WalkForwardResult:
    """Execute the daily walk-forward over a build_dataset() frame.

    For each prediction date D: train on rows with date in the IS window
    (strictly before D), score D's rows per child, rank, pick top-k. The ML
    stream realizes the sum of the picked children's PNL on D.
    """
    cfg = cfg or WalkForwardConfig()
    feature_cols = [c for c in dataset.columns if c != TARGET]
    dates = dataset.index.get_level_values("date")
    data_start = dates.min()

    pred_dates = _prediction_dates(dataset, cfg, start, end)
    if not pred_dates:
        raise ValueError("no prediction dates: not enough history for the IS window")

    # honest as-of-start baseline: best child on data before the first prediction
    pre = dataset[dates < pred_dates[0]][TARGET].groupby("child").sum()
    best_child_name = pre.idxmax()

    cycle_rows, score_rows = [], []
    for d in pred_dates:
        window_start = data_start if cfg.anchored else d - pd.DateOffset(months=cfg.is_months)
        train = dataset[(dates >= window_start) & (dates < d)].dropna()
        test = dataset.loc[[d]]
        if len(train) < cfg.min_train_rows:
            continue

        model = make_model(hp_set)
        model.fit(train[feature_cols], train[TARGET])
        preds = pd.Series(
            model.predict(test[feature_cols].fillna(0.0)),
            index=test.index.get_level_values("child"),
        )

        picks = preds.nlargest(cfg.top_k).index.tolist()
        realized = test[TARGET].droplevel("date")
        cycle_rows.append(
            {
                "date": d,
                "picks": picks,
                "ml_pnl": float(realized.reindex(picks).sum()),
                "baseline_pnl": float(realized.mean()),
                "best_child_pnl": float(realized.get(best_child_name, 0.0)),
            }
        )
        score_rows.append(preds.rename(d))

    cycles = pd.DataFrame(cycle_rows).set_index("date")
    scores = pd.DataFrame(score_rows)
    scores.index.name = "date"
    return WalkForwardResult(cycles=cycles, scores=scores, config=cfg, best_child_name=best_child_name)
