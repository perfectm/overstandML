"""Portfolio analytics metrics (spec section 7, Portfolio Analytics panel).

Headline stats observed in the reference UI: total PNL, MaxDD%, MAR, CAGR,
Sharpe (ann), win rate, trades, profit factor, skewness, kurtosis, tail
ratio, %PNL from top 3/5 days. All functions take a daily-PNL Series (or a
trade-level PNL Series where noted) so they work identically for a child, a
parent, or an ML equity stream.

Conventions: 252 trading days/year; Sharpe uses raw daily PNL (no risk-free
subtraction) annualized by sqrt(252); DD% is measured against the running
equity peak. These match common practice but are [INFERRED] for the
reference tool — recheck against its numbers during the acceptance test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def equity_curve(daily_pnl: pd.Series, initial_equity: float = 100_000.0) -> pd.Series:
    return initial_equity + daily_pnl.cumsum()


def drawdown(equity: pd.Series) -> pd.Series:
    """Dollar drawdown from the running peak (<= 0)."""
    return equity - equity.cummax()


def max_drawdown(equity: pd.Series) -> dict:
    """Max DD in dollars, in % of the running peak, and the longest
    peak-to-recovery spell in days."""
    dd = drawdown(equity)
    dd_pct = dd / equity.cummax()
    underwater = dd < 0
    # longest consecutive run of underwater days
    runs = underwater.groupby((~underwater).cumsum()).cumsum()
    return {
        "max_dd": float(dd.min()),
        "max_dd_pct": float(dd_pct.min()),
        "max_dd_days": int(runs.max()) if len(runs) else 0,
    }


def sharpe(daily_pnl: pd.Series, annualize: bool = True) -> float:
    sd = daily_pnl.std()
    if sd == 0 or np.isnan(sd):
        return float("nan")
    s = daily_pnl.mean() / sd
    return float(s * np.sqrt(TRADING_DAYS)) if annualize else float(s)


def cagr(equity: pd.Series) -> float:
    n = len(equity)
    if n < 2 or equity.iloc[0] <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (TRADING_DAYS / (n - 1)) - 1)


def mar(equity: pd.Series) -> float:
    """CAGR / |MaxDD%|."""
    dd_pct = max_drawdown(equity)["max_dd_pct"]
    if dd_pct == 0:
        return float("nan")
    return float(cagr(equity) / abs(dd_pct))


def trade_stats(trade_pnl: pd.Series) -> dict:
    """Trade-level stats from a per-trade PNL series."""
    wins = trade_pnl[trade_pnl > 0]
    losses = trade_pnl[trade_pnl < 0]
    gross_loss = float(-losses.sum())
    return {
        "trades": int(len(trade_pnl)),
        "total_pnl": float(trade_pnl.sum()),
        "win_rate": float(len(wins) / len(trade_pnl)) if len(trade_pnl) else float("nan"),
        "profit_factor": float(wins.sum() / gross_loss) if gross_loss > 0 else float("inf"),
        "avg_win": float(wins.mean()) if len(wins) else float("nan"),
        "avg_loss": float(losses.mean()) if len(losses) else float("nan"),
        "max_1d_loss": float(trade_pnl.min()) if len(trade_pnl) else float("nan"),
    }


def tail_stats(daily_pnl: pd.Series) -> dict:
    """Distribution-shape stats: skew, excess kurtosis, tail ratio
    (|p95/p5|), and PNL concentration in the best 3/5 days."""
    total = daily_pnl.sum()
    top = daily_pnl.sort_values(ascending=False)
    p5, p95 = daily_pnl.quantile(0.05), daily_pnl.quantile(0.95)
    return {
        "skew": float(daily_pnl.skew()),
        "kurtosis": float(daily_pnl.kurtosis()),
        "tail_ratio": float(abs(p95 / p5)) if p5 != 0 else float("inf"),
        "pct_pnl_top3": float(top.head(3).sum() / total) if total > 0 else float("nan"),
        "pct_pnl_top5": float(top.head(5).sum() / total) if total > 0 else float("nan"),
    }


def summary(
    daily_pnl: pd.Series,
    trade_pnl: pd.Series | None = None,
    initial_equity: float = 100_000.0,
) -> dict:
    """The Portfolio Analytics headline block for one PNL stream."""
    eq = equity_curve(daily_pnl, initial_equity)
    out = {
        "total_pnl": float(daily_pnl.sum()),
        "cagr": cagr(eq),
        "mar": mar(eq),
        "sharpe": sharpe(daily_pnl),
        **max_drawdown(eq),
        **tail_stats(daily_pnl),
    }
    if trade_pnl is not None:
        out.update(trade_stats(trade_pnl))
    return out
