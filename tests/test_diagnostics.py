"""Phase 2: metrics + Parent Lab, on small frames with hand-computed answers."""
import numpy as np
import pandas as pd
import pytest

from mlcpo.diagnostics import metrics, parent_lab


def dates(n):
    return pd.bdate_range("2022-05-16", periods=n)


# Day:            1    2   3   4  (day 4: all-child zero)
PNL = pd.DataFrame(
    {"A": [10.0, 0.0, 5.0, 0.0], "B": [0.0, 20.0, 1.0, 0.0]}, index=dates(4)
)


# ------------------------------------------------------------------ metrics

def test_equity_and_drawdown():
    eq = metrics.equity_curve(pd.Series([100.0, -300.0, 100.0, 300.0], index=dates(4)), 1000.0)
    assert list(eq) == [1100.0, 800.0, 900.0, 1200.0]
    dd = metrics.max_drawdown(eq)
    assert dd["max_dd"] == -300.0
    assert dd["max_dd_pct"] == pytest.approx(-300.0 / 1100.0)
    assert dd["max_dd_days"] == 2  # underwater on days 3 and 4? no: days 2,3

def test_max_dd_days_counts_underwater_run():
    eq = pd.Series([100.0, 90.0, 95.0, 101.0, 99.0], index=dates(5))
    assert metrics.max_drawdown(eq)["max_dd_days"] == 2


def test_sharpe_zero_variance_is_nan():
    assert np.isnan(metrics.sharpe(pd.Series([5.0, 5.0, 5.0])))


def test_trade_stats():
    ts = metrics.trade_stats(pd.Series([100.0, -50.0, 200.0, -50.0]))
    assert ts["trades"] == 4
    assert ts["win_rate"] == 0.5
    assert ts["profit_factor"] == 3.0
    assert ts["max_1d_loss"] == -50.0


def test_summary_keys():
    s = metrics.summary(PNL["A"], trade_pnl=PNL["A"])
    for key in ("total_pnl", "cagr", "mar", "sharpe", "max_dd", "win_rate"):
        assert key in s


# --------------------------------------------------------------- parent lab

def test_all_child_zero_days():
    assert parent_lab.all_child_zero_days(PNL) == 1


def test_winner_series_excludes_zero_days():
    w = parent_lab.winner_series(PNL)
    assert list(w) == ["A", "B", "A"]


def test_winner_frequency():
    f = parent_lab.winner_frequency(PNL)
    assert f["A"] == pytest.approx(2 / 3)
    assert f["B"] == pytest.approx(1 / 3)


def test_rotation_entropy():
    w = parent_lab.winner_series(PNL)
    expected = -(2 / 3 * np.log(2 / 3) + 1 / 3 * np.log(1 / 3)) / np.log(2)
    assert parent_lab.rotation_entropy(w, n_children=2) == pytest.approx(expected)
    # degenerate: one child always wins -> 0
    assert parent_lab.rotation_entropy(pd.Series(["A", "A"]), n_children=2) == 0.0


def test_oracle_gap():
    # oracle: 10+20+5+0 = 35; best static: A=15, B=21 -> 21; gap = 14
    assert parent_lab.oracle_gap(PNL) == pytest.approx(14.0)


def test_oracle_gap_sit_out_clips_losses():
    pnl = pd.DataFrame({"A": [-10.0, 5.0], "B": [-20.0, 1.0]}, index=dates(2))
    # oracle picks A both days: -5; sit-out oracle: 0 + 5 = 5; best static A = -5
    assert parent_lab.oracle_gap(pnl) == pytest.approx(0.0)
    assert parent_lab.oracle_gap(pnl, allow_sit_out=True) == pytest.approx(10.0)


def test_decisive_days_and_win_gap():
    # gaps on active days: 10, 20, 4
    gaps = parent_lab.win_gap_series(PNL)
    assert list(gaps) == [10.0, 20.0, 4.0]
    assert parent_lab.decisive_days(PNL) == 1.0
    assert parent_lab.decisive_days(PNL, min_gap=5.0) == pytest.approx(2 / 3)
    stats = parent_lab.win_gap_stats(PNL)
    assert stats["median_win_gap"] == 10.0
    assert stats["longest_nonpositive_run"] == 0


def test_daily_dispersion_range():
    d = parent_lab.daily_dispersion(PNL)
    assert list(d) == [10.0, 20.0, 4.0]


def test_strategy_jaccard():
    jac = parent_lab.strategy_jaccard({"A": ["S1", "S2"], "B": ["S2", "S3"]})
    assert jac.loc["A", "A"] == 1.0
    assert jac.loc["A", "B"] == pytest.approx(1 / 3)


def test_single_active_pct():
    s = parent_lab.single_active_pct(PNL)
    assert s["A"] == pytest.approx(1 / 2)  # alone on day 1, shared day 3
    assert s["B"] == pytest.approx(1 / 2)  # alone on day 2, shared day 3


def test_joint_losing_days():
    pnl = pd.DataFrame({"A": [-1.0, -2.0, 3.0], "B": [-1.0, 2.0, -3.0]}, index=dates(3))
    j = parent_lab.joint_losing_days(pnl)
    assert j.loc["A", "B"] == 1
    assert j.loc["A", "A"] == 2


def test_rolling_oracle_edge():
    edge = parent_lab.rolling_oracle_edge(PNL, window=2)
    # window days 1-2: oracle 30, best static max(10, 20) = 20 -> 10
    assert edge.iloc[1] == pytest.approx(10.0)


def test_parent_report_end_to_end():
    from mlcpo.data import portfolio
    from tests.test_data import make_parent

    report = parent_lab.parent_report(make_parent())
    assert report["all_child_zero_days"] == 0
    assert 0.0 <= report["rotation_entropy"] <= 1.0
    assert report["oracle_gap"] > 0  # oracle can't be worse than best static
    assert set(report["per_child"].index) == {"A", "B"}
    assert report["per_child"]["trade_days_pct"].between(0, 1).all()


# ----------------------------------------------------------- market features

def test_market_features_leakage_alignment():
    from mlcpo.features import market

    idx = dates(3)
    mk = lambda o, c: pd.DataFrame({"Open": o, "Close": c}, index=idx)
    ohlc = {
        "spx": mk([100.0, 102.0, 101.0], [101.0, 103.0, 102.0]),
        "vix": mk([20.0, 21.0, 22.0], [20.5, 21.5, 22.5]),
        "vix9d": mk([19.0, 19.5, 20.0], [19.2, 19.7, 20.2]),
        "vix3m": mk([23.0, 23.5, 24.0], [23.2, 23.7, 24.2]),
        "vix6m": mk([24.0, 24.5, 25.0], [24.2, 24.7, 25.2]),
    }
    f = market.build_market_features(ohlc)
    # first date dropped (no D-1 close)
    assert f.index[0] == idx[1]
    # day-of columns use day-of OPEN
    assert f.loc[idx[1], "opening_vix"] == 21.0
    assert f.loc[idx[1], "opening_spx"] == 102.0
    # gap = D open / D-1 close - 1
    assert f.loc[idx[1], "opening_gap"] == pytest.approx(102.0 / 101.0 - 1)
    # prior-close columns use D-1 CLOSE, never day-of
    assert f.loc[idx[1], "vix"] == 20.5
    assert f.loc[idx[2], "vix9d"] == 19.7
    # no VIX1D column (deliberately excluded)
    assert "vix1d" not in f.columns
