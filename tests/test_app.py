"""Web app: figures and layout build from a synthetic cache (no server)."""
import json

import numpy as np
import pandas as pd
import pytest

from mlcpo import app as webapp


@pytest.fixture
def cache():
    dates = pd.bdate_range("2022-08-16", periods=60)
    rng = np.random.default_rng(0)
    streams = pd.DataFrame(
        {name: rng.normal(200, 400, 60) for name in webapp.STREAM_COLORS}, index=dates
    )
    cycles = pd.DataFrame(
        {
            "pick": rng.choice(["S1", "S2", "S3"], 60),
            "ml_pnl": rng.normal(200, 400, 60),
            "baseline_pnl": rng.normal(150, 200, 60),
            "best_child_pnl": rng.normal(150, 300, 60),
        },
        index=dates,
    )
    from mlcpo.diagnostics import metrics

    return {
        "meta": {
            "refreshed_at": "2026-08-26 12:00:00",
            "ts_csv": "x.csv",
            "data_range": ["2022-05-16", "2026-08-10"],
            "n_cycles": 60,
            "hp_sets": ["P3_v2_2"],
            "diagnostics": {},
        },
        "summary": {name: metrics.summary(streams[name]) for name in streams.columns},
        "children": {
            "S1": {"strategies": ["a", "b"], "realized_pnl": 1000.0, "sharpe": 1.2,
                   "winner_pct": 0.4, "max_dd": -100.0, "max_1d_loss": -50.0,
                   "trade_days_pct": 0.6, "zero_days_pct": 0.4, "single_active_pct": 0.1},
        },
        "streams": streams,
        "cycles": cycles,
        "decision": {
            "date": "2026-08-26", "picks": ["S1"], "method": "vote", "top_k": 1,
            "combined_scores": {"S1": 3.0, "S2": 2.0},
            "per_set_scores": {"P3_v2_2": {"S1": 300.0, "S2": 200.0}},
            "enable": {"S1": ["a", "b"]}, "disable": {"S2": ["c"]},
        },
    }


def test_figures_build(cache):
    for fig in (
        webapp.equity_fig(cache["streams"]),
        webapp.drawdown_fig(cache["streams"]),
        webapp.picks_fig(cache["cycles"]),
        webapp.child_pnl_fig(cache["cycles"]),
    ):
        d = fig.to_dict()
        assert d["data"], "figure has no traces"
        assert d["layout"]["paper_bgcolor"] == webapp.SURFACE


def test_app_layout_builds(cache):
    app = webapp.build_app(cache)
    layout = str(app.layout)
    assert "ML CPO" in layout
    assert "S1" in layout


def test_app_layout_without_decision(cache):
    cache["decision"] = None
    app = webapp.build_app(cache)
    assert "No decision file yet" in str(app.layout)
