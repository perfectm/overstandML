"""Phase 3: descriptors, dataset assembly, walk-forward engine end-to-end."""
import numpy as np
import pandas as pd
import pytest

from mlcpo.features.child_descriptors import build_child_descriptors
from mlcpo.model import boosters
from mlcpo.model.walkforward import (
    TARGET,
    WalkForwardConfig,
    build_dataset,
    run_walkforward,
)


def dates(n):
    return pd.bdate_range("2022-05-16", periods=n)


# ------------------------------------------------------------- descriptors

def test_descriptors_are_shifted_one_day():
    pnl = pd.DataFrame({"A": [100.0, 200.0, 300.0]}, index=dates(3))
    desc = build_child_descriptors(pnl)
    d = dates(3)
    # row for day 2 sees only day 1
    assert desc.loc[(d[1], "A"), "pnl_5d"] == 100.0
    # row for day 3 sees days 1-2
    assert desc.loc[(d[2], "A"), "pnl_5d"] == 300.0
    # day 1 has no history
    assert np.isnan(desc.loc[(d[0], "A"), "pnl_5d"])


def test_descriptors_drawdown_state():
    pnl = pd.DataFrame({"A": [100.0, -50.0, -25.0, 200.0]}, index=dates(4))
    desc = build_child_descriptors(pnl).xs("A", level="child")
    d = dates(4)
    # as of day 3 close-of-D-1 (= day 2): equity 50, peak 100 -> dd -50, 1 day below
    assert desc.loc[d[2], "dd_dollars"] == -50.0
    assert desc.loc[d[2], "days_since_high"] == 1
    # as of day 4 (= through day 3): dd -75, 2 days below
    assert desc.loc[d[3], "dd_dollars"] == -75.0
    assert desc.loc[d[3], "days_since_high"] == 2


# ----------------------------------------------------------------- dataset

def make_regime_data(n=260, seed=7):
    """Synthetic learnable world: regime feature x alternates; child A earns
    when x=1, child B when x=0. Noise small relative to signal."""
    rng = np.random.default_rng(seed)
    idx = dates(n)
    x = (np.arange(n) // 5) % 2  # regime flips every 5 days
    noise = rng.normal(0, 5, (n, 2))
    pnl = pd.DataFrame(
        {
            "A": np.where(x == 1, 100.0, -40.0) + noise[:, 0],
            "B": np.where(x == 0, 100.0, -40.0) + noise[:, 1],
        },
        index=idx,
    )
    feats = pd.DataFrame({"regime": x.astype(float)}, index=idx)
    return pnl, feats


def test_build_dataset_shape_and_target():
    pnl, feats = make_regime_data(30)
    ds = build_dataset(pnl, feats)
    assert ds.index.names == ["date", "child"]
    assert TARGET in ds.columns and "regime" in ds.columns
    d0 = dates(30)[3]
    assert ds.loc[(d0, "A"), TARGET] == pnl.loc[d0, "A"]


def test_make_model_lightgbm_and_xgboost():
    from mlcpo.model.hp import load_hp_sets

    hp = load_hp_sets()["P3_v2_2"]
    m = boosters.make_model(hp)
    assert type(m).__name__ == "LGBMRegressor"
    assert m.n_estimators == 240 and m.random_state == 4998

    xgb = boosters.make_model({**hp, "model": "xgboost"})
    assert type(xgb).__name__ == "XGBRegressor"
    assert xgb.max_leaves == 36 and xgb.max_depth == 0

    with pytest.raises(ValueError, match="unknown model"):
        boosters.make_model({"model": "catboost"})


# ---------------------------------------------------------------- end-to-end

SMALL_HP = {
    "model": "lightgbm",
    "n_estimators": 60,
    "learning_rate": 0.1,
    "num_leaves": 7,
    "min_child_samples": 5,
    "random_state": 0,
}


def test_walkforward_learns_regime():
    pnl, feats = make_regime_data()
    ds = build_dataset(pnl, feats)
    cfg = WalkForwardConfig(is_months=3, top_k=1, min_train_rows=40)
    res = run_walkforward(ds, SMALL_HP, cfg)

    assert res.n_cycles > 100
    # the regime is trivially learnable: ML should pick the right child
    # almost always and beat both baselines decisively
    right = (res.cycles["ml_pnl"] > 0).mean()
    assert right > 0.9
    tbl = res.summary_table()
    assert tbl.loc["ML", "total_pnl"] > tbl.loc["BASELINE", "total_pnl"]
    assert tbl.loc["ML", "total_pnl"] > tbl.loc["BEST CHILD", "total_pnl"]


def test_walkforward_no_leakage_of_prediction_day():
    """Poison test: make one child hugely profitable ONLY on the final day
    with no feature signal for it — a leaky engine would learn from the
    prediction day itself; an honest one cannot know."""
    pnl, feats = make_regime_data(120, seed=1)
    last = pnl.index[-1]
    pnl.loc[last, "B"] = 1_000_000.0
    feats["regime"] = 0.5  # feature carries no information at all
    ds = build_dataset(pnl, feats)
    res = run_walkforward(ds, SMALL_HP, WalkForwardConfig(min_train_rows=40))
    # scores for the last day were produced by a model trained strictly
    # before it; with uninformative features the predicted score for B on
    # the poisoned day must not reflect the million-dollar outcome
    assert res.scores.loc[last, "B"] < 10_000


def test_walkforward_anchored_vs_rolling_window():
    pnl, feats = make_regime_data(140, seed=3)
    ds = build_dataset(pnl, feats)
    rolling = run_walkforward(ds, SMALL_HP, WalkForwardConfig(anchored=False, min_train_rows=40))
    anchored = run_walkforward(ds, SMALL_HP, WalkForwardConfig(anchored=True, min_train_rows=40))
    # same prediction dates either way; both learn this easy world
    assert list(rolling.cycles.index) == list(anchored.cycles.index)


def test_walkforward_top_k_two_sums_picks():
    pnl, feats = make_regime_data(130, seed=5)
    ds = build_dataset(pnl, feats)
    res = run_walkforward(ds, SMALL_HP, WalkForwardConfig(top_k=2, min_train_rows=40))
    d = res.cycles.index[0]
    realized = pnl.loc[d]
    assert res.cycles.loc[d, "ml_pnl"] == pytest.approx(realized.sum())
    assert len(res.cycles.loc[d, "picks"]) == 2


def test_best_child_is_chosen_as_of_start_only():
    """The honest-baseline rule: BEST CHILD uses pre-start data only."""
    pnl, feats = make_regime_data(130, seed=9)
    ds = build_dataset(pnl, feats)
    res = run_walkforward(ds, SMALL_HP, WalkForwardConfig(min_train_rows=40))
    first_pred = res.cycles.index[0]
    pre = pnl.loc[: first_pred - pd.Timedelta(days=1)].sum()
    assert res.best_child_name == pre.idxmax()
