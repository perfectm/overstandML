"""Phase 5: partition persistence, daily decision, execution plan."""
import json

import pandas as pd
import pytest

from mlcpo import live
from mlcpo.model.walkforward import WalkForwardConfig
from tests.test_data import synthetic_log
from tests.test_walkforward import make_regime_data


def make_assignment():
    return pd.DataFrame(
        {"child": ["A", "A", "B"]}, index=pd.Index(["S1", "S2", "S3"], name="strategy")
    )


def test_partition_roundtrip(tmp_path):
    path = tmp_path / "partition.json"
    live.save_partition(make_assignment(), path)
    part = live.load_partition(path)
    assert part == {"A": ["S1", "S2"], "B": ["S3"]}


def test_apply_partition_warns_on_unassigned(capsys):
    from mlcpo.data.oo_contract import validate_oo

    oo = validate_oo(
        pd.concat([synthetic_log(s, 10, i) for i, s in enumerate(["S1", "S2", "S3", "SNEW"])],
                  ignore_index=True)
    )
    parent = live.apply_partition(oo, {"A": ["S1", "S2"], "B": ["S3"]})
    assert [c.name for c in parent.children] == ["A", "B"]
    assert "SNEW" in capsys.readouterr().out  # loudly reported, not silent


def regime_decision(pick_date, method="vote"):
    """Decision on the synthetic regime world with injected features."""
    pnl, feats = make_regime_data(200, seed=21)
    from mlcpo.data.oo_contract import validate_oo

    frames = []
    for child, strat in [("A", "SA"), ("B", "SB")]:
        f = synthetic_log(strat, 200, seed=1)
        f["Date Opened"] = f["Date Closed"] = list(pnl.index)
        f["P/L"] = pnl[child].values
        frames.append(f)
    oo = validate_oo(pd.concat(frames, ignore_index=True))
    hp = {"model": "lightgbm", "n_estimators": 60, "learning_rate": 0.1,
          "num_leaves": 7, "min_child_samples": 5, "random_state": 0}
    return live.daily_decision(
        oo, {"A": ["SA"], "B": ["SB"]}, {"h1": hp, "h2": {**hp, "random_state": 9}},
        WalkForwardConfig(min_train_rows=40), method=method,
        pick_date=pick_date, market_features=feats,
    ), pnl, feats


def test_daily_decision_picks_regime_child():
    pnl, feats0 = make_regime_data(200, seed=21)
    d = pnl.index[-1]
    decision, pnl, feats = regime_decision(d)
    # regime on the last day determines which child should be picked
    expected = "A" if feats.loc[d, "regime"] == 1 else "B"
    assert decision.picks == [expected]
    assert decision.date == str(d.date())
    assert set(decision.combined_scores) == {"A", "B"}
    assert set(decision.per_set_scores) == {"h1", "h2"}
    # execution plan covers every child exactly once
    assert set(decision.enable) | set(decision.disable) == {"A", "B"}
    assert not set(decision.enable) & set(decision.disable)


def test_decision_save_and_checklist(tmp_path):
    decision, _, _ = regime_decision(pd.bdate_range("2022-05-16", periods=200)[-1])
    path = decision.save(tmp_path)
    saved = json.loads(path.read_text())
    assert saved["picks"] == decision.picks
    text = decision.checklist()
    assert "VERIFY BY HAND" in text
    assert f"ENABLE  {decision.picks[0]}" in text


def test_decision_requires_history():
    with pytest.raises(ValueError, match="not enough history"):
        regime_decision(pd.bdate_range("2022-05-16", periods=200)[5])
