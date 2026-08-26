"""Phase 4b: HP-set ensemble aggregation."""
import pandas as pd
import pytest

from mlcpo.model import ensemble


def dates(n):
    return pd.bdate_range("2022-05-16", periods=n)


def score_frame(rows):
    return pd.DataFrame(rows, index=dates(len(rows)), columns=["A", "B", "C"])


def test_mean_rank_combines():
    # set1 says A best; set2 says B best but A second — A wins on mean rank
    s1 = score_frame([[3.0, 1.0, 2.0]])
    s2 = score_frame([[2.0, 3.0, 1.0]])
    picks = ensemble.ensemble_pick({"s1": s1, "s2": s2}, top_k=1)
    assert picks.iloc[0] == ["A"]


def test_mean_rank_scale_free():
    # one set's scores are 1000x larger — mean_rank must not care
    s1 = score_frame([[3.0, 1.0, 2.0]])
    s2 = score_frame([[2000.0, 3000.0, 1000.0]])
    s3 = score_frame([[0.003, 0.001, 0.002]])
    picks = ensemble.ensemble_pick({"s1": s1, "s2": s2, "s3": s3}, top_k=1)
    assert picks.iloc[0] == ["A"]  # ranks: A 1,2,1  B 3,1,3  C 2,3,2


def test_vote_majority_wins():
    s1 = score_frame([[3.0, 1.0, 2.0]])   # votes A
    s2 = score_frame([[3.0, 2.0, 1.0]])   # votes A
    s3 = score_frame([[1.0, 3.0, 2.0]])   # votes B
    picks = ensemble.ensemble_pick({"s1": s1, "s2": s2, "s3": s3}, method="vote")
    assert picks.iloc[0] == ["A"]


def test_vote_tie_broken_by_mean_rank():
    s1 = score_frame([[3.0, 2.0, 1.0]])   # votes A; B second
    s2 = score_frame([[1.0, 3.0, 2.0]])   # votes B; C second
    picks = ensemble.ensemble_pick({"s1": s1, "s2": s2}, method="vote")
    # A and B tied 1-1; mean ranks: A 2.0, B 1.5 -> B
    assert picks.iloc[0] == ["B"]


def test_mean_score():
    s1 = score_frame([[1.0, 5.0, 0.0]])
    s2 = score_frame([[2.0, 0.0, 1.0]])
    combined = ensemble.combine_scores({"s1": s1, "s2": s2}, method="mean_score")
    assert combined.iloc[0].tolist() == [1.5, 2.5, 0.5]


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown method"):
        ensemble.combine_scores({"s": score_frame([[1.0, 2.0, 3.0]])}, method="magic")


def test_run_ensemble_end_to_end():
    from mlcpo.model.walkforward import WalkForwardConfig, build_dataset
    from tests.test_walkforward import SMALL_HP, make_regime_data

    pnl, feats = make_regime_data(140, seed=11)
    ds = build_dataset(pnl, feats)
    hp2 = {**SMALL_HP, "random_state": 99, "n_estimators": 40}
    res, per_set = ensemble.run_ensemble(
        ds, {"a": SMALL_HP, "b": hp2}, WalkForwardConfig(min_train_rows=40)
    )
    assert set(per_set) == {"a", "b"}
    assert res.n_cycles == per_set["a"].n_cycles
    # ensemble of two good sets still learns the easy regime world
    assert (res.cycles["ml_pnl"] > 0).mean() > 0.8
    # realized pnl matches the picks against the dataset target
    d = res.cycles.index[0]
    assert res.cycles.loc[d, "ml_pnl"] == pytest.approx(
        pnl.loc[d, res.cycles.loc[d, "picks"]].sum()
    )
