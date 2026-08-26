"""Phase 1: co-firing clustering and balanced child construction."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlcpo.data import children as ch
from tests.test_data import synthetic_log

REAL_EXPORT = Path(__file__).resolve().parents[1] / "data" / "RUN_G-trades-20260826-171604.csv"


def make_universe():
    """6 strategies: A1/A2 co-fire (same days), B1/B2 co-fire, C and the
    hedge fire on their own patterns."""
    frames = []
    d = pd.bdate_range("2022-05-16", periods=120)
    patterns = {
        "A1": d[::2], "A2": d[::2],          # co-firing pair
        "B1": d[1::2], "B2": d[1::2],        # co-firing pair
        "C": d[::3],
        "Tail Hedge X": d,
    }
    rng = np.random.default_rng(0)
    for name, days in patterns.items():
        f = synthetic_log(name, len(days), seed=hash(name) % 2**31)
        f["Date Opened"] = f["Date Closed"] = list(days)
        frames.append(f)
    from mlcpo.data.oo_contract import validate_oo

    return validate_oo(pd.concat(frames, ignore_index=True))


def test_is_hedge():
    assert ch.is_hedge("EOD Put Butterfly Hedge")
    assert ch.is_hedge("Long Hedge Strangle 0DTE")
    assert not ch.is_hedge("MOC SPX Condor v2")


def test_cofire_groups_pair_up():
    oo = make_universe()
    groups = ch.cofire_groups(oo[~oo["Strategy"].map(ch.is_hedge)])
    as_sets = [set(g) for g in groups]
    assert {"A1", "A2"} in as_sets
    assert {"B1", "B2"} in as_sets
    assert {"C"} in as_sets


def test_build_children_excludes_hedges_and_keeps_groups_together():
    oo = make_universe()
    parent, assignment = ch.build_children(oo, n_children=2)
    assert "Tail Hedge X" not in assignment.index
    # co-firing pairs stay in one child
    assert assignment.loc["A1", "child"] == assignment.loc["A2", "child"]
    assert assignment.loc["B1", "child"] == assignment.loc["B2", "child"]
    assert len(parent.children) == 2
    # every non-hedge strategy assigned exactly once
    assert sorted(assignment.index) == ["A1", "A2", "B1", "B2", "C"]


def test_build_children_include_hedges_flag():
    oo = make_universe()
    _, assignment = ch.build_children(oo, n_children=2, include_hedges=True)
    assert "Tail Hedge X" in assignment.index


def test_pack_groups_balances_pnl():
    stats = pd.DataFrame(
        {"pnl": [100.0, 90.0, 60.0, 50.0], "days": [10, 10, 10, 10]},
        index=[0, 1, 2, 3],
    )
    packing = ch.pack_groups(stats, 2)
    totals = sorted(
        sum(stats.loc[g, "pnl"] for g in gids) for gids in packing.values()
    )
    assert totals == [150.0, 150.0]  # 100+50 / 90+60 — greedy optimum


@pytest.mark.skipif(not REAL_EXPORT.exists(), reason="real TS export not present")
def test_real_partition_similar_size():
    from mlcpo.data.ts_to_oo import convert_file

    parent, assignment = ch.build_children(convert_file(REAL_EXPORT), n_children=6)
    assert len(parent.children) == 6
    assert len(assignment) == 60  # 79 strategies minus 19 hedges
    assert parent.check_similar_size()
    rep = parent.size_report()
    # PNL balanced tightly by the greedy pack
    assert rep["total_pnl"].max() / rep["total_pnl"].min() < 1.1
