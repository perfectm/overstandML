"""Phase 0: TS -> OO adapter, on a synthetic TS frame + the real export."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlcpo.data import oo_contract, ts_to_oo

REAL_EXPORT = Path(__file__).resolve().parents[1] / "data" / "RUN_G-trades-20260826-171604.csv"


def synthetic_ts_frame() -> pd.DataFrame:
    """Two trades + the end-of-file marker row, in TS export shape."""
    rows = pd.DataFrame(
        {
            "Backtick UID": ["abc", "def"],
            "Position Name": ["Strat One", "Strat Two"],
            "Entry Date": ["2026-08-03", "2026-08-04"],
            "Entry Time": ["09:46:35", "13:30:05"],
            "Exit Date": ["2026-08-03", "2026-08-04"],
            "Exit Time": ["16:00:00", "15:00:00"],
            "Exit Reason": ["Assigned", "Stop Loss (Algo)"],
            "Trade P/L": [150.0, -300.0],
            "Quantity": [2.0, 1.0],
            "Buying Power": [1000.0, 2500.0],
            "Entry Price": [1.5, -2.0],
            "Exit Price": [0.0, 1.0],
            "Total Entry Fee": [2.0, 1.0],
            "Total Exit Fee": [0.0, 1.0],
            "Total Running P/L": [150.0, -150.0],
            "VIX Entry": [15.0, 16.0],
            "VIX Exit": [14.5, 16.5],
            "Underlying Open": [7400.0, 7410.0],
            "Underlying Previous Close": [7390.0, 7420.0],
            "Underlying Close": [7420.0, 7400.0],
            "Opt1-Type": ["Short Put", "Long Call"],
            "Opt1-Quantity": [2.0, 1.0],
            "Opt1-Strike": [7300.0, 7450.0],
            "Opt1-Expiration": ["2026-08-03", "2026-08-08"],
            "Opt1-Entry Price": [1.5, -2.0],
        }
    )
    marker = pd.DataFrame({"Backtick UID": ["# end-of-file: 2 rows"]})
    return pd.concat([rows, marker], ignore_index=True)


def test_convert_synthetic():
    oo = ts_to_oo.convert(synthetic_ts_frame())
    assert len(oo) == 2  # marker row dropped
    assert list(oo.columns[: len(oo_contract.OO_COLUMNS)]) == oo_contract.OO_COLUMNS
    assert oo["P/L"].tolist() == [150.0, -300.0]
    assert oo["Strategy"].tolist() == ["Strat One", "Strat Two"]
    # status mapping: cash-settled Assigned -> Expired; algo stop -> Stop Loss
    assert oo["Reason For Close"].tolist() == ["Expired", "Stop Loss"]
    # gap in percent vs previous close
    assert oo["Gap"].iloc[0] == pytest.approx((7400 / 7390 - 1) * 100)
    assert oo["Legs"].iloc[0] == "2 2026-08-03 7300 P STO 1.5"
    assert oo["Legs"].iloc[1] == "1 2026-08-08 7450 C BTO 2"


def test_split_by_strategy(tmp_path):
    oo = ts_to_oo.convert(synthetic_ts_frame())
    groups = ts_to_oo.split_by_strategy(oo, tmp_path)
    assert set(groups) == {"Strat One", "Strat Two"}
    assert (tmp_path / "Strat One.csv").exists()


needs_real_file = pytest.mark.skipif(
    not REAL_EXPORT.exists(), reason="real TS export not present (data/ is gitignored)"
)


@needs_real_file
def test_convert_real_export():
    oo = ts_to_oo.convert_file(REAL_EXPORT)
    ts = pd.read_csv(REAL_EXPORT)
    ts = ts[ts["Entry Date"].notna()]
    # nothing dropped, P/L conserved to the cent
    assert len(oo) == len(ts) == 12142
    assert oo["P/L"].sum() == pytest.approx(ts["Trade P/L"].sum(), abs=0.01)
    assert oo["Strategy"].nunique() == 79
    # every status label resolved
    assert not oo["Reason For Close"].isna().any()
    # data range matches Mauro's (spec s5): starts 2022-05-16
    assert oo["Date Opened"].min() == pd.Timestamp("2022-05-16")


@needs_real_file
def test_real_export_pnl_identity():
    """Trade P/L = entry value + exit value - exit fee, to the cent."""
    ts = pd.read_csv(REAL_EXPORT)
    ts = ts[ts["Entry Date"].notna()]
    calc = ts["Total Entry Value"] + ts["Exit Price"] * 100 * ts["Quantity"] - ts["Total Exit Fee"]
    assert (ts["Trade P/L"] - calc).abs().max() < 0.01
