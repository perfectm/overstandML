"""Phase 0-1 data layer: contract validation and child/parent assembly."""
import numpy as np
import pandas as pd
import pytest

from mlcpo.data import oo_contract, portfolio


def synthetic_log(strategy: str, n: int, seed: int) -> pd.DataFrame:
    """Minimal contract-valid OO trade log."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-05-16", periods=n)
    df = oo_contract.empty_oo_frame()
    return pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "Date Opened": dates,
                    "Time Opened": "09:33:00",
                    "Date Closed": dates,
                    "Time Closed": "16:00:00",
                    "P/L": rng.normal(50, 300, n).round(2),
                    "No. of Contracts": 1,
                    "Margin Req.": 5000.0,
                    "Strategy": strategy,
                    "Reason For Close": oo_contract.STATUS_EXPIRED,
                }
            ),
        ],
        ignore_index=True,
    )


def test_validate_oo_missing_column():
    df = synthetic_log("S1", 5, 0).drop(columns=["P/L"])
    with pytest.raises(oo_contract.ContractError, match="P/L"):
        oo_contract.validate_oo(df)


def test_validate_oo_coerces_dtypes():
    out = oo_contract.validate_oo(synthetic_log("S1", 5, 0))
    assert pd.api.types.is_datetime64_any_dtype(out["Date Closed"])
    assert pd.api.types.is_numeric_dtype(out["P/L"])


def make_parent() -> portfolio.Parent:
    a = portfolio.Child.from_logs(
        "A",
        {
            "S1": oo_contract.validate_oo(synthetic_log("S1", 60, 1)),
            "S2": oo_contract.validate_oo(synthetic_log("S2", 60, 2)),
        },
    )
    b = portfolio.Child.from_logs(
        "B",
        {
            "S3": oo_contract.validate_oo(synthetic_log("S3", 60, 3)),
            "S4": oo_contract.validate_oo(synthetic_log("S4", 60, 4)),
        },
    )
    return portfolio.Parent("parent_test", [a, b])


def test_parent_daily_pnl_shape():
    wide = make_parent().daily_pnl()
    assert list(wide.columns) == ["A", "B"]
    assert not wide.isna().any().any()
    # per-child totals survive aggregation
    parent = make_parent()
    assert wide["A"].sum() == pytest.approx(parent.children[0].total_pnl())


def test_size_report_and_constraint():
    parent = make_parent()
    rep = parent.size_report()
    assert set(rep.index) == {"A", "B"}
    assert (rep["active_days"] == 60).all()


def test_parent_save_load_roundtrip(tmp_path):
    parent = make_parent()
    portfolio.save_parent(parent, tmp_path)
    loaded = portfolio.load_parent("parent_test", tmp_path)
    assert [c.name for c in loaded.children] == ["A", "B"]
    assert loaded.children[0].strategies == ["S1", "S2"]
    pd.testing.assert_frame_equal(
        loaded.daily_pnl(), parent.daily_pnl(), check_freq=False
    )


def test_ts_adapter_unfilled_map_raises():
    from mlcpo.data import ts_to_oo

    with pytest.raises(NotImplementedError, match="COLUMN_MAP"):
        ts_to_oo.convert(pd.DataFrame({"anything": [1]}))
