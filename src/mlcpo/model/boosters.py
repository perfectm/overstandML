"""LightGBM / XGBoost regressor construction from an HP-set dict (spec s2, s6).

HP sets (configs/known_params.yaml) use LightGBM sklearn-API names; the
XGBoost path translates the ones that differ. `model` selects the backend —
the reference UI's "LGBM | XGB" toggle.
"""
from __future__ import annotations

# LightGBM name -> XGBoost name (approximate role equivalents)
_XGB_RENAME = {
    "num_leaves": "max_leaves",
    "min_child_samples": "min_child_weight",
    "colsample_bytree": "colsample_bytree",
    "subsample": "subsample",
}


def make_model(hp_set: dict):
    """Build an unfitted regressor from an HP-set dict ({'model': 'lightgbm'
    | 'xgboost', <sklearn-API params>})."""
    hp = dict(hp_set)
    kind = hp.pop("model", "lightgbm")

    if kind == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(**hp, verbosity=-1)

    if kind == "xgboost":
        from xgboost import XGBRegressor

        xgb_hp = {_XGB_RENAME.get(k, k): v for k, v in hp.items()}
        if xgb_hp.get("max_depth", 0) == -1:  # LightGBM's "no limit"
            xgb_hp["max_depth"] = 0
        if "max_leaves" in xgb_hp:
            xgb_hp["grow_policy"] = "lossguide"
        return XGBRegressor(**xgb_hp)

    raise ValueError(f"unknown model kind {kind!r}")
