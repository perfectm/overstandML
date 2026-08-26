"""Hyperparameter sets and Optuna studies (spec section 6).

Workflow: evergreen HP sets validate any NEW parent first; only promising
parents get an Optuna study, deliberately capped at <=50 trials to limit
overfitting. Known sets live in configs/known_params.yaml.

Note the regularization posture of the reference sets (small leaves, high
min_child_samples, aggressive sub/colsample): a 3-month daily window across
4 children is only ~250-350 training rows.
"""
from __future__ import annotations
import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "known_params.yaml"


def load_hp_sets() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)["hp_sets"]


def optuna_study(
    dataset,
    cfg=None,
    n_trials: int = 50,
    metric: str = "total_pnl",
    storage: str | None = None,
    study_name: str = "mlcpo",
    seed: int = 42,
    n_jobs: int = 1,
):
    """Phase 4 Optuna study over the full walk-forward (spec section 6).

    Deliberately capped at <=50 trials — the reference author's overfitting
    guard; each trial IS an in-sample selection over the whole history, so
    treat the best set as a candidate for the ensemble, not truth.

    metric: 'total_pnl' (headline) or 'mar'/'sharpe' — all three are stored
    as user attrs on every trial regardless. Search space mirrors the
    regularization posture of the evergreen sets (small leaves, high
    min_child_samples, aggressive sub/colsampling).
    """
    import optuna

    from ..diagnostics import metrics as m
    from .walkforward import run_walkforward

    def objective(trial: "optuna.Trial") -> float:
        hp = {
            "model": "lightgbm",
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "max_depth": trial.suggest_categorical("max_depth", [-1, 4, 6, 8]),
            "num_leaves": trial.suggest_int("num_leaves", 8, 64),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            # LightGBM ignores subsample unless subsample_freq > 0
            "subsample_freq": trial.suggest_categorical("subsample_freq", [0, 1]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "random_state": 4998,
        }
        res = run_walkforward(dataset, hp, cfg)
        s = m.summary(res.stream("ml"))
        for key in ("total_pnl", "sharpe", "mar", "max_dd"):
            trial.set_user_attr(key, float(s[key]))
        return float(s[metric])

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage=storage,
        study_name=study_name,
        load_if_exists=storage is not None,
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    return study
