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


def optuna_study(dataset, n_trials: int = 50):
    raise NotImplementedError("Phase 4")
