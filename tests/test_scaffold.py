"""Smoke tests: package imports and config parses."""
import yaml
from pathlib import Path


def test_imports():
    import mlcpo  # noqa: F401
    from mlcpo.data import oo_contract, ts_to_oo, portfolio  # noqa: F401
    from mlcpo.features import market, child_descriptors  # noqa: F401
    from mlcpo.model import walkforward, hp, ensemble  # noqa: F401
    from mlcpo.diagnostics import parent_lab  # noqa: F401


def test_known_params_config():
    cfg = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "known_params.yaml").read_text()
    )
    assert cfg["hp_sets"]["P3_v2_2"]["num_leaves"] == 36
    assert cfg["walkforward"]["is_months"] == 3
    assert cfg["optuna"]["max_trials"] == 50
