"""Strategy CSVs -> children -> parent assembly (spec section 3, Phase 1).

Hierarchy: trade log (strategy) -> child (4-7 strategies trading as a unit,
independently tradable) -> parent (N children = the ML candidate set).

Attribution matters downstream (spec section 7, open question #6): realized
equity uses Date Closed; entry-cohort analysis uses Date Opened. Daily PNL
here supports both bases.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .oo_contract import validate_oo


def load_strategy_log(path: str | Path) -> pd.DataFrame:
    """Load one OO-format trade log CSV and validate it against the contract."""
    return validate_oo(pd.read_csv(path))


@dataclass
class Child:
    """A named selection of strategies that trades as a unit."""

    name: str
    strategies: list[str]
    trades: pd.DataFrame = field(repr=False, default=None)

    @classmethod
    def from_logs(cls, name: str, logs: dict[str, pd.DataFrame]) -> "Child":
        """Assemble from {strategy name: validated OO frame}."""
        trades = pd.concat(logs.values(), ignore_index=True)
        return cls(name=name, strategies=list(logs), trades=trades)

    def daily_pnl(self, basis: str = "closed") -> pd.Series:
        """Daily child PNL. basis='closed' (realized, Date Closed) or
        'opened' (entry cohorts, Date Opened)."""
        col = {"closed": "Date Closed", "opened": "Date Opened"}[basis]
        return self.trades.groupby(col)["P/L"].sum().rename(self.name)

    def total_pnl(self) -> float:
        return float(self.trades["P/L"].sum())

    def active_days(self, basis: str = "closed") -> int:
        return int(self.daily_pnl(basis).shape[0])


@dataclass
class Parent:
    """N children assembled into one ML candidate set — 'a technicality',
    but it is the unit the ML runs against."""

    name: str
    children: list[Child]

    def daily_pnl(self, basis: str = "closed") -> pd.DataFrame:
        """Date x child daily-PNL matrix, missing days filled with 0
        (a child that didn't trade that day made nothing)."""
        wide = pd.concat(
            [c.daily_pnl(basis) for c in self.children], axis=1
        ).sort_index()
        return wide.fillna(0.0)

    def size_report(self) -> pd.DataFrame:
        """Per-child total PNL and activation frequency — the two axes of
        the similar-size constraint (spec section 3): children of one parent
        must be comparable on both, or the learner biases toward the
        larger/more active child."""
        return pd.DataFrame(
            {
                "total_pnl": [c.total_pnl() for c in self.children],
                "active_days": [c.active_days() for c in self.children],
                "n_strategies": [len(c.strategies) for c in self.children],
                "n_trades": [len(c.trades) for c in self.children],
            },
            index=[c.name for c in self.children],
        )

    def check_similar_size(self, max_ratio: float = 3.0) -> bool:
        """True when max/min total-PNL magnitude and activation frequency
        both stay within max_ratio. Advisory threshold, not from the spec."""
        rep = self.size_report()
        pnl = rep["total_pnl"].abs()
        act = rep["active_days"]
        pnl_ok = pnl.min() > 0 and pnl.max() / pnl.min() <= max_ratio
        act_ok = act.min() > 0 and act.max() / act.min() <= max_ratio
        return bool(pnl_ok and act_ok)


def save_parent(parent: Parent, directory: str | Path) -> Path:
    """Write parent trades to <name>.csv (with a Child column) and register
    it in parents.json — mirroring the reference layout (parent_N.csv +
    parents.json index)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    frames = []
    for child in parent.children:
        f = child.trades.copy()
        f["Child"] = child.name
        frames.append(f)
    csv_path = directory / f"{parent.name}.csv"
    pd.concat(frames, ignore_index=True).to_csv(csv_path, index=False)

    index_path = directory / "parents.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    index[parent.name] = {
        "file": csv_path.name,
        "children": {c.name: c.strategies for c in parent.children},
    }
    index_path.write_text(json.dumps(index, indent=2))
    return csv_path


def load_parent(name: str, directory: str | Path) -> Parent:
    """Reload a parent saved by save_parent()."""
    directory = Path(directory)
    index = json.loads((directory / "parents.json").read_text())
    entry = index[name]
    trades = validate_oo(pd.read_csv(directory / entry["file"]))
    children = [
        Child(
            name=child_name,
            strategies=strategies,
            trades=trades[trades["Child"] == child_name].drop(columns="Child"),
        )
        for child_name, strategies in entry["children"].items()
    ]
    return Parent(name=name, children=children)
