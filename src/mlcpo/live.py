"""LIVE daily operation + execution hook (spec section 7 LIVE, section 9 Phase 5).

Morning loop: refresh features -> one TRAIN->PREDICT cycle per HP set on the
trailing IS window -> ensemble-combine -> today's pick -> execution plan.

The execution plan is the Phase 5 hook: which TS Bot Portfolios to enable
(the picked children's strategies) and which to disable. Execution is
deliberately human-verified — James's requirement — so the plan is emitted
as a JSON decision file + printed checklist rather than pushed straight to
TradeSteward. When TS API access is configured, an executor can consume the
same decision file.

Partition stability: live children must be STABLE day to day, so the
partition (child -> strategies) is persisted to JSON once and loaded
thereafter — never re-clustered implicitly (re-clustering silently
reshuffles children and invalidates the trained history).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from .data.portfolio import Child, Parent
from .features.market import build_market_features, fetch_ohlc
from .model.ensemble import combine_scores
from .model.walkforward import WalkForwardConfig, build_dataset, predict_day

DEFAULT_PARTITION_PATH = Path("data/partition_live.json")


# ---------------------------------------------------------------- partition

def save_partition(assignment: pd.DataFrame, path: str | Path = DEFAULT_PARTITION_PATH) -> Path:
    """Persist a build_children()/build_style_children() assignment table
    ({strategy -> child}) as the stable live partition."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, list[str]] = {}
    for strategy, row in assignment.iterrows():
        mapping.setdefault(row["child"], []).append(strategy)
    path.write_text(json.dumps(mapping, indent=2, sort_keys=True))
    return path


def load_partition(path: str | Path = DEFAULT_PARTITION_PATH) -> dict[str, list[str]]:
    return json.loads(Path(path).read_text())


def apply_partition(oo_df: pd.DataFrame, partition: dict[str, list[str]]) -> Parent:
    """Build the Parent from a persisted partition. Strategies in the data
    but not in the partition (new bots since the partition was cut) are
    reported loudly rather than silently dropped."""
    known = {s for members in partition.values() for s in members}
    present = set(oo_df["Strategy"].unique())
    unassigned = sorted(present - known)
    if unassigned:
        print(f"WARNING: {len(unassigned)} strategies not in partition (ignored): {unassigned}")
    children = [
        Child.from_logs(name, {s: oo_df[oo_df["Strategy"] == s] for s in members if s in present})
        for name, members in sorted(partition.items())
    ]
    return Parent(name="live", children=children)


# ----------------------------------------------------------------- decision

@dataclass
class LiveDecision:
    date: str
    picks: list[str]
    combined_scores: dict[str, float]
    per_set_scores: dict[str, dict[str, float]]
    method: str
    top_k: int
    enable: dict[str, list[str]] = field(default_factory=dict)   # child -> strategies ON
    disable: dict[str, list[str]] = field(default_factory=dict)  # child -> strategies OFF

    def save(self, directory: str | Path = "data/live") -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.date}.json"
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    def checklist(self) -> str:
        lines = [
            f"== ML CPO decision for {self.date} (method={self.method}, top_k={self.top_k}) ==",
            f"PICK: {', '.join(self.picks)}",
            "",
            "scores (combined): "
            + "  ".join(f"{c}={v:,.0f}" for c, v in sorted(self.combined_scores.items(), key=lambda kv: -kv[1])),
            "",
            "TS Bot Portfolios — VERIFY BY HAND BEFORE MARKET:",
        ]
        for child, strats in sorted(self.enable.items()):
            lines.append(f"  ENABLE  {child} ({len(strats)} bots)")
        for child, strats in sorted(self.disable.items()):
            lines.append(f"  disable {child} ({len(strats)} bots)")
        return "\n".join(lines)


def daily_decision(
    oo_df: pd.DataFrame,
    partition: dict[str, list[str]],
    hp_sets: dict[str, dict],
    cfg: WalkForwardConfig | None = None,
    method: str = "vote",
    pick_date: str | pd.Timestamp | None = None,
    market_features: pd.DataFrame | None = None,
) -> LiveDecision:
    """Produce the decision for pick_date (default: the last date with
    features). market_features may be injected for tests/replays; live use
    fetches through today — day-of rows need today's opens, so run after
    9:31 ET."""
    parent = apply_partition(oo_df, partition)
    pnl = parent.daily_pnl()

    if market_features is None:
        market_features = build_market_features(
            fetch_ohlc(str(pnl.index.min().date()), None)
        )

    # prediction rows need features for D even though PNL for D is unknown
    # yet: extend the PNL frame through the feature dates with zeros so
    # build_dataset emits rows for D (target 0 is ignored at predict time)
    all_dates = pnl.index.union(market_features.index)
    ds = build_dataset(pnl.reindex(all_dates, fill_value=0.0), market_features)

    cfg = cfg or WalkForwardConfig()
    d = pd.Timestamp(pick_date) if pick_date else ds.index.get_level_values("date").max()

    per_set = {}
    for name, hp in hp_sets.items():
        preds = predict_day(ds, hp, cfg, d)
        if preds is None:
            raise ValueError(f"not enough history before {d.date()} for set {name}")
        per_set[name] = preds.to_frame(name=d).T

    combined = combine_scores(per_set, method, cfg.top_k).iloc[0]
    picks = list(combined.nlargest(cfg.top_k).index)

    return LiveDecision(
        date=str(d.date()),
        picks=picks,
        combined_scores={c: float(v) for c, v in combined.items()},
        per_set_scores={
            n: {c: float(v) for c, v in f.iloc[0].items()} for n, f in per_set.items()
        },
        method=method,
        top_k=cfg.top_k,
        enable={c: partition[c] for c in picks},
        disable={c: partition[c] for c in partition if c not in picks},
    )


def main(argv=None):
    import argparse

    from .data.ts_to_oo import convert_file
    from .model.hp import load_hp_sets

    ap = argparse.ArgumentParser(description="LIVE: today's child pick + execution checklist")
    ap.add_argument("ts_csv", help="latest TradeSteward export CSV")
    ap.add_argument("--partition", default=str(DEFAULT_PARTITION_PATH))
    ap.add_argument("--sets", default="P3_v2_2,optuna_s6_pnl,optuna_s6_mar")
    ap.add_argument("--method", default="vote", choices=["mean_rank", "vote", "mean_score"])
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--date", help="pick date YYYY-MM-DD (default: latest with features)")
    ap.add_argument("--out-dir", default="data/live")
    args = ap.parse_args(argv)

    hp = load_hp_sets()
    decision = daily_decision(
        convert_file(args.ts_csv),
        load_partition(args.partition),
        {n: hp[n] for n in args.sets.split(",")},
        WalkForwardConfig(top_k=args.top_k),
        method=args.method,
        pick_date=args.date,
    )
    path = decision.save(args.out_dir)
    print(decision.checklist())
    print(f"\ndecision file: {path}")


if __name__ == "__main__":
    main()
