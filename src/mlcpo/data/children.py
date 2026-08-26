"""Child construction (spec section 9, Phase 1).

Partition the strategy universe into n similar-size children along
co-firing lines:

  1. hedge strategies (name contains 'hedge') are excluded by default —
     James's head-to-head design runs ML-no-hedges against the full
     with-hedges portfolio as control
  2. strategies that co-fire (high Jaccard of active days) are clustered
     into groups — they behave as one unit, splitting them across children
     would just duplicate exposure
  3. groups are greedily bin-packed into n children balancing total PNL,
     with activation frequency as the tie-break — the similar-size
     constraint (spec section 3): comparable total PNL AND comparable
     activation frequency, or the learner biases toward the big child
"""
from __future__ import annotations

import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .portfolio import Child, Parent


def is_hedge(strategy_name: str) -> bool:
    return "hedge" in strategy_name.lower()


def activation_jaccard(oo_df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Jaccard similarity of strategies' active-day sets
    (Date Closed basis) — the co-firing measure."""
    active = (
        oo_df.groupby(["Date Closed", "Strategy"])["P/L"].size().unstack(fill_value=0).gt(0)
    )
    a = active.astype(float)
    both = a.T @ a
    days = a.sum()
    union = days.values[:, None] + days.values[None, :] - both
    return both / union


def cofire_groups(oo_df: pd.DataFrame, min_jaccard: float = 0.5) -> list[list[str]]:
    """Cluster strategies into co-firing groups: average-linkage
    agglomerative clustering on 1 - Jaccard, cut at 1 - min_jaccard."""
    jac = activation_jaccard(oo_df)
    if len(jac) == 1:
        return [list(jac.index)]
    dist = 1.0 - jac.values
    dist = (dist + dist.T) / 2.0  # exact symmetry for squareform
    labels = fcluster(
        linkage(squareform(dist, checks=False), method="average"),
        t=1.0 - min_jaccard,
        criterion="distance",
    )
    groups: dict[int, list[str]] = {}
    for name, lab in zip(jac.index, labels):
        groups.setdefault(lab, []).append(name)
    return list(groups.values())


def pack_groups(
    group_stats: pd.DataFrame, n_children: int
) -> dict[str, list[int]]:
    """Greedy balanced bin-packing: groups (rows with total pnl + active
    days), largest PNL first, each into the child with the lowest running
    PNL (ties: lowest running activation). Returns {child name: group ids}."""
    bins = {f"C{i+1}": {"groups": [], "pnl": 0.0, "days": 0} for i in range(n_children)}
    for gid, row in group_stats.sort_values("pnl", ascending=False).iterrows():
        target = min(bins, key=lambda c: (bins[c]["pnl"], bins[c]["days"]))
        bins[target]["groups"].append(gid)
        bins[target]["pnl"] += row["pnl"]
        bins[target]["days"] += row["days"]
    return {c: b["groups"] for c, b in bins.items()}


def build_children(
    oo_df: pd.DataFrame,
    n_children: int = 6,
    min_jaccard: float = 0.5,
    include_hedges: bool = False,
) -> tuple[Parent, pd.DataFrame]:
    """Full Phase 1: filter -> co-firing groups -> balanced pack -> Parent.

    Returns (parent, assignment table: one row per strategy with its group
    id and child).
    """
    df = oo_df if include_hedges else oo_df[~oo_df["Strategy"].map(is_hedge)]

    groups = cofire_groups(df, min_jaccard)
    per_strategy = df.groupby("Strategy").agg(
        pnl=("P/L", "sum"), days=("Date Closed", "nunique")
    )
    group_stats = pd.DataFrame(
        [
            {
                "gid": i,
                "pnl": per_strategy.loc[g, "pnl"].sum(),
                # group activation ~ union of members' active days
                "days": int(
                    df[df["Strategy"].isin(g)]["Date Closed"].nunique()
                ),
                "strategies": g,
            }
            for i, g in enumerate(groups)
        ]
    ).set_index("gid")

    packing = pack_groups(group_stats, n_children)

    children = []
    rows = []
    for child_name, gids in sorted(packing.items()):
        members = [s for gid in gids for s in group_stats.loc[gid, "strategies"]]
        logs = {s: df[df["Strategy"] == s] for s in members}
        children.append(Child.from_logs(child_name, logs))
        rows += [
            {"strategy": s, "gid": gid, "child": child_name}
            for gid in gids
            for s in group_stats.loc[gid, "strategies"]
        ]

    parent = Parent(name=f"parent_{n_children}c", children=children)
    assignment = pd.DataFrame(rows).set_index("strategy").sort_values(["child", "gid"])
    return parent, assignment


def strategy_daily_pnl(oo_df: pd.DataFrame) -> pd.DataFrame:
    """Date x strategy daily-PNL frame (Date Closed basis), zeros filled."""
    return (
        oo_df.groupby(["Date Closed", "Strategy"])["P/L"].sum().unstack(fill_value=0.0)
    )


def style_groups(oo_df: pd.DataFrame, n_groups: int = 15) -> list[list[str]]:
    """Cluster strategies into style groups by daily-PNL correlation
    (average linkage on 1 - corr, cut to n_groups clusters). Unlike
    cofire_groups (activation overlap), this groups strategies that BEHAVE
    alike — same-family tranches and different systems with the same
    regime exposure."""
    corr = strategy_daily_pnl(oo_df).corr().fillna(0.0)
    if len(corr) <= n_groups:
        return [[s] for s in corr.index]
    dist = 1.0 - corr.values
    dist = (dist + dist.T) / 2.0
    labels = fcluster(
        linkage(squareform(dist, checks=False), method="average"),
        t=n_groups,
        criterion="maxclust",
    )
    groups: dict[int, list[str]] = {}
    for name, lab in zip(corr.index, labels):
        groups.setdefault(lab, []).append(name)
    return list(groups.values())


def build_style_children(
    oo_df: pd.DataFrame,
    n_children: int = 6,
    n_style_groups: int = 15,
    include_hedges: bool = False,
    tolerance: float = 1.25,
) -> tuple[Parent, pd.DataFrame]:
    """Style-coherent variant of build_children: children are internally
    correlated, mutually different portfolio identities — the hypothesis
    (from the first walk-forward run) being that mixed-composition children
    are mini-indexes with too little conditional differentiation to select
    between.

    Packing: the n_children largest style groups seed one child each; every
    remaining group goes to the child its members correlate with most,
    among children whose PNL stays under (total/n)*tolerance (fallback:
    lowest-PNL child). Keeps the spec s3 similar-size constraint soft
    rather than exact — style purity is the point here.
    """
    df = oo_df if include_hedges else oo_df[~oo_df["Strategy"].map(is_hedge)]

    daily = strategy_daily_pnl(df)
    corr = daily.corr().fillna(0.0)
    per_strategy = df.groupby("Strategy")["P/L"].sum()

    # refine granularity until no single style group busts the size cap —
    # sub-pieces of one family remain style-coherent but can be packed
    # evenly (a 14-strategy family may legitimately span two children)
    cap_pnl = per_strategy.sum() / n_children * tolerance
    if cap_pnl <= 0:  # degenerate universe (non-positive total) — no cap
        cap_pnl = float("inf")
    groups = style_groups(df, n_style_groups)
    while (
        max(per_strategy.loc[g].sum() for g in groups) > cap_pnl
        and n_style_groups < len(per_strategy)
    ):
        n_style_groups += 3
        groups = style_groups(df, n_style_groups)

    stats = sorted(
        ((float(per_strategy.loc[g].sum()), g) for g in groups), reverse=True
    )
    cap = per_strategy.sum() / n_children * tolerance

    bins: list[dict] = [
        {"name": f"S{i+1}", "members": list(g), "pnl": p}
        for i, (p, g) in enumerate(stats[:n_children])
    ]
    for pnl, g in stats[n_children:]:
        fits = [b for b in bins if b["pnl"] + pnl <= cap] or [
            min(bins, key=lambda b: b["pnl"])
        ]
        target = max(
            fits, key=lambda b: corr.loc[g, b["members"]].values.mean()
        )
        target["members"] += g
        target["pnl"] += pnl

    children = [
        Child.from_logs(b["name"], {s: df[df["Strategy"] == s] for s in b["members"]})
        for b in bins
    ]
    assignment = pd.DataFrame(
        [{"strategy": s, "child": b["name"]} for b in bins for s in b["members"]]
    ).set_index("strategy").sort_values("child")
    return Parent(name=f"style_parent_{n_children}c", children=children), assignment


def main(argv=None):
    import argparse

    from ..diagnostics import parent_lab
    from .ts_to_oo import convert_file

    ap = argparse.ArgumentParser(description="Phase 1: propose a child partition")
    ap.add_argument("ts_csv", help="TradeSteward export CSV")
    ap.add_argument("--children", type=int, default=6)
    ap.add_argument("--min-jaccard", type=float, default=0.5)
    ap.add_argument("--include-hedges", action="store_true")
    ap.add_argument(
        "--style",
        action="store_true",
        help="style-coherent construction (PNL-correlation clusters) instead of balance-packed mixing",
    )
    ap.add_argument("--save-dir", help="save parent_N.csv + parents.json here")
    args = ap.parse_args(argv)

    oo = convert_file(args.ts_csv)
    if args.style:
        parent, assignment = build_style_children(
            oo, args.children, include_hedges=args.include_hedges
        )
    else:
        parent, assignment = build_children(
            oo, args.children, args.min_jaccard, args.include_hedges
        )

    print(f"== assignment ({len(assignment)} strategies) ==")
    print(assignment.to_string())
    print("\n== size report (similar-size constraint, spec s3) ==")
    print(parent.size_report().round(0).to_string())
    print(f"\nsimilar-size ok (3x ratio): {parent.check_similar_size()}")

    report = parent_lab.parent_report(parent)
    print("\n== parent diagnostics ==")
    for k, v in report.items():
        if k != "per_child":
            print(f"  {k}: {v:,.3f}" if isinstance(v, float) else f"  {k}: {v}")
    print(report["per_child"].round(3).to_string())

    if args.save_dir:
        from .portfolio import save_parent

        path = save_parent(parent, args.save_dir)
        print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
