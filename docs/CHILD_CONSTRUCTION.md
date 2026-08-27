# How the child portfolios were determined

Audience: a reviewer (human or Claude) troubleshooting the partition. Every step
below states exactly what the code does (`src/mlcpo/data/children.py`), with
tuning constants and known weaknesses called out. Statements tagged **[CHOICE]**
are our design decisions, not confirmed properties of Mauro's reference system —
these are the things to challenge and correct.

## Input

The full TradeSteward export (`RUN_G-trades-20260826-171604.csv`: 12,142 SPX
trades, 2022-05-16 → 2026-08-10, 79 strategies), converted to the internal OO
frame by `mlcpo.data.ts_to_oo.convert()`.

**Step 0 — hedge exclusion.** Any strategy whose name contains `"hedge"`
(case-insensitive) is dropped: 19 of 79, leaving 60 income strategies.
Rationale: the hedges net only ~+$13K of the $1.44M total — they are insurance,
not income — and James's head-to-head design runs ML-no-hedges against the
with-hedges portfolio as control. **[CHOICE]** The filter is name-based
(`is_hedge()`); a hedge not named "hedge" would slip through.

## Two algorithms were built; only one survived

### v1 — co-firing + balance-packing (`build_children`) — REJECTED for ML use

Clustered strategies by activation-day overlap (Jaccard of active-day sets,
average linkage, cut at Jaccard ≥ 0.5), then greedy bin-packed the groups into
6 children to equalize total P&L. Result: P&L balanced within 1%, and it
**failed** — walk-forward ML made $180.9K vs the $215.6K equal-weight baseline.
Diagnosis: spreading every style across every child produces six mini-indexes;
they differ too little *conditionally* for any feature to select between.
Kept in the codebase because the co-firing clustering itself is sound (it
recovered the SMA-condor / iron-fly-tranche / PM-puts-zone families from
activation data alone) and it's the cautionary baseline.

### v2 — style-coherent construction (`build_style_children`) — CURRENT

The live partition. Principle: children must be **internally similar,
mutually different portfolio identities** — the same property Mauro's
hand-built children have (SMA condors vs EMA skewed condors etc.).

Exact procedure:

1. **Per-strategy daily P&L frame** — `strategy_daily_pnl()`: date × strategy,
   Date **Closed** attribution, inactive days = 0.
2. **Similarity** — Pearson correlation of those daily P&L columns
   (NaN → 0). **[CHOICE]** Zeros on inactive days mean this correlation blends
   *when a strategy trades* with *how it does when it trades*. A
   correlation computed only on co-active days would be a purer behavior
   measure; we haven't tested it.
3. **Style groups** — average-linkage agglomerative clustering on
   `1 − corr` (`scipy.cluster.hierarchy.linkage` + `fcluster`,
   `criterion="maxclust"`), starting at **15 groups**.
4. **Size-cap refinement loop** — the spec's similar-size constraint (§3):
   children in a parent need comparable total P&L and activation, or the
   model biases toward the big child. Cap = `total P&L / n_children ×
   tolerance` with **tolerance = 1.25**. While any single style group's
   P&L exceeds the cap, re-cluster with 3 more groups (15 → 18 → 21 …).
   This lets a huge family (the 14-strategy Afternoon-SMA cluster,
   ~$421K) split into sub-pieces that are still internally coherent but
   packable. Degenerate guard: if total P&L ≤ 0 the cap is disabled.
5. **Affinity packing** — sort groups by P&L descending; the **6 largest
   seed one child each**; every remaining group goes to the child whose
   current members it correlates with most (mean pairwise corr), among
   children that stay under the cap (fallback: lowest-P&L child).
6. **Persistence** — the resulting {child: [strategies]} map is saved once to
   `data/partition_live.json` (`mlcpo.live.save_partition`) and **never
   re-clustered implicitly**. Re-running the clustering on new data would
   silently reshuffle children and invalidate all trained history. New
   strategies not in the partition are reported loudly and ignored until a
   human re-cuts.

## The resulting partition (RUN G data, 6 children)

| Child | P&L | Strategies | Active days | Identity |
|---|---|---|---|---|
| S1 | $294.6K | 9 | 655 | EMA-skewed condors + SMA butterflies |
| S2 | $293.5K | 12 | 609 | 0DTE iron-fly tranches + power hour |
| S3 | $291.6K | 13 | 632 | VIX-up condors + AM call spreads + ORB |
| S4 | $224.3K | 5 | 767 | High-activation mixed core (RIC5, MOC, Friday fly) |
| S5 | $115.9K | 9 | 267 | Bearish: PM long puts under-SMA + inside-down |
| S6 | $203.9K | 12 | 437 | PM puts over-SMA + weekend/EOM condors |

Diagnostics (`parent_lab.parent_report`): mean pairwise child P&L correlation
**0.006** (max 0.24), rotation entropy 0.947, decisive days 89.6%, zero
all-child-zero days, median daily win-gap $733, oracle gap $1.40M.

Validation: identical walk-forward (P3_v2_2, IS=3m rolling, top-1, 979
cycles) on v1 vs v2 partitions — ML $180.9K (below both baselines) vs
**$280.6K (above both)**. Composition was the single changed variable.

## Known weaknesses / things to correct

1. **S5 violates size-similarity in spirit** ($116K, 267 active days vs
   S4's 767): the bearish-puts style is inherently small and infrequent.
   Style purity vs size balance is a real tension; tolerance=1.25 caps the
   top but doesn't lift the bottom.
2. **Correlation conflates activation and behavior** (step 2 note above).
3. **Cluster count / tolerance / seed count are untuned constants** —
   n_style_groups=15, +3 per refinement, tolerance=1.25 were chosen once and
   never swept.
4. **Hedge filter is name-based.**
5. **Date Closed attribution only** — the spec's Parent Lab uses both bases;
   an entry-cohort (Date Opened) partition sanity check hasn't been done.
6. **No human review step** — Mauro composes children by hand in his Parent
   Lab; ours is fully automatic. The partition is a candidate, not gospel:
   re-cut with `python -m mlcpo.data.children <ts.csv> --style` and diff.
