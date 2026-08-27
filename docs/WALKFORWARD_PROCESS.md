# The IS/OoS walk-forward process, exactly as implemented

Audience: a reviewer (human or Claude) troubleshooting the training/prediction
loop. Code: `src/mlcpo/model/walkforward.py` (engine), `src/mlcpo/features/`
(inputs), `src/mlcpo/model/ensemble.py`, `src/mlcpo/live.py`. Tags:
**[CONFIRMED]** = stated by Mauro; **[INFERRED]** = our reading of evidence;
**[CHOICE]** = our design decision pending correction. The **[CHOICE]** and
**[INFERRED]** items are what to troubleshoot.

## 1. The dataset

One row per **(date D, child c)**, built by `build_dataset()`:

**Market features** (shared by all children on D, `features/market.py`) —
the leakage rule is the core design constraint **[CONFIRMED]**: a row for D
holds only what is knowable at ~9:31 ET on D.

| Column | Definition | Timing |
|---|---|---|
| `opening_vix` | VIX daily-bar Open for D | day-of |
| `opening_spx` | SPX (^GSPC) daily-bar Open for D | day-of |
| `opening_gap` | SPX Open(D) / Close(D-1) − 1 | day-of |
| `vix`, `vix9d`, `vix3m`, `vix6m` | index Close, **shifted one day** (D-1's close) | prior-close |

Sources: CBOE public daily CSVs for the VIX family (Yahoo dropped
^VIX9D/^VIX3M/^VIX6M), yfinance for SPX. **[CHOICE]** Daily-bar Open is a
proxy for the true 9:31 capture — fine for research; live should capture real
9:31 quotes. VIX1D deliberately excluded **[CONFIRMED]**.

**Child descriptors** (per child, `features/child_descriptors.py`), all
computed from the child's daily P&L and then `.shift(1)`-ed so row D sees
only data through D-1: trailing P&L sums over 5/10/20 days, activation rate
(20d), hit rate of active days (20d), drawdown dollars below running equity
high, consecutive days since high. **[CHOICE]** This list is our guess at
spec §4's unknown descriptor set (open question #2) — untested candidates:
composition summaries, market-beta descriptors ("Beta_features" in the
reference UI).

**Target** — `target` = child c's realized P&L on D (Date Closed basis,
0 if inactive). **[CHOICE — the biggest one]** The reference target is
unknown (spec §10 Q1): could be normalized P&L, multi-day, or paired with the
"D-Day threshold 0.65" probability gate. We use raw same-day P&L and have
**no D-Day gate implemented**. Any correction here likely changes results
materially.

## 2. The loop

`run_walkforward()` — the same cycle Mauro described directly (26 Aug 2026)
**[CONFIRMED]**: TRAIN on history (features + target) → PREDICT the next day
(features only) → shift one day → repeat.

For each prediction date D (all dataset dates ≥ data start + IS months):

1. **IS window**: rolling (unanchored, default): `[D − 3 months, D)` by
   calendar `DateOffset`, ≈ 62 trading days ≈ **370 training rows** with 6
   children. Anchored option: `[data start, D)` (expanding). Mauro's constant
   3-month IS implies rolling **[INFERRED]**; his demo UI showed an
   Anchored toggle.
2. **Train**: `dropna()` on training rows (early rows lack full trailing
   descriptors); skip the cycle entirely if fewer than `min_train_rows=50`
   remain **[CHOICE]**. Fit a fresh LightGBM regressor
   (`boosters.make_model`) on feature columns → target.
3. **Predict**: score D's rows (one per child; prediction-row NaNs filled
   with 0 **[CHOICE]**), rank descending, take **top-k** (k=1 live).
4. **Realize**: ML P&L for D = sum of picked children's actual P&L on D.
5. Advance one day. First run on real data: 979 cycles, first prediction
   2022-08-16, ~70s wall-clock for the evergreen HP set.

Leakage guarantees: training strictly excludes D; a poison test
(`tests/test_walkforward.py::test_walkforward_no_leakage_of_prediction_day`)
plants a $1M outcome on the final day with uninformative features and asserts
the engine cannot see it. `predict_day()` is the extracted single-cycle step
shared verbatim by the backtest and `mlcpo.live` — live predictions cannot
drift from the validated path.

## 3. Baselines (the honesty rules) [CONFIRMED as discipline, definitions partly INFERRED]

Every run reports three streams over the identical prediction dates:

- **BASELINE** = equal-weight mean of all children's P&L each day.
  **[INFERRED]** — spec Q4; the reference's exact pool definition is unknown.
- **BEST CHILD** = the single child with the highest total P&L using **only
  data strictly before the first prediction date**, held statically. This is
  the as-of-start honest pick (the reference's Dec-31 "P4 would have been
  rational" rule).
- **ML** = the walk-forward selection.

Never quote an ML result without both. Current standing (6 style children,
979 cycles): ML $280.6K > BEST CHILD $257.8K > BASELINE $215.6K on total
P&L; BASELINE still wins risk-adjusted (Sharpe 7.7 vs 3.7).

## 4. Hyperparameters and Optuna

HP sets in `configs/known_params.yaml`. Evergreen `P3_v2_2` (from the
reference UI **[CONFIRMED]**): est=240, lr=0.05, depth=-1, leaves=36,
minleaf=100, subsample=0.9, colsample=0.5, seed=4998. Gotcha: LightGBM
**ignores `subsample` unless `subsample_freq > 0`** — the reference numbers
don't show that field, so bagging may be silently off in both systems.

Optuna (`model/hp.py::optuna_study`): TPE, ≤50 trials **[CONFIRMED cap]**,
objective = full-walk-forward total P&L **[CHOICE]**; each trial is in-sample
selection over the whole history, so winners are ensemble candidates, not
truth. Empirical findings on this data size: 0/50 trials beat the evergreen
set on total P&L; 20/50 trials produced **bit-identical pick sequences**
(with ~370-row windows and heavy regularization the HP landscape is plateaus,
not peaks).

## 5. Ensemble [aggregation method is spec Q3 — UNKNOWN in reference]

`model/ensemble.py`: run the loop once per HP set, combine per-day scores:
`mean_rank` (scale-free average of per-day ranks), `vote` (each set votes its
top-k; mean-rank tie-break scaled to never outweigh a vote — **default**), or
`mean_score`. Measured motivation: single sets agree on as little as 18% of
daily picks while landing within 2.5% of each other's totals — pick sequences
are fragile, aggregates aren't. In-sample the ensemble cannot beat the best
single set (selection bias); its value is fragility reduction (worst month
−$3.6K vs −$7.2K) and can only be honestly judged out-of-sample.

## 6. Delta vs the reference system (troubleshooting checklist)

1. Target definition + D-Day gate: unknown vs our raw same-day P&L, no gate.
2. BASELINE row definition: inferred equal-weight.
3. Anchored vs rolling: we default rolling; reference demo showed anchored.
4. Descriptor list: our guess; reference list unknown.
5. 9:31 capture: daily-bar Open proxy.
6. Ensemble aggregation: our vote/mean-rank; reference method unknown.
7. Multi-tranche rollup: we sum trades to daily child P&L on Date Closed;
   reference attribution details unknown (spec Q6).
8. Sanity anchor still to run: feed the same children through the reference
   tool and check both systems pick similar children on similar days (spec §9).
