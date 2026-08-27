# Hyperparameters in use

Source of truth: `configs/known_params.yaml` (loaded by `mlcpo.model.hp.load_hp_sets`).
All models are LightGBM regressors (sklearn API, `LGBMRegressor`); the XGBoost
toggle exists (`boosters.make_model`) but nothing live uses it.

## The live ensemble (what "ML is using")

The default in both `mlcpo.live` and the dashboard cache is a **vote ensemble of
three sets** — each set independently ranks the children per day, votes for its
top-k, most votes wins (mean-rank tie-break):

| Parameter | `P3_v2_2` (evergreen) | `optuna_s6_pnl` | `optuna_s6_mar` |
|---|---|---|---|
| `n_estimators` | 240 | 250 | 308 |
| `learning_rate` | 0.05 | 0.13321981713769523 | 0.020762404986637867 |
| `max_depth` | -1 (no limit) | 8 | 6 |
| `num_leaves` | 36 | 24 | 30 |
| `min_child_samples` | 100 | 191 | 20 |
| `subsample` | 0.9 | 0.8442175459606699 | 0.6103482291446671 |
| `subsample_freq` | — (unset = 0) | 1 | 0 |
| `colsample_bytree` | 0.5 | 0.4421156110631817 | 0.30799079042609434 |
| `random_state` | 4998 | 4998 | 4998 |

Everything else is LightGBM defaults (`verbosity=-1` set by the wrapper).

### Provenance and standalone performance (979 cycles, 6 style children, top-1)

- **`P3_v2_2`** — read directly off the reference system's MLCPO run panel
  [CONFIRMED, spec §6]; Mauro's demo/live-adjacent evergreen set.
  Standalone: **$280.6K**, Sharpe 3.65, MAR 11.10, MaxDD −$12.8K.
- **`optuna_s6_pnl`** — best of the 50-trial Optuna study
  (`style6_topk1_v1`, objective = total P&L). Standalone: $273.7K,
  Sharpe 4.04, MAR 13.63, MaxDD −$9.3K. 20/50 trials produced bit-identical
  pick sequences to this one (HP plateau).
- **`optuna_s6_mar`** — trial 13 of the same study, best MAR. Standalone:
  $268.2K, Sharpe 3.50, MAR 15.55, MaxDD −$5.9K (half the evergreen's DD).

Optuna values are written programmatically from the study DB and asserted
exact — don't round them when porting.

## Walk-forward settings (`WalkForwardConfig` defaults)

| Setting | Value | Status |
|---|---|---|
| `is_months` | 3 (≈62 trading days ≈ 370 rows) | CONFIRMED (Mauro direct) |
| `oos_days` | 1 | CONFIRMED |
| `anchored` | False (rolling window) | INFERRED from constant 3-month IS |
| `top_k` | 1 | CONFIRMED (live; MMC experiment uses 2) |
| `d_day_threshold` | 0.65 — **carried but UNUSED** (mechanism unknown, spec Q1) | UNKNOWN |
| `min_train_rows` | 50 (skip cycle below this) | CHOICE |
| `initial_equity` | 100,000 | CONFIRMED (demo) |
| ensemble method | `vote` | CHOICE (reference method unknown, spec Q3) |

## Second reference set (not in the ensemble)

`static1` — also visible in the reference UI but with shaky OCR on two
values: est=213, lr=0.06, depth=4, leaves=380(?), minleaf=400(?), sub=0.8,
col=0.7, seed=8849. Kept in the config flagged `verify: true`; unused.

## Notes for troubleshooting

1. **`subsample` is inert without `subsample_freq > 0`** in LightGBM. The
   evergreen set has no `subsample_freq`, so its `subsample: 0.9` does
   nothing — and if Mauro's tool passes params the same sklearn way, his
   does nothing either. Worth asking him.
2. The regularization posture (small leaves, high min_child_samples,
   aggressive column sampling) is what ~370-row training windows demand;
   `optuna_s6_mar`'s min_child_samples=20 is the outlier and is ~5× slower
   to fit.
3. `random_state=4998` everywhere makes runs bit-reproducible; the Optuna
   study itself used TPE seed 42, sqlite storage `data/optuna.db3`
   (gitignored, on the dev machine).
4. Optuna search space, if re-running: n_estimators 100–400, lr 0.02–0.15
   (log), max_depth {-1,4,6,8}, num_leaves 8–64, min_child_samples 20–200,
   subsample 0.6–1.0, subsample_freq {0,1}, colsample_bytree 0.3–1.0,
   ≤50 trials (the reference's overfitting cap).
