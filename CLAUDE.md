# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ML CPO: each trading day, a walk-forward-trained gradient-boosting model scores a set of candidate portfolios ("children") and the top-k trade that day. Reverse-engineered from Mauro's reference system — `docs/ML_CPO_Reverse_Engineering_Spec.md` is the authoritative spec; statements there are tagged [CONFIRMED]/[INFERRED]/[UNKNOWN], and §10 lists the open questions (Q1, the exact target variable + "D-Day threshold" mechanism, is the biggest unresolved one — the current target is a documented placeholder: same-day child P&L).

## Commands

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && pip install -e .
# macOS: LightGBM needs `brew install libomp` or imports fail on libomp.dylib

.venv/bin/python -m pytest -q                       # full suite (~30s; real-data tests auto-skip without data/)
.venv/bin/python -m pytest tests/test_walkforward.py -q          # one file
.venv/bin/python -m pytest tests/test_diagnostics.py::test_oracle_gap -q  # one test

python -m mlcpo.data.children <ts.csv> [--style] [--children 6]  # propose a child partition + diagnostics
python -m mlcpo.live <ts.csv> [--date YYYY-MM-DD]   # daily pick + enable/disable checklist -> data/live/<date>.json
python -m mlcpo.app --refresh <ts.csv>              # rebuild web caches (slow: full walk-forwards, ~10 min)
python -m mlcpo.app                                 # serve dashboard from caches on :8050
```

## Data rules (non-negotiable)

- `data/` is gitignored and must stay that way: TradeSteward exports are real trade logs. Never commit anything derived from them (caches, cycles, decisions, Optuna DBs all live in `data/`).
- Input is always a **TradeSteward export** (103 columns, `Backtick UID`…). The "OO column contract" (`mlcpo/data/oo_contract.py`) is purely the internal interchange schema — there are no real Option Omega files and no OO-format compatibility requirements. `ts_to_oo.convert()` is the only TS-specific code; everything downstream consumes the OO frame.
- Verified TS identity (tested): `Trade P/L = Total Entry Value + Exit Price*100*Quantity - Total Exit Fee` — P/L is net of all fees. The export's last row is an end-of-file marker, dropped by the adapter.

## Architecture: one data flow

```
TS export ─ts_to_oo─► OO frame ─children.py─► Parent (6 children) ─daily_pnl()─► date×child P&L
                                                                                     │
features/market.py (CBOE VIX family + yfinance SPX) ──┐                              │
features/child_descriptors.py (trailing stats, D-1-shifted) ──┤                      │
                                                              ▼                      ▼
                                    walkforward.build_dataset(): rows (date, child), target = that day's P&L
                                                              │
                          walkforward.run_walkforward() / predict_day(): train on IS window strictly < D
                                                              │
                       ensemble.py (vote across HP sets) ──► picks ──► live.py decision / app.py dashboard
```

Key invariants that span files:

- **Leakage timing is the core design rule** (spec §4). A row for date D may only contain what is knowable at ~9:31 ET on D: market opens for D, everything else as of D-1 close (term-structure features and all child descriptors are `.shift(1)`-ed). `run_walkforward` trains strictly before D; `tests/test_walkforward.py::test_walkforward_no_leakage_of_prediction_day` is the poison test that guards this — keep it passing.
- **`predict_day()` is shared** by the backtest loop and `live.py` so live predictions cannot drift from the validated path. Don't fork the logic.
- **The live partition is persisted** (`data/partition_live.json`, via `live.save_partition`) and never re-clustered implicitly — re-clustering silently reshuffles children and invalidates history. `build_style_children` (P&L-correlation clusters, size-capped) is the validated construction; balance-packed mixing (`build_children`) measurably underperforms (children become mini-indexes with nothing to select between).
- **Honest baselines** (spec §5): every result reports BASELINE (equal-weight all children) and BEST CHILD (chosen with pre-start data only) alongside ML. Never present an ML result without them; ML currently beats both on total P&L but not equal-weight's Sharpe.
- **Hedge strategies** (name contains "hedge") are excluded from children by default — James's design runs ML-no-hedges vs the with-hedges portfolio as control.
- HP sets live in `configs/known_params.yaml` (evergreen `P3_v2_2` + Optuna candidates). Optuna studies are capped at ≤50 trials by design (overfitting guard, spec §6); with ~370-row training windows the HP landscape is plateaus — 20/50 trials produced bit-identical pick sequences.

## Testing conventions

Real-data tests are marked `skipif` on the presence of `data/RUN_G-trades-20260826-171604.csv` and skip cleanly in CI/clones. Synthetic fixtures: `tests/test_data.py::synthetic_log` (contract-valid OO frames), `tests/test_walkforward.py::make_regime_data` (a learnable regime world where the right pick is known). New pipeline features should prove themselves on the regime world the way the existing end-to-end tests do.
