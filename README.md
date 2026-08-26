# ML CPO

An independent implementation of a daily child-portfolio-selection ("CPO") system for
0DTE/short-dated SPX options portfolios, targeting **TradeSteward** trade data.

Each day, a walk-forward-trained gradient-boosting model scores a set of candidate
portfolios ("children") assembled into a parent, ranks them, and selects the top-k to
trade that day. The design is reverse-engineered from a working system by mauro3000 —
see [docs/ML_CPO_Reverse_Engineering_Spec.md](docs/ML_CPO_Reverse_Engineering_Spec.md)
for the full specification, evidence, and confidence levels.

## Architecture

```
TradeSteward export ──► TS→OO adapter ──► strategy CSVs (OO column contract)
                                              │
                                   child construction (similar-size constraint)
                                              │
                                        parent (candidate set)
                                              │
        features (D-open: VIX/SPX/gap ─┐      │
                  D-1 close: VIX9D/3M/6M etc.)│
                                       ▼      ▼
                          walk-forward LightGBM/XGBoost (IS=3m, OoS=1d)
                                              │
                                    rank children → top-k pick
                                              │
                              TS Bot Portfolios enable/disable
```

## Repository layout

```
docs/           Reverse-engineering spec (start here)
src/mlcpo/
  data/         TS→OO format adapter, OO log loader, child/parent assembly
  features/     Market features (VIX suite, SPX, gap) + child descriptors
  model/        Walk-forward engine, LightGBM/XGBoost wrappers, Optuna, ensemble
  diagnostics/  Parent Lab metrics: oracle gap, rotation entropy, Jaccard, dispersion
configs/        Known hyperparameter sets and run configs
tests/          Test suite
```

## Build phases (from the spec, §9)

0. **Data adapter** — TS export → OO column contract (only TS-specific code in the system)
1. **Child construction** — partition strategies into 6–7 similar-size children by co-firing groups
2. **Metrics + Parent Lab lite** — analytics first; validates the adapter and the candidate set before any ML
3. **Walk-forward CPO engine** — features, daily rows per child, rolling retrain, rank → top-k, honest baselines
4. **Optuna + ensemble** — ≤50-trial studies on promising parents only
5. **Execution hook** — daily pick → TS Bot Portfolios toggle, with morning human verification

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -e .
```

## Daily workflow

```bash
# one-time (or when re-cutting children): build + persist the live partition,
# compute the app's caches (slow — runs full walk-forwards)
python -m mlcpo.app --refresh data/<ts_export>.csv

# each morning after 9:31 ET, with a fresh TS export:
python -m mlcpo.live data/<ts_export>.csv      # prints pick + enable/disable
                                               # checklist, writes data/live/<date>.json

# web interface (reads caches + latest decision)
python -m mlcpo.app                            # http://127.0.0.1:8050
python -m mlcpo.app --host 0.0.0.0 --port 8050 # expose on LAN / reverse proxy
```

The execution hook is deliberately human-verified (Phase 5): the decision file
lists which TS Bot Portfolios to enable/disable; toggling them in TradeSteward is
a manual morning step until TS API access is wired to consume the same file.

To serve at a domain (e.g. `closet.cottonmike.com`): run the app with
`--host 0.0.0.0` (or `MLCPO_HOST=0.0.0.0`), point the domain's A record at this
machine, and terminate TLS with a reverse proxy (Caddy: `closet.cottonmike.com {
reverse_proxy 127.0.0.1:8050 }`). The Dash dev server is fine for personal use;
put it behind auth before exposing it publicly — the dashboard shows P&L data.

## Acceptance test

Before trusting this implementation: run it against the reference system's own child
CSVs and verify both systems pick similar children on similar days (spec §9, "sanity
anchor").

## Status

Working pipeline on synthetic data; awaiting first TradeSteward export to fill the
adapter column map (`src/mlcpo/data/ts_to_oo.py`).

- **Phase 0** — OO contract + child/parent assembly done; TS→OO column map pending
  first real export (`python -m mlcpo.data.ts_to_oo <file.csv>` prints the worksheet)
- **Phase 2** — analytics + Parent Lab diagnostics implemented (oracle gap, rotation
  entropy, decisive days, win-gap stats, Jaccard, correlations, `parent_report()`)
- **Phase 3** — market features live (CBOE VIX family + yfinance SPX, leakage-safe),
  child descriptors, LGBM/XGB wrappers, walk-forward engine with honest baselines
  (BASELINE / BEST CHILD / ML table); verified on synthetic regime data
- **Phase 1** — co-firing + style-coherent child construction, validated on the
  real export (style children beat both honest baselines end-to-end)
- **Phase 4** — Optuna studies (≤50 trials) + HP-set ensemble (mean-rank/vote)
- **Phase 5** — LIVE daily decision + human-verified execution checklist
  (`python -m mlcpo.live`), stable persisted partition; TS-API executor pending
  API access
- **Web interface** — Plotly Dash app (`python -m mlcpo.app`): overview equity/
  drawdown, cycle picks, children, live decision

Open questions for the reference author are ranked in spec §10 — #1 is the exact
target-variable definition and the "D-Day threshold" mechanism.
