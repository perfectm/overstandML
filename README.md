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

## Acceptance test

Before trusting this implementation: run it against the reference system's own child
CSVs and verify both systems pick similar children on similar days (spec §9, "sanity
anchor").

## Status

Scaffold + spec. No working pipeline yet. Open questions for the reference author are
ranked in spec §10 — #1 is the exact target-variable definition and the "D-Day
threshold" mechanism.
