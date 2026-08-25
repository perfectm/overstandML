# ML CPO — Reverse-Engineering Specification

**Target:** Mauro's "Portfolio26 – Research & ML" application (the ML CPO system)
**Sources:** 19 Aug 2026 status-update video (transcript + ~30 UI screenshots), Discord ML Discussion thread (8/21–8/25), Mauro's direct explanations of his methodology
**Purpose:** Build an independent implementation that runs against TradeSteward data
**Confidence key:** Statements are marked [CONFIRMED] (stated directly by Mauro or clearly visible), [INFERRED] (strong evidence, unverified), or [UNKNOWN] (gap to fill). OCR-derived numbers should be re-verified against the screenshots before being treated as exact.

---

## 1. What the tool is

A local, single-user research web application that answers one daily question: **given a set of candidate portfolios ("children"), which one(s) should trade today?** A supervised model is retrained on a rolling window and each morning scores every child; the top-ranked child (k=1 live) or top-k children (k=2 in the MMC experiment) trade that day, the rest sit out. Mauro calls this CPO — the daily "child portfolio" prediction — and the whole loop (data ingest, portfolio construction, walk-forward ML, diagnostics, live operation) lives inside one app.

The core claim being validated in the video: YTD 2026, walk-forward ML selection produced **+$15,535** against a candidate pool whose best member made ~$7.3K statically and whose "rational pick" as of Dec 31 (P4) lost ~$4–5K. [CONFIRMED]

## 2. Technology stack

[INFERRED] The app is a Python web application served locally on `127.0.0.1`, ports 8051–8069 across the screenshots — each "Open New" spawns a fresh instance on the next port with its own timestamped log file (`.../P26_prd/.../20260819_133437.log` pattern visible). Sequential ports starting near 8050 plus this multi-instance pattern strongly suggests **Plotly Dash** (default port 8050); Streamlit (8501) is the alternative but the port range fits Dash better. Runs on Windows (`C:\Users\mauro\MAURO\P26_prd\l\parents\parent_3.csv` visible in run summaries).

[CONFIRMED] Models: **LightGBM and XGBoost** (model toggle "LGBM | XGB" in the ML run panel; the HP sets are unmistakably LightGBM-style parameters). Hyperparameter optimization via **Optuna**, capped at ~50 trials per study specifically to limit overfitting. Everything else is standard Python data stack (pandas/numpy assumed).

[CONFIRMED] There is no external ML service and no cloud dependency — "mostly local to my machine."

## 3. Data model

The hierarchy, bottom to top:

**Trade log (strategy level).** The atomic input is an **Option Omega backtest/live CSV log** — one row per trade with all OO columns. Mauro: "as long as a file has that format with all OO columns in it then it will work." This is the entire ingestion contract, which is what makes a TradeSteward port feasible: build a TS→OO column translator (ordering and labels differ, content largely maps) including status fields like "Backtest Completed" for non-0DTE. [CONFIRMED]

**Strategy.** A named trading system (e.g. `WM-0DTE-350-PCS-10W-GAP-UP-HIGH-ORB-TRANCH-VIX`, `RoD-EMA-Skewed-Condors-2-times`, `RIC-12DTE-Below-SMA`). The app maintains a **strategy universe** (~88 strategies visible in one screenshot) with rich metadata per strategy: qualification (e.g. "Core candidate", "Experimental/review"), rating (1.0–3.0 scale visible), tags (Credit/Debit, Long-IC, Bullish/Bearish/Neutral, R1/R2 revision markers), payoff class, total P&L, max loss, activity, and free-text risk flags ("win rate outside preferred zone; average win < average loss", "2026 negative | win rate…"). [CONFIRMED]

**Child (portfolio).** A selection of strategies (4–7 for the live children; 3–4 for MMC minis) that trades as a unit. A child is representable as a CSV of its constituent trades. The live constraint to date: *every child must be independently tradable* — a real portfolio you'd be happy to run on its own. [CONFIRMED]

**Parent.** A file assembling N children into one candidate set — "a technicality" in Mauro's words, but it is the unit the ML runs against. Named like `CPO_source_1_v2 (parent_3) – 4 children` and stored as `parent_3.csv`, registered in a `parents.json` index (the dataset dropdown shows entries like `Ltest_1 (parent_19) – 4 children`, `Npairwise_1 (parent_20)`, `Beta_features_1 (parent_23)`, `Regt_po_vises (parent_24)` — note these names imply ongoing feature/pairing experiments). The live parent holds children A–D: **Child 069, ML_P1_v2, Opt_Pbo_P3, Opt_Pbo_P4**. [CONFIRMED]

**Critical construction constraint** [CONFIRMED]: children in a parent should be of *similar aggregate size* — comparable total P&L and comparable activation frequency — because the model picks one per day and a $300K/daily-multi-tranche child alongside a $30K/once-a-week child biases the learner heavily toward the former.

## 4. Feature engineering

From Mauro's Discord explanation (8/22), features split by information timing — this is the leakage-control backbone:

**Day-of (D) features, captured just after market open and just before prediction:** Opening VIX (VIX value at ~9:31 ET), Opening SPX, Opening Gap. [CONFIRMED]

**Prior-close (D-1) features — everything else**, including the longer-term VIX term structure: **VIX9D, VIX3M, VIX6M** as of previous RTH close. VIX1D deliberately excluded so far ("may test it one day"). [CONFIRMED]

**Child descriptor features.** ML runs at the child level, so strategy-level data from the OO logs is "either aggregated up or recalculated" into child descriptors. [CONFIRMED] Exact descriptor list [UNKNOWN] — plausibly recent child P&L/equity statistics, activation counts, and composition summaries; one parent is literally named `Beta_features_1`, suggesting market-beta descriptors have been tried.

**Target variable** [PARTIALLY UNKNOWN — the biggest gap]. What is confirmed: a **regressor** produces "a prediction for each of them, then ranks them, and I take the first two" (MMC) or first one (live). Yet the run panel also shows a **"D-Day threshold 0.65"** and selection "Top-K per day by predicted prob," which is classifier language. Most consistent reading: the primary daily selector is a regressor on next-day child P&L (or a normalized variant) used for ranking, with a separate/auxiliary probability output ("D-Day" = danger/drawdown-day probability at 0.65 threshold) acting as a trade/no-trade gate. Treat the exact label definition and the D-Day mechanism as the first thing to pin down (see §10).

## 5. Walk-forward protocol

The evaluation and live procedure are the same loop [CONFIRMED]:

Train on an in-sample window of **IS months** (3 in the demo; a 2 appears in one config), predict the next **OoS days** (1), advance one day, repeat. **Anchored/Unanchored** toggle controls whether the train window start is fixed (expanding window) or rolls. The demo run: start date override 2025-10-01, first prediction 2026-01-01, data through 2026-08-18, producing **154 cycles** at initial equity $100,000. Modes: Monthly (Static) / Weekly (CPO) / **Daily (CPO)** — daily is the live mode.

**Baseline discipline** — the part Mauro is most emphatic about: the benchmark is chosen *with information available at comparison start only*. On Dec 31 the best-looking static child was P4; that is the honest baseline, and it went on to be the worst performer. The results table therefore reports three rows: **BASELINE** (196 trades, -$2,149 — [INFERRED] the pool/all-candidates reference), **BEST CHILD** (170 trades, -$4,098 — the as-of-start pick, P4), and **ML** (217 trades, **+$15,535**, ~21.4% figures visible), with per-row: PCR, max DD $ and %, daily Sharpe, max DD days, max 1-day loss, win-month %, avg/median/best/worst month P&L; plus a participation/normalization block (nominal units, trades/day stats, unique-UID/day stats, p95/max margin per day). OCR on this table is unreliable — verify exact values from `chrome_ombXmQPkTy.png` / `chrome_SaD5ECHgK4.png` before quoting.

**Ensemble.** Live trading uses an **ensemble of HP sets**, not the single demo set; ensemble composition is reviewed/re-saved only every month or two. The app has a dedicated "ML Ensemble" module. Aggregation method (vote/mean-rank/etc.) [UNKNOWN].

## 6. Hyperparameters

Two concrete LightGBM-style sets are visible:

`P3_v2_2 | est=240, lr=0.05, depth=-1, leaves=36, minleaf=100, subsample=0.9, colsample=0.5, seed=4998` (the demo/live-adjacent set)

`static1 | est=213, lr=0.06, depth=4, leaves=380(?), minleaf=400, sub=0.8, col=0.7, seed=8849` (OCR shaky on leaves/minleaf)

Workflow [CONFIRMED]: a small library of "evergreen" HP sets found over time validates any *new* parent first (is it promising at all?); only promising parents get an Optuna study, deliberately ≤50 trials. Note the regularization posture — shallow-ish trees, small leaf counts, high min-leaf, aggressive subsampling — consistent with tiny tabular datasets (a 3-month window of daily rows across 4 children is only ~250–350 training rows).

## 7. Application modules

Left-nav, in order [CONFIRMED]: **Strategies R&D, Portfolio Analytics, Portfolio Comparisons, CPO Optimizer, Parent Lab, ML Utilities, MLCPO, ML Ensemble, Data Management, LIVE, Settings.**

**Portfolio Analytics** — load a portfolio (active strategies + searchable universe with include/exclude), date-filterable, with a Strategy Analytics panel: tabs Overview / Subperiods / Rolling / Tail Risk / Parameters / Regimes / Overfitting. Headline stats observed: total P&L, MaxDD%, MAR, CAGR, Sharpe (ann), win rate, trades, profit factor, skewness, kurtosis, tail ratio, %P&L from top 3 and top 5 days, plus cumulative equity and drawdown charts (display mode: cumulative, all-selected overlay). Example values seen: Child 069 → $162,898 / -3.21% MaxDD / MAR 7.96 / CAGR 25.58% / Sharpe 4.74 / WR 72.5% / 1,296 trades; P3 → $143,599 / -3.70% / 6.28 / 23.26% / 4.82 / 72.1% / 2,007 trades.

**Parent Lab** — "Build the choice set before testing the model." Interactive child composition against the strategy universe with explicit **activation, maturity and realized-equity semantics**: an analysis window (e.g. 2026-01-02 → 2026-08-19), a **mature cutoff** (2026-06-17, "153 active dates"), and **provisional cohorts** (recent child-date cohorts excluded from mature comparisons — trades too young to have settled outcomes). Membership is a strategy×child matrix with click-to-toggle cells, batch add/remove, per-strategy Move/Copy/Remove between children, quick filters (qualification, labels, tags, payoff, activity, risk flags), Commit Children → +Parent, snapshot hashes, and a live results monitor. Baseline modes: "Common start window / Equal / Current / Editable."

**MLCPO** — the run screen. Scope: Portfolio | Strategy. Model: LGBM | XGB | **MMC** toggle. Dataset (parent) picker from parents.json; ML HP Set picker; mode Monthly/Weekly/Daily; start/end overrides; FWD OoS prediction date (for a one-day live replay); IS months; OoS days; Anchored/Unanchored; D-Day threshold; Selection mode Top-K (+K value); "Save MM data" toggle; Run → Run Summary (dataset path, cycles, initial equity, run time) → Open Analysis / Save Results.

**MLCPO analysis** — Equity & Drawdown (Baseline vs ML) with equity views: Raw P&L cumulative, P&L per UID/day cumulative, P&L per Margin/day cumulative, PCR% daily / rolling 20d / rolling 60d, raw-vs-exposure-normalized equity; drawdown view; Baseline vs ML metrics tables (§5); **Cycle Summary** table (per-cycle rows: date, picked child, …).

**Parent/MMC diagnostics dashboard** — tabs Overview / Activation / Risk & Contribution / Cross-Child ML Opportunity / Baseline Delta. Headline cards: **Oracle Gap** ($80,300.74 for the 10-child MMC parent; $67,441.20 for the live 4-child parent), **Decisive Days** (56.6% / 59.5%), **Rotation Entropy** (0.960 / 0.983), **All-Child Zero Days** (12 / 3), **ML Opportunity Score** (0.724 / 0.745, "Components" expandable). Child summary table: strategies count, realized P&L, annualized P&L, max DD, max 1d loss, Sharpe, trade days %, zero days %, single-active %, mature cohorts, provisional count. Charts: realized equity by child (Date Closed basis), mature entry-cohort cumulative P&L (Date Opened basis), realized child winner frequency, child strategy-overlap **Jaccard** matrix, child tail-day Jaccard, pairwise strategy diagnostics (correlations: mature P&L, close P&L, activation; co-activation rates, union activity, incremental contribution, neither-active rate, joint losing cohort days), unachievable **oracle upper bound vs best static** (mature Date-Opened cohorts), rolling 60-day oracle edge, rolling 60-day cross-child outcome dispersion (median dispersion $358.08, positive windows 100%, quarterly breadth 66.7–100%), and best-minus-second-best daily gap distribution (median win-gap $218.76, lower quartile $71.80, longest non-positive run 0 days). Parent-level stats: child count, unique/total/duplicated strategy counts, max strategy appearances, repeated family count, mean/max pairwise strategy Jaccard, all-child-zero days/rate, exactly-one-active days, mean active children, common mature active days, mean/max pairwise child P&L correlation.

These diagnostics ARE the "reverse-engineer ML without running it" effort Mauro described: quantifying, pre-ML, whether a candidate set has enough *exploitable dispersion* (oracle gap, dispersion, win-gap), enough *rotation* (winner frequency, entropy, decisive days), and enough *independence* (Jaccard, correlations) to be worth training on.

**LIVE** — daily operation: refresh open-of-day features, run FWD prediction for today, output which child trades. Details [UNKNOWN]; execution itself is manual/semi-manual (Mauro toggles TAT schedules by hand; the local db3 automation is designed but not built).

## 8. MMC (Multiple Mini Children) — current research direction

Instead of 4 full-size independently-tradable children with k=1: **~10 small children of ~3–4 strategies each** (some strategies deliberately shared across two children), select **top-2 per day** by predicted rank. Purpose: control aggregate size *non-linearly* — via how many children fire, not just a lot multiplier — and admit different strategy mixes. Experiment name "Power 10", parent `MMC_par1 (parent_10)`. MMC child cards show: realized P&L, mature-cohort P&L, MaxDD, trade participation %, zero-activity %, contribution score (0.26–0.68 range observed), provisional cohorts. Early results "encouraging"; explicitly research-stage. [CONFIRMED]

## 9. Rebuild plan (TradeSteward edition)

Phase 0 — **Data adapter.** Export TS trade logs; write a TS→OO column translator (Cotton Mike: "just ordering and labels"). Emit one CSV per strategy or per child with all OO columns; add OO status labels for non-0DTE. This adapter is the only TS-specific code in the whole system.

Phase 1 — **Child construction.** Partition the 36-strategy / ~85-bot portfolio into 6–7 children along co-firing lines (the implied groups from the earlier Claude analysis — SMA condors, EMA skewed condors, etc.). Enforce the similar-size constraint (§3). Optionally build parent variants per James's head-to-head design: current-port-with-hedges control vs ML-no-hedges at descending filter levels.

Phase 2 — **Metrics + Parent Lab lite.** Implement the analytics layer first (equity, DD, Sharpe/MAR/CAGR, activation calendars, Jaccard overlap, child correlation, oracle gap, dispersion, winner frequency). This validates the data adapter *and* tells you whether the candidate set is even worth training on, before any ML exists.

Phase 3 — **Walk-forward CPO engine.** Feature builder (D-open: VIX/SPX/gap; D-1 close: VIX9D/3M/6M + market features + child descriptors), daily-row dataset per child, LightGBM regressor, anchored/unanchored rolling retrain (IS=3m, OoS=1d), rank → top-k, baseline framework (pool + as-of-start best child), results tables. Start with the published evergreen HP set (est 240, lr 0.05, leaves 36, minleaf 100, sub 0.9, col 0.5).

Phase 4 — **Optuna + ensemble.** ≤50-trial studies on promising parents only; small HP-set ensemble for live; monthly review cadence.

Phase 5 — **Execution hook.** TS already supports Bot Portfolios (grouping + enable/disable together, cross-account). Daily ML output → enable the picked child's portfolio, disable the rest — via API if available, else a Claude scheduled task / browser automation, with the morning human-verification step James insisted on.

Sanity anchor: Emet's "step 1 is basic algorithm reproduction" — before trusting a reimplementation, run it on Mauro's own children (he offered to run shared CSVs through his tool) and check both systems pick similar children on similar days.

## 10. Open questions for Mauro (ranked)

1. **Exact target variable**: what does the regressor predict (raw next-day child P&L? normalized? multi-day?), and what exactly is the "D-Day threshold 0.65" — a separate no-trade classifier, or a probability transform of the ranker?
2. Full feature list beyond the VIX/SPX/gap set — which child descriptor features, any calendar features, any regime features?
3. Ensemble mechanics: how many HP sets, how are their picks combined?
4. Baseline row definition: is BASELINE equal-weight all children, or something else?
5. Anchored vs unanchored: which is live, and why?
6. Data hygiene details: how are same-day partial fills / multi-tranche strategies rolled up to a daily child P&L row; Date Opened vs Date Closed attribution (the diagnostics use both).
7. What lives in the "Overfitting" and "Regimes" analytics tabs.

---
*Compiled 25 Aug 2026 from the ML Project shared folder. OCR-derived figures (especially the Baseline-vs-ML table) should be verified against the original screenshots before external use.*
