# Phase status: P3 to P8 against the PRD exit criteria

Status as of 2026-10-30. Each criterion from PRD §15 is marked **Met**, **Not met** (the
work is done but the result falls short), **Blocked** (needs something outside the code: a
data vendor, a news source), or **Time-gated** (needs months of live
operation). Evidence is the command or file that shows it.

## Summary

| Phase | Scope built | Exit criteria |
|---|---|---|
| P3 Signals + backtester | Done | 3 of 4 met; the "8 signals pass every gate" criterion is not met |
| P4 Expected returns + ensemble | Done except the structural model (needs fundamentals) | Not met: holdout skill below the targets |
| P5 Global expansion | Countries, FX, rates, commodities, credit, crypto done; global sectors blocked | Not met: global sectors need paid data |
| P6 NLP + LLM agent | Done except transcripts and news (blocked: no source) | 2 of 3 met; sending is switched off for now |
| P7 Product hardening | Done | Met, with a web app instead of Next.js (decision below) |
| P8 Live paper track | Done | Time-gated: the 12-month track completes about October 2027 |

## P3: signals and backtester

| Criterion | Status | Evidence |
|---|---|---|
| Feature registry with 40+ signals | **Met** | 34 signal definitions, 118 signal x universe combinations across 10 universes (`atlas/signals/`) |
| CPCV, deflated Sharpe, PBO | **Met** | `atlas/backtest/` (purged / combinatorial CV, DSR with every logged trial, CSCV PBO per family) |
| Every signal has a tearsheet and card | **Met** | 118 cards with SVG tearsheets in `docs/signals/<universe>/` (`ramon research cards`) |
| At least 8 signals pass every promotion gate | **Not met** | 0 pass all six; the deflated-Sharpe gate binds (see `docs/phase3_status.md`). The gate is not loosened after the fact |
| User story 3: a signal from your own file | **Met** | `ramon research signal --file docs/examples/my_signal.py`: IC, decay, turnover, DSR, marginal ICIR, family PBO; the run is logged as a trial |

## P4: expected returns and ensemble

| Item | Status | Evidence |
|---|---|---|
| Ensemble (IC-weighted) and gradient-boosted challenger | **Done** | `atlas/models/expected_returns/`; challenger `ens_v1_gbm` frozen daily |
| Calibration (isotonic) and conformal ranges | **Done** | every view has a calibrated probability and an 80% range |
| View publisher, write-once, reproducible | **Done** | `gold/views/`; `ramon reproduce <view_id>` (real check: cmdty 2026-09, 17/17 identical) |
| View object per PRD §3.2 | **Done** | views now carry `regime_context`, `historical_analogs` and `risks` (`atlas/views/context.py`) |
| Risk model and constrained optimizer | **Done** | `atlas/models/risk/`, `atlas/models/portfolio/optimizer.py` |
| Black-Litterman | **Done, not adopted** | `black_litterman=True`: equal-weight equilibrium (zero active prior), view uncertainty from the conformal ranges. Before 2021 it did worse than the direct views in 7 of 9 universes (it shrinks views so far the portfolio barely trades), so the direct views stay the default |
| Structural (fundamentals) model | **Blocked** | needs P1 company fundamentals |
| OOS sector 1M rank IC >= 0.04 | **Not met** | US sectors' 3-month ensemble on the opened 2021-2025 holdout: IC +0.024 (t 0.33). A separate 1-month opening would be a reopening of the same family |
| Model portfolio IR >= 0.3 net | **Not met** | US sectors (ind12) 2021-2025: net IR -0.17. Other universes range from -0.61 (developed bonds) to +0.98 (EM currencies) |

## P5: global expansion

| Asset class (§3.1) | Status | Universe |
|---|---|---|
| US sectors | **Done** | `ind12`, `ind49` (Ken French industries as sector proxies) |
| Global sectors | **Blocked** | needs MSCI / global sector data (paid) |
| Countries (~45) | **Done (33)** | `ctry_world`: 33 markets in USD; `ctry_usd` (14), `ctry_dev_lcl` (9, local currency). Chile and Colombia lack an ECB exchange rate |
| Factors | **Done** | Fama-French 5 + momentum (inputs to signals and scenarios) |
| Rates | **Done** | `rates_dev` (9 developed), `rates_em` (9 emerging) 10-year bonds, hedged |
| Credit | **Done (approximate)** | `CREDIT.AAA` / `CREDIT.BAA` vs Treasuries from Moody's spreads (time-series model). The ICE BofA indices on FRED now keep only 3 years |
| FX | **Done** | `fx_g10` (8), `fx_em` (9) |
| Commodities | **Done (spot)** | `cmdty`: 17 IMF monthly-average spot prices; futures returns with roll yield need futures data |
| Crypto | **Done (short history)** | `CRYPTO.BTC` / `CRYPTO.ETH` vs cash (time-series model, `ramon views sleeves`) |
| Exit: every §3.1 asset class produces views nightly | **Not met** | all but global sectors do |

Holdout openings (3-month ensemble, 2021-2025; "clean" = researched before 2021 only):

| Universe | IC | t | Opening |
|---|---|---|---|
| fx_em | +0.299 | 4.57 | clean (mostly the known interest income; see the README) |
| cmdty | +0.091 | 1.73 | clean |
| rates_em | +0.073 | 1.30 | clean |
| fx_g10 | +0.044 | 0.62 | contaminated |
| ind12 | +0.024 | 0.33 | contaminated |
| ind49 | +0.018 | 0.36 | contaminated |
| ctry_dev_lcl | +0.018 | 0.25 | contaminated |
| ctry_world | -0.005 | -0.10 | contaminated |
| rates_dev | -0.044 | -0.54 | contaminated |
| ctry_usd | -0.065 | -1.14 | contaminated |
| credit + crypto | Brier +0.002 vs base rate | n/a | clean: no better than base rates |

## P6: NLP and the LLM agent

| Criterion | Status | Evidence |
|---|---|---|
| Grounding 100% on the golden set | **Met** | baseline run: every answer fully grounded (`data/evals/baseline.json`) |
| Eval score >= 85 / 100 | **Met** | 97.3 on 160 questions; 96.7 on the executive questions |
| Weekly report auto-sent for 3 consecutive weeks | **Paused (owner's choice)** | ntfy delivery was tested on 2026-10-30, then switched off at the owner's request: `ATLAS_NTFY_TOPIC_URL` is commented out in `.env`. The Monday job still writes the briefing, and the Briefing page shows every alert. Uncomment the line to resume; the 3-week count starts then |
| Central-bank text | **Done** | FOMC (2006-) and ECB (1998-) statements, minutes, press conferences; search and tone |
| Earnings-call transcripts, news | **Blocked** | no licensed source chosen |

## P7: product hardening

| Criterion | Status | Evidence |
|---|---|---|
| Dashboard | **Met (decision)** | a FastAPI-served web app (`atlas/web/`: Briefing, Your business, watchlist, What if, Ask Ramon, analyst pages) instead of Next.js: one process, no build step, same API |
| Alerts, scenario lab, analogs, track record | **Met** | `config/alerts.yaml`; What if and custom scenarios; analogs; System page and `ramon pipeline track` |
| Observability, auth | **Met** | structlog, Prometheus `/metrics`, drift monitor; bearer token for non-local serving |
| User story 1: Monday report with views heatmap, top changes, risks | **Met** | `data/reports/<date>/report.md` (heatmap section built by code) |
| User story 2: "why underweight X?" | **Met** | views carry drivers, analogs, risks and the invalidation rule; Ask Ramon explains them |
| User story 4: alert within 15 minutes of a regime cross or view flip | **Met** | alerts run about a minute after each release slot brings new data (release-day ingest); view flips are checked when views are frozen. Shown in the Briefing; phone delivery is off for now |
| User story 5: scenarios with confidence bands | **Met** | What if presets, `POST /v1/scenario` (80% bootstrap bands) |
| Runbooks | **Met** | `docs/runbooks.md` |

## P8: live paper track

| Criterion | Status | Evidence |
|---|---|---|
| Frozen views scored daily | **Met** | `track:score` in every daily run; `gold/views/` and `gold/sleeves/` are write-once |
| Monthly model review | **Met** | `ramon review monthly`; the daily pipeline writes the previous month's review in each month's first week (`data/reports/review-YYYY-MM.md`) |
| 12-month live track for evaluation | **Time-gated** | first views frozen October 2026; the first scores arrive once their 3-month outcome is published; 12 months completes about October 2027 |

## What would close the remaining gaps

1. **A paid market-data vendor** (P1): fundamentals for the structural model, global sectors, futures for commodity roll yield.
2. **A news / transcript source** (P6).
3. **Anthropic API credits**: Ask Ramon, the written weekly note, statement extraction and evals stop when they run out (they did on 2026-10-29). The briefing, digest, lens and alerts need none.
4. **Delivery**: the owner turns phone messages back on (P6).
5. **Time**: the live track (P8) and, through it, fresh out-of-sample evidence for P3 and P4, which cannot come from the already-opened 2021-2025 holdout.
