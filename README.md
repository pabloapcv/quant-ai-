# Atlas — global quant AI research assistant

Spec: [PRD.md](PRD.md). Current status: **Phase 0** (foundations), **Phase 2** (macro regime engine), **Phase 3** in progress (signals + backtester). Phase 1 waits on a paid market-data vendor.

## What exists today
- `atlas/pit/` — append-only bitemporal Parquet store. Every read is point-in-time
  (`as_of`, `realtime_panel`, `history`); snapshots pin exact data + config + code.
- `atlas/connectors/fred.py` — FRED/ALFRED connector with full vintage history, rate
  limiting, retries, raw-payload archiving, and API-key redaction.
- `atlas/quality/` — Pandera data contracts that block bad writes; staleness checks.
- `config/series.yaml` — canonical series catalog (PRD Appendix A + NBER recession dates).
- `atlas/features/` — causal transforms (YoY, annualized, diff, trailing z) and a
  point-in-time monthly feature matrix with a bounded ragged-edge fill.
- `atlas/models/regime/` — growth / inflation / financial-conditions composites,
  growth×inflation quadrant with probabilities, Gaussian HMM (BIC-selected states),
  12-month recession probit with NBER label lag, a walk-forward evaluator, and historical
  analog search (k-NN, Mahalanobis on the macro state; PRD §7.6).
- `atlas/connectors/kenfrench.py` — Ken French data library (FF5 + momentum factors,
  12 and 49 industry portfolios back to 1926), no key needed.
- `atlas/connectors/sdmx.py` — ECB, OECD, BIS and IMF over SDMX: euro FX and yield curve,
  14 country CLIs, share prices and 10Y yields, real effective FX, CPIs (no keys).
- `atlas/signals/country.py`, `universes.py` — 14 country equity markets (USD since 1999;
  9 developed markets in local currency since 1955): momentum, leading-indicator change,
  falling yields, value and volatility signals through the same gates, views and portfolio.
- `atlas/backtest/` — purged K-fold and combinatorial purged CV, IC / Newey-West t,
  quantile long-short backtests with costs, PSR, deflated Sharpe, PBO (CSCV), trial log.
- `atlas/signals/industry.py` — momentum (incl. vol-targeted), reversal, low-volatility,
  long-term reversal, industry value (BE/ME, Fama-French June timing), value+momentum,
  residual momentum vs FF3/FF5.
- `docs/signals/` — generated signal cards (`atlas research cards`), one per signal.
- `atlas/backtest/gates.py` — PRD §8.4 promotion gates; `atlas/models/regime/market.py`
  gives the trend x volatility market regimes the regime gate uses.
- `atlas/agent/` — the research assistant (Phase 6): Claude (`claude-opus-5-5`) calls the
  quant engine through 12 tools (views, track records, gates, portfolios, series, regime,
  historical analogs, scenarios, FOMC document search and statements);
  a deterministic guard checks every number in the answer against the tool results and
  gives Claude one repair turn. Needs `ANTHROPIC_API_KEY` or an `ant auth login` profile.
- `atlas/agent/report.py` — `atlas report weekly`: engine facts gathered deterministically,
  a Claude writer and a red-team pass, every number grounding-checked; saved to
  `data/reports/<date>/` (`--dry-run` gathers facts without calling Claude).
- `atlas/pipeline.py` — `atlas pipeline daily`: ingest every source, staleness check,
  snapshot, freeze this month's views (write-once), score the live track, run alerts. Run it
  daily: snapshot-only sources gain true point-in-time history only from recorded days.
- `atlas/agent/usage.py` — LLM cost controls (PRD §10.6): every Claude call is metered to
  `data/usage/<date>.jsonl`; a daily budget (`ATLAS_LLM_DAILY_TOKEN_BUDGET`, weighted tokens)
  is a hard stop; system prompt, tools and the tool loop use prompt caching. `atlas usage`.
- `atlas/agent/evals.py` — `atlas eval run`: the 160-question golden set
  (`config/evals/golden.yaml`: views, drivers, track record, gates, portfolio, regime, series,
  analogs, scenarios, documents, unavailable, advice, adversarial) scored 0-100 on grounding (30), tool
  choice (20) and an Opus judge rubric (50); fails on a > 3-point drop vs
  `data/evals/baseline.json` (`--set-baseline`). Every case calls Claude: filter with
  `--category`, `--ids` or `--limit` for cheap checks.
- Ensemble combiner option `--combiner gbm` (views build/evaluate, portfolio): a
  walk-forward gradient-boosted ranker on the component z-scores (PRD §7.2's LightGBM ranker,
  via scikit-learn). Better than the IC-weighted default on 3 of 6 universes, worse on 3, so
  the default is unchanged; the comparison table is in the ensemble module's docstring.
- `atlas/signals/fx.py` — G10 FX vs the US dollar (`fx_g10`, 8 currencies, 1999 onward): ECB
  month-end rates plus the carry earned at OECD 3-month rates; carry, momentum and long-term
  reversal signals through the same gates, views, portfolio and live track. Carry is the one
  strong signal (IC t 3.7, positive in 4 of 4 regimes); the ensemble's probabilities are not
  yet informative (Brier at 0.25).
- `atlas/signals/rates.py` — 9 developed-market 10-year government bonds (`rates_dev`),
  hedged, excess over cash, returns from yields (duration 8): term-spread carry, momentum,
  yield value and growth-slowdown signals. Carry is again the strongest (IC t 2.9) but the
  ensemble shows no skill yet (walk-forward IC t 1.4, portfolio IR -0.11); views are frozen
  daily so the live track judges it.
- `atlas/text/` — central-bank text (PRD §6.6): FOMC statements and minutes from
  federalreserve.gov (2006 onward; `atlas text sync`, daily in the pipeline), stored point in
  time; BM25 passage search (`atlas text search`), statement-to-statement changes
  (`atlas text show`), and a Claude Haiku reading of each statement (action, size, stance -2..+2,
  dissents; size checked against the text; `atlas text tone`). Tools: `search_documents`,
  `get_fomc_statement`.
- `atlas/models/nowcast.py` — GDP nowcast research: mixed-frequency DFM + bridge on 15
  monthly series, evaluated in real time vs GDPNow (`atlas nowcast evaluate`). It does not
  beat a constant forecast (RMSE 1.6 vs GDPNow 0.86 one hour before the advance release,
  2012-2025 ex-2020), so Atlas reports GDPNow (point in time) as its nowcast in the regime
  tool; the Phase 2 "within 10% of GDPNow" criterion is not met. Details in the docstring.
- `atlas/models/scenario.py` — `atlas scenario run --shock us10y=0.5`: same-month asset
  response to factor shocks (rates, breakeven, Baa credit spread, USD, oil, VIX); unshocked
  factors move by their conditional expectation; 80% block-bootstrap bands (PRD §7.6).
- `atlas/alerts.py` — rules in `config/alerts.yaml` (regime-probability crossings, recession
  > 40%, HY OAS +50bp/week, VIX/VIX3M > 1, view flips, pipeline failures), evaluated at the
  end of every daily run with cooldown and rate limit. Logged to `data/alerts/`; ntfy and
  Slack delivery are off until configured (`atlas alerts check` runs them by hand).
- `atlas/api.py` — `atlas api serve`: read-only FastAPI over the same tools (regime, views,
  frozen views, track record, portfolio, gates, analogs, scenario, series, alerts,
  snapshots, usage) plus `POST /v1/chat` (Server-Sent Events: tool calls, then the grounded
  answer); OpenAPI docs at `/docs`. Localhost only unless `ATLAS_API_TOKEN` is set.
- `atlas/views/reproduce.py` — `atlas reproduce <view_id>`: rebuilds a frozen view from its
  pinned data snapshot only (later data invisible), verifies the snapshot's hashes, and
  compares every view with what was published; flags a different or uncommitted code state.
  The daily pipeline and reproduce share one view-building function.
- `atlas/views/track.py` — `atlas pipeline track`: frozen views scored once their horizon
  has elapsed (the live paper track, immune to hindsight).
- `atlas/quality/drift.py` — live-vs-backtest IC monitor (alert rule `live_ic_drift`) and a
  feature PSI diagnostic in every run report (not an alert: see the module docstring).
- `docs/runbooks.md` — operations: scheduled jobs, reading a pipeline run, alerts, recovery,
  evals, API.
- `atlas/backtest/holdout.py` — the locked 2021-2025 holdout (PRD §8.2): research with
  `atlas research gates --exclude-holdout`; `atlas research holdout open --reason ...` scores
  the ensemble and each signal on holdout months only, logs the opening
  (`data/research/holdout_log.jsonl`) and refuses a second one. Every universe built so far
  was developed on the full sample, so its logged opening is marked contaminated. Holdout
  3-month ensemble IC: ind12 +0.024, ind49 +0.018, ctry_dev_lcl +0.018, fx_g10 +0.044,
  ctry_usd -0.065, rates_dev -0.044 (all |t| < 1.2): no reliable skill in 2021-2025.
- `tests/test_leakage.py` — the look-ahead leakage suite (blocking in CI): future-vintage
  injection, as-of vs brute force, shuffled-target IC, timestamp audits (store, features, text).

## Quickstart
```bash
uv sync --all-groups
cp .env.example .env            # add ATLAS_FRED_API_KEY
uv run atlas catalog list
uv run atlas ingest fred US.CPI.CORE US.LABOR.UNRATE
uv run atlas pit as-of US.CPI.CORE --at 2020-04-30
uv run atlas pit history US.LABOR.NFP 2020-04-01
uv run atlas ingest fred --all          # whole catalog with ALFRED vintages
uv run atlas regime now
uv run atlas regime now --at 2008-09-30   # point-in-time: only what was known then
uv run atlas regime backtest --start 1995-01-01 --out regime_bt.csv
uv run atlas regime analogs --universe ind12 --k 10   # closest past macro states + what sectors did next
uv run atlas ingest kenfrench --all     # free, no key
uv run atlas ingest sdmx                # ECB, OECD, BIS, IMF; free, no keys
uv run atlas research signal mom_12_1 --universe ind49
uv run atlas research family momentum --universe ind49   # PBO across configs
uv run atlas research gates --universe ind49             # PRD promotion gates
uv run atlas views build --universe ctry_dev_lcl         # country views
uv run atlas research cards --universe ind49             # regenerate docs/signals
uv run atlas ask "Which developed markets do the models favour, and how reliable is that?"
uv run atlas pipeline daily                 # the whole daily run
uv run atlas report weekly                  # weekly research note (writer + red team)
uv run atlas pipeline track                 # live track record of frozen views
uv run atlas scenario run --shock credit_spread=1 --universe ind12
uv run atlas alerts check                   # evaluate alert rules now
uv run atlas api serve                      # http://127.0.0.1:8710/docs
uv run atlas eval run --category advice     # assistant eval on a slice of the golden set
uv run atlas snapshot create
```

## Checks (same as CI)
```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```
