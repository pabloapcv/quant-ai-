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
  quant engine through 10 tools (views, track records, gates, portfolios, series, regime,
  historical analogs, scenarios);
  a deterministic guard checks every number in the answer against the tool results and
  gives Claude one repair turn. Needs `ANTHROPIC_API_KEY` or an `ant auth login` profile.
- `atlas/agent/report.py` — `atlas report weekly`: engine facts gathered deterministically,
  a Claude writer and a red-team pass, every number grounding-checked; saved to
  `data/reports/<date>/` (`--dry-run` gathers facts without calling Claude).
- `atlas/pipeline.py` — `atlas pipeline daily`: ingest every source, staleness check,
  snapshot, freeze this month's views (write-once), score the live track, run alerts. Run it
  daily: snapshot-only sources gain true point-in-time history only from recorded days.
- `atlas/agent/evals.py` — `atlas eval run`: the 150-question golden set
  (`config/evals/golden.yaml`: views, drivers, track record, gates, portfolio, regime, series,
  analogs, scenarios, unavailable, advice, adversarial) scored 0-100 on grounding (30), tool
  choice (20) and an Opus judge rubric (50); fails on a > 3-point drop vs
  `data/evals/baseline.json` (`--set-baseline`). Every case calls Claude: filter with
  `--category`, `--ids` or `--limit` for cheap checks.
- `atlas/models/scenario.py` — `atlas scenario run --shock us10y=0.5`: same-month asset
  response to factor shocks (rates, breakeven, Baa credit spread, USD, oil, VIX); unshocked
  factors move by their conditional expectation; 80% block-bootstrap bands (PRD §7.6).
- `atlas/alerts.py` — rules in `config/alerts.yaml` (regime-probability crossings, recession
  > 40%, HY OAS +50bp/week, VIX/VIX3M > 1, view flips, pipeline failures), evaluated at the
  end of every daily run with cooldown and rate limit. Logged to `data/alerts/`; ntfy and
  Slack delivery are off until configured (`atlas alerts check` runs them by hand).
- `atlas/api.py` — `atlas api serve`: read-only FastAPI over the same tools (regime, views,
  frozen views, track record, portfolio, gates, analogs, scenario, series, alerts,
  snapshots); OpenAPI docs at `/docs`. Localhost only unless `ATLAS_API_TOKEN` is set.
- `atlas/views/track.py` — `atlas pipeline track`: frozen views scored once their horizon
  has elapsed (the live paper track, immune to hindsight).
- `atlas/quality/drift.py` — live-vs-backtest IC monitor (alert rule `live_ic_drift`) and a
  feature PSI diagnostic in every run report (not an alert: see the module docstring).
- `docs/runbooks.md` — operations: scheduled jobs, reading a pipeline run, alerts, recovery,
  evals, API.
- `tests/test_leakage.py` — the look-ahead leakage suite (blocking in CI).

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
