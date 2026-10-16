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
  12-month recession probit with NBER label lag, and a walk-forward evaluator.
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
  quant engine through 8 tools (views, track records, gates, portfolios, series, regime);
  a deterministic guard checks every number in the answer against the tool results and
  gives Claude one repair turn. Needs `ANTHROPIC_API_KEY` or an `ant auth login` profile.
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
uv run atlas ingest kenfrench --all     # free, no key
uv run atlas ingest sdmx                # ECB, OECD, BIS, IMF; free, no keys
uv run atlas research signal mom_12_1 --universe ind49
uv run atlas research family momentum --universe ind49   # PBO across configs
uv run atlas research gates --universe ind49             # PRD promotion gates
uv run atlas views build --universe ctry_dev_lcl         # country views
uv run atlas research cards --universe ind49             # regenerate docs/signals
uv run atlas ask "Which developed markets do the models favour, and how reliable is that?"
uv run atlas snapshot create
```

## Checks (same as CI)
```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```
