# Atlas — global quant AI research assistant

Spec: [PRD.md](PRD.md). Current status: **Phase 0** (foundations) + **Phase 2** (macro regime engine).

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
uv run atlas snapshot create
```

## Checks (same as CI)
```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```
