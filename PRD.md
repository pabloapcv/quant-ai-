# PRD — "Atlas": Global Quant AI Research Assistant

| Field | Value |
|---|---|
| Status | Draft v1.0 |
| Date | 2026-10-14 |
| Owner | Pablo Peña |
| Doc type | Product + Technical Requirements (build spec) |
| Audience | Engineers, data scientists, and the LLM agents that will help build it |

---

## 0. TL;DR

Atlas is a research system that ingests point-in-time global macro, market, fundamental, positioning, and text data. It runs a layered quant stack on that data: a macro regime engine, expected-return models for each asset class, a risk model, and portfolio construction. On top sits an LLM reasoning layer (Claude) that **explains and debates** the quant output but **never invents numbers**. It produces ranked, scored, falsifiable views (overweight / neutral / underweight) on:

- **11 GICS sectors / 25 industry groups** across US, Europe, Japan, and EM
- **~45 country equity markets**
- **Rates** (curve level, slope, real yields) for G10 and key EM markets
- **Credit** (IG / HY / EM sovereign)
- **FX** (G10 + 15 EM pairs vs USD)
- **Commodities** (energy, metals, ags)
- **Crypto** (BTC, ETH) as a liquidity-sensitive sleeve

Views come at four horizons (1W, 1M, 3M, 12M). Each view carries an expected excess return, a confidence level, its drivers, its risks, and an explicit invalidation condition. Every view can be reproduced from an immutable data snapshot ID.

v1 is a **decision-support tool with no automated execution**.

---

## 1. Problem & Goals

### 1.1 Problem
Retail investors and small funds have the same free data as institutions (FRED, EDGAR, IMF, ECB, and others). What they lack:
1. **Point-in-time discipline.** Most DIY backtests leak revised data and suffer survivorship bias.
2. **Integrated cross-asset reasoning.** Sector calls are rarely tied to the macro regime, rates, the dollar, and liquidity at the same time.
3. **Rigorous validation.** Overfit backtests (no purged CV, no deflated Sharpe) are the norm.
4. **Explainability.** Model output lacks a narrative that can be argued with and falsified.

### 1.2 Goals (measurable)
| # | Goal | Metric | Target (after 12-month live paper track) |
|---|---|---|---|
| G1 | Sector ranking skill | Mean monthly rank IC (Spearman) of 1M sector views vs realized excess return | ≥ 0.05, ICIR ≥ 0.5 |
| G2 | Regime detection | Balanced accuracy of nowcast regime vs ex-post (final vintage) regime | ≥ 65% |
| G3 | Calibration | Brier score on directional calls vs naive 50/50 | ≥ 10% improvement |
| G4 | Model portfolio | Net-of-cost Sharpe of the tilt portfolio vs equal-weight benchmark, OOS | Information ratio ≥ 0.4 |
| G5 | Robustness | Deflated Sharpe Ratio p-value of the promoted strategy | < 0.05 |
| G6 | Faithfulness | Share of numeric claims in LLM output that trace to a tool result | 100% (hard gate) |
| G7 | Freshness | Data available within SLA after official release (see §9) | ≥ 99% of releases |
| G8 | Reproducibility | Recommendations reproducible bit-for-bit from `snapshot_id` | 100% |

### 1.3 Non-Goals (v1)
- Automated order execution or broker integration (possible in v3; the existing Kalshi bot is a separate system).
- Single-stock picking for >500 names. v1 uses stocks only as building blocks for sector and factor aggregates.
- High-frequency or intraday signals (< 1 day).
- Personalized advice. Atlas produces model research views, not suitability-assessed advice.

---

## 2. Users & Core Use Cases

| Persona | Need |
|---|---|
| **Owner / PM** | Weekly "state of the world" brief, sector and asset allocation tilts, alerts on regime changes |
| **Researcher** | Add a new signal, backtest it with point-in-time rigor, and see its marginal contribution to the ensemble |
| **Conversational user** | Ask "What happens to European banks if the ECB cuts 50bp and the curve bull-steepens?" and get a quant-grounded answer |

### Top user stories (v1 must-have)
1. *As a PM*, every Monday at 07:00 local I receive a report covering the macro regime, a heatmap of views by asset × horizon, the top 5 changes since last week, and key risks.
2. *As a PM*, I can ask "why underweight Utilities?" and get the factor/driver decomposition, historical analogs, and the invalidation level.
3. *As a researcher*, I can define a signal in a Python file, run `atlas backtest signal=my_signal`, and get a tearsheet with IC decay, turnover, DSR, and PBO.
4. *As a PM*, I get an alert within 15 minutes when a regime probability crosses a threshold or a view flips.
5. *As any user*, I can run a scenario ("oil +30%", "10Y +100bp", "USD −10%") and get the estimated impact per sector and asset with confidence bands.

---

## 3. Product Output Specification

### 3.1 Coverage universe (v1)
| Asset class | Instruments (proxies for return measurement) |
|---|---|
| US sectors | 11 SPDR sector ETFs (XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLB, XLRE, XLC) plus equal-weight variants (RSP family) as a robustness check |
| Global sectors | MSCI ACWI sector indices; iShares global sector ETFs (IXN, IXG, IXC, IXJ, EXI, RXI, KXI, JXI, MXI, REET, IXP) |
| Countries | iShares MSCI country ETFs (EWJ, EWG, EWU, EWZ, INDA, MCHI, EWY, EWT, …), ~45 in total |
| Factors | Fama-French 5 + Momentum (Ken French library); ETF proxies MTUM, QUAL, USMV, VLUE, SIZE |
| Rates | US 2Y/5Y/10Y/30Y, Bund, Gilt, JGB, BTP-Bund spread; futures-equivalent total-return series |
| Credit | LQD, HYG, EMB; ICE BofA OAS indices (via FRED) |
| FX | DXY, broad trade-weighted USD; EURUSD, USDJPY, GBPUSD, AUDUSD, USDCAD, USDCHF, NZDUSD, USDNOK, USDSEK; USDCNH, USDINR, USDBRL, USDMXN, USDZAR, USDKRW, USDTRY, USDIDR |
| Commodities | WTI, Brent, NatGas, Gold, Silver, Copper, Iron ore, Wheat, Corn, Soybeans (continuous-futures returns with roll yield) |
| Crypto | BTC, ETH |

### 3.2 The `View` object (canonical output)
Every recommendation is a typed, versioned object. Pydantic v2 model, persisted to Postgres:

```json
{
  "view_id": "uuid",
  "snapshot_id": "snap_2026-10-12T22:00Z_7f3a",
  "created_at": "2026-10-13T06:00:00Z",
  "asset_id": "SECTOR.US.GICS45",          // Information Technology, US
  "asset_class": "equity_sector",
  "horizon": "3M",
  "stance": "overweight",                  // strong_underweight|underweight|neutral|overweight|strong_overweight
  "score": 0.62,                           // ensemble z-score, clipped to [-3, 3]
  "expected_excess_return": {"mean": 0.021, "p10": -0.034, "p90": 0.071, "benchmark": "ACWI"},
  "prob_outperform": 0.61,                 // calibrated (isotonic) probability
  "confidence": "medium",                  // from model agreement + calibration bucket
  "regime_context": {"regime": "late_expansion_disinflation", "prob": 0.58},
  "drivers": [
    {"name": "earnings_revisions_3m", "contribution": 0.24, "value": 1.8, "unit": "z"},
    {"name": "real_yield_10y_change_3m", "contribution": -0.11, "value": 0.35, "unit": "pp"},
    {"name": "price_momentum_12_1", "contribution": 0.19, "value": 1.1, "unit": "z"}
  ],
  "model_breakdown": {"structural": 0.40, "ml_ranker": 0.71, "trend": 0.55, "llm_qualitative": 0.20},
  "risks": ["AI capex guidance cut", "10Y real yield > 2.5%"],
  "invalidation": {"rule": "score < 0 for 2 consecutive weekly runs OR XLK/ACWI relative drawdown > 6%"},
  "historical_analogs": [{"period": "1997-Q2", "similarity": 0.81, "fwd_3m_excess": 0.045}],
  "narrative_md": "…LLM-generated, every number tagged with a source tool-call id…",
  "provenance": {"model_versions": {"ml_ranker": "lgbm_sector_v14"}, "code_sha": "abc123"}
}
```

### 3.3 Stance mapping
Stance comes from calibrated `prob_outperform` (P) **and** model agreement (A = share of sub-models with the same sign):

| Stance | Rule |
|---|---|
| strong_overweight | P ≥ 0.65 and A ≥ 0.75 |
| overweight | P ≥ 0.55 |
| neutral | 0.45 < P < 0.55, or A < 0.5 |
| underweight | P ≤ 0.45 |
| strong_underweight | P ≤ 0.35 and A ≥ 0.75 |

Hysteresis: a stance changes only when the new P crosses its threshold by ≥ 0.02. This cuts flip-flopping.

---

## 4. System Architecture

```
                         ┌───────────────────────────────────────────────┐
                         │                ORCHESTRATION                  │
                         │  Dagster (software-defined assets, sensors,   │
                         │  release-calendar schedules, backfills)       │
                         └───────────────────────────────────────────────┘
 ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
 │ 1. INGEST    │──▶│ 2. STORAGE   │──▶│ 3. FEATURES  │──▶│ 4. MODELS    │──▶│ 5. DECISION  │
 │ connectors   │   │ raw→bronze→  │   │ point-in-time│   │ regime, ER,  │   │ ensemble,    │
 │ (40+ sources)│   │ silver→gold  │   │ feature store│   │ risk, PC     │   │ calibration, │
 │ httpx/async  │   │ Parquet+     │   │ (as-of joins)│   │ MLflow       │   │ View objects │
 └──────────────┘   │ DuckDB/Arctic│   └──────────────┘   └──────────────┘   └──────┬───────┘
                    │ Postgres meta│                                                │
                    └──────────────┘                                                ▼
 ┌──────────────┐   ┌───────────────────────────────────────┐          ┌───────────────────────┐
 │ TEXT INGEST  │──▶│ NLP: chunk → embed (pgvector) → LLM   │─────────▶│ 6. LLM REASONING      │
 │ FOMC/ECB/BoJ │   │ extraction (Haiku) → structured       │          │ Claude agent w/ tools │
 │ filings, news│   │ sentiment/hawkishness features        │          │ (quant engine = tools)│
 └──────────────┘   └───────────────────────────────────────┘          └──────────┬────────────┘
                                                                                   ▼
                                     ┌─────────────────────────────────────────────────────┐
                                     │ 7. INTERFACES: FastAPI · Next.js dashboard · chat · │
                                     │    weekly HTML/PDF report · alerts (Slack/email/push)│
                                     └─────────────────────────────────────────────────────┘
          Cross-cutting: Pandera data contracts · OpenTelemetry · Sentry · Prometheus/Grafana
```

### 4.1 Tech stack decisions
| Concern | Choice | Why / alternatives rejected |
|---|---|---|
| Language | Python 3.12 (core), TypeScript (dashboard) | Ecosystem. Rust/Polars for hot paths only if needed |
| Dataframes | **Polars** for ETL and features; pandas only at library boundaries (statsmodels, PyPortfolioOpt) | 5–20× faster and lazy evaluation; pandas index semantics cause as-of bugs |
| Orchestration | **Dagster** | Asset lineage, partitions (date × source), sensors on release calendars. Airflow rejected: task-centric and heavier |
| Lake format | **Parquet**, Hive-partitioned, zstd; Apache Iceberg in v2 for time travel | Columnar, cheap, DuckDB-native |
| Analytical engine | **DuckDB** | Embedded, fast as-of joins (`ASOF JOIN`), zero ops |
| Tick/bar time series | **ArcticDB** (Man Group, open source) | Versioned dataframes, built for quant time series, S3 backend |
| Metadata / views / users | **Postgres 16 + pgvector + TimescaleDB extension** | One DB for relational data, vectors, and alert time series |
| Object store | Local MinIO (dev) → S3 / Cloudflare R2 (prod) | R2 has no egress fees |
| Experiment tracking | **MLflow** (tracking + model registry) | Champion/challenger promotion |
| Hyperparameter search | **Optuna** with purged-CV objective | |
| Data quality | **Pandera** schemas on every silver/gold asset, plus custom anomaly checks | Great Expectations rejected: too heavy for a team of 1–3 |
| API | **FastAPI** + Pydantic v2 | |
| Front-end | **Next.js 15 + Tailwind + Plotly/ECharts**; Streamlit for internal research only | |
| LLM | **Anthropic Claude**: `claude-opus-5-5` (synthesis, debate, reports), `claude-haiku-5-5` (bulk extraction/classification), `claude-sonnet-5-5` (chat default) | Prompt caching for system prompts and tool schemas; Batch API for nightly extraction |
| Embeddings | Voyage `voyage-finance-2` (domain-tuned) or `voyage-3-large` | |
| Infra | Docker Compose (dev) → single Hetzner/AWS box + managed Postgres (v1); k8s only if needed | |
| CI/CD | GitHub Actions: ruff, mypy --strict, pytest, data-contract tests, nightly backtest regression | |
| Secrets | 1Password CLI / AWS Secrets Manager. Never in `.env` committed | |

---

## 5. Data Layer (most important section)

### 5.1 Golden rule: bitemporal, point-in-time everything
Every fact is stored with **two timestamps**:
- `event_time`: the period the value describes (e.g. CPI for 2026-08). Follows the FRED convention: period *start* for M/Q/SA/A, the stated date for D/W
- `knowledge_time`: when the value became publicly known (release timestamp, UTC)

Every model input is retrieved with `as_of(t)` semantics: `WHERE knowledge_time <= t`, keeping the latest vintage per `event_time`. **No feature may be computed without an `as_of`.** A unit test injects a "future" vintage and asserts that no feature changes (the leak test, §8.6).

Canonical silver schema (`observations`):

| column | type | notes |
|---|---|---|
| series_id | str | internal canonical ID, e.g. `US.CPI.CORE.YOY` |
| source | str | `fred`, `alfred`, `bls`, `ecb`, … |
| source_series_id | str | e.g. `CPILFESL` |
| event_time | date | period start (FRED convention) |
| knowledge_time | ts (UTC) | release time; ALFRED `realtime_start` for history |
| value | float64 | |
| unit | str | normalized via `pint` |
| frequency | enum | D/W/M/Q/A |
| vintage_id | str | |
| sa_flag | bool | seasonally adjusted |
| ingest_run_id | str | lineage |

When an observation predates ALFRED vintage coverage, a synthetic row is added at `knowledge_time = period_end + publication_lag(series)`, flagged `is_approx_vintage=true`. Lags and release times live in `config/series.yaml`.

### 5.2 Data sources

#### Tier A — free, official (v1 core)
| Domain | Source | Access | Key content |
|---|---|---|---|
| US macro | **FRED** + **ALFRED** (vintages) | REST API, key | ~800k series; ALFRED gives real-time vintages, which point-in-time work depends on |
| US labor/prices | BLS API v2 | REST, key | CPI components, CES, JOLTS, PPI |
| US output | BEA API | REST, key | GDP, PCE, corporate profits, NIPA tables |
| US housing/trade | Census API | REST | Retail sales, housing starts, trade, M3 durable goods |
| US fiscal/liquidity | Treasury FiscalData, Daily Treasury Statement | REST | TGA balance, auctions, debt |
| US rates | Fed H.15, NY Fed Markets API | REST | SOFR, EFFR, RRP, SOMA holdings |
| Term premium | NY Fed ACM term-premium model | CSV | 1–10Y term premium |
| Nowcasts | Atlanta Fed GDPNow, NY Fed Staff Nowcast, Cleveland Fed inflation nowcast | Scrape/CSV | Benchmarks for our own nowcast |
| Supply chain | NY Fed GSCPI | CSV | Global supply-chain pressure |
| Financial conditions | Chicago Fed NFCI/ANFCI, St. Louis Fed STLFSI | FRED | |
| Energy | EIA API v2 | REST, key | Inventories, production, STEO |
| Ags | USDA (WASDE, NASS QuickStats) | REST | Supply/demand balances |
| Positioning | CFTC Commitments of Traders (legacy, disaggregated, TFF) | Socrata API | Weekly futures positioning |
| Filings | **SEC EDGAR**: XBRL `companyfacts`, `frames`, 13F, Form 4, 8-K | REST (User-Agent required, ≤10 req/s) | Point-in-time fundamentals, insider and institutional flows |
| Global macro | **IMF** (IFS, WEO, DOTS, COFER) via SDMX 2.1/3.0 | `sdmx1` Python lib | Country GDP, reserves, trade |
| Global macro | **OECD** (Composite Leading Indicators, MEI) | SDMX | CLIs for ~40 countries, a key leading indicator |
| Global macro | **World Bank WDI** | REST | Structural data |
| Global finance | **BIS** (credit-to-GDP gaps, property prices, effective exchange rates, global liquidity) | SDMX | |
| Euro area | **ECB Data Portal** | SDMX | Rates, M3, bank lending survey |
| EU | Eurostat | SDMX | HICP, industrial production |
| UK | ONS, Bank of England database | REST/CSV | |
| Japan | BoJ, e-Stat | CSV/API | Tankan |
| China | NBS (via IMF/OECD/BIS mirrors), PBoC | Mixed | Treat as lower reliability (flag `quality=low`) |
| Academic factors | **Kenneth French Data Library** | ZIP/CSV | FF3/FF5/Mom, industry portfolios back to 1926 |
| Academic factors | **AQR datasets** (QMJ, BAB, time-series momentum, century of factor premia) | XLSX | |
| Academic factors | Open Source Asset Pricing (Chen & Zimmermann) | CSV | 200+ replicated anomaly signals |
| Valuation | Shiller CAPE dataset; Damodaran (ERP, country risk premia) | XLSX | |
| Volatility | CBOE (VIX, VIX9D, VIX3M, VVIX, SKEW, put/call) | CSV | |
| Prediction markets | **Kalshi** (reuse existing bot's client), Polymarket | REST | Market-implied odds for FOMC, CPI, recession, elections. Unique edge |
| Events/news (free) | GDELT 2.0 | BigQuery/CSV | Geopolitical risk tone |
| Geopolitical risk | Caldara–Iacoviello GPR index; Baker–Bloom–Davis EPU | CSV | |
| Central bank text | FOMC statements/minutes/SEP/pressers, ECB, BoE, BoJ, PBoC, RBA, BoC | Scrape (HTML/PDF) | Hawkish/dovish scoring |

#### Tier B — paid, recommended for v1 production (budget ~$150–600/mo, verify current pricing)
| Domain | Primary pick | Alternative | Why |
|---|---|---|---|
| Daily equity/ETF OHLCV, adjusted, **survivorship-free** | **Norgate Data** or **Sharadar SEP** (Nasdaq Data Link) | EODHD, Tiingo | Delisted securities included; corporate actions |
| Point-in-time fundamentals | **Sharadar SF1** (PIT, as-reported, ARQ/ART dimensions) | FMP, EDGAR-derived (free but needs heavy cleaning) | `datekey` = filing date, which is real point-in-time |
| Intraday/real-time + options + futures | **Polygon.io** or **Databento** | IBKR API | Databento for CME futures with roll-quality data |
| Futures continuous series | Databento (CME Globex) or Norgate futures | | Needed for commodity carry and rates |
| FX spot/forwards | Databento / OANDA / Polygon FX | | Forward points for carry |
| Options-implied data | **ORATS** | CBOE DataShop | Implied vol surface, skew, term structure per ETF |
| Estimates and revisions | FMP (analyst estimates), Zacks via Nasdaq Data Link | LSEG I/B/E/S (enterprise) | Revision signals are among the strongest sector signals |
| PMIs | S&P Global PMI (licensed); fallback: ISM headlines via press releases, regional Fed surveys (free) | | PMIs are proprietary. Do not scrape |
| News | Benzinga / Polygon news / Tiingo news | RavenPack (enterprise) | Timestamped headlines for sentiment |
| Transcripts | FMP earnings-call transcripts | | LLM-extracted guidance tone |

#### Tier C — enterprise (optional adapters; same interface)
Bloomberg (BLPAPI / B-PIPE), LSEG Workspace / Datastream, FactSet, CRSP/Compustat via WRDS, MSCI Barra. Build a `Connector` interface so these drop in without touching downstream code.

> **Licensing:** `yfinance` and scraping of Yahoo, Investing.com, or TradingView are **allowed only in notebooks/prototypes** and are **forbidden in production pipelines** (ToS risk, silent data errors). Enforce with a CI lint rule that bans `import yfinance` under `atlas/`.

### 5.3 Connector contract
```python
class Connector(Protocol):
    source: str

    def list_series(self) -> list[SeriesMeta]: ...
    async def fetch(
        self, series: SeriesMeta, start: date, end: date, vintages: bool = True
    ) -> pl.DataFrame: ...  # returns bronze schema
    def release_calendar(self, start: date, end: date) -> list[Release]: ...

    rate_limit: RateLimit  # token bucket, honored by a shared aiolimiter
    retry: RetryPolicy  # tenacity: exp backoff, jitter, max 6, retry on 429/5xx
```
- All HTTP goes through `httpx.AsyncClient` with HTTP/2, a 30s timeout, and an ETag/Last-Modified cache.
- Every raw response is stored **immutably** at `s3://atlas-raw/{source}/{yyyy}/{mm}/{dd}/{run_id}/{hash}.json.zst` before parsing. That makes any parse re-runnable.
- Idempotent upserts keyed on `(series_id, event_time, vintage_id)`.

### 5.4 Medallion layers
| Layer | Content | Format | Retention |
|---|---|---|---|
| raw | Untouched API payloads | zstd JSON/CSV/PDF | Forever |
| bronze | Parsed, typed, source schema | Parquet | Forever |
| silver | Canonical bitemporal `observations`, `prices`, `fundamentals_pit`, `documents` | Parquet + ArcticDB | Forever |
| gold | Features (as-of), model-ready panels, `views` | Parquet + Postgres | Forever (snapshotted) |

### 5.5 Security master and reference data
- **Identifiers:** internal `atlas_id`, mapped to FIGI (OpenFIGI API), ticker history (tickers change), CIK, ISIN where licensed.
- **Corporate actions:** splits, dividends, spinoffs, and ticker changes go into `corp_actions`. Store both adjusted and unadjusted prices and recompute adjustments ourselves.
- **Classification history:** GICS sector membership is time-varying. Communication Services was created in Sept 2018 and Real Estate split out in 2016. Store `classification(atlas_id, scheme, code, valid_from, valid_to)` and **map historical sectors to current definitions with an explicit backcast rule** (e.g. pre-2018 XLC is reconstructed from constituents).
- **Futures roll:** back-adjusted (ratio) continuous contracts with a volume/OI-based roll and a stored roll calendar. Carry = log(F1/F2) annualized.

### 5.6 Data quality gates (Pandera + custom)
Each gold asset fails the Dagster run (blocking downstream) on any of:
- Schema and dtype mismatch; nulls in non-nullable columns
- Price jump > 8σ (rolling 250d MAD) without a matching corp action → quarantine
- Stale series: no update past `expected_frequency × 1.5` → alert, keep last value flagged `stale=True`
- Vintage regression: `knowledge_time` decreases for the same `vintage_id`
- Cross-source reconciliation: FRED `DGS10` vs Treasury yield curve differ > 3bp → warn
- Coverage: < 95% of universe present on a trading day → block

---

## 6. Feature / Signal Library

Every feature is a registered Python function with metadata. The registry auto-generates documentation and tearsheets.

```python
@feature(
    name="credit_hy_oas_z_3y",
    inputs=["US.CREDIT.HY.OAS"],  # BAMLH0A0HYM2
    frequency="D",
    lookback="3Y",
    transform="zscore",
    lag_policy="release",
    tags=["macro", "credit", "risk_appetite"],
)
def credit_hy_oas_z(ctx: AsOfContext) -> pl.DataFrame: ...
```

### 6.1 Macro block (global, country level)
| Pillar | Example features (FRED IDs where applicable) |
|---|---|
| Growth | Nowcast GDP (own DFM), INDPRO, RSAFS, PAYEMS 3m avg, ICSA 4wk avg & YoY, CCSA, JTSJOL, PERMIT, HOUST, OECD CLI level & 6m diff, PMI new orders − inventories, Sahm rule (`SAHMREALTIME`), Citi-style surprise index (own: actual − consensus, z-scored, 3m exp-decay) |
| Inflation | CPIAUCSL, CPILFESL, PCEPILFE 3m/6m annualized, supercore, Cleveland Fed nowcast, `T10YIE`/`T5YIFR` breakevens, GSCPI, wage growth (Atlanta Fed tracker), commodity impulse |
| Policy | `DFF`/SOFR; 2Y − policy rate (market-implied path); Kalshi/Polymarket FOMC odds; central-bank text hawkishness (§6.6); QT pace (`WALCL` Δ) |
| Liquidity | Net liquidity = `WALCL − WTREGEN − RRPONTSYD`; M2 YoY (`M2SL`); BIS global liquidity; dollar funding (cross-currency basis, if licensed) |
| Financial conditions | `NFCI`, `STLFSI4`, HY OAS (`BAMLH0A0HYM2`), IG OAS (`BAMLC0A0CM`), `VIXCLS`, MOVE (licensed), broad USD (`DTWEXBGS`) |
| Curve | `T10Y2Y`, `T10Y3M`, 5s30s, ACM term premium, real 10Y (`DFII10`), curve regime (bull/bear steepener/flattener from Δlevel × Δslope signs) |
| External / global | China credit impulse (BIS/PBoC TSF), EM FX index, trade (IMF DOTS), copper/gold ratio, Baltic Dry |
| Risk / uncertainty | EPU, GPR, GDELT tone, VIX term structure (VIX/VIX3M) |

### 6.2 Cross-sectional equity signals (computed at stock level, aggregated to sector/country by cap weight and median)
| Family | Signals |
|---|---|
| Momentum | 12-1M return, 6-1M, residual momentum (vs FF factors), industry momentum (Moskowitz–Grinblatt), 52w-high proximity |
| Value | E/P, B/P, FCF/EV, EBITDA/EV; sector-relative and vs own 10y history |
| Quality | Gross profitability (Novy-Marx), ROE stability, accruals (Sloan), Piotroski F, leverage |
| Revisions / earnings | FY1 EPS revision breadth (up−down)/total, 3m Δ FY1 EPS, SUE, earnings-announcement drift |
| Investment | Asset growth, net share issuance (buybacks) |
| Risk | Beta, idiosyncratic vol, low-vol anomaly, downside beta |
| Flows / positioning | 13F crowding (HHI of holders), short interest/days-to-cover, ETF flows (shares outstanding Δ), insider net buying (Form 4) |
| Options | 25Δ put-call skew, IV-RV spread, IV term slope (ORATS) |
| Breadth | % above 200dma, advance/decline, new highs − lows per sector |
| Text | Transcript guidance tone, 10-K risk-factor change (Lazy Prices: cosine similarity YoY), news sentiment |

Aggregation: rank-normalize signals cross-sectionally within the region first, winsorize at 1/99%, then aggregate to sector by **cap-weighted mean** and **median** (both kept as features).

### 6.3 Time-series / cross-asset signals
- **Trend:** multi-horizon TSMOM (1, 3, 6, 12M) with volatility scaling (Moskowitz–Ooi–Pedersen); EWMA crossovers (8/24, 16/48, 32/96).
- **Carry:** equities (dividend yield − cash plus buyback yield), bonds (yield + roll-down from a fitted Nelson-Siegel-Svensson curve via QuantLib), FX (forward-implied rate differential), commodities (F1/F2 roll yield).
- **Value:** FX vs PPP and BEER fair value; bonds vs a nominal-GDP-growth anchor; commodities vs 5y real-price mean.
- **Defensive / BAB:** low beta.
- **Seasonality:** monthly sector seasonality (Heston–Sadka), FOMC-cycle drift. Low weight; feature only.

### 6.4 Macro-to-sector sensitivity features
Rolling (5y window, monthly) and regime-conditional betas of each sector's excess return to: Δ10Y real yield, Δbreakeven, Δslope, ΔHY OAS, ΔUSD, Δoil, Δcopper, growth-surprise index, inflation-surprise index. These power **scenario analysis** (§7.6) and driver attribution.

### 6.5 Feature engineering rules
- All rolling stats use **only past data** (Polars `rolling_*` with `closed="left"` on as-of panels).
- Normalization: expanding or rolling z-score (`min_periods` ≥ 36 months); never a full-sample z-score.
- Mixed frequencies: as-of forward-fill only up to `max_staleness(frequency)` (D: 5d, W: 14d, M: 45d, Q: 120d), then null.
- Missing data: features are allowed to be NaN. LightGBM handles them natively; linear models use median imputation plus an `is_missing` indicator.

### 6.6 NLP / text features
| Corpus | Pipeline | Output feature |
|---|---|---|
| FOMC/ECB/BoE/BoJ statements, minutes, speeches | Sentence split → `claude-haiku-5-5` classify each sentence {hawkish, dovish, neutral} × topic {inflation, labor, growth, financial stability}; validated against FinBERT-FOMC on a 500-sentence hand-labelled set (target κ ≥ 0.7) | Hawkishness index per bank, Δ vs prior meeting |
| Earnings transcripts | Section split (prepared vs Q&A) → extract guidance direction, demand commentary, capex, pricing power as JSON | Sector guidance-tone index; AI-capex mentions count |
| 10-K/10-Q | Risk-factor diff (YoY cosine similarity on TF-IDF + embeddings) | "Lazy prices" signal |
| News headlines | Dedupe (MinHash) → entity link to sector/country → sentiment (FinBERT for cost; Haiku for ambiguous cases) | 5d/20d exp-decayed sentiment, attention (volume) |
| GDELT | Country-level tone and conflict event counts | Geopolitical stress per country |

All LLM extraction runs through the **Message Batches API** nightly (≈50% cost), with prompt caching on the instruction block. Every extraction stores `(prompt_version, model_id, input_hash, output_json)` for reproducibility.

---

## 7. Modeling Stack

### 7.1 Layer 1 — Macro Regime Engine
**Goal:** a probability distribution over regimes, updated daily.

1. **Nowcasting (growth and inflation):** mixed-frequency **dynamic factor model**, `statsmodels.tsa.statespace.DynamicFactorMQ`. Inputs are ~40 monthly series and quarterly GDP from ALFRED vintages. Output: current-quarter GDP nowcast and a common growth factor. Benchmarked against GDPNow and the NY Fed nowcast (target: RMSE within 10% of GDPNow on 2010–2025 real-time vintages). Same approach for an inflation factor.
2. **Regime classification:** two complementary methods.
   - **Rules-based quadrant** (interpretable): sign of growth momentum × sign of inflation momentum → {Reflation, Goldilocks, Stagflation, Deflationary slowdown}. Liquidity and financial conditions (NFCI) act as a modifier (easing/tightening).
   - **Statistical:** Gaussian **HMM** (`hmmlearn`, 4–6 states, full covariance, chosen by BIC) on [growth factor, inflation factor, ΔNFCI, curve slope, HY OAS z]. Also a Markov-switching regression (`statsmodels MarkovRegression`) on equity excess returns for a risk-on/risk-off probability. States are relabelled by their mean characteristics so labels stay stable across refits.
3. **Change-point detection:** `ruptures` (PELT, RBF cost) on the growth and inflation factors to flag structural breaks that should trigger a model refit.
4. **Recession probability:** probit on term spread (`T10Y3M`), Sahm, HY OAS, NFCI, Kalshi recession market. Calibrated on NBER dates (with real-time vintages).

Output table `regime_probs(as_of, model, regime, prob)`.

### 7.2 Layer 2 — Expected Return Models (per asset class)
Four sub-models; the ensemble combines them (§7.5).

| Sub-model | Method | Targets |
|---|---|---|
| **A. Structural / premia** | Linear factor model: E[r] = Σ βₖ · λₖ(regime), where λₖ are regime-conditional factor premia (estimated with Bayesian shrinkage toward the long-run premium) | All asset classes |
| **B. ML cross-sectional ranker** | **LightGBM** `lambdarank` (sector/country ranking) and `regression` (excess return) on the §6 feature panel. Monotone constraints where theory demands it (e.g. revisions ↑ → return ↑). Hyperparameters by Optuna with **purged K-fold + embargo** | Sectors, countries, industries |
| **C. Trend / carry (time-series)** | Volatility-targeted TSMOM + carry composite (risk-parity weighted) | Rates, FX, commodities, crypto, sector absolute |
| **D. LLM qualitative overlay** | Claude scores each asset on a −2..+2 rubric from the week's text corpus (policy, guidance, geopolitics). **Weight capped at 10%** and only allowed in the ensemble if its OOS paper-track IC > 0 after 6 months | All |

Training details for B:
- Universe-month panel from 1995 (ETF proxies reconstructed from Ken French 49-industry portfolios and constituents pre-ETF).
- Target: forward 1M / 3M excess return vs the regional benchmark, **rank-transformed** per date (robust to outliers).
- CV: **Combinatorial Purged CV** (López de Prado, *AFML* ch. 7 & 12): N = 10 groups, k = 2 test groups, embargo = horizon length. Gives a distribution of OOS paths for PBO.
- Feature selection: Clustered MDA feature importance (to handle multicollinearity); drop features with negative mean OOS importance.
- Explainability: **SHAP** (TreeExplainer) per prediction → the `drivers` field in `View`.
- Retrain monthly. Champion/challenger in the MLflow registry. A challenger is promoted only if it beats the champion on the last 36M walk-forward IC with a Diebold-Mariano p < 0.10.

### 7.3 Layer 3 — Risk Model
- **Covariance:** blend of (a) EWMA (half-life 63d) and (b) **Ledoit-Wolf** shrinkage on 3y daily returns. Optional DCC-GARCH (`arch` + custom DCC) for crisis-sensitive correlations.
- **Factor risk model** (Barra-style, built in-house): factors = market, region, sector, style (value, momentum, quality, size, low-vol), macro (rates, USD, oil). Cross-sectional WLS regression daily → factor returns → factor covariance (Newey-West, half-life 90d) + specific risk (Bayesian shrinkage).
- **Tail:** historical and filtered-historical-simulation VaR/CVaR (97.5%), drawdown distribution by block bootstrap.
- **Stress tests (historical replay):** 1987 crash, 1994 bond rout, 1998 LTCM, 2000–02 dot-com, 2008 GFC, 2011 euro crisis, 2013 taper tantrum, 2015 CNY deval, 2018 Q4, Mar-2020 COVID, 2022 inflation/rate shock, Aug-2024 yen carry unwind.
- **Hypothetical shocks:** user-defined factor shocks propagated via the factor covariance (conditional expectation of all other factors given the shocked ones: E[f₂|f₁] = Σ₂₁Σ₁₁⁻¹f₁).

### 7.4 Layer 4 — Portfolio Construction (model portfolio, illustrative)
Turns views into a coherent tilt portfolio so that skill can be measured as returns, not just IC.
- **Black-Litterman** (PyPortfolioOpt / skfolio): prior = market-cap equilibrium (reverse optimization, δ = 2.5). Views = ensemble expected excess returns, with confidence Ω derived from each model's historical OOS error variance (Idzorek method).
- **Optimizer:** `cvxpy` (CLARABEL solver). Maximize μᵀw − λ·wᵀΣw − κ·‖w − w_prev‖₁ (turnover penalty). Constraints: |w − w_bench| ≤ 5% per sector, tracking error ≤ 4% annualized, long-only, fully invested.
- **Robustness alternatives**, reported side by side: HRP (Hierarchical Risk Parity) and equal-risk-contribution of the tilts.
- **Costs:** spread + square-root market impact (η·σ·√(Q/ADV)); ETF commissions 0.

### 7.5 Ensemble and calibration
1. Each sub-model outputs a cross-sectional z-score per (asset, horizon).
2. Combination weights: **inverse OOS-error-variance**, shrunk 50% toward equal weight, re-estimated quarterly. Also conditioned on regime: weights per regime cluster when ≥ 60 obs exist.
3. `prob_outperform` = **isotonic regression** on the ensemble score → realized sign, fit on walk-forward OOS predictions only.
4. `expected_excess_return` p10/p90 from **conformal prediction** (split-conformal on OOS residuals, per horizon), which gives coverage guarantees.
5. `confidence` = f(model agreement A, calibration bucket reliability, data-staleness penalty).

### 7.6 Scenario engine
Input: shocks on any of {10Y, 2Y, breakeven, HY OAS, USD, oil, copper, VIX, growth surprise, inflation surprise}. Output per asset: expected return = Σ βᵢ(current regime) · shockᵢ + conditional moves of unshocked factors (§7.3). Bands come from bootstrap of the beta estimates. Also **historical analog search**: k-NN (k = 10) on the standardized macro state vector (Mahalanobis distance) over 1970–present, returning forward returns per asset after each analog date.

---

## 8. Validation & Backtesting Framework (non-negotiable)

### 8.1 Engines
| Use | Tool |
|---|---|
| Signal research (IC, quantiles, decay) | `alphalens-reloaded` + own Polars IC engine |
| Fast vectorized portfolio sims | `vectorbt` (open-source) |
| Event-driven, realistic | Own lightweight engine (daily bars, next-open fills, cost model). `zipline-reloaded` as an optional cross-check |
| Performance tearsheets | `quantstats` + `empyrical-reloaded` |

### 8.2 Protocol
1. **Data:** point-in-time only (ALFRED vintages, Sharadar `datekey`, survivorship-free prices).
2. **Split:** in-sample design 1995–2014, validation 2015–2020, **locked holdout 2021–2025**. The holdout is opened once per model family, and the opening is logged in MLflow.
3. **Walk-forward:** expanding window, monthly refit, 1-period lag between signal date and trade date.
4. **Purging and embargo** for every CV (§7.2).
5. **Costs:** always on. Report gross and net.

### 8.3 Required metrics per signal / model
- Rank IC mean, IC std, ICIR, t-stat (Newey-West), IC decay at 1/5/21/63 days
- Quintile spread returns, monotonicity (Spearman of quintile vs return)
- Turnover, breakeven cost (bps)
- Sharpe, Sortino, Calmar, max drawdown, hit rate, skew, kurtosis
- **Deflated Sharpe Ratio** (Bailey & López de Prado 2014), using the number of trials logged in MLflow
- **Probability of Backtest Overfitting** (CSCV), which must be < 0.2
- Regime-conditional performance (it must not be all from one regime)
- Brier score and reliability diagram for `prob_outperform`
- Conformal interval empirical coverage (target 80% ± 3%)

### 8.4 Promotion gates (signal → production ensemble)
| Gate | Threshold |
|---|---|
| OOS rank IC t-stat | ≥ 2.0 (Harvey–Liu–Zhu suggest ≥ 3.0 for novel factors. Use 3.0 when there is no published prior) |
| DSR p-value | < 0.05 |
| PBO | < 0.20 |
| Positive IC in ≥ 3 of 4 regimes | required |
| Marginal contribution to ensemble ICIR | > 0 |
| Economic rationale documented | required (no pure data-mined signals) |

### 8.5 Live paper-trading track
From day 1 of production, every `View` is frozen and scored against realized returns automatically. The **live track is the source of truth** for the G1–G5 goals. Dashboard page: "Track record", with no cherry-picking possible because views are immutable.

### 8.6 Leakage test suite (CI, blocking)
- **Future-vintage injection:** add a fake future vintage and assert features at `t` are unchanged.
- **Shuffle test:** shuffle the target and assert IC ≈ 0 (|IC| < 0.02).
- **Timestamp audit:** assert every feature's max `knowledge_time` ≤ `as_of`.
- **Survivorship test:** the universe at date t includes later-delisted names.

---

## 9. Pipeline Schedules & SLAs

All times US/Eastern. The orchestrator is driven by a **release calendar** (FRED release dates API + an official calendar scrape), not just cron.

| Job | Trigger | SLA |
|---|---|---|
| Macro release ingest (CPI, NFP, GDP, …) | Release-calendar sensor at T+0 with 1-min polling for 30 min | Ingested ≤ 5 min after release |
| Daily market close prices (US) | 17:30 ET | Gold features by 19:00 ET |
| Global markets (Asia/EU close) | 04:00 / 12:30 ET | |
| CFTC COT | Fri 15:30 ET | |
| EDGAR filings (8-K, Form 4, 13F) | Every 10 min (RSS/full-index) | ≤ 15 min |
| Nightly NLP batch | 01:00 ET | Done by 05:00 ET |
| Nightly model scoring → Views | 05:30 ET | Views published 06:30 ET |
| Weekly report | Sun 22:00 ET build → Mon 07:00 local send | |
| Monthly retrain + champion/challenger | 1st Sat of month | |
| Regime/alert evaluator | After every gold update | Alert ≤ 15 min |

**Snapshotting:** each nightly run writes an immutable `snapshot_id` that pins the gold Parquet manifest hash, model versions (MLflow URIs), code SHA, and config hash. `atlas reproduce <view_id>` rebuilds exactly.

---

## 10. LLM Reasoning Layer

### 10.1 Principles
1. **The LLM is an analyst, not a calculator.** All numbers come from tool calls into the quant engine.
2. **Grounding is mandatory.** Every numeric token in output must reference a `tool_call_id`. A post-processor validates this (§10.4).
3. **Adversarial by design.** Every major view goes through a bull/bear debate before publication.
4. **Not advice.** The system prompt forbids personalized recommendations, and output carries a disclaimer.

### 10.2 Agent architecture (Claude tool use, Anthropic Python SDK)
Tools exposed (JSON-schema, thin wrappers around FastAPI internals):

| Tool | Description |
|---|---|
| `get_views(asset_filter, horizon)` | Current View objects |
| `get_regime(as_of)` | Regime probabilities, nowcasts, recession probability |
| `get_series(series_id, start, end, as_of)` | Point-in-time time series |
| `explain_view(view_id)` | SHAP drivers, sub-model breakdown |
| `run_scenario(shocks)` | Scenario engine (§7.6) |
| `find_analogs(as_of, k)` | Historical analogs + forward returns |
| `backtest_signal(signal_name, params)` | Async; returns tearsheet summary |
| `search_documents(query, source, date_range)` | Hybrid BM25 + pgvector search over CB statements, transcripts, filings, news |
| `get_track_record(asset, horizon)` | Live paper-track stats |
| `compare_to_consensus(asset)` | Prediction-market odds, consensus estimates, our nowcast vs GDPNow |

Model routing:
- Chat: `claude-sonnet-5-5`. Escalates to `claude-opus-5-5` for multi-step "why / what-if" questions or when the user asks for a deep dive.
- Weekly report and debate: `claude-opus-5-5` with extended thinking.
- Extraction and classification (nightly batch): `claude-haiku-5-5`.

### 10.3 Weekly report generation flow
1. **Planner** (Opus) reads the regime, top view changes, and the week's surprises, and outlines sections.
2. **Section writers** (parallel) each pull tools for their asset class.
3. **Red team** (separate Opus call with a "find the flaw" prompt) challenges the top 5 calls with counter-evidence from `search_documents` and `find_analogs`.
4. **Editor** merges, keeps disagreements visible ("Bear case:"), and enforces the template.
5. **Fact-check pass** (§10.4), then render to HTML/PDF (Jinja2 + WeasyPrint), email, and dashboard.

### 10.4 Hallucination guard (deterministic)
- Regex/AST-extract every number, %, date, and ticker from the output.
- Each must match a value in the turn's tool results within tolerance (rounding ±1 in the last displayed digit), or carry an inline `[src:tool_call_id]`.
- On failure: one auto-repair attempt ("these claims are unsupported: … revise"). If it still fails, the section is blocked and flagged for human review.
- Logged metric: `% grounded numeric claims` (target 100%, G6).

### 10.5 LLM evaluation harness
- 150-question golden set (macro, sector, scenario, adversarial "trick" questions such as asking for a price target the system does not produce).
- Scored by rubric (LLM-as-judge with Opus and a fixed rubric, plus 20% human spot-check): grounding, correctness vs tool data, calibration language ("likely" maps to the probability bucket), refusal of personalized advice.
- Runs in CI on prompt or model changes. Regression > 3 points blocks the merge.

### 10.6 Cost controls
- Prompt caching on system prompt + tool schemas (90%+ cache hit target).
- Batch API for all non-interactive extraction.
- Per-day token budget with a hard stop. Estimated v1 LLM spend: $50–250/mo depending on chat volume (verify against current pricing).

---

## 11. Interfaces

### 11.1 API (FastAPI, OpenAPI 3.1, JWT auth)
```
GET  /v1/regime?as_of=
GET  /v1/views?asset_class=&region=&horizon=&as_of=
GET  /v1/views/{view_id}
GET  /v1/views/{view_id}/explain
POST /v1/scenario           {shocks: {...}, horizon}
GET  /v1/analogs?as_of=&k=
GET  /v1/series/{series_id}?start=&end=&as_of=
POST /v1/chat               (SSE streaming)
GET  /v1/track-record?asset=&horizon=
POST /v1/backtests          (async job) → GET /v1/backtests/{id}
GET  /v1/snapshots/{snapshot_id}
```

### 11.2 Dashboard pages (Next.js)
1. **World Map**: regime gauge, growth/inflation quadrant with trail, recession probability, net liquidity, financial conditions.
2. **Views Heatmap**: assets × horizons, colored by score, with click-through to explain.
3. **Sector Rotation**: relative rotation graph (RS-ratio vs RS-momentum) plus model views.
4. **Macro Dashboard**: nowcasts vs GDPNow, surprise indices, curve, credit, USD, commodities.
5. **Scenario Lab**: sliders for shocks and a resulting impact table/chart.
6. **Analogs**: today vs top-10 historical analogs and their forward paths.
7. **Track Record**: live IC, hit rate, calibration plot, model portfolio vs benchmark.
8. **Chat**: side panel available on every page, context-aware (current page state passed as context).
9. **Research**: signal registry, tearsheets, MLflow links.

### 11.3 Alerts
Rules in YAML (`config/alerts.yaml`): regime prob crosses 0.5 / 0.7; view flip at the 3M horizon; recession prob > 40%; HY OAS 1-week Δ > 50bp; VIX/VIX3M > 1 (inversion); data-quality failure. Channels: email (Resend/SES), Slack webhook, push (ntfy.sh). Rate-limited and deduplicated.

### 11.4 CLI (Typer)
`atlas ingest --source fred --since 2026-01-01` · `atlas features build --as-of 2026-10-13` · `atlas backtest signal=revisions_breadth` · `atlas views publish` · `atlas reproduce <view_id>` · `atlas report weekly --dry-run`

---

## 12. Non-Functional Requirements

| Area | Requirement |
|---|---|
| Performance | Full nightly pipeline ≤ 90 min on 8 vCPU / 32 GB. Chat first-token ≤ 2s, tool round-trip p95 ≤ 800 ms. Scenario run ≤ 3s |
| Storage | ~50–150 GB by end of v1 (prices + fundamentals + text) |
| Availability | 99% for dashboard/API (single region acceptable in v1). Pipelines retried automatically |
| Reproducibility | Deterministic seeds; pinned deps (`uv.lock`); snapshot IDs (§9) |
| Observability | Structured logs (structlog, JSON) → Loki. Metrics (Prometheus): ingest lag per source, DQ failures, model drift (PSI per feature > 0.25 → alert), LLM tokens/cost, grounding rate. Traces: OpenTelemetry. Errors: Sentry |
| Drift monitoring | Feature PSI weekly; rolling 6M live IC vs backtest IC (alert if live < backtest − 2σ) |
| Testing | ≥ 85% unit coverage on `atlas/features`, `atlas/models`, `atlas/pit`; property-based tests (Hypothesis) for as-of joins; golden-file tests for connectors (recorded with `vcrpy`) |
| Code quality | ruff, mypy --strict, pre-commit, conventional commits |
| Cost (v1 monthly) | Infra $40–120, data $150–600 (Tier B), LLM $50–250 |

---

## 13. Security, Compliance, Legal

- **Disclaimer** on every output: research/education only; not investment advice; past performance and backtests do not guarantee future results.
- **No personalization in v1:** no collection of user portfolios or risk profiles, which avoids suitability obligations. If added later, get a legal review (RIA rules in the US, MiFID II in the EU).
- **Data licensing register** (`docs/data_licenses.md`): per source, record redistribution rights. The dashboard must not redistribute raw licensed data (e.g. PMIs, ORATS) to unlicensed users; show only derived signals.
- **SEC EDGAR fair access:** declared User-Agent with contact email, ≤ 10 req/s.
- **Secrets:** never in the repo; gitleaks pre-commit hook.
- **Auth:** single-user + invite. JWT with short TTL, Argon2 password hashing, or OAuth (Google) via Auth.js.
- **LLM data handling:** only public data and derived signals go to the LLM API. No PII.

---

## 14. Repository Structure

```
quant-ai/
├── pyproject.toml / uv.lock
├── config/
│   ├── universe.yaml            # assets, proxies, benchmarks
│   ├── series.yaml              # canonical series_id ↔ source mapping
│   ├── publication_lags.yaml
│   ├── alerts.yaml
│   └── models/*.yaml            # hyperparams, feature sets per model
├── atlas/
│   ├── connectors/              # fred.py, alfred.py, bls.py, bea.py, edgar.py, imf_sdmx.py,
│   │                            # oecd.py, ecb.py, bis.py, cftc.py, eia.py, kalshi.py, sharadar.py, ...
│   ├── pit/                     # bitemporal store, as_of(), vintage logic  ← most-tested module
│   ├── refdata/                 # security master, FIGI, corp actions, GICS history, futures roll
│   ├── quality/                 # pandera schemas, anomaly checks
│   ├── features/                # registry + macro/, equity/, cross_asset/, text/
│   ├── nlp/                     # chunking, embeddings, LLM extraction prompts (versioned)
│   ├── models/
│   │   ├── regime/              # dfm_nowcast.py, hmm.py, quadrant.py, recession_probit.py
│   │   ├── expected_returns/    # structural.py, lgbm_ranker.py, trend_carry.py, llm_overlay.py
│   │   ├── risk/                # covariance.py, factor_model.py, stress.py
│   │   ├── portfolio/           # black_litterman.py, optimizer.py, hrp.py
│   │   ├── ensemble.py, calibration.py, conformal.py
│   │   └── scenario.py, analogs.py
│   ├── backtest/                # cv.py (purged/CPCV), metrics.py (DSR, PBO), engine.py, costs.py
│   ├── views/                   # View schema, stance logic, publisher, track record scorer
│   ├── agent/                   # tools.py, prompts/, report.py, grounding_guard.py, evals/
│   ├── api/                     # FastAPI app
│   ├── orchestration/           # Dagster definitions, sensors, schedules
│   └── cli.py
├── web/                         # Next.js dashboard
├── notebooks/                   # research only; yfinance allowed here only
├── tests/                       # unit, property, leakage, connector cassettes, llm evals
├── docs/                        # data_licenses.md, signal cards, model cards, runbooks
└── infra/                       # docker-compose.yml, terraform/, grafana dashboards
```

Every model and signal ships with a **card** (`docs/models/<name>.md`): purpose, economic rationale, data, training window, OOS metrics, known failure regimes, owner.

---

## 15. Delivery Plan & Acceptance Criteria

| Phase | Weeks | Scope | Exit criteria |
|---|---|---|---|
| **P0: Foundations** | 1–3 | Repo, CI, Docker, Postgres, MinIO, Dagster; `pit` module; FRED/ALFRED, BLS, Treasury, Ken French connectors; Pandera gates | `as_of()` leakage suite green; 200 US macro series with full vintage history; reproducible snapshot |
| **P1: Market data + refdata** | 4–6 | Sharadar/Norgate prices + SF1 fundamentals; security master; GICS history; sector ETF + reconstructed history to 1995; CFTC, EIA, CBOE | Survivorship test passes; sector returns reconcile to ETF returns within 5bp/month after 1999 |
| **P2: Regime engine** | 7–9 | DFM nowcast, quadrant, HMM, recession probit, change points; World Map page (Streamlit first) | Nowcast RMSE within 10% of GDPNow (2012–2025 real-time); regime balanced accuracy ≥ 65% on holdout |
| **P3: Signals + backtester** | 10–14 | Feature registry with ≥ 40 signals; CPCV, DSR, PBO; tearsheets; alphalens integration | Each signal has a tearsheet and card; ≥ 8 signals pass promotion gates |
| **P4: Expected returns + ensemble** | 15–19 | Structural, LGBM ranker, trend/carry; ensemble, isotonic, conformal; View publisher; risk model; BL optimizer | OOS (2021–2025 holdout, opened once) sector 1M rank IC ≥ 0.04, model portfolio IR ≥ 0.3 net |
| **P5: Global expansion** | 20–23 | IMF/OECD/ECB/BIS/Eurostat/BoJ connectors; countries, FX, rates, commodities, credit views | All §3.1 asset classes produce Views nightly |
| **P6: NLP + LLM agent** | 24–28 | CB text, transcripts, news pipelines; tools; chat; weekly report with red team; grounding guard; eval harness | Grounding 100% on golden set; eval score ≥ 85/100; weekly report auto-sent for 3 consecutive weeks |
| **P7: Product hardening** | 29–32 | Next.js dashboard, alerts, scenario lab, analogs, track record, observability, auth | All §2 user stories demoable; NFRs met; runbooks written |
| **P8: Live paper track** | 33+ | Frozen views scored daily; monthly model review | 12-month live track for G1–G5 evaluation |

**Definition of Done (any feature):** typed, tested (unit + leakage where relevant), documented card, Pandera contract, observability metrics, reproducible via snapshot.

---

## 16. Key Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Look-ahead / revision leakage | Fake alpha | Bitemporal store, ALFRED, leakage CI suite, locked holdout |
| Overfitting across many signals | Live underperformance | Trial counting in MLflow → DSR; PBO; economic-rationale gate; shrinkage-heavy ensemble |
| Regime shift breaks models | Drawdown | Regime-conditional weights; change-point triggered refits; live-vs-backtest IC drift alerts |
| LLM hallucinated numbers or overconfident narrative | User misled | Deterministic grounding guard; red-team pass; LLM weight capped at 10% and earned OOS |
| Vendor data errors / outages | Wrong views | Cross-source reconciliation; quarantine; stale flags; confidence penalty for stale inputs |
| Licensing violation | Legal | License register; derived-only display; yfinance ban in prod |
| Scope creep ("whole global economy") | Never ships | Phased plan: US sectors first (P0–P4), then global (P5) |
| Small sample for monthly sector models (~360 months × 11 sectors) | Noisy estimates | Pool across regions/industries (49 FF industries), predict ranks not levels, strong regularization, use weekly as auxiliary target |
| Cost overrun (data + LLM) | Budget | Tier A first; batch + caching; daily token budget |

---

## 17. Open Questions

1. **Budget for Tier B data?** Determines Sharadar+Norgate (~low hundreds $/mo) vs an EDGAR-only fundamentals path (free, about 3 extra weeks of cleaning work).
2. **Primary benchmark** for sector views: S&P 500 equal-sector, ACWI, or cap-weighted SPX?
3. **Single user or multi-user** in v1? This affects auth and licensing scope.
4. **Integration with the existing Kalshi bot:** share infrastructure (Postgres, connectors), or keep it fully isolated and consume its odds via API only? Recommendation: share the `connectors/kalshi.py` client package and keep the deployments separate.
5. **Execution in v3?** If yes, choose a broker (IBKR API) early so the instrument mapping aligns.

---

## Appendix A — Core FRED series list (v1 seed)

| Canonical ID | FRED ID | Description |
|---|---|---|
| US.GDP.REAL | GDPC1 | Real GDP (Q) |
| US.CPI.HEADLINE | CPIAUCSL | CPI-U SA |
| US.CPI.CORE | CPILFESL | Core CPI SA |
| US.PCE.CORE | PCEPILFE | Core PCE price index |
| US.LABOR.UNRATE | UNRATE | Unemployment rate |
| US.LABOR.NFP | PAYEMS | Nonfarm payrolls |
| US.LABOR.CLAIMS.INIT | ICSA | Initial claims |
| US.LABOR.CLAIMS.CONT | CCSA | Continuing claims |
| US.LABOR.JOLTS | JTSJOL | Job openings |
| US.LABOR.SAHM | SAHMREALTIME | Real-time Sahm rule |
| US.IP | INDPRO | Industrial production |
| US.RETAIL | RSAFS | Retail & food services sales |
| US.HOUSING.STARTS | HOUST | Housing starts |
| US.HOUSING.PERMITS | PERMIT | Building permits |
| US.SENTIMENT.UMICH | UMCSENT | UMich consumer sentiment |
| US.RATES.FF | DFF | Effective fed funds (daily) |
| US.RATES.SOFR | SOFR | SOFR |
| US.RATES.2Y | DGS2 | 2Y Treasury CMT |
| US.RATES.10Y | DGS10 | 10Y Treasury CMT |
| US.RATES.10Y.REAL | DFII10 | 10Y TIPS yield |
| US.RATES.BE10 | T10YIE | 10Y breakeven |
| US.RATES.5Y5Y | T5YIFR | 5Y5Y forward inflation |
| US.CURVE.10Y2Y | T10Y2Y | 10Y–2Y spread |
| US.CURVE.10Y3M | T10Y3M | 10Y–3M spread |
| US.CREDIT.HY.OAS | BAMLH0A0HYM2 | ICE BofA US HY OAS |
| US.CREDIT.IG.OAS | BAMLC0A0CM | ICE BofA US Corp OAS |
| US.FCI.NFCI | NFCI | Chicago Fed NFCI |
| US.FCI.STLFSI | STLFSI4 | St. Louis Fed stress index |
| US.FED.ASSETS | WALCL | Fed total assets |
| US.FED.RRP | RRPONTSYD | Overnight reverse repo |
| US.TGA | WTREGEN | Treasury General Account |
| US.MONEY.M2 | M2SL | M2 |
| US.USD.BROAD | DTWEXBGS | Broad trade-weighted USD |
| US.VOL.VIX | VIXCLS | VIX |
| CMDTY.OIL.WTI | DCOILWTICO | WTI spot |
| US.MORTGAGE.30Y | MORTGAGE30US | 30Y mortgage rate |

## Appendix B — Key references (methodology)
- López de Prado, *Advances in Financial Machine Learning* (2018): purged CV, CPCV, meta-labelling, feature importance
- Bailey & López de Prado (2014), "The Deflated Sharpe Ratio"; Bailey et al. (2017), "The Probability of Backtest Overfitting"
- Harvey, Liu & Zhu (2016), "…and the Cross-Section of Expected Returns" (t-stat ≥ 3 hurdle)
- Moskowitz, Ooi & Pedersen (2012), "Time Series Momentum"; Koijen, Moskowitz, Pedersen & Vrugt (2018), "Carry"
- Asness, Moskowitz & Pedersen (2013), "Value and Momentum Everywhere"
- Fama & French (2015), five-factor model; Novy-Marx (2013), gross profitability
- Black & Litterman (1992); Idzorek (2005); López de Prado (2016), HRP
- Bańbura, Giannone & Reichlin, nowcasting with dynamic factor models
- Adrian, Crump & Moench (2013), term-premium model
- Cohen, Malloy & Nguyen (2020), "Lazy Prices"
- Chen & Zimmermann (2022), "Open Source Cross-Sectional Asset Pricing"
- Ledoit & Wolf (2004), covariance shrinkage
- Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction"
