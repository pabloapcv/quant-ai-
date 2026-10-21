# Atlas operations runbook

Everything runs from the repo root (`/Users/pablopena/quant ai`) with `.venv/bin/atlas`
(or `uv run atlas`). Secrets live only in the gitignored `.env`: `ATLAS_FRED_API_KEY`,
`ANTHROPIC_API_KEY`, and optionally `ATLAS_SLACK_WEBHOOK_URL` and `ATLAS_API_TOKEN`.

## Scheduled jobs (launchd, this Mac)

| Job | Label | When | Log |
|---|---|---|---|
| Daily pipeline | `com.atlas.pipeline-daily` | Mon–Fri 06:30 local | `data/runs/launchd.log`, one JSON per run in `data/runs/` |
| Weekly note | `com.atlas.report-weekly` | Mon 07:00 local | `data/reports/launchd.log`, note in `data/reports/<date>/` |

Plists are in `~/Library/LaunchAgents/`. A Mac that is asleep at the scheduled time runs the
job at the next wake.

```bash
launchctl print gui/$(id -u)/com.atlas.pipeline-daily | grep -E "state|last exit"
```

```bash
launchctl kickstart gui/$(id -u)/com.atlas.pipeline-daily
```

```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.atlas.pipeline-daily.plist
```

Re-enable with `launchctl bootstrap gui/$(id -u) <plist>`.

## Release-day ingest (optional)

`atlas ingest releases` ingests the day's timed FRED releases (CPI, payrolls, retail sales,
claims, ...) a minute after each Eastern-time release and polls for up to 30 minutes, then
exits; `--dry-run` prints the plan. Results go to `data/runs/releases-<date>.json`. A launchd
template that starts it on weekdays at 06:45 local is in `deploy/launchd/`; install it with:

```bash
cp "deploy/launchd/com.atlas.ingest-releases.plist" ~/Library/LaunchAgents/ && launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.atlas.ingest-releases.plist
```

If the Mac's local 06:45 is after 08:30 ET (US west coast), the first slot runs as a
catch-up; the daily pipeline still picks everything up the next morning.

## Daily pipeline: reading a run

`atlas pipeline daily` prints one line per step; the same is in the latest `data/runs/*.json`
(`ok`, `steps`, `stale`, `snapshot_id`, `track`, `alerts`, `ic_monitor`, `feature_psi`). Every
step is isolated: one failure is recorded and the rest still run, and the `data_quality`
alert reports the failed steps.

| Step | Failure means | What to do |
|---|---|---|
| `ingest:fred` | Key missing/invalid, FRED down, or rate limit | Check `ATLAS_FRED_API_KEY`; rerun later. Incremental fetches resume from the last release minus 3 days, so a missed day self-heals. |
| `ingest:kenfrench`, `ingest:ecb/oecd/bis/imf` | Source outage or format change | Rerun the single source (`atlas ingest kenfrench`, `atlas ingest sdmx --agency <agency>`). These sources only publish current values: a missed day is a day of point-in-time history lost, not recoverable later. |
| `ingest:fed_text` | federalreserve.gov unreachable or page layout changed | Rerun `atlas text sync`; a parse failure lists the page. Stored documents are never refetched or rewritten. |
| `ingest:ecb_text` | ecb.europa.eu unreachable or page layout changed | The daily sync covers this and last year; `atlas text sync --bank ecb --all-years` refetches anything missing since 1998 (about 25 minutes at the ECB's crawl delay). |
| `text:tone` | Budget reached, or a reply that did not parse | `atlas text tone` retries statements without a reading. "skipped" means no Anthropic credentials. |
| `quality:staleness` | Never fails; lists stale series | `US.SENTIMENT.UMICH` is expected (FRED's licence delay). Anything else stale for days: check the source. |
| `snapshot` | Disk or permission problem | Free disk; snapshots pin data + config + code SHA. |
| `views:<universe>:3M` | Panel could not be built (missing data) | Usually follows an ingest failure. "frozen earlier; today's recompute differs (kept first)" is **not** an error: views are write-once, revised data never rewrites a published month. |
| `track:score` | Corrupt view file | Inspect `data/gold/views/<universe>/<h>M/`; never edit published files. |
| `quality:drift` | Macro data missing | PSI is a diagnostic only (see `atlas/quality/drift.py` for why it is not an alert). |
| `alerts` | Bad `config/alerts.yaml` | Validate by running `atlas alerts check`. |

Series that fail during ingest get one more try two minutes later (step `ingest:retry`);
a step whose failures all recover is marked ok with "recovered on retry", so a transient
5xx from a source no longer fails the run. Rerun without network:
`atlas pipeline daily --skip-ingest`.

## Alerts

Rules: `config/alerts.yaml`. Log: `data/alerts/<date>.jsonl` (every alert, with status `sent`,
`cooldown` or `rate_limited`). State: `data/alerts/state.json` (previous quadrant
probabilities, keys sent in the cooldown window); deleting it resets cooldowns and makes the
next run skip regime-crossing checks once.

- Turn on phone push: set `ntfy_topic_url` (use a hard-to-guess topic; ntfy topics are public).
- Turn on Slack: put the incoming-webhook URL in `.env` as `ATLAS_SLACK_WEBHOOK_URL`.
- `live_ic_drift` needs 6 scored live months per universe; monthly IC is noisy (s.d. ~0.25–0.4),
  so it only catches a severe breakdown. Read the track record, not just the alert.

## Data recovery and reproduction

- Raw payloads are archived in `data/raw/`; the store (`data/silver/`) is append-only Parquet.
- `atlas snapshot verify <snapshot_id>` checks every pinned file still exists unchanged.
- `atlas reproduce <view_id>` rebuilds a published view from its snapshot and reports
  how many views match exactly (views frozen before 2026-10-19 carry no snapshot id).
- Re-ingest one FRED series with full vintages: `atlas ingest fred <SERIES.ID>`.
- Point-in-time questions: `atlas pit as-of <SERIES.ID> --at YYYY-MM-DD`,
  `atlas pit history <SERIES.ID> <event_date>`.
- New catalog series: add a line to `config/series.yaml`, then `atlas ingest fred <ID>`.

## Assistant, report and evals

- Weekly note failure: run `atlas report weekly --dry-run` (facts only, no Claude). Missing
  Anthropic credentials: set `ANTHROPIC_API_KEY` or run `ant auth login`.
- Before changing a prompt, the model or a tool's output: run `atlas eval run` (all 160 cases,
  real tokens) or a slice (`--category scenario`, `--ids ...`). The run exits 1 when the score
  falls more than 3 points below `data/evals/baseline.json`; set a new baseline with
  `--set-baseline` only after reviewing the run.
- Token spend: `atlas usage` (per day, weighted tokens, cache hit rate, by purpose). When the
  daily budget is reached every Claude call (ask, chat, report, eval) stops with a budget
  error until 00:00 UTC; raise `ATLAS_LLM_DAILY_TOKEN_BUDGET` in `.env` if that is intended.
  A full 160-case eval (assistant + judge) needs a few million weighted tokens, above the
  default 3M budget: raise it for that run.
- A grounding miss in an answer (`[grounding] not traceable ...`) is reported, never hidden.

## Monitoring

`atlas metrics` (or `GET /metrics` on the API, bearer token if set) prints Prometheus text.
Useful alarms: `atlas_pipeline_last_run_timestamp_seconds` older than 26 hours on a weekday,
`atlas_pipeline_last_run_ok == 0`, `atlas_source_data_age_seconds{source="fred"}` over 3 days,
`atlas_llm_weighted_tokens_today` near `atlas_llm_daily_budget_tokens`.

## API

`atlas api serve` binds `127.0.0.1:8710` (docs at `/docs`). Binding any other host requires
`ATLAS_API_TOKEN`; clients then send `Authorization: Bearer <token>`. The tool cache refreshes
hourly, so a morning pipeline run shows up without a restart.

## Champion and challenger

Challenger views (`gold/challengers/<version>/`) are frozen every day next to the champion's
and never published to the assistant, alerts or reports. Promote a challenger only when, on
the same scored live months (at least 12), its mean rank IC beats the champion's with a
paired t-stat of 2 or more; do it as a new model version (a code change to
`atlas/views/models.py`), never by editing frozen files. The first challenger months
(2026-08/09) were frozen on 2026-10-21, a few days after the champion's, so they saw slightly
more data; compare from 2026-10 onward for a strict test.

## Research discipline

- New signals, components or settings are researched with `--exclude-holdout`; the
  2021-2025 holdout is opened once per family with `atlas research holdout open --clean
  --reason "..."`, and `atlas research holdout status` lists every opening. A reopening needs
  `--allow-reopen` and is logged as one.

## Monthly

- Check `atlas pipeline track` (live paper track) against the walk-forward figures (`atlas views evaluate`).
- Review `atlas research gates --universe ind12` after data revisions; gate results are
  recorded with the number of trials (multiple-testing charge).
- `git log` / `git status`: the scheduled jobs run the checked-out code.
