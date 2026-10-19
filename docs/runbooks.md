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
| `text:tone` | Budget reached, or a reply that did not parse | `atlas text tone` retries statements without a reading. "skipped" means no Anthropic credentials. |
| `quality:staleness` | Never fails; lists stale series | `US.SENTIMENT.UMICH` is expected (FRED's licence delay). Anything else stale for days: check the source. |
| `snapshot` | Disk or permission problem | Free disk; snapshots pin data + config + code SHA. |
| `views:<universe>:3M` | Panel could not be built (missing data) | Usually follows an ingest failure. "frozen earlier; today's recompute differs (kept first)" is **not** an error: views are write-once, revised data never rewrites a published month. |
| `track:score` | Corrupt view file | Inspect `data/gold/views/<universe>/<h>M/`; never edit published files. |
| `quality:drift` | Macro data missing | PSI is a diagnostic only (see `atlas/quality/drift.py` for why it is not an alert). |
| `alerts` | Bad `config/alerts.yaml` | Validate by running `atlas alerts check`. |

Rerun without network: `atlas pipeline daily --skip-ingest`.

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

## API

`atlas api serve` binds `127.0.0.1:8710` (docs at `/docs`). Binding any other host requires
`ATLAS_API_TOKEN`; clients then send `Authorization: Bearer <token>`. The tool cache refreshes
hourly, so a morning pipeline run shows up without a restart.

## Monthly

- Check `atlas pipeline track` (live paper track) against the walk-forward figures (`atlas views evaluate`).
- Review `atlas research gates --universe ind12` after data revisions; gate results are
  recorded with the number of trials (multiple-testing charge).
- `git log` / `git status`: the scheduled jobs run the checked-out code.
