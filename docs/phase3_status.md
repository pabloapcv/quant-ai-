# Phase 3 (signals + backtester): exit criteria

Status as of 2026-10-20. Research window: data before the locked 2021-2025 holdout
(`atlas research gates --exclude-holdout`); the holdout was opened once for the signal
library (`signal_library_v2`, clean for the 16 signals added that day).

| Criterion (PRD §15) | Status |
|---|---|
| Feature registry with at least 40 signals | **Met**: 33 signal definitions, 49 signal x asset-class combinations across 6 universes (industries 18, countries 17-18, FX 6, government bonds 7), each with 1-12 configurations logged as trials |
| CPCV, deflated Sharpe, PBO | **Met**: purged/embargoed CV, DSR with every logged configuration as a trial, CSCV PBO per family |
| Every signal has a tearsheet and a card | **Met**: 84 cards in `docs/signals/<universe>/`, each with an SVG tearsheet (growth of $1 long-short net, 24-month rolling IC), IC decay, quantile returns, regime ICs, family PBO table |
| Alphalens integration | **Met**: `tests/test_alphalens.py` checks our rank IC against `alphalens-reloaded` (equal to 1e-6) |
| At least 8 signals pass all promotion gates | **Not met**: 0 of 84 pass all six. The deflated Sharpe gate is binding: 14 signal-universe pairs pass the other five (IC t-stat, PBO, regimes, marginal ICIR, rationale), led by 12-1 momentum on 49 industries (t 8.6), 12-7 momentum (t 6.8), the 52-week high (t 6.7) and leading-indicator change on developed markets |

Why DSR binds: it compares each Sharpe ratio with the expected maximum of N unskilled
trials, with N the raw count of configurations logged (57 for industries) and the variance
taken across all of them, including sign-flipped ideas such as reversal. A clustered
"effective number of trials" (López de Prado 2019) would lower the bar, but changing the
gate after seeing that nothing passes would be moving the goalposts, so it is unchanged.

Holdout (2021-2025, clean for the new signals): of the 10 new signal-universe pairs with a
research IC t-stat of 2 or more, 6 kept a positive IC and 2 stayed significant: the 52-week
high (t 2.09) and same-month seasonality (t 2.82) on developed-market equities. The strong
industry results did not hold (e.g. seasonality on 49 industries: t 7.9 in research, -0.3 in
the holdout), consistent with the ensemble's weak 2021-2025 record.

Ensembles are unchanged: each universe's ensemble components are pinned, so the new
signals are research inputs only until a new model version adopts them.

## Commodities (added 2026-10-23)

17 commodities from IMF PCPS (monthly-average USD spot prices, 1992 onwards). Research
window only (`--exclude-holdout`): momentum failed (12-1 t -0.22, 6-1 t -1.92; averaged spot
prices, unlike futures returns), 5-year reversal passed every gate but DSR (t 2.89, positive
in all four regimes), low volatility t 1.60. The ensemble was pinned to `ltr_60_12` +
`low_vol_36` on that evidence, then the holdout was opened once (clean): ensemble 3-month IC
+0.091, t 1.73, hit rate 60% over 57 months; `ltr_60_12` alone +0.095 (t 2.62). The best
holdout result of any universe, but below t 2, so reported as promising, not validated.
