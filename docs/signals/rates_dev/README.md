# Signal cards — `rates_dev`

Ensemble ICIR 0.26; 11 configurations tried. Generated 2026-10-19.

| Signal | Gates passed | IC t-stat | Sharpe net | DSR | Verdict |
|---|---|---|---|---|---|
| [`term_spread`](term_spread.md) | 4/6 | 2.86 | 0.19 | 0.01 | fails dsr, pbo |
| [`mom_12_1`](mom_12_1.md) | 3/6 | -0.30 | -0.28 | 0.00 | fails ic_tstat, dsr, regimes |
| [`yield_vs_60m`](yield_vs_60m.md) | 2/6 | 0.65 | 0.12 | 0.00 | fails ic_tstat, dsr, pbo, regimes |
| [`cli_fall_6`](cli_fall_6.md) | 1/6 | 1.14 | -0.23 | 0.00 | fails ic_tstat, dsr, pbo, regimes, marginal |
