# Signal cards — `rates_dev`

Ensemble ICIR 0.29; 16 configurations tried. Generated 2026-10-21.

| Signal | Gates passed | IC t-stat | Sharpe net | DSR | Verdict |
|---|---|---|---|---|---|
| [`term_spread`](term_spread.md) | 4/6 | 3.21 | 0.25 | 0.01 | fails dsr, pbo |
| [`mom_12_1`](mom_12_1.md) | 3/6 | 0.24 | -0.23 | 0.00 | fails ic_tstat, dsr, regimes |
| [`yield_vs_60m`](yield_vs_60m.md) | 2/6 | 0.49 | 0.09 | 0.00 | fails ic_tstat, dsr, pbo, regimes |
| [`cli_fall_6`](cli_fall_6.md) | 1/6 | 1.61 | -0.05 | 0.00 | fails ic_tstat, dsr, pbo, regimes, marginal |
| [`curve_flattening_12`](curve_flattening_12.md) | 1/6 | -2.60 | -0.56 | 0.00 | fails ic_tstat, dsr, pbo, regimes, marginal |
| [`short_rate_fall_12`](short_rate_fall_12.md) | 3/6 | 1.37 | 0.12 | 0.00 | fails ic_tstat, dsr, pbo |
| [`equity_weakness_12`](equity_weakness_12.md) | 1/6 | -1.16 | -0.69 | 0.00 | fails ic_tstat, dsr, pbo, regimes, marginal |
