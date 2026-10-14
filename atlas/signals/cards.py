"""Signal cards (PRD §14): one Markdown page per signal, generated from a gate-suite run.

A card states what the signal is, why it should work, exactly what data it uses and when
that data is known, how it performed (net of costs), every configuration that was tried,
the promotion-gate verdict, and where it breaks. Cards are regenerated, never hand-edited,
so they cannot drift from the code and data.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from atlas.backtest import gates as g
from atlas.backtest.crosssection import SignalReport
from atlas.backtest.metrics import max_drawdown
from atlas.signals.industry import SIGNALS, SignalDef
from atlas.signals.research import GateSuite

RATIONALE = {
    "momentum": (
        "Investors under-react to industry news, so prices drift in the direction of recent "
        "performance; Moskowitz & Grinblatt find industry momentum explains much of "
        "stock-level momentum."
    ),
    "residual_momentum": (
        "Raw momentum loads on whatever factors did well recently and crashes when they "
        "reverse. Ranking on returns after removing factor exposure keeps the under-reaction "
        "signal and drops most of the crash risk."
    ),
    "reversal": (
        "At the stock level, last month's losers bounce back (liquidity provision). At the "
        "industry level the effect is weaker and usually flips to continuation."
    ),
    "low_volatility": (
        "Leverage-constrained investors bid up high-beta, high-volatility assets, leaving "
        "low-volatility assets with higher risk-adjusted returns."
    ),
    "long_term_reversal": (
        "Investors over-extrapolate multi-year performance; long-run losers become too cheap "
        "and mean-revert."
    ),
    "value": (
        "Cheap assets (high book-to-market) earn a premium, as compensation for distress risk "
        "or because investors over-extrapolate growth."
    ),
    "value_momentum": (
        "Value and momentum are negatively correlated, so a blend has a better risk-adjusted "
        "return than either alone."
    ),
}

TIMING = {
    "value": "Annual BE/ME for fiscal year y is used from the end of June y+1 (Fama-French).",
    "value_momentum": (
        "Momentum uses returns through month t; BE/ME follows the Fama-French June rule."
    ),
    "residual_momentum": (
        "Factor betas are estimated on the trailing window ending at month t; only returns "
        "through t enter."
    ),
}
DEFAULT_TIMING = "Uses returns through the end of month t; the forecast is month t+1's return."


def _f(v: float | None, spec: str = ".2f") -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    return format(v, spec)


def _gate_row(name: str, threshold: str, value: str, ok: bool) -> str:
    return f"| {name} | {threshold} | {value} | {'pass' if ok else 'fail'} |"


def _worst_drawdown(ls: pl.DataFrame) -> tuple[date | None, date | None, float]:
    net = ls.sort("date")["net"].to_numpy()
    if not net.size:
        return None, None, 0.0
    equity = np.cumprod(1 + net)
    peak = np.maximum.accumulate(equity)
    trough_i = int(np.argmin(equity / peak))
    peak_i = int(np.argmax(equity[: trough_i + 1]))
    dates = ls.sort("date")["date"].to_list()
    return dates[peak_i], dates[trough_i], max_drawdown(net)


def render_card(sig: SignalDef, suite: GateSuite, *, generated: date, cost_bps: float) -> str:
    rep: SignalReport = suite.reports[sig.name]
    gate = suite.gates[sig.name]
    s = rep.summary()
    ls = rep.long_short.sort("date")
    post2000 = ls.filter(pl.col("date") >= date(2000, 1, 1))["net"].to_numpy()
    sr_post = (
        float(post2000.mean() / post2000.std(ddof=1) * math.sqrt(12))
        if post2000.size > 2 and post2000.std(ddof=1) > 0
        else math.nan
    )
    peak, trough, mdd = _worst_drawdown(ls)
    worst12 = (
        ls.with_columns(
            pl.col("net").log1p().rolling_sum(12, min_samples=12).exp().sub(1).alias("r12")
        )
        .drop_nulls("r12")
        .sort("r12")
    )
    verdict = (
        "**Pass** — eligible for the ensemble"
        if gate.passed
        else (
            f"**Not promoted** — passes {sum(gate.checks.values())} of {len(gate.checks)} gates; "
            f"fails {', '.join(gate.failed)}"
        )
    )
    t_needed = g.T_PUBLISHED if sig.reference else g.T_NOVEL
    family = suite.family_reports.get(sig.family, [])

    lines = [
        f"# Signal card: `{sig.name}`",
        "",
        f"{sig.description}. Universe: `{suite.universe}`, {s['start']:%Y-%m} to {s['end']:%Y-%m} "
        f"({s['periods']} months), net of {cost_bps:g} bp per unit of turnover.",
        "",
        f"Verdict: {verdict}.",
        "",
        "## Why it should work",
        "",
        RATIONALE.get(sig.family, "No rationale recorded."),
        "",
        f"Reference: {sig.reference or 'none (novel signal: t-stat bar is 3.0)'}",
        "",
        "## Data and timing",
        "",
        f"- {TIMING.get(sig.family, DEFAULT_TIMING)}",
        "- Source: Ken French data library (industry portfolios, factors), latest vintage.",
        f"- Parameters: `{sig.params}`"
        + (f"; volatility target {sig.vol_target:.0%}" if sig.vol_target else ""),
        "",
        "## Results",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Mean rank IC | {_f(s['ic_mean'], '.3f')} |",
        f"| IC t-stat (Newey-West) | {_f(s['ic_tstat_nw'])} |",
        f"| IC information ratio (annual) | {_f(s['icir_annual'])} |",
        f"| IC hit rate | {_f(s['ic_hit_rate'], '.0%')} |",
        f"| Long-short return, annual, net | {_f(s['ls_return_annual_net'], '.1%')} |",
        f"| Long-short volatility, annual | {_f(s['ls_vol_annual'], '.1%')} |",
        f"| Sharpe, gross / net | {_f(s['sharpe_gross'])} / {_f(s['sharpe_net'])} |",
        f"| Sharpe since 2000, net | {_f(sr_post)} |",
        f"| Probabilistic Sharpe (vs 0) | {_f(s['psr'])} |",
        f"| Deflated Sharpe ({suite.n_trials} trials) | {_f(s['dsr'])} |",
        f"| Turnover per month | {_f(s['turnover_mean'])} |",
        f"| Break-even cost | {_f(s['breakeven_cost_bps'], '.0f')} bp |",
        f"| Quintile monotonicity | {_f(s['monotonicity'])} |",
        "",
        "IC decay (signal at t vs return k months later): "
        + ", ".join(f"{k}m {v:+.3f}" for k, v in rep.ic_decay.items()),
        "",
        "Annual return by quantile: "
        + ", ".join(f"Q{k} {v:+.1%}" for k, v in rep.quantile_mean_annual.items()),
        "",
        "## Promotion gates (PRD §8.4)",
        "",
        "| Gate | Threshold | Value | Result |",
        "|---|---|---|---|",
        _gate_row("IC t-stat", f"≥ {t_needed:g}", _f(gate.ic_tstat), gate.checks["ic_tstat"]),
        _gate_row("Deflated Sharpe", f"≥ {g.DSR_MIN:g}", _f(gate.dsr), gate.checks["dsr"]),
        _gate_row("Family PBO", f"< {g.PBO_MAX:.0%}", _f(gate.pbo, ".0%"), gate.checks["pbo"]),
        _gate_row(
            "Regimes with IC > 0",
            f"≥ {g.REGIMES_REQUIRED} of 4",
            f"{gate.regimes_positive} of {gate.regimes_total}",
            gate.checks["regimes"],
        ),
        _gate_row(
            "Ensemble ICIR contribution",
            "> 0",
            _f(gate.marginal_icir, "+.3f"),
            gate.checks["marginal"],
        ),
        _gate_row(
            "Economic rationale",
            "reference",
            "yes" if sig.reference else "no",
            gate.checks["rationale"],
        ),
        "",
        "IC by market regime: "
        + (", ".join(f"{k} {v:+.3f}" for k, v in sorted(rep.regime_ic.items())) or "n/a"),
        "",
        f"## Every configuration tried (`{sig.family}` family)",
        "",
        f"PBO across {len(family)} configurations: {_f(suite.pbo.get(sig.family), '.0%')} "
        f"over {suite.pbo_months.get(sig.family, 0)} common months.",
        "",
        "| Configuration | IC | t-stat | Sharpe net | DSR |",
        "|---|---|---|---|---|",
        *[
            f"| `{r.name}` | {r.ic_mean:.3f} | {_f(r.ic_tstat)} | {_f(r.sharpe_net)} "
            f"| {_f(r.dsr)} |"
            for r in sorted(family, key=lambda r: -r.sharpe_net)
        ],
        "",
        "## Where it breaks",
        "",
        f"- Worst drawdown: {mdd:.0%}"
        + (f", from {peak:%b %Y} to {trough:%b %Y}." if peak and trough else "."),
    ]
    if worst12.height:
        w = worst12.row(0, named=True)
        lines.append(f"- Worst 12 months: {w['r12']:.0%}, ending {w['date']:%b %Y}.")
    lines += [
        f"- Since 2000 the net Sharpe is {_f(sr_post)} versus {_f(s['sharpe_net'])} over the "
        "full sample.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "uv run atlas ingest kenfrench --all",
        f"uv run atlas research gates --universe {suite.universe}",
        f"uv run atlas research signal {sig.name} --universe {suite.universe}",
        "```",
        "",
        f"_Generated by `atlas research cards` on {generated.isoformat()}. Do not edit by hand._",
        "",
    ]
    return "\n".join(lines)


def write_cards(suite: GateSuite, out_dir: Path, *, generated: date, cost_bps: float) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    index = [
        f"# Signal cards — `{suite.universe}`",
        "",
        f"Ensemble ICIR {suite.ensemble_icir:.2f}; {suite.n_trials} configurations tried. "
        f"Generated {generated.isoformat()}.",
        "",
        "| Signal | Gates passed | IC t-stat | Sharpe net | DSR | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for name, sig in SIGNALS.items():
        if name not in suite.gates:
            continue  # inputs missing for this universe
        path = out_dir / f"{name}.md"
        path.write_text(render_card(sig, suite, generated=generated, cost_bps=cost_bps))
        paths.append(path)
        gate, rep = suite.gates[name], suite.reports[name]
        index.append(
            f"| [`{name}`]({name}.md) | {sum(gate.checks.values())}/{len(gate.checks)} | "
            f"{_f(gate.ic_tstat)} | {_f(rep.sharpe_net)} | {_f(gate.dsr)} | "
            f"{'pass' if gate.passed else 'fails ' + ', '.join(gate.failed)} |"
        )
    readme = out_dir / "README.md"
    readme.write_text("\n".join(index) + "\n")
    return [readme, *paths]
