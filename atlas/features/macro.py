"""US macro feature set for the regime engine (PRD §6.1, §7.1).

Pillar features are trailing 10-year z-scores (min 3 years) so they are comparable and
can be averaged into composites. Recession-model inputs stay in natural units.
"""

from __future__ import annotations

from atlas.features.registry import FeatureSpec, Pillar
from atlas.features.transforms import Annualized, Diff, PctChange, RollingZ

Z = RollingZ(window=120, min_samples=36)

GROWTH = (
    FeatureSpec(
        "ip_yoy",
        "US.IP",
        Pillar.GROWTH,
        (PctChange(12), Z),
        description="Industrial production YoY",
    ),
    FeatureSpec(
        "payrolls_6m_ann",
        "US.LABOR.NFP",
        Pillar.GROWTH,
        (Annualized(6), Z),
        description="Payrolls 6m annualized",
    ),
    FeatureSpec(
        "unrate_12m_chg",
        "US.LABOR.UNRATE",
        Pillar.GROWTH,
        (Diff(12), Z),
        sign=-1,
        description="Unemployment rate 12m change",
    ),
    FeatureSpec(
        "claims_yoy",
        "US.LABOR.CLAIMS.INIT",
        Pillar.GROWTH,
        (PctChange(12), Z),
        sign=-1,
        description="Initial claims (monthly avg) YoY",
    ),
    FeatureSpec(
        "retail_yoy", "US.RETAIL", Pillar.GROWTH, (PctChange(12), Z), description="Retail sales YoY"
    ),
    FeatureSpec(
        "permits_yoy",
        "US.HOUSING.PERMITS",
        Pillar.GROWTH,
        (PctChange(12), Z),
        description="Building permits YoY",
    ),
    FeatureSpec(
        "sentiment", "US.SENTIMENT.UMICH", Pillar.GROWTH, (Z,), description="UMich sentiment level"
    ),
)

INFLATION = (
    FeatureSpec(
        "core_cpi_6m_ann",
        "US.CPI.CORE",
        Pillar.INFLATION,
        (Annualized(6), Z),
        description="Core CPI 6m annualized",
    ),
    FeatureSpec(
        "cpi_yoy",
        "US.CPI.HEADLINE",
        Pillar.INFLATION,
        (PctChange(12), Z),
        description="Headline CPI YoY",
    ),
    FeatureSpec(
        "core_pce_yoy",
        "US.PCE.CORE",
        Pillar.INFLATION,
        (PctChange(12), Z),
        description="Core PCE YoY",
    ),
    FeatureSpec(
        "infl_exp_5y5y",
        "US.RATES.5Y5Y",
        Pillar.INFLATION,
        (Z,),
        description="5y5y forward breakeven",
    ),
    FeatureSpec(
        "oil_yoy", "CMDTY.OIL.WTI", Pillar.INFLATION, (PctChange(12), Z), description="WTI YoY"
    ),
)

FINANCIAL = (
    FeatureSpec(
        "nfci",
        "US.FCI.NFCI",
        Pillar.FINANCIAL,
        (),
        description="Chicago Fed NFCI (already standardized)",
    ),
    FeatureSpec("vix", "US.VOL.VIX", Pillar.FINANCIAL, (Z,), description="VIX monthly avg"),
    FeatureSpec("hy_oas", "US.CREDIT.HY.OAS", Pillar.FINANCIAL, (Z,), description="HY OAS"),
    FeatureSpec(
        "stlfsi",
        "US.FCI.STLFSI",
        Pillar.FINANCIAL,
        (),
        description="St. Louis Fed stress (standardized)",
    ),
)

# For the recession probit, ``sign`` is the economically expected sign of the coefficient:
# an inverted curve, a triggered Sahm rule, rising unemployment and tight financial
# conditions all raise recession odds. Inputs fitted against their sign are dropped.
RECESSION = (
    FeatureSpec(
        "curve_10y3m",
        "US.CURVE.10Y3M",
        Pillar.RECESSION,
        (),
        sign=-1,
        description="10Y-3M spread, pp",
    ),
    FeatureSpec(
        "sahm",
        "US.LABOR.SAHM",
        Pillar.RECESSION,
        (),
        aggregate="last",
        description="Real-time Sahm rule, pp",
    ),
    FeatureSpec("nfci_level", "US.FCI.NFCI", Pillar.RECESSION, (), description="NFCI level"),
    FeatureSpec(
        "unrate_12m_chg_pp",
        "US.LABOR.UNRATE",
        Pillar.RECESSION,
        (Diff(12),),
        description="Unemployment 12m change, pp",
    ),
)

PILLAR_FEATURES = (*GROWTH, *INFLATION, *FINANCIAL)
