"""Research universes: how to build each panel and which signals apply to it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import polars as pl

from atlas.pit.store import PITStore
from atlas.signals.country import (
    COUNTRY_FAMILIES,
    COUNTRY_NAMES,
    COUNTRY_SIGNALS,
    DEVELOPED,
    country_panel,
)
from atlas.signals.fx import FX_FAMILIES, FX_NAMES, FX_SIGNALS, fx_panel
from atlas.signals.industry import FAMILIES, SIGNALS, SignalDef, research_panel
from atlas.signals.rates import RATES_FAMILIES, RATES_NAMES, RATES_SIGNALS, rates_panel

INDUSTRY_COMPONENTS = (
    "mom_12_1",
    "resmom_ff3_36",
    "value_log_bm",
    "low_vol_36",
    "ltr_60_12",
    "reversal_1m",
)

INDUSTRY12_NAMES = {  # keys are catalog asset ids (upper-case Ken French names)
    "NODUR": "Consumer non-durables",
    "DURBL": "Consumer durables",
    "MANUF": "Manufacturing",
    "ENRGY": "Energy",
    "CHEMS": "Chemicals",
    "BUSEQ": "Business equipment (tech)",
    "TELCM": "Telecom",
    "UTILS": "Utilities",
    "SHOPS": "Retail and wholesale",
    "HLTH": "Healthcare",
    "MONEY": "Finance",
    "OTHER": "Other",
}


@dataclass(frozen=True)
class Universe:
    name: str
    description: str
    build_panel: Callable[[PITStore], pl.DataFrame]
    signals: dict[str, SignalDef]
    families: dict[str, list[SignalDef]]
    asset_names: dict[str, str]
    asset_class: str
    asset_id_prefix: str
    # ensemble components; None = every distinct ranking in ``signals``
    ensemble_components: tuple[str, ...] | None = None

    def asset_name(self, asset: str) -> str:
        return self.asset_names.get(asset, asset)


UNIVERSES: dict[str, Universe] = {
    "ind12": Universe(
        "ind12",
        "12 US industries (Ken French)",
        lambda s: research_panel(s, "ind12"),
        SIGNALS,
        FAMILIES,
        INDUSTRY12_NAMES,
        "equity_industry",
        "INDUSTRY.IND12",
        INDUSTRY_COMPONENTS,
    ),
    "ind49": Universe(
        "ind49",
        "49 US industries (Ken French)",
        lambda s: research_panel(s, "ind49"),
        SIGNALS,
        FAMILIES,
        {},
        "equity_industry",
        "INDUSTRY.IND49",
        INDUSTRY_COMPONENTS,
    ),
    "ctry_usd": Universe(
        "ctry_usd",
        "14 country equity markets, USD returns (OECD, ECB FX)",
        lambda s: country_panel(s, usd=True),
        COUNTRY_SIGNALS,
        COUNTRY_FAMILIES,
        COUNTRY_NAMES,
        "equity_country",
        "COUNTRY",
    ),
    "ctry_dev_lcl": Universe(
        "ctry_dev_lcl",
        "9 developed equity markets, local-currency returns (OECD)",
        lambda s: country_panel(s, usd=False, countries=DEVELOPED),
        COUNTRY_SIGNALS,
        COUNTRY_FAMILIES,
        COUNTRY_NAMES,
        "equity_country",
        "COUNTRY",
    ),
    "fx_g10": Universe(
        "fx_g10",
        "8 G10 currencies vs the US dollar: spot plus carry (ECB, OECD rates)",
        fx_panel,
        FX_SIGNALS,
        FX_FAMILIES,
        FX_NAMES,
        "fx",
        "FX",
    ),
    "rates_dev": Universe(
        "rates_dev",
        "9 developed-market 10-year government bonds, hedged, excess over cash (OECD)",
        rates_panel,
        RATES_SIGNALS,
        RATES_FAMILIES,
        RATES_NAMES,
        "rates",
        "RATES",
    ),
}


def get_universe(name: str) -> Universe:
    try:
        return UNIVERSES[name]
    except KeyError:
        raise ValueError(f"unknown universe {name!r}: {', '.join(UNIVERSES)}") from None
