from __future__ import annotations

from atlas.signals.country import ALL_COUNTRIES, COUNTRIES, EXTRA_COUNTRIES, WORLD
from atlas.signals.rates import EM_AREAS, RATES_EM_SIGNALS
from atlas.signals.universes import UNIVERSES, get_universe


def test_every_universe_pins_components_it_has() -> None:
    for name, uni in UNIVERSES.items():
        assert uni.ensemble_components is not None, name
        assert set(uni.ensemble_components) <= set(uni.signals), name


def test_world_country_universe() -> None:
    assert len(WORLD) == 33
    assert set(COUNTRIES) <= set(WORLD)
    assert not set(EXTRA_COUNTRIES) & set(COUNTRIES)
    uni = get_universe("ctry_world")
    assert uni.asset_name("NZL") == "New Zealand"
    assert all(cur for _, cur in ALL_COUNTRIES.values())


def test_em_bond_universe() -> None:
    uni = get_universe("rates_em")
    assert len(EM_AREAS) == 9
    assert uni.asset_name("MEX") == "Mexico 10Y government bond"
    assert "cli_fall_6" not in RATES_EM_SIGNALS  # too few EM leading indicators
    assert uni.ensemble_components == ("term_spread", "yield_vs_60m")
