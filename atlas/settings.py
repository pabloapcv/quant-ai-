"""Runtime settings, loaded from environment / .env (prefix ``ATLAS_``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ATLAS_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    config_dir: Path = Path("config")
    log_level: str = "INFO"
    log_json: bool = False

    fred_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("ATLAS_FRED_API_KEY", "FRED_API_KEY")
    )
    # FRED allows 120 req/min per key; keep headroom for ad-hoc use.
    fred_max_calls_per_minute: int = 100
    http_timeout_s: float = 30.0

    # Assistant (Phase 6). Credentials come from ANTHROPIC_API_KEY or an `ant auth login`
    # profile, resolved by the Anthropic SDK itself.
    assistant_model: str = "claude-opus-5-5"
    assistant_effort: str = "medium"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
