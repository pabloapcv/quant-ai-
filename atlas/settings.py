"""Runtime settings, loaded from environment / .env (prefix ``ATLAS_``)."""

from __future__ import annotations

import os
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

    # Assistant (Phase 6). The key may live in .env like the FRED key; when it is absent the
    # Anthropic SDK resolves credentials itself (environment, then an `ant auth login` profile).
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("ATLAS_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    )
    assistant_model: str = "claude-opus-5-5"
    assistant_effort: str = "medium"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # ATLAS_ENV_FILE lets tests (and deployments) point at a different file than ./.env
    return Settings(_env_file=os.environ.get("ATLAS_ENV_FILE", ".env"))
