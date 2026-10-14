"""Shared connector plumbing: rate-limited, retrying HTTP + the raw (bronze-input) zone."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import httpx
import polars as pl
import zstandard
from aiolimiter import AsyncLimiter
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from atlas.catalog import SeriesSpec


class ConnectorError(Exception):
    """A source returned an error we should not retry."""


class TransientHTTPError(ConnectorError):
    """Network failure, 429 or 5xx — safe to retry."""


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 6
    initial_wait_s: float = 1.0
    max_wait_s: float = 60.0


def redact(text: str, secrets: Sequence[str]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text


class HttpClient:
    """Async HTTP with a token-bucket limiter and exponential-backoff retries.

    Error messages never include request URLs: query strings carry API keys.
    """

    def __init__(
        self,
        *,
        base_url: str,
        max_calls: int,
        period_s: float,
        timeout_s: float = 30.0,
        retry: RetryPolicy | None = None,
        secrets: Sequence[str] = (),
        user_agent: str = "atlas-quant/0.1",
        accept: str = "application/json",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout_s,
            headers={"User-Agent": user_agent, "Accept": accept},
            transport=transport,
        )
        self._limiter = AsyncLimiter(max_calls, period_s)
        self._retry = retry or RetryPolicy()
        self._secrets = tuple(secrets)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def get(self, path: str, params: Mapping[str, str | int]) -> httpx.Response:
        """GET with retries. Returns any non-retryable response (including 4xx) to the caller."""
        response: httpx.Response | None = None
        retrying = AsyncRetrying(
            stop=stop_after_attempt(self._retry.max_attempts),
            wait=wait_random_exponential(
                multiplier=self._retry.initial_wait_s, max=self._retry.max_wait_s
            ),
            retry=retry_if_exception_type(TransientHTTPError),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                async with self._limiter:
                    try:
                        response = await self._client.get(path, params=dict(params))
                    except httpx.TransportError as exc:
                        msg = redact(f"{type(exc).__name__} calling {path}: {exc}", self._secrets)
                        raise TransientHTTPError(msg) from None
                if response.status_code == 429 or response.status_code >= 500:
                    raise TransientHTTPError(f"HTTP {response.status_code} calling {path}")
        assert response is not None
        return response


class RawStore:
    """Immutable landing zone for untouched payloads (PRD §5.3), so every parse is re-runnable.

    ``{root}/raw/{source}/{yyyy}/{mm}/{dd}/{run_id}/{sha16}.json.zst``
    """

    def __init__(self, data_dir: Path) -> None:
        self.root = data_dir / "raw"
        self._compressor = zstandard.ZstdCompressor(level=10)

    def put(
        self,
        *,
        source: str,
        run_id: str,
        request: Mapping[str, Any],
        body: bytes,
        fetched_at: datetime | None = None,
    ) -> Path:
        fetched = fetched_at or datetime.now(UTC)
        digest = hashlib.sha256(body).hexdigest()
        envelope = {
            "source": source,
            "run_id": run_id,
            "fetched_at": fetched.isoformat(),
            "request": dict(request),  # caller must pass params WITHOUT credentials
            "body_sha256": digest,
            "body": body.decode("utf-8"),
        }
        path = self.root / source / f"{fetched:%Y/%m/%d}" / run_id / f"{digest[:16]}.json.zst"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self._compressor.compress(json.dumps(envelope).encode()))
        return path

    @staticmethod
    def read(path: Path) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(
            zstandard.ZstdDecompressor().decompress(path.read_bytes())
        )
        return data


class Connector(Protocol):
    """Contract every data source implements (PRD §5.3)."""

    source: str

    async def fetch(self, spec: SeriesSpec, *, vintages: bool = True) -> pl.DataFrame:
        """Return rows in the canonical silver observation schema."""
        ...

    async def aclose(self) -> None: ...
