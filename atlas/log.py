"""Structured logging setup."""

from __future__ import annotations

import logging

import structlog


def configure_logging(level: str = "INFO", *, json: bool = False) -> None:
    numeric = logging.getLevelNamesMapping()[level.upper()]
    logging.basicConfig(level=numeric, format="%(message)s")
    # httpx logs full request URLs at INFO, which would include API keys in query strings.
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        cache_logger_on_first_use=True,
    )
