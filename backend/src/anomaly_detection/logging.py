"""Structured logging configuration using structlog."""

from __future__ import annotations

import logging
import sys

import structlog


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structured logging for the application.

    Uses structlog with JSON output in production and pretty console output
    in development (determined by log level).
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    is_debug = numeric_level <= logging.DEBUG

    # Configure structlog processors
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if is_debug:
        # Human-readable console output for development
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer(colors=True)
    else:
        # JSON output for production
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure stdlib logging to use structlog formatting
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(numeric_level)

    # Quiet down noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.INFO if is_debug else logging.WARNING
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a bound structured logger for a module."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]
