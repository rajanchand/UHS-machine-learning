"""Async SQLAlchemy engine factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

if TYPE_CHECKING:
    from anomaly_detection.config import Settings


def create_engine(settings: Settings) -> AsyncEngine:
    """Create an async SQLAlchemy engine from application settings.

    Args:
        settings: Application settings containing the database URL.

    Returns:
        Configured async engine with connection pooling.
    """
    from sqlalchemy import event

    engine = create_async_engine(
        settings.database_url,
        echo=settings.log_level.upper() == "DEBUG",
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=300,
    )

    if "sqlite" in settings.database_url:
        @event.listens_for(engine.sync_engine, "connect")
        def register_sqlite_functions(dbapi_connection, connection_record):
            def date_trunc(field, dt_str):
                if not dt_str:
                    return dt_str
                # Truncate to minute 'YYYY-MM-DD HH:MM:00'
                # Replacing 'T' with ' ' for consistency if ISO formatted
                val = str(dt_str).replace("T", " ")
                return val[:16] + ":00"
            dbapi_connection.create_function("date_trunc", 2, date_trunc)

    return engine
