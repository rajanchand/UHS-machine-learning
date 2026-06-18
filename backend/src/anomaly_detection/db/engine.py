"""Async SQLAlchemy engine factory."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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

    is_sqlite = "sqlite" in settings.database_url
    extra = (
        {}
        if is_sqlite
        else {
            "pool_size": settings.db_pool_size,
            "max_overflow": settings.db_max_overflow,
            "pool_pre_ping": True,
            "pool_recycle": 300,
        }
    )
    engine = create_async_engine(
        settings.database_url,
        echo=settings.log_level.upper() == "DEBUG",
        **extra,
    )

    if "sqlite" in settings.database_url:
        @event.listens_for(engine.sync_engine, "connect")
        def register_sqlite_functions(dbapi_connection: Any, connection_record: Any) -> None:
            def date_trunc(field: Any, dt_str: Any) -> Any:
                if not dt_str:
                    return dt_str
                # Truncate to minute 'YYYY-MM-DD HH:MM:00'
                # Replacing 'T' with ' ' for consistency if ISO formatted
                val = str(dt_str).replace("T", " ")
                return val[:16] + ":00"
            dbapi_connection.create_function("date_trunc", 2, date_trunc)

    return engine
