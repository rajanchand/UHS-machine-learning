"""Alembic environment configuration for database migrations."""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from anomaly_detection.db.models import Base

# Alembic Config object
config = context.config

# Set up logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set the SQLAlchemy URL from environment
database_url = os.environ.get(
    "DATABASE_URL_SYNC",
    "postgresql+psycopg2://anomaly:changeme_in_production@localhost:5432/anomaly_detection",
)
config.set_main_option("sqlalchemy.url", database_url)

# Target metadata for autogenerate
target_metadata = Base.metadata


def include_name(name: str | None, type_: str, parent_names: dict[str, str | None]) -> bool:
    """Filter out TimescaleDB internal indexes from autogenerate."""
    if type_ == "index" and name is not None:
        # TimescaleDB creates internal indexes that we don't want Alembic to manage
        if name.startswith("_timescaledb") or name.endswith("_time_idx"):
            return False
    return True


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode — generates SQL without a DB connection."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_name=include_name,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode — connects to the database."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_name=include_name,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
