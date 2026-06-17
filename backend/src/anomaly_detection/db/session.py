"""Async session factory and FastAPI dependency."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


def create_session_factory(engine: object) -> async_sessionmaker[AsyncSession]:
    """Create an async session factory bound to the given engine.

    Args:
        engine: The async SQLAlchemy engine.

    Returns:
        A session factory that produces AsyncSession instances.
    """
    from sqlalchemy.ext.asyncio import AsyncEngine

    assert isinstance(engine, AsyncEngine), "Expected an AsyncEngine instance"
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


async def get_session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a database session.

    Commits on success, rolls back on exception, always closes.

    Args:
        session_factory: The session factory to create sessions from.

    Yields:
        An async database session.
    """
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
