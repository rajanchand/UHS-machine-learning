import asyncio

from anomaly_detection.config import get_settings
from anomaly_detection.db.engine import create_engine
from anomaly_detection.db.models import Base


async def init_db() -> None:
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from anomaly_detection.authentication import hash_password
    from anomaly_detection.db.models import User

    settings = get_settings()
    print(f"Initializing database at: {settings.database_url}")
    engine = create_engine(settings)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Database tables created successfully!")

    # Seed the analyst user
    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        result = await session.execute(select(User).where(User.username == "analyst"))
        existing_user = result.scalar_one_or_none()
        if not existing_user:
            print("Seeding analyst user...")
            analyst = User(
                username="analyst",
                email="analyst@anomalyguard.local",
                password_hash=hash_password("password123"),
                full_name="Security Analyst",
            )
            session.add(analyst)
            await session.commit()
            print("Analyst user seeded successfully!")
        else:
            print("Analyst user already exists.")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(init_db())
