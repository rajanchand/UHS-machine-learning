"""Typed application configuration loaded from environment variables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database
    database_url: str = (
        "postgresql+asyncpg://anomaly:changeme_in_production@localhost:5432/anomaly_detection"
    )
    database_url_sync: str = (
        "postgresql+psycopg2://anomaly:changeme_in_production@localhost:5432/anomaly_detection"
    )

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    cors_origins: list[str] = ["http://localhost:5173"]

    # Model registry — overridable via MODEL_REGISTRY_PATH env var
    model_registry_path: Path = Path("/app/models")

    # Data directory — overridable via DATA_DIR env var
    data_dir: Path = Path("/app/data")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> list[str]:
        """Parse CORS origins from JSON string or list."""
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except json.JSONDecodeError:
                return [s.strip() for s in v.split(",")]
        if isinstance(v, list):
            return [str(item) for item in v]
        return [str(v)]

    @model_validator(mode="after")
    def ensure_directories_exist(self) -> "Settings":
        """Create model registry and data directories if they don't exist.
        Silently skips on read-only file systems (e.g. inside Docker with
        externally mounted volumes, or during unit tests with temp dirs).
        """
        for path in (self.model_registry_path, self.data_dir):
            try:
                path.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError):
                # Directory either already exists or is on a read-only FS;
                # the application will fail later only if it actually needs
                # to write there.
                pass
        return self


def get_settings() -> Settings:
    """Factory function for settings — enables dependency injection and testing."""
    return Settings()
