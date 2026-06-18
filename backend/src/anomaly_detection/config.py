"""Typed application configuration loaded from environment variables."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator
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
    db_pool_size: int = 10
    db_max_overflow: int = 20

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    cors_origins: list[str] = ["http://localhost:5173"]

    # Security — SESSION_SECRET_KEY must be set in production
    session_secret_key: str = Field(
        default="dev-only-secret-change-in-production-min-32-chars",
        min_length=32,
    )
    simulator_api_key: str = "simulator-secret"
    environment: Literal["development", "staging", "production"] = "development"

    # Rate limiting
    login_rate_limit: int = 10  # max login attempts per minute per IP
    batch_max_size: int = 500   # max flows per batch request

    # Model registry and data paths
    model_registry_path: Path = Path("/app/models")
    data_dir: Path = Path("/app/data")

    # Evaluated thresholds at ~1% FPR (fallback if not in metrics.json)
    default_thresholds: dict[str, float] = {
        "autoencoder": 0.0035,
        "isolation_forest": 0.39,
        "halfspace_trees": 0.976,
        "lightgbm_benchmark": 0.56,
    }

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> list[str]:
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
    def ensure_directories_exist(self) -> Settings:
        for path in (self.model_registry_path, self.data_dir):
            with contextlib.suppress(OSError, PermissionError):
                path.mkdir(parents=True, exist_ok=True)
        return self


def get_settings() -> Settings:
    return Settings()
