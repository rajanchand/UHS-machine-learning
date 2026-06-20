"""Typed application configuration loaded from environment variables."""

from __future__ import annotations

import contextlib
import json
import os
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
        "sqlite+aiosqlite:////tmp/anomaly_detection.db"
        if os.environ.get("VERCEL") == "1"
        else "postgresql+asyncpg://anomaly:changeme_in_production@localhost:5432/anomaly_detection"
    )
    database_url_sync: str = (
        "sqlite:////tmp/anomaly_detection.db"
        if os.environ.get("VERCEL") == "1"
        else "postgresql+psycopg2://anomaly:changeme_in_production@localhost:5432/anomaly_detection"
    )
    db_pool_size: int = 10
    db_max_overflow: int = 20

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:8000"]

    # JWT Authentication
    jwt_secret_key: str = Field(
        default="dev-jwt-secret-change-in-production-min-32-characters-long",
        min_length=32,
    )
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60
    jwt_refresh_expiry_days: int = 7

    # Security
    session_secret_key: str = Field(
        default="dev-only-secret-change-in-production-min-32-chars",
        min_length=32,
    )
    simulator_api_key: str = "simulator-secret"
    environment: Literal["development", "staging", "production"] = "development"
    alert_webhook_url: str | None = None

    # Rate limiting
    login_rate_limit: int = 10  # max login attempts per minute per IP
    batch_max_size: int = 500  # max flows per batch request

    # File paths
    model_registry_path: Path = Path("models")
    data_dir: Path = Path("data")
    upload_dir: Path = (
        Path("/tmp/uploads") if os.environ.get("VERCEL") == "1" else Path("uploads")
    )
    reports_dir: Path = (
        Path("/tmp/reports") if os.environ.get("VERCEL") == "1" else Path("reports")
    )
    frontend_dir: Path = Path("frontend")

    # ML defaults
    default_thresholds: dict[str, float] = {
        "isolation_forest": 0.39,
        "random_forest": 0.5,
        "xgboost": 0.5,
        "decision_tree": 0.5,
    }

    # Email (optional)
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    smtp_from: str = "noreply@anomalyguard.local"

    @field_validator("database_url", mode="before")
    @classmethod
    def validate_database_url(cls, v: Any) -> str:
        if isinstance(v, str):
            if v.startswith("postgres://"):
                v = v.replace("postgres://", "postgresql+asyncpg://", 1)
            elif v.startswith("postgresql://"):
                v = v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return str(v)

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
        for path in (
            self.model_registry_path,
            self.data_dir,
            self.upload_dir,
            self.reports_dir,
        ):
            with contextlib.suppress(OSError, PermissionError):
                path.mkdir(parents=True, exist_ok=True)
        return self


def get_settings() -> Settings:
    """Factory function to create Settings instance."""
    return Settings()
