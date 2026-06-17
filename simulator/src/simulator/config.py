"""Simulator configuration loaded from environment."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class SimulatorSettings(BaseSettings):
    """Simulator service settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    simulator_api_url: str = "http://api:8000"
    simulator_rate_per_sec: float = 10.0
    simulator_data_path: str = "/app/data/fixtures/cicids2017_sample.csv"
    log_level: str = "INFO"


def get_simulator_settings() -> SimulatorSettings:
    """Factory for simulator settings."""
    return SimulatorSettings()
