"""
Configuration settings for the Ayurvedic Clinical Bridge system.

Uses pydantic-settings (pydantic v2 compatible). Install with:
    pip install pydantic-settings
"""

import os
from pathlib import Path
from typing import Optional

try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    # Fallback for environments that still ship pydantic v1
    from pydantic import BaseSettings, Field  # type: ignore[no-redef]


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    # Application
    app_name: str = "Ayurvedic Clinical Bridge"
    app_version: str = "3.0.0"
    debug: bool = Field(default=False)
    testing: bool = Field(default=False)
    environment: str = Field(default="development")

    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=4)

    # Security
    secret_key: str = Field(default="change-me-in-production")
    algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=1440)

    # Database (optional)
    database_url: str = Field(default="sqlite:///./ayurvedic_bridge.db")

    # ChromaDB (optional)
    chromadb_host: str = Field(default="localhost")
    chromadb_port: int = Field(default=8001)

    # Redis (optional)
    redis_url: Optional[str] = Field(default=None)

    # Logging
    log_level: str = Field(default="INFO")
    log_file: str = Field(default="logs/app.log")
    enable_audit_logging: bool = Field(default=True)
    anonymize_logs: bool = Field(default=True)

    # ML
    model_cache_dir: str = Field(default="./models")
    data_cache_dir: str = Field(default="./data/cache")
    huggingface_api_token: Optional[str] = Field(default=None)

    # Performance
    max_workers: int = Field(default=4)
    batch_size: int = Field(default=32)
    max_sequence_length: int = Field(default=512)

    # Compliance
    data_retention_days: int = Field(default=90)

    # CORS
    allowed_origins: str = Field(
        default="http://localhost:5173,http://localhost:3000,http://localhost:8080"
    )

    model_config = {"env_file": ".env", "case_sensitive": False, "extra": "ignore"}

    def model_post_init(self, __context) -> None:  # pydantic v2 hook
        self._ensure_dirs()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_dirs()

    def _ensure_dirs(self):
        for d in [self.model_cache_dir, self.data_cache_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)


# Lazy singleton — only instantiated when imported
try:
    settings = Settings()
except Exception:
    # Minimal fallback so other modules can always import `settings`
    settings = Settings(secret_key="fallback-secret-for-dev")