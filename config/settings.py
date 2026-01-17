"""
Configuration settings for the Ayurvedic Clinical Bridge system.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    app_name: str = "Ayurvedic Clinical Bridge"
    app_version: str = "0.1.0"
    debug: bool = Field(default=False, env="DEBUG")
    testing: bool = Field(default=False, env="TESTING")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8080, env="API_PORT")
    api_workers: int = Field(default=4, env="API_WORKERS")
    
    # Security
    secret_key: str = Field(env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Database Configuration
    neo4j_uri: str = Field(env="NEO4J_URI")
    neo4j_user: str = Field(env="NEO4J_USER")
    neo4j_password: str = Field(env="NEO4J_PASSWORD")
    
    # ChromaDB Configuration
    chromadb_host: str = Field(default="localhost", env="CHROMADB_HOST")
    chromadb_port: int = Field(default=8000, env="CHROMADB_PORT")
    
    # Redis Configuration (optional)
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/app.log", env="LOG_FILE")
    enable_audit_logging: bool = Field(default=True, env="ENABLE_AUDIT_LOGGING")
    anonymize_logs: bool = Field(default=True, env="ANONYMIZE_LOGS")
    
    # Model Configuration
    model_cache_dir: str = Field(default="./models", env="MODEL_CACHE_DIR")
    huggingface_cache_dir: str = Field(default="./cache/huggingface", env="HUGGINGFACE_CACHE_DIR")
    datasets_cache_dir: str = Field(default="./cache/datasets", env="DATASETS_CACHE_DIR")
    huggingface_api_token: Optional[str] = Field(default=None, env="HUGGINGFACE_API_TOKEN")
    
    # Performance
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    batch_size: int = Field(default=32, env="BATCH_SIZE")
    max_sequence_length: int = Field(default=512, env="MAX_SEQUENCE_LENGTH")
    
    # Data and Compliance
    data_retention_days: int = Field(default=90, env="DATA_RETENTION_DAYS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create necessary directories
        Path(self.model_cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.huggingface_cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.datasets_cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()