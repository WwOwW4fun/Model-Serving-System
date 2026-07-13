"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import torch
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated settings for the model-serving API."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )

    app_name: str = "ML Model Serving API"
    app_version: str = "1.0.0"
    environment: str = "development"

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = Field(default=1, ge=1)
    reload: bool = False

    model_name: str = "resnet18"
    model_path: Optional[str] = None
    device: str = "cpu"
    batch_size: int = Field(default=1, ge=1)
    warmup_iterations: int = Field(default=10, ge=0)

    max_image_size_mb: int = Field(default=10, gt=0)
    allowed_image_types: List[str] = Field(
        default_factory=lambda: ["image/jpeg", "image/png", "image/jpg"]
    )
    default_top_k: int = Field(default=5, ge=1)
    confidence_threshold: float = 0.0

    log_level: str = "INFO"
    log_format: str = "json"

    enable_metrics: bool = True
    metrics_port: int = 8000

    enable_cache: bool = False
    cache_size: int = Field(default=1000, ge=1)
    cache_ttl_seconds: int = Field(default=300, ge=0)

    enable_rate_limit: bool = False
    rate_limit_requests: int = Field(default=100, ge=1)
    rate_limit_window_seconds: int = Field(default=60, ge=1)

    enable_cors: bool = True
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    health_check_interval_seconds: int = Field(default=30, ge=1)

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, value: str) -> str:
        allowed = {"development", "staging", "production"}
        normalized = value.lower()
        if normalized not in allowed:
            raise ValueError(f"environment must be one of: {sorted(allowed)}")
        return normalized

    @field_validator("port", "metrics_port")
    @classmethod
    def validate_port(cls, value: int) -> int:
        if not 1 <= value <= 65535:
            raise ValueError("port must be between 1 and 65535")
        return value

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        supported = {"resnet18", "resnet50", "mobilenet_v2"}
        if value not in supported:
            raise ValueError(f"model_name must be one of: {sorted(supported)}")
        return value

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        if value == "cpu" or value == "cuda":
            return value
        if value.startswith("cuda:") and value.removeprefix("cuda:").isdigit():
            return value
        raise ValueError("device must be 'cpu', 'cuda', or 'cuda:N'")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        normalized = value.upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in allowed:
            raise ValueError(f"log_level must be one of: {sorted(allowed)}")
        return normalized

    @field_validator("confidence_threshold")
    @classmethod
    def validate_confidence_threshold(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        return value

    def is_production(self) -> bool:
        return self.environment == "production"

    def is_development(self) -> bool:
        return self.environment == "development"

    def get_device(self) -> torch.device:
        return torch.device(self.device)

    def get_model_path(self) -> Path:
        if self.model_path:
            return Path(self.model_path)
        return Path("models") / f"{self.model_name}.pth"

    def to_dict(self) -> dict:
        return self.model_dump()


def load_settings() -> Settings:
    """Load and validate settings from the current environment."""
    return Settings()


@lru_cache
def get_settings() -> Settings:
    """Return one cached settings instance per process."""
    return load_settings()


def override_settings(**kwargs) -> Settings:
    """Return settings with selected values overridden."""
    values = load_settings().model_dump()
    values.update(kwargs)
    return Settings(**values)


def get_development_settings() -> Settings:
    return override_settings(
        environment="development",
        log_level="DEBUG",
        reload=True,
        batch_size=1,
        device="cpu",
    )


def get_production_settings() -> Settings:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return override_settings(
        environment="production",
        log_level="INFO",
        reload=False,
        batch_size=4,
        device=device,
        enable_metrics=True,
        enable_rate_limit=True,
    )


def validate_configuration(settings: Settings) -> List[str]:
    """Return cross-field configuration problems not handled by Pydantic."""
    errors: List[str] = []
    if settings.model_path and not settings.get_model_path().is_file():
        errors.append(f"model file does not exist: {settings.model_path}")
    if settings.device.startswith("cuda") and not torch.cuda.is_available():
        errors.append("CUDA was requested but is not available")
    if settings.reload and settings.workers > 1:
        errors.append("reload cannot be used with multiple workers")
    return errors
