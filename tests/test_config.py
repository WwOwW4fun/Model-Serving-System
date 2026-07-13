"""Tests for application configuration."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import (
    Settings,
    get_development_settings,
    get_production_settings,
    get_settings,
    load_settings,
    override_settings,
    validate_configuration,
)


def test_default_settings_and_helpers():
    settings = Settings()

    assert settings.environment == "development"
    assert settings.port == 8000
    assert settings.is_development()
    assert not settings.is_production()
    assert str(settings.get_device()) == "cpu"
    assert settings.get_model_path() == Path("models/resnet18.pth")
    assert settings.to_dict()["model_name"] == "resnet18"


def test_settings_read_environment(monkeypatch):
    monkeypatch.setenv("PORT", "9000")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("ENVIRONMENT", "STAGING")

    settings = load_settings()

    assert settings.port == 9000
    assert settings.log_level == "DEBUG"
    assert settings.environment == "staging"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("environment", "local"),
        ("port", 0),
        ("metrics_port", 65536),
        ("model_name", "unknown"),
        ("device", "gpu"),
        ("log_level", "verbose"),
        ("confidence_threshold", 1.1),
        ("workers", 0),
    ],
)
def test_invalid_settings_are_rejected(field, value):
    with pytest.raises(ValidationError):
        Settings(**{field: value})


def test_model_path_override(tmp_path):
    model_path = tmp_path / "custom.pt"
    settings = Settings(model_path=str(model_path))

    assert settings.get_model_path() == model_path


def test_cached_and_overridden_settings(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("PORT", "8100")

    first = get_settings()
    second = get_settings()
    overridden = override_settings(port=8200, model_name="resnet50")

    assert first is second
    assert first.port == 8100
    assert overridden.port == 8200
    assert overridden.model_name == "resnet50"
    get_settings.cache_clear()


def test_environment_specific_settings(monkeypatch):
    monkeypatch.setattr("src.config.torch.cuda.is_available", lambda: False)

    development = get_development_settings()
    production = get_production_settings()

    assert development.reload
    assert development.log_level == "DEBUG"
    assert production.is_production()
    assert production.device == "cpu"
    assert production.enable_metrics
    assert production.enable_rate_limit


def test_validate_configuration(tmp_path, monkeypatch):
    missing = tmp_path / "missing.pt"
    settings = Settings(
        model_path=str(missing),
        device="cuda",
        reload=True,
        workers=2,
    )
    monkeypatch.setattr("src.config.torch.cuda.is_available", lambda: False)

    errors = validate_configuration(settings)

    assert len(errors) == 3
    assert "model file does not exist" in errors[0]
    assert "CUDA" in errors[1]
    assert "multiple workers" in errors[2]


def test_validate_configuration_accepts_valid_model_path(tmp_path):
    model_path = tmp_path / "model.pt"
    model_path.touch()

    assert validate_configuration(Settings(model_path=str(model_path))) == []
