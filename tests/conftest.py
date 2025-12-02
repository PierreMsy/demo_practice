# tests/conftest.py
from __future__ import annotations

import pytest

from simple_lightning_classifier.config import AppConfig
from simple_lightning_classifier.datamodule import BreastCancerDataModule


@pytest.fixture
def config() -> AppConfig:
    """Return base configuration fixture.

    Uses the normal AppConfig resolution (env, .env, yaml),
    but tests can still override via monkeypatch if needed.
    """
    return AppConfig()


@pytest.fixture
def datamodule(config: AppConfig) -> BreastCancerDataModule:
    """Return data_module for tests to use.

    DataModule fixture that has run setup() and is ready to use.
    """
    data_module = BreastCancerDataModule(config=config)
    data_module.setup()
    return data_module
