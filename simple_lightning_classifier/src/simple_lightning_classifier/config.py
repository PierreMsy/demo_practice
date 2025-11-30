from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Tuple, Type

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    PydanticBaseSettingsSource,
    YamlConfigSettingsSource,
)


class DataConfig(BaseModel):
    test_size: float = Field(0.2, description="Fraction of data used for the test split.")
    random_state: int = Field(42, description="Random seed used for train/test split.")
    standardize: bool = Field(
        True,
        description="Whether to apply standardization (mean 0, std 1) to features.",
    )


class ModelConfig(BaseModel):
    hidden_dim: int = Field(16, ge=1, description="Hidden units in the hidden layer.")
    dropout: float = Field(0.1, ge=0.0, le=1.0, description="Dropout probability.")


class TrainingConfig(BaseModel):
    max_epochs: int = Field(5, ge=1, description="Number of training epochs.")
    batch_size: int = Field(32, ge=1, description="Batch size.")
    learning_rate: float = Field(1e-3, gt=0.0, description="Learning rate.")
    num_workers: int = Field(0, ge=0, description="DataLoader workers.")
    log_every_n_steps: int = Field(10, ge=1, description="Logging frequency.")


class AppConfig(BaseSettings):
    """
    Top-level application configuration.

    Precedence (highest → lowest):
    1. Environment variables (prefix ``APP_``, nested with ``__``)
    2. YAML file (``APP_CONFIG_FILE`` or ``configs/default.yaml``)
    3. Init kwargs
    4. Secrets (unused here)
    """
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Define the order of configuration sources.

        We inject a YAML source between env and init kwargs:

        - env vars
        - yaml file (configs/default.yaml or APP_CONFIG_FILE)
        - init kwargs
        - file secrets
        """
        yaml_path = os.getenv("APP_CONFIG_FILE", "configs/default.yaml")
        yaml_source = YamlConfigSettingsSource(
            settings_cls,
            yaml_file=Path(yaml_path),
        )

        return (
            env_settings,
            yaml_source,
            init_settings,
            file_secret_settings,
        )
