#from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Tuple, Type

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    PydanticBaseSettingsSource,
    YamlConfigSettingsSource,
)

# what the domain needs
class DataConfig(BaseModel):
    test_size: float = Field(default=0.2, description="Fraction of data used for the test split.")
    random_state: int = Field(default=42, description="Random seed used for train/test split.")
    standardize: bool = Field(
        default=True,
        description="Whether to apply standardization (mean 0, std 1) to features.",
    )


class ModelConfig(BaseModel):
    hidden_dim: int = Field(default=16, ge=1, description="Hidden units in the hidden layer.")
    dropout: float = Field(default=0.1, ge=0.0, le=1.0, description="Dropout probability.")


class TrainingConfig(BaseModel):
    max_epochs: int = Field(default=5, ge=1, description="Number of training epochs.")
    batch_size: int = Field(default=16, ge=1, description="Batch size.")
    learning_rate: float = Field(default=1e-3, gt=0.0, description="Learning rate.")
    num_workers: int = Field(default=0, ge=0, description="DataLoader workers.")
    log_every_n_steps: int = Field(default=10, ge=1, description="Logging frequency.")
    checkpoint_dir : Path = Field(
        default=Path(__file__).parent.parent.parent / "checkpoints",
        description="Where to store checkpoints",
        )

    @field_validator('checkpoint_dir')
    @classmethod
    def _create_dir(cls, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path


# how we load settings
class AppConfig(BaseSettings):
    """Top-level application configuration.

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
        env_file=".env",
        env_file_encoding="utf-8",
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
        """Define the order of configuration sources.

        - env vars
        - .env file
        - yaml file
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
            dotenv_settings,
            yaml_source,
            init_settings,
            file_secret_settings,
        )
