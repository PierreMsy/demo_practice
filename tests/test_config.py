from __future__ import annotations

from pathlib import Path

import pytest

from simple_lightning_classifier.config import AppConfig


def test_default_config_from_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Check AppConfig loads values from a custom YAML file."""
    # create a temporary YAML file
    yaml_content = """
data:
  test_size: 0.3
  random_state: 123
  standardize: false
model:
  hidden_dim: 8
  dropout: 0.2
training:
  max_epochs: 2
  batch_size: 16
  learning_rate: 0.01
  num_workers: 0
  log_every_n_steps: 5
"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml_content, encoding="utf-8")

    # point APP_CONFIG_FILE to this temp file
    monkeypatch.setenv("APP_CONFIG_FILE", str(cfg_path))

    cfg = AppConfig()

    assert cfg.data.test_size == 0.3
    assert cfg.data.random_state == 123
    assert cfg.data.standardize is False
    assert cfg.model.hidden_dim == 8
    assert cfg.training.batch_size == 16
    assert cfg.training.max_epochs == 2


@pytest.mark.parametrize(
    "env_var, attr_path, expected",
    [
        ("APP_TRAINING__MAX_EPOCHS", ("training", "max_epochs"), 7),
        ("APP_MODEL__HIDDEN_DIM", ("model", "hidden_dim"), 64),
        ("APP_TRAINING__BATCH_SIZE", ("training", "batch_size"), 128),
    ],
)
def test_env_overrides_yaml_and_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    env_var: str,
    attr_path: tuple[str, str],
    expected: int,
) -> None:
    """Check that environment variables override YAML/defaults."""
    yaml_content = """
model:
  hidden_dim: 16
training:
  max_epochs: 3
"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml_content, encoding="utf-8")

    monkeypatch.setenv("APP_CONFIG_FILE", str(cfg_path))
    monkeypatch.setenv(env_var, str(expected))

    cfg = AppConfig()

    section, field = attr_path
    value = getattr(getattr(cfg, section), field)
    assert value == expected


def test_dotenv_is_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Check that values from .env default env vars."""
    # create a temp .env file
    env_content = "APP_TRAINING__BATCH_SIZE=99\n"
    env_path = tmp_path / ".env"
    env_path.write_text(env_content, encoding="utf-8")

    monkeypatch.delenv("APP_TRAINING__BATCH_SIZE", raising=False)

    # Change cwd so the .envis read
    monkeypatch.chdir(tmp_path)

    cfg = AppConfig()
    assert cfg.training.batch_size == 99
