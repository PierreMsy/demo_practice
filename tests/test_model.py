# tests/test_data_and_model.py
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import torch

from simple_lightning_classifier import datamodule
from simple_lightning_classifier.config import AppConfig
from simple_lightning_classifier.datamodule import BreastCancerDataModule
from simple_lightning_classifier.model import BinaryClassifier


def test_datamodule_shapes(datamodule: BreastCancerDataModule) -> None:
    """Check that train/val dataloaders yield the expected shapes."""
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    x, y = batch

    assert x.ndim == 2  # (batch, features)
    assert y.ndim == 2  # (batch, 1)
    assert set(torch.unique(y).tolist()).issubset({0.0, 1.0})

def test_model_forward_pass(config: AppConfig) -> None:
    """Check forward pass with random input."""
    input_dim = 30
    model = BinaryClassifier(input_dim=input_dim, config=config)

    batch_size = 4
    x = torch.randn(batch_size, input_dim)
    logits = model(x)

    assert logits.shape == (batch_size, 1)

def test_datamodule_with_mocked_dataset() -> None:
    """Demonstrates mocking: replace load_breast_cancer with fake data."""
    def fake_load_breast_cancer() -> Any:
        # 10 samples, 5 features
        X = np.random.rand(10, 5).astype(np.float32)
        y = np.random.randint(0, 2, size=(10,)).astype(np.float32)
        return type(
            "FakeBunch",
            (),
            {"data": X, "target": y},
        )()

    with patch.object(datamodule, "load_breast_cancer", new=fake_load_breast_cancer):
        cfg = AppConfig()
        dm = BreastCancerDataModule(config=cfg)
        dm.setup()

        train_loader = dm.train_dataloader()
        x, y = next(iter(train_loader))

        assert x.shape[1] == 5 # 5 features
        assert y.shape[1] == 1
