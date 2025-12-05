from __future__ import annotations

"""
LightningModule implementing a very small MLP for binary classification.
"""

from typing import Any, Dict

import torch
from lightning import pytorch as pl
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig, TrainingConfig


class BinaryClassifier(pl.LightningModule):
    """
    Simple neural network for binary classification on tabular data.

    Architecture:

    - Linear(input_dim -> hidden_dim)
    - ReLU
    - Dropout
    - Linear(hidden_dim -> 1)  (logit output)

    Loss: BCEWithLogitsLoss
    Metric: accuracy on 0/1 predictions
    """
    def __init__(
            self,
            input_dim: int,
            model_config: ModelConfig,
            training_config: TrainingConfig,
        ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.model_config = model_config
        self.training_config = training_config

        hidden_dim = model_config.hidden_dim
        dropout = model_config.dropout

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits for input features."""
        return self.net(x)

    def _compute_loss_and_acc(self, batch: Any) -> Dict[str, torch.Tensor]:
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        correct = (preds == y).float().sum()
        acc = correct / y.numel()
        return {"loss": loss, "acc": acc}

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        metrics = self._compute_loss_and_acc(batch)
        self.log("train_loss", metrics["loss"], on_step=True, on_epoch=True)
        self.log("train_acc", metrics["acc"], on_step=True, on_epoch=True)
        return metrics["loss"]

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        metrics = self._compute_loss_and_acc(batch)
        self.log("val_loss", metrics["loss"], prog_bar=True, on_epoch=True)
        self.log("val_acc", metrics["acc"], prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.training_config.learning_rate,
        )
        return optimizer
