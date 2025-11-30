from __future__ import annotations

"""
Data module for the breast cancer dataset.

This wraps scikit-learn's breast cancer dataset into a PyTorch
Lightning DataModule.
"""

from typing import Optional

import numpy as np
import torch
from lightning import pytorch as pl
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from simple_lightning_classifier.config import AppConfig


class BreastCancerDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the breast cancer dataset.

    The dataset is loaded from scikit-learn, optionally standardized,
    and split into train and validation sets.
    """

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config

        self._scaler: Optional[StandardScaler] = None
        self._train_dataset: Optional[TensorDataset] = None
        self._val_dataset: Optional[TensorDataset] = None

    @property
    def train_dataset(self) -> TensorDataset:
        return self._train_dataset

    @property
    def val_dataset(self) -> TensorDataset:
        return self._val_dataset

    def prepare_data(self) -> None:
        """Download or generate data if needed (noop here)."""
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Create train/validation datasets.

        Called by the Trainer on ``fit``, ``validate``, etc.
        """
        data = load_breast_cancer()
        X = data.data.astype(np.float32)
        y = data.target.astype(np.float32)  # binary labels 0/1

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=self.config.data.test_size,
            random_state=self.config.data.random_state,
            stratify=y,
        )

        if self.config.data.standardize:
            self._scaler = StandardScaler()
            X_train = self._scaler.fit_transform(X_train)
            X_val = self._scaler.transform(X_val)

        X_train_tensor = torch.from_numpy(X_train)
        X_val_tensor = torch.from_numpy(X_val)
        y_train_tensor = torch.from_numpy(y_train).unsqueeze(1)  # shape (N, 1)
        y_val_tensor = torch.from_numpy(y_val).unsqueeze(1)

        self._train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        self._val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
        )
