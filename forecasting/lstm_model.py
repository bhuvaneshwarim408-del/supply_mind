"""
forecasting/lstm_model.py

PyTorch nn.LSTM, sequence length = 28, rolled forward for multi-day
horizon prediction. MSELoss + early stopping on a validation split.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

SEQ_LEN = 28


class SequenceDataset(Dataset):
    """
    Builds sliding windows of length SEQ_LEN over a single SKU's
    (feature, target) history. X is a window of engineered features,
    y is the next-day target.
    """

    def __init__(self, features: np.ndarray, target: np.ndarray, seq_len: int = SEQ_LEN):
        assert len(features) == len(target)
        self.seq_len = seq_len
        self.features = features.astype(np.float32)
        self.target = target.astype(np.float32)

    def __len__(self):
        return max(0, len(self.features) - self.seq_len)

    def __getitem__(self, idx):
        x = self.features[idx: idx + self.seq_len]
        y = self.target[idx + self.seq_len]
        return torch.from_numpy(x), torch.tensor(y)


class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


@dataclass
class LSTMTrainConfig:
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 40
    patience: int = 5
    seq_len: int = SEQ_LEN


class LSTMForecasterWrapper:
    """Thin trainer/predictor wrapper around LSTMForecaster."""

    def __init__(self, n_features: int, config: LSTMTrainConfig | None = None, seed: int = 42):
        torch.manual_seed(seed)
        self.config = config or LSTMTrainConfig()
        self.n_features = n_features
        self.model = LSTMForecaster(
            n_features=n_features,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        )
        self.feature_mean_: np.ndarray | None = None
        self.feature_std_: np.ndarray | None = None

    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.feature_mean_ = X.mean(axis=0)
            self.feature_std_ = X.std(axis=0) + 1e-8
        return (X - self.feature_mean_) / self.feature_std_

    def fit(self, X_train, y_train, X_val, y_val):
        cfg = self.config
        X_train = self._normalize(X_train, fit=True)
        X_val = self._normalize(X_val, fit=False)

        train_ds = SequenceDataset(X_train, y_train, cfg.seq_len)
        val_ds = SequenceDataset(X_val, y_val, cfg.seq_len)
        if len(train_ds) == 0 or len(val_ds) == 0:
            raise ValueError(
                f"Not enough history for seq_len={cfg.seq_len}: "
                f"train rows={len(X_train)}, val rows={len(X_val)}"
            )

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        loss_fn = nn.MSELoss()

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(cfg.max_epochs):
            self.model.train()
            for xb, yb in train_loader:
                opt.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    pred = self.model(xb)
                    val_losses.append(loss_fn(pred, yb).item())
            val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= cfg.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict_horizon(self, last_window: np.ndarray, horizon_days: int) -> np.ndarray:
        """
        Roll the model forward: predict day t+1 from the last seq_len
        rows, append the prediction (naively repeating the last known
        exogenous feature row, since future exogenous values such as
        price/weather aren't known here) and slide the window.

        last_window: array of shape (seq_len, n_features), already in
        the SAME raw feature space the model was normalized on.
        """
        cfg = self.config
        window = self._normalize(last_window.copy(), fit=False)
        preds = []
        self.model.eval()
        with torch.no_grad():
            for _ in range(horizon_days):
                x = torch.from_numpy(window[None, :, :].astype(np.float32))
                pred = self.model(x).item()
                preds.append(pred)
                # slide window: drop oldest row, append a new row that
                # repeats the last exogenous features but updates the
                # implicit "target lag" position (position 0 by convention
                # is assumed to be the most recent lag_1 feature; callers
                # supplying a different feature layout should override
                # this method).
                new_row = window[-1].copy()
                window = np.vstack([window[1:], new_row[None, :]])
        return np.array(preds)
