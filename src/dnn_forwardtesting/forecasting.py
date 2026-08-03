from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class ForecastingError(ValueError):
    """Raised for invalid forecasting inputs or unavailable optional backends."""


def make_supervised_windows(
    values: np.ndarray | pd.Series,
    lookback: int = 5,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create contiguous one-step or multi-step windows without future leakage."""
    array = np.asarray(values, dtype=float).reshape(-1)
    if lookback < 1 or horizon < 1:
        raise ForecastingError("lookback and horizon must be positive")
    if len(array) < lookback + horizon:
        raise ForecastingError("not enough observations for requested windows")
    starts = range(lookback, len(array) - horizon + 1)
    x = np.stack([array[start - lookback : start] for start in starts])
    y = np.stack([array[start : start + horizon] for start in starts])
    return x, y


@dataclass
class LastValueForecaster:
    """Deterministic random-walk baseline used by the core pipeline."""

    values: np.ndarray | None = None

    def fit(self, values: np.ndarray | pd.Series) -> LastValueForecaster:
        array = np.asarray(values, dtype=float).reshape(-1)
        if len(array) == 0 or not np.isfinite(array).all():
            raise ForecastingError("values must be non-empty and finite")
        self.values = array.copy()
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self.values is None:
            raise ForecastingError("forecaster is not fitted")
        if horizon < 1:
            raise ForecastingError("horizon must be positive")
        return np.repeat(self.values[-1], horizon)


class TorchMLPForecaster:
    """Optional paper-inspired MLP forecaster; imports torch only when used."""

    def __init__(
        self,
        lookback: int = 5,
        hidden_layers: tuple[int, ...] = (50, 25),
        dropout: float = 0.2,
        epochs: int = 100,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        seed: int = 42,
    ) -> None:
        self.lookback = lookback
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self._model: Any = None
        self._torch: Any = None

    def _load_torch(self) -> Any:
        try:
            import torch
        except ImportError as exc:
            raise ForecastingError(
                "TorchMLPForecaster requires the optional 'ml' dependency: "
                "pip install dnn-forwardtesting[ml]"
            ) from exc
        return torch

    def fit(self, values: np.ndarray | pd.Series) -> TorchMLPForecaster:
        torch = self._load_torch()
        array = np.asarray(values, dtype=np.float32).reshape(-1)
        x, y = make_supervised_windows(array, self.lookback, 1)
        torch.manual_seed(self.seed)
        layers: list[Any] = []
        input_size = self.lookback
        for width in self.hidden_layers:
            layers.extend([torch.nn.Linear(input_size, width), torch.nn.ReLU()])
            if self.dropout:
                layers.append(torch.nn.Dropout(self.dropout))
            input_size = width
        layers.append(torch.nn.Linear(input_size, 1))
        self._model = torch.nn.Sequential(*layers)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.L1Loss()
        features = torch.tensor(x, dtype=torch.float32)
        targets = torch.tensor(y[:, 0], dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(features, targets)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self._model.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                loss = loss_fn(self._model(batch_x).squeeze(-1), batch_y)
                loss.backward()
                optimizer.step()
        self._history = array.copy()
        self._torch = torch
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._model is None or self._torch is None:
            raise ForecastingError("forecaster is not fitted")
        if horizon < 1:
            raise ForecastingError("horizon must be positive")
        torch = self._torch
        history = list(self._history[-self.lookback :])
        self._model.eval()
        predictions: list[float] = []
        with torch.no_grad():
            for _ in range(horizon):
                x = torch.tensor([history[-self.lookback :]], dtype=torch.float32)
                value = float(self._model(x).squeeze().item())
                predictions.append(value)
                history.append(value)
        return np.asarray(predictions, dtype=float)


def forecast_ohlc(
    history: pd.DataFrame,
    dates: pd.Series,
    horizon: int,
    model_factory: type | Any = LastValueForecaster,
    model_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Forecast each OHLC column independently and repair impossible OHLC bounds."""
    if len(dates) != horizon:
        raise ForecastingError("dates length must equal horizon")
    kwargs = model_kwargs or {}
    result = {"date": pd.to_datetime(dates).reset_index(drop=True)}
    for column in ("open", "high", "low", "close"):
        model = model_factory(**kwargs)
        model.fit(history[column].to_numpy())
        result[column] = model.predict(horizon)
    frame = pd.DataFrame(result)
    bounds = frame[["open", "close"]].max(axis=1)
    floors = frame[["open", "close"]].min(axis=1)
    frame["high"] = frame["high"].combine(bounds, max)
    frame["low"] = frame["low"].combine(floors, min)
    return frame
