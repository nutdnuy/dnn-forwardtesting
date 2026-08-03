"""Research-only DNN forward-testing primitives."""

from .backtesting import BacktestResult, run_backtest
from .data import DataQualityError, load_ohlcv, validate_ohlcv
from .forecasting import (
    ForecastingError,
    LastValueForecaster,
    TorchMLPForecaster,
    make_supervised_windows,
)
from .pipeline import (
    ExperimentConfig,
    ForwardTestingResult,
    StrategySpec,
    default_strategies,
    run_forwardtesting,
    select_strategy,
)

__all__ = [
    "BacktestResult",
    "DataQualityError",
    "ExperimentConfig",
    "ForecastingError",
    "ForwardTestingResult",
    "LastValueForecaster",
    "StrategySpec",
    "TorchMLPForecaster",
    "default_strategies",
    "load_ohlcv",
    "make_supervised_windows",
    "run_backtest",
    "run_forwardtesting",
    "select_strategy",
    "validate_ohlcv",
]
