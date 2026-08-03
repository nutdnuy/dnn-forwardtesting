import numpy as np
import pandas as pd
import pytest

from dnn_forwardtesting import (
    DataQualityError,
    LastValueForecaster,
    StrategySpec,
    make_supervised_windows,
    run_backtest,
    select_strategy,
    validate_ohlcv,
)


def make_frame(closes: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    close = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "date": dates,
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
    })


def test_validation_rejects_bad_order_and_ohlc() -> None:
    frame = make_frame([10, 11, 12])
    frame.loc[1, "date"] = frame.loc[0, "date"]
    with pytest.raises(DataQualityError, match="duplicates"):
        validate_ohlcv(frame)
    frame = make_frame([10, 11, 12])
    frame.loc[1, "high"] = 5
    with pytest.raises(DataQualityError, match="high"):
        validate_ohlcv(frame)


def test_windows_use_only_prior_observations() -> None:
    x, y = make_supervised_windows([1, 2, 3, 4, 5, 6], lookback=2, horizon=2)
    np.testing.assert_array_equal(x, [[1, 2], [2, 3], [3, 4]])
    np.testing.assert_array_equal(y, [[3, 4], [4, 5], [5, 6]])


def test_last_value_forecaster_is_deterministic() -> None:
    model = LastValueForecaster().fit([1, 2, 3])
    np.testing.assert_array_equal(model.predict(3), [3, 3, 3])


def test_signal_at_close_applies_on_next_bar() -> None:
    frame = make_frame([100, 110, 120])
    result = run_backtest(frame, pd.Series([1, 0, 0]), initial_capital=100)
    assert result.equity.loc[0, "position"] == 0
    assert result.equity.loc[1, "position"] == 1
    assert result.metrics["total_return"] == pytest.approx(0.10)


def test_turnover_cost_is_charged() -> None:
    frame = make_frame([100, 100, 100])
    result = run_backtest(frame, pd.Series([1, 1, 1]), initial_capital=100, fee_bps=100)
    assert result.metrics["total_cost"] == pytest.approx(1.0)
    assert result.equity.loc[1, "cost"] == pytest.approx(0.01)


def test_trade_direction_uses_full_position_path() -> None:
    frame = make_frame([100, 100, 100, 100, 100, 100])
    result = run_backtest(frame, pd.Series([1, 1, 0, 0, 1, 1]))
    assert result.trades["position_change"].tolist() == [1, -1, 1]


def test_selector_uses_only_supplied_data() -> None:
    frame = make_frame([100, 101, 102, 103, 104, 105])
    strategies = [
        StrategySpec("cash", signal_function=lambda data: pd.Series(0, index=data.index)),
        StrategySpec("long", signal_function=lambda data: pd.Series(1, index=data.index)),
    ]
    selected = select_strategy(frame, strategies, objective="total_return")
    assert selected.selected.name == "long"
