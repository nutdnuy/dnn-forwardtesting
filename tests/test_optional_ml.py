import pytest

from dnn_forwardtesting.forecasting import ForecastingError, TorchMLPForecaster


def test_torch_backend_has_clear_optional_dependency_boundary() -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        with pytest.raises(ForecastingError, match="optional 'ml'"):
            TorchMLPForecaster(epochs=1).fit([1, 2, 3, 4, 5, 6, 7])
    else:
        model = TorchMLPForecaster(epochs=1, lookback=2, hidden_layers=(4,), batch_size=2)
        model.fit([1, 2, 3, 4, 5, 6, 7])
        assert len(model.predict(2)) == 2
