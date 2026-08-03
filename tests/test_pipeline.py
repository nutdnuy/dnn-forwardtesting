import json

import numpy as np
import pandas as pd

from dnn_forwardtesting import ExperimentConfig, StrategySpec, run_forwardtesting


def make_data(rows: int = 50) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    close = np.linspace(100, 130, rows)
    return pd.DataFrame({
        "date": dates,
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
    })


def test_forward_pipeline_writes_artifacts_and_keeps_selection_boundary(tmp_path) -> None:
    data = make_data()
    strategies = [
        StrategySpec("cash", signal_function=lambda frame: pd.Series(0, index=frame.index)),
        StrategySpec("long", signal_function=lambda frame: pd.Series(1, index=frame.index)),
    ]
    config = ExperimentConfig(horizon=10, selection_window=10, objective="total_return")
    result = run_forwardtesting(data, "2024-02-09", config, strategies, tmp_path)
    assert result.selected_forward.name == "cash" or result.selected_forward.name == "long"
    assert result.selected_traditional.name in {"cash", "long"}
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "forecast.csv").exists()
    assert (tmp_path / "metrics.json").exists()
    audit = json.loads((tmp_path / "audit.json").read_text())
    assert audit["ok"] is True

    # Changing actual OOS values cannot alter either selection score table.
    altered = data.copy()
    altered.loc[altered["date"] > pd.Timestamp("2024-02-09"), "close"] *= 0.5
    altered.loc[altered["date"] > pd.Timestamp("2024-02-09"), "open"] *= 0.5
    altered.loc[altered["date"] > pd.Timestamp("2024-02-09"), "high"] *= 0.5
    altered.loc[altered["date"] > pd.Timestamp("2024-02-09"), "low"] *= 0.5
    altered_result = run_forwardtesting(altered, "2024-02-09", config, strategies)
    assert altered_result.selected_forward.name == result.selected_forward.name
    assert altered_result.selected_traditional.name == result.selected_traditional.name
