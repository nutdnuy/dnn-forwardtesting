from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .backtesting import BacktestResult, run_backtest
from .data import validate_ohlcv
from .forecasting import LastValueForecaster, forecast_ohlc
from .indicators import add_indicators

SignalFunction = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class StrategySpec:
    name: str
    kind: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    signal_function: SignalFunction | None = field(default=None, compare=False, repr=False)

    def signal(self, frame: pd.DataFrame) -> pd.Series:
        if self.signal_function is not None:
            return self.signal_function(frame)
        indicator_params = {k: v for k, v in self.params.items() if k in {"fast", "slow"}}
        data = add_indicators(frame, **indicator_params)
        kind = self.kind or self.name
        if kind == "sma_cross":
            return (data["sma_fast"] > data["sma_slow"]).fillna(False).astype(int)
        if kind == "ema_cross":
            return (data["ema_fast"] > data["ema_slow"]).fillna(False).astype(int)
        if kind == "rsi_reversion":
            entry = float(self.params.get("entry", 30))
            exit_level = float(self.params.get("exit", 55))
            if not 0 <= entry < exit_level <= 100:
                raise ValueError("RSI entry/exit must satisfy 0 <= entry < exit <= 100")
            in_market = False
            positions: list[int] = []
            for value in data["rsi_14"]:
                if pd.notna(value):
                    if not in_market and value < entry:
                        in_market = True
                    elif in_market and value > exit_level:
                        in_market = False
                positions.append(int(in_market))
            return pd.Series(positions, index=data.index)
        if kind == "macd_trend":
            return (data["macd"] > data["macd_signal"]).fillna(False).astype(int)
        raise ValueError(f"unsupported strategy kind: {kind}")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "params": self.params}


def default_strategies() -> list[StrategySpec]:
    return [
        StrategySpec("sma_cross", "sma_cross"),
        StrategySpec("ema_cross", "ema_cross"),
        StrategySpec("rsi_reversion", "rsi_reversion", {"entry": 30, "exit": 55}),
        StrategySpec("macd_trend", "macd_trend"),
    ]


@dataclass(frozen=True)
class ExperimentConfig:
    lookback: int = 5
    horizon: int = 30
    selection_window: int | None = None
    initial_capital: float = 100.0
    fee_bps: float = 0.0
    slippage_bps: float = 0.0
    objective: str = "sharpe"
    forecast_backend: str = "last_value"
    seed: int = 42
    indicator_warmup: int = 30
    model_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectionResult:
    selected: StrategySpec
    scores: pd.DataFrame
    backtests: dict[str, BacktestResult]


@dataclass(frozen=True)
class ForwardTestingResult:
    forecast: pd.DataFrame
    selected_forward: StrategySpec
    selected_traditional: StrategySpec
    forward_scores: pd.DataFrame
    traditional_scores: pd.DataFrame
    forward_actual: BacktestResult
    traditional_actual: BacktestResult
    artifacts: dict[str, str] = field(default_factory=dict)


def _metric(result: BacktestResult, objective: str) -> float:
    if objective not in result.metrics:
        raise ValueError(f"unsupported objective: {objective}")
    value = float(result.metrics[objective])
    return value if pd.notna(value) else float("-inf")


def select_strategy(
    data: pd.DataFrame,
    strategies: list[StrategySpec],
    objective: str = "sharpe",
    initial_capital: float = 100.0,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    context: pd.DataFrame | None = None,
) -> SelectionResult:
    """Score candidates on the target data while optionally warming indicators from context."""
    if not strategies:
        raise ValueError("at least one strategy is required")
    backtests: dict[str, BacktestResult] = {}
    rows: list[dict[str, Any]] = []
    for strategy in strategies:
        signal_data = data
        if context is not None and not context.empty:
            signal_data = pd.concat([context, data], ignore_index=True)
        signal = strategy.signal(signal_data).tail(len(data)).reset_index(drop=True)
        result = run_backtest(
            data,
            signal,
            initial_capital=initial_capital,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )
        backtests[strategy.name] = result
        rows.append({"strategy": strategy.name, **result.metrics})
    scores = pd.DataFrame(rows)
    selected_name = max(
        strategies,
        key=lambda strategy: (
            _metric(backtests[strategy.name], objective),
            -strategies.index(strategy),
        ),
    ).name
    selected = next(strategy for strategy in strategies if strategy.name == selected_name)
    return SelectionResult(selected=selected, scores=scores, backtests=backtests)


def _target_signal(
    strategy: StrategySpec,
    target: pd.DataFrame,
    context: pd.DataFrame,
) -> pd.Series:
    signal_data = pd.concat([context, target], ignore_index=True)
    return strategy.signal(signal_data).tail(len(target)).reset_index(drop=True)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def _write_artifacts(
    run_dir: Path,
    config: ExperimentConfig,
    forecast: pd.DataFrame,
    forward: SelectionResult,
    traditional: SelectionResult,
    actual_forward: BacktestResult,
    actual_traditional: BacktestResult,
) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    config_path = run_dir / "config.json"
    _write_json(config_path, asdict(config))
    paths["config"] = str(config_path)
    forecast_path = run_dir / "forecast.csv"
    forecast.to_csv(forecast_path, index=False)
    paths["forecast"] = str(forecast_path)
    score_tables = (("forward_scores", forward.scores), ("traditional_scores", traditional.scores))
    for name, table in score_tables:
        path = run_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = str(path)
    selected_path = run_dir / "selected_strategy.json"
    _write_json(
        selected_path,
        {"forward": forward.selected.to_dict(), "traditional": traditional.selected.to_dict()},
    )
    paths["selected_strategy"] = str(selected_path)
    actual_results = (
        ("forward_actual", actual_forward),
        ("traditional_actual", actual_traditional),
    )
    for name, result in actual_results:
        path = run_dir / f"{name}_equity.csv"
        result.equity.to_csv(path, index=False)
        paths[f"{name}_equity"] = str(path)
        trades_path = run_dir / f"{name}_trades.csv"
        result.trades.to_csv(trades_path, index=False)
        paths[f"{name}_trades"] = str(trades_path)
    metrics_path = run_dir / "metrics.json"
    _write_json(
        metrics_path,
        {
            "forward_actual": actual_forward.metrics,
            "traditional_actual": actual_traditional.metrics,
        },
    )
    paths["metrics"] = str(metrics_path)
    audit_path = run_dir / "audit.json"
    _write_json(audit_path, {"ok": True, "errors": [], "warnings": [
        "Results are historical research artifacts; no live orders are executed."
    ]})
    paths["audit"] = str(audit_path)
    return paths


def run_forwardtesting(
    data: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    config: ExperimentConfig | None = None,
    strategies: list[StrategySpec] | None = None,
    run_dir: str | Path | None = None,
) -> ForwardTestingResult:
    """Run isolated forecast-selection and traditional-selection experiments."""
    cfg = config or ExperimentConfig()
    frame = validate_ohlcv(data, min_rows=cfg.lookback + cfg.horizon + 2)
    cutoff_ts = pd.Timestamp(cutoff)
    history = frame.loc[frame["date"] <= cutoff_ts].reset_index(drop=True)
    actual = frame.loc[frame["date"] > cutoff_ts].head(cfg.horizon).reset_index(drop=True)
    window = cfg.selection_window or cfg.horizon
    if cfg.indicator_warmup < 0:
        raise ValueError("indicator_warmup must be non-negative")
    if len(history) < cfg.lookback + 2 or len(actual) < cfg.horizon:
        raise ValueError("data must contain enough history and a complete actual horizon")
    if len(history) < window + cfg.indicator_warmup:
        raise ValueError(
            "data must contain enough pre-selection history for indicator warm-up; "
            "reduce indicator_warmup or provide more history"
        )
    if cfg.horizon < 1 or cfg.lookback < 1:
        raise ValueError("lookback and horizon must be positive")
    selected_strategies = strategies or default_strategies()
    backend_map = {"last_value": LastValueForecaster}
    if cfg.forecast_backend not in backend_map:
        if cfg.forecast_backend == "torch_mlp":
            from .forecasting import TorchMLPForecaster

            backend_map["torch_mlp"] = TorchMLPForecaster
        else:
            raise ValueError(f"unsupported forecast backend: {cfg.forecast_backend}")
    model_factory = backend_map[cfg.forecast_backend]
    model_kwargs = dict(cfg.model_kwargs)
    if cfg.forecast_backend == "torch_mlp":
        model_kwargs.setdefault("lookback", cfg.lookback)
        model_kwargs.setdefault("seed", cfg.seed)
    forecast = forecast_ohlc(
        history,
        actual["date"],
        cfg.horizon,
        model_factory=model_factory,
        model_kwargs=model_kwargs,
    )
    forward = select_strategy(
        forecast,
        selected_strategies,
        objective=cfg.objective,
        initial_capital=cfg.initial_capital,
        fee_bps=cfg.fee_bps,
        slippage_bps=cfg.slippage_bps,
        context=history.tail(cfg.indicator_warmup),
    )
    traditional_data = history.tail(window).reset_index(drop=True)
    traditional_context = history.iloc[
        len(history) - window - cfg.indicator_warmup : len(history) - window
    ]
    traditional = select_strategy(
        traditional_data,
        selected_strategies,
        objective=cfg.objective,
        initial_capital=cfg.initial_capital,
        fee_bps=cfg.fee_bps,
        slippage_bps=cfg.slippage_bps,
        context=traditional_context,
    )
    forward_actual = run_backtest(
        actual,
        _target_signal(forward.selected, actual, history.tail(cfg.indicator_warmup)),
        initial_capital=cfg.initial_capital,
        fee_bps=cfg.fee_bps,
        slippage_bps=cfg.slippage_bps,
    )
    traditional_actual = run_backtest(
        actual,
        _target_signal(traditional.selected, actual, history.tail(cfg.indicator_warmup)),
        initial_capital=cfg.initial_capital,
        fee_bps=cfg.fee_bps,
        slippage_bps=cfg.slippage_bps,
    )
    artifacts = {}
    if run_dir is not None:
        artifacts = _write_artifacts(
            Path(run_dir), cfg, forecast, forward, traditional, forward_actual, traditional_actual
        )
    return ForwardTestingResult(
        forecast=forecast,
        selected_forward=forward.selected,
        selected_traditional=traditional.selected,
        forward_scores=forward.scores,
        traditional_scores=traditional.scores,
        forward_actual=forward_actual,
        traditional_actual=traditional_actual,
        artifacts=artifacts,
    )
