from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float | int]


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1).min())


def run_backtest(
    frame: pd.DataFrame,
    signal: pd.Series,
    initial_capital: float = 100.0,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> BacktestResult:
    """Run long/cash close-to-close backtest; a close signal applies next bar.

    Turnover and its fee/slippage deduction are recorded on the row where the
    shifted position becomes active, making the execution timing explicit.
    """
    if len(signal) != len(frame):
        raise ValueError("signal length must equal frame length")
    if initial_capital <= 0 or fee_bps < 0 or slippage_bps < 0:
        raise ValueError("capital must be positive and costs non-negative")
    data = frame.reset_index(drop=True).copy()
    close = pd.to_numeric(data["close"], errors="raise")
    clean_signal = signal.fillna(0).astype(int).clip(0, 1).reset_index(drop=True)
    position = clean_signal.shift(1).fillna(0).astype(int)
    turnover = position.diff().abs().fillna(position.abs())
    cost_rate = (fee_bps + slippage_bps) / 10_000
    asset_return = close.pct_change().fillna(0.0)
    costs = turnover * cost_rate
    strategy_return = position * asset_return - costs
    equity_value = initial_capital * (1 + strategy_return).cumprod()
    drawdown = equity_value / equity_value.cummax() - 1

    equity = data[["date", "close"]].copy()
    equity["signal"] = clean_signal
    equity["position"] = position
    equity["turnover"] = turnover
    equity["cost"] = costs
    equity["asset_return"] = asset_return
    equity["strategy_return"] = strategy_return
    equity["equity"] = equity_value
    equity["drawdown"] = drawdown

    mask = turnover > 0
    trades = equity.loc[mask, ["date", "close", "position", "turnover", "cost"]].copy()
    position_change = position.diff().fillna(position)
    trades["position_change"] = position_change.loc[mask]
    trades = trades.reset_index(drop=True)

    periodic_vol = float(strategy_return.std(ddof=0))
    downside = strategy_return.where(strategy_return < 0, 0.0)
    downside_dev = float(np.sqrt(np.mean(np.square(downside))))
    mean_return = float(strategy_return.mean())
    sharpe = mean_return / periodic_vol * np.sqrt(TRADING_DAYS_PER_YEAR) if periodic_vol else 0.0
    sortino = mean_return / downside_dev * np.sqrt(TRADING_DAYS_PER_YEAR) if downside_dev else 0.0
    total_return = float(equity_value.iloc[-1] / initial_capital - 1)
    periods = len(strategy_return)
    annualized = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / periods) - 1 if periods else 0.0
    mdd = _max_drawdown(equity_value)
    metrics: dict[str, float | int] = {
        "total_return": total_return,
        "annualized_return": float(annualized),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(annualized / abs(mdd)) if mdd < 0 else 0.0,
        "max_drawdown": mdd,
        "trade_count": int(mask.sum()),
        "total_cost": float(costs.sum() * initial_capital),
        "expectancy": float(strategy_return.mean()),
    }
    return BacktestResult(equity=equity, trades=trades, metrics=metrics)
