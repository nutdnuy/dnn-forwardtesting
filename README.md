# dnn-forwardtesting

`dnn-forwardtesting` is a research-only Python library inspired by the DNN-forwardtesting method in the paper `DNN-Forward Testing: A New Trading Strategy Validation Using Statistical Timeseries Analysis and Deep Neural Networks`.

Reference paper: [arXiv:2210.11532](https://arxiv.org/abs/2210.11532).

It separates two decisions:

1. Select a strategy by evaluating candidates on a forecasted OHLC horizon.
2. Evaluate that selected strategy on the untouched actual out-of-sample horizon.

The package also selects a traditional-backtest strategy on the historical lookback window so the two selection methods can be compared fairly.

Step-by-step tutorial: [`voo-ema20-200-webull-tutorial.ipynb`](output/jupyter-notebook/voo-ema20-200-webull-tutorial.ipynb). It demonstrates read-only Webull VOO daily-bar loading, EMA 20/200 crossover signals, next-bar backtesting, audit checks, and a 50/200 exercise. It runs in offline demo mode by default; live mode supports Webull Thailand (`WEBULL_ENV=th`, `WEBULL_REGION=th`) after token verification in the Webull App.

Verified live run: [`voo-ema20-200-webull-tutorial.executed.ipynb`](output/jupyter-notebook/voo-ema20-200-webull-tutorial.executed.ipynb) contains the executed, read-only Webull Thailand VOO result using 1,200 daily bars from 2021-10-19 through 2026-07-31. It contains rendered metrics and charts only; credentials, access tokens, and the downloaded CSV remain local and gitignored.

## Install

```bash
python -m pip install -e ".[dev]"
# Optional paper-inspired MLP backend:
python -m pip install -e ".[ml]"
```

The default `last_value` forecaster keeps the core package runnable without PyTorch. `torch_mlp` is optional and uses configurable hidden layers, dropout, Adam, and L1 loss.

## Python API

```python
from dnn_forwardtesting import ExperimentConfig, run_forwardtesting

config = ExperimentConfig(
    lookback=5,
    horizon=30,
    initial_capital=100.0,
    fee_bps=5,
    slippage_bps=2,
    objective="sharpe",
    forecast_backend="last_value",  # use torch_mlp after installing [ml]
)

result = run_forwardtesting(
    data=ohlcv,
    cutoff="2021-10-16",
    config=config,
    run_dir="outputs/example-run",
)

print(result.selected_forward.name)
print(result.forward_actual.metrics)
```

## CLI

```bash
dnn-forwardtesting validate --data data/ANF.csv
dnn-forwardtesting run \
  --data data/ANF.csv \
  --cutoff 2021-10-16 \
  --horizon 30 \
  --output outputs/anf-run
```

## Research boundary

- Signals are generated from a bar and applied on the next bar in the long/cash backtest.
- Fees and slippage are explicit basis-point assumptions and are deducted on the row where the shifted next-bar position becomes active. The MVP uses a close-to-close return model; exact open-price execution is a future extension.
- Actual out-of-sample OHLC values are not used to choose the forward-tested strategy.
- Technical indicators are warmed with configurable pre-target context so short selection windows do not discard their historical burn-in period.
- The current MVP uses a single-origin experiment; rolling walk-forward validation is the next major extension.
- Results are historical research artifacts. They are not investment advice, do not guarantee future returns, and the package does not connect to brokers or execute live orders.
- The paper contains ambiguous metric formulas and experimental details; this implementation keeps those settings configurable rather than treating the paper's single result as a production claim.
