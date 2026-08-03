"""Build the Webull-powered visual teaching notebook without embedding credentials."""

# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "output/jupyter-notebook/dnn-forwardtesting-webull-visual-lab.ipynb"
CELL_INDEX = 0


def cell(cell_type: str, source: str) -> dict:
    global CELL_INDEX
    CELL_INDEX += 1
    result = {
        "id": f"visual-lab-{CELL_INDEX:03d}",
        "cell_type": cell_type,
        "metadata": {},
        "source": dedent(source).strip("\n").splitlines(keepends=True),
    }
    if cell_type == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


def build() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    notebook["cells"] = [
        cell("markdown", r'''
        # DNN-ForwardTesting Visual Lab — Webull API + paper-inspired charts

        A read-only, visual walkthrough of the method in
        [Letteri et al. (2022)](https://arxiv.org/abs/2210.11532):

        ```text
        Webull OHLCV → understand the data → forecast a held-out horizon →
        score strategy candidates → evaluate the selected rules on the same untouched actual horizon
        ```

        This notebook downloads historical VOO daily bars only. It never places, previews, or submits orders.
        Results are historical research artifacts, not investment advice.
        '''),
        cell("markdown", r'''
        ## What you will learn

        This tutorial is for a reader with basic pandas knowledge who wants to see what
        **DNN-forwardtesting** changes relative to a normal backtest.

        By the end, you can:

        1. Load VOO daily OHLCV from the read-only Webull Market Data API.
        2. Recreate paper-inspired OHLC, volatility, forecast, and strategy-comparison charts.
        3. Keep the prediction horizon separate from the actual out-of-sample (OOS) evaluation period.
        4. Compare forecast-scored and history-scored strategy choices fairly.

        **Outline:** Setup → Webull data → OHLC diagnostics → volatility → split → forecast → selection → actual OOS → audit → exercise.
        '''),
        cell("markdown", r'''
        ## Step 0 — Install and configure

        From the repository root:

        ```bash
        pip install -e ".[webull]"
        pip install matplotlib jupyterlab

        # Optional: paper-inspired MLP forecaster
        pip install -e ".[ml]"
        ```

        Default execution uses a deterministic `last_value` (random-walk) baseline so the notebook is reproducible without PyTorch.
        Set `DNN_VISUAL_LAB_BACKEND=torch_mlp` only after installing `.[ml]`; that label changes the model used, not the research caveats.
        '''),
        cell("code", r'''
        from pathlib import Path
        import logging
        import os
        import sys

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from IPython.display import display

        try:
            get_ipython().run_line_magic("matplotlib", "inline")
        except NameError:
            pass

        def find_repo_root() -> Path:
            for candidate in (Path.cwd(), *Path.cwd().parents):
                if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
                    return candidate
            raise RuntimeError("Open Jupyter from the dnn-forwardtesting repository.")

        REPO_ROOT = find_repo_root()
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from dnn_forwardtesting import ExperimentConfig, StrategySpec, run_backtest, run_forwardtesting, validate_ohlcv

        QS = {
            "background": "#121212", "surface": "#1E1E1E", "text": "#FFFFFF",
            "muted": "#BDBDBD", "primary": "#BB86FC", "secondary": "#03DAC6",
            "error": "#CF6679", "benchmark": "#90A4AE", "warning": "#FFB74D",
        }
        plt.rcParams.update({
            "figure.facecolor": QS["background"], "axes.facecolor": QS["surface"],
            "axes.edgecolor": QS["muted"], "axes.labelcolor": QS["text"],
            "axes.titlecolor": QS["text"], "xtick.color": QS["muted"],
            "ytick.color": QS["muted"], "text.color": QS["text"],
            "legend.facecolor": QS["surface"], "legend.edgecolor": QS["muted"],
        })

        SYMBOL, BAR_COUNT, PAPER_HORIZON = "VOO", 1200, 30
        INITIAL_CAPITAL, FEE_BPS, SLIPPAGE_BPS = 10_000.0, 5.0, 2.0
        USE_LIVE_WEBULL = os.getenv("WEBULL_TUTORIAL_LIVE", "0") == "1"
        FORECAST_BACKEND = os.getenv("DNN_VISUAL_LAB_BACKEND", "last_value")
        DATA_DIR = REPO_ROOT / "data" / "private"
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        DATA_PATH = DATA_DIR / f"{SYMBOL.lower()}-webull-d.csv"
        print({"symbol": SYMBOL, "mode": "live Webull API" if USE_LIVE_WEBULL else "offline demo", "forecast_backend": FORECAST_BACKEND})
        '''),
        cell("markdown", r'''
        ## Step 1 — Connect to Webull Market Data safely

        Set credentials in your terminal, never in the notebook:

        ```bash
        export WEBULL_TUTORIAL_LIVE=1
        export WEBULL_ENV=th
        export WEBULL_REGION=th
        export WEBULL_APP_KEY="your_app_key"
        export WEBULL_APP_SECRET="your_app_secret"
        export WEBULL_TOKEN_DIR="data/private/.webull-token-voo-th"
        jupyter lab
        ```

        The Thailand endpoint and an already verified local token are reused when available, avoiding unnecessary 2FA prompts. The token and CSV remain in gitignored `data/private/`.
        '''),
        cell("code", r'''
        WEBULL_ENV = os.getenv("WEBULL_ENV", "th").lower()
        WEBULL_REGION = os.getenv("WEBULL_REGION", "th" if WEBULL_ENV == "th" else "us").lower()
        WEBULL_APP_KEY = os.getenv("WEBULL_APP_KEY", "")
        WEBULL_APP_SECRET = os.getenv("WEBULL_APP_SECRET", "")
        WEBULL_TOKEN_DIR = Path(os.getenv("WEBULL_TOKEN_DIR", str(DATA_DIR / ".webull-token-voo-th")))
        WEBULL_ENDPOINTS = {"th": "api.webull.co.th", "uat": "us-openapi-alb.uat.webullbroker.com", "prod": "api.webull.com"}

        display(pd.Series({
            "live_mode": USE_LIVE_WEBULL, "environment": WEBULL_ENV, "region": WEBULL_REGION,
            "app_key_present": bool(WEBULL_APP_KEY), "app_secret_present": bool(WEBULL_APP_SECRET),
            "token_directory": str(WEBULL_TOKEN_DIR),
        }).to_frame("value"))
        if USE_LIVE_WEBULL and not (WEBULL_APP_KEY and WEBULL_APP_SECRET):
            raise RuntimeError("Live mode requires WEBULL_APP_KEY and WEBULL_APP_SECRET.")

        def fetch_webull_daily_bars(symbol: str = SYMBOL, count: int = BAR_COUNT) -> pd.DataFrame:
            from webull.core.client import ApiClient
            from webull.data.data_client import DataClient
            from webull.data.quotes.market_data import MarketData

            api_client = ApiClient(WEBULL_APP_KEY, WEBULL_APP_SECRET, WEBULL_REGION)
            api_client.add_endpoint(WEBULL_REGION, WEBULL_ENDPOINTS[WEBULL_ENV])
            WEBULL_TOKEN_DIR.mkdir(parents=True, exist_ok=True)
            api_client.set_token_dir(str(WEBULL_TOKEN_DIR))
            for logger_name in ("webull.core", "webull.data"):
                logger = logging.getLogger(logger_name)
                logger.handlers.clear(); logger.addHandler(logging.NullHandler()); logger.propagate = False
            token_file = WEBULL_TOKEN_DIR / "token.txt"
            token_lines = token_file.read_text(encoding="utf-8").splitlines() if token_file.exists() else []
            market_data = MarketData(api_client) if len(token_lines) >= 3 and token_lines[2] == "NORMAL" else DataClient(api_client).market_data
            if token_lines and len(token_lines) >= 3 and token_lines[2] == "NORMAL":
                api_client.set_token(token_lines[0])
            response = market_data.get_history_bar(symbol, "US_ETF", "D", count=count, real_time_required="true")
            if getattr(response, "status_code", None) != 200:
                raise RuntimeError(f"Webull API returned HTTP {getattr(response, 'status_code', 'unknown')}")
            raw = pd.DataFrame(response.json()).rename(columns={"time": "date"})
            columns = [column for column in ["date", "open", "high", "low", "close", "volume"] if column in raw]
            raw = raw.loc[:, columns]
            raw["date"] = pd.to_datetime(raw["date"], format="mixed", errors="coerce")
            for column in columns[1:]: raw[column] = pd.to_numeric(raw[column], errors="coerce")
            return validate_ohlcv(raw.sort_values("date").reset_index(drop=True), min_rows=300)
        '''),
        cell("markdown", r'''
        ## Step 2 — Load VOO OHLCV and state the evidence window

        Live mode calls Webull once and stores only the downloaded bars locally. Offline mode creates deterministic synthetic OHLCV with the identical data contract, so the lesson remains runnable without credentials.
        '''),
        cell("code", r'''
        def make_demo_ohlcv(rows: int = BAR_COUNT, seed: int = 42) -> pd.DataFrame:
            rng = np.random.default_rng(seed); dates = pd.date_range("2019-01-02", periods=rows, freq="B")
            close = 250 * np.exp(np.cumsum(0.00025 + 0.009 * rng.normal(size=rows)))
            open_ = close * (1 + rng.normal(0, 0.002, rows))
            high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.012, rows))
            low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.012, rows))
            return pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low, "close": close, "volume": rng.integers(1_000_000, 8_000_000, rows)})

        voo = fetch_webull_daily_bars() if USE_LIVE_WEBULL else validate_ohlcv(make_demo_ohlcv(), min_rows=300)
        if USE_LIVE_WEBULL: voo.to_csv(DATA_PATH, index=False)
        evidence = pd.Series({"source": "Webull Market Data API" if USE_LIVE_WEBULL else "deterministic offline demo", "observations": len(voo), "start": voo.date.min().date(), "cutoff": voo.date.max().date(), "frequency": "daily OHLCV", "limitations": "single ETF, historical daily bars, no corporate-action verification"})
        display(evidence.to_frame("value")); display(voo.tail(3))
        '''),
        cell("markdown", r'''
        ## Step 3 — Paper-style OHLC chart

        The paper begins from OHLC data. This compact candlestick view makes the input to technical indicators visible. Green/red bodies denote close above/below open; wicks show high/low. It is a descriptive chart, not a prediction.
        '''),
        cell("code", r'''
        recent = voo.tail(100).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(15, 6))
        for i, row in recent.iterrows():
            color = QS["secondary"] if row.close >= row.open else QS["error"]
            ax.vlines(i, row.low, row.high, color=color, linewidth=0.8)
            ax.add_patch(Rectangle((i - 0.32, min(row.open, row.close)), 0.64, max(abs(row.close - row.open), 0.02), color=color, alpha=0.9))
        ticks = np.linspace(0, len(recent) - 1, 7, dtype=int)
        ax.set_xticks(ticks, recent.date.dt.strftime("%Y-%m-%d").iloc[ticks], rotation=25, ha="right")
        ax.set_title(f"{SYMBOL} daily OHLC candlesticks — latest 100 Webull bars")
        ax.set_ylabel("Price (USD)"); ax.grid(alpha=0.15); fig.tight_layout(); plt.show()
        '''),
        cell("markdown", r'''
        ## Step 4 — Return distribution and OHLC historical-volatility estimators

        The paper examines close-to-close returns and Parkinson (PK), Garman-Klass (GK), Rogers-Satchell (RS), and Yang-Zhang (YZ) historical-volatility estimators. The chart below is a rolling 21-trading-day annualized estimate. These are diagnostics, not forecasts.
        '''),
        cell("code", r'''
        def add_volatility_estimators(frame: pd.DataFrame, window: int = 21) -> pd.DataFrame:
            data = frame.copy(); prev_close = data.close.shift(1)
            log_hl = np.log(data.high / data.low); log_co = np.log(data.close / data.open)
            log_oc = np.log(data.open / prev_close)
            rs = np.log(data.high / data.open) * np.log(data.high / data.close) + np.log(data.low / data.open) * np.log(data.low / data.close)
            scale = np.sqrt(252)
            data["close_vol"] = data.close.pct_change().rolling(window).std() * scale
            data["pk_vol"] = np.sqrt(log_hl.pow(2).rolling(window).mean() / (4 * np.log(2))) * scale
            gk_var = (0.5 * log_hl.pow(2) - (2 * np.log(2) - 1) * log_co.pow(2)).rolling(window).mean()
            data["gk_vol"] = np.sqrt(gk_var.clip(lower=0)) * scale
            data["rs_vol"] = np.sqrt(rs.rolling(window).mean().clip(lower=0)) * scale
            k = 0.34 / (1.34 + (window + 1) / (window - 1))
            yz_var = log_oc.rolling(window).var() + k * log_co.rolling(window).var() + (1 - k) * rs.rolling(window).mean()
            data["yz_vol"] = np.sqrt(yz_var.clip(lower=0)) * scale
            return data

        diagnostics = add_volatility_estimators(voo)
        fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
        returns = diagnostics.close.pct_change().dropna() * 100
        axes[0].hist(returns, bins=45, color=QS["primary"], alpha=0.8)
        axes[0].axvline(returns.mean(), color=QS["secondary"], label=f"mean = {returns.mean():.2f}%")
        axes[0].set_title("Daily close-to-close return distribution"); axes[0].set_xlabel("Return (%)"); axes[0].legend(); axes[0].grid(alpha=0.15)
        for column, label, color in [("close_vol", "Close-to-close", QS["benchmark"]), ("pk_vol", "Parkinson", QS["primary"]), ("gk_vol", "Garman-Klass", QS["secondary"]), ("rs_vol", "Rogers-Satchell", QS["warning"]), ("yz_vol", "Yang-Zhang", QS["error"])]:
            axes[1].plot(diagnostics.date, diagnostics[column] * 100, label=label, linewidth=1.2, color=color)
        axes[1].set_title("Rolling 21-day annualized volatility estimators"); axes[1].set_ylabel("Volatility (%)"); axes[1].legend(ncol=2); axes[1].grid(alpha=0.15)
        fig.tight_layout(); plt.show()
        '''),
        cell("markdown", r'''
        ## Step 5 — The important split: selection data versus untouched actual OOS

        This is the methodological point. Both approaches use the same actual future dates for evaluation. Only the information used to **select** the rule differs. The forecast is trained from data at or before the cutoff; the actual OOS bars are never used to choose the forecast-scored strategy.
        '''),
        cell("code", r'''
        cutoff = voo.date.iloc[-PAPER_HORIZON - 1]
        history = voo.loc[voo.date <= cutoff].reset_index(drop=True)
        actual = voo.loc[voo.date > cutoff].head(PAPER_HORIZON).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(15, 3.8))
        ax.axvspan(history.date.min(), history.date.max(), color=QS["benchmark"], alpha=0.23, label="Training history")
        ax.axvspan(actual.date.min(), actual.date.max(), color=QS["secondary"], alpha=0.25, label="Untouched actual OOS")
        ax.axvline(cutoff, color=QS["warning"], linewidth=2, label=f"Cutoff: {cutoff.date()}")
        ax.plot(voo.date, voo.close, color=QS["text"], linewidth=1)
        ax.set_title(f"Paper-style {PAPER_HORIZON}-trading-day holdout: one cutoff, one actual OOS window")
        ax.set_ylabel("VOO close (USD)"); ax.legend(ncol=3, loc="upper left"); ax.grid(alpha=0.15); fig.tight_layout(); plt.show()
        print({"history_rows": len(history), "actual_oos_rows": len(actual), "cutoff": str(cutoff.date()), "actual_start": str(actual.date.min().date()), "actual_end": str(actual.date.max().date())})
        '''),
        cell("markdown", r'''
        ## Step 6 — Forecast the holdout horizon, then inspect forecast error

        The paper trains separate OHLC forecasts for a 30-day future horizon. This implementation mirrors that structure. In the default baseline, all forecast lines are flat because a last-value model is intentionally simple. That makes its limitations obvious and provides a fair reference before enabling the optional MLP.
        '''),
        cell("code", r'''
        candidates = [
            StrategySpec("EMA 20/200", "ema_cross", {"fast": 20, "slow": 200}),
            StrategySpec("EMA 50/200", "ema_cross", {"fast": 50, "slow": 200}),
            StrategySpec("SMA 50/200", "sma_cross", {"fast": 50, "slow": 200}),
            StrategySpec("MACD trend", "macd_trend"),
            StrategySpec("RSI reversion", "rsi_reversion", {"entry": 30, "exit": 55}),
        ]
        config = ExperimentConfig(lookback=5, horizon=PAPER_HORIZON, selection_window=PAPER_HORIZON, indicator_warmup=200, initial_capital=INITIAL_CAPITAL, fee_bps=FEE_BPS, slippage_bps=SLIPPAGE_BPS, objective="sharpe", forecast_backend=FORECAST_BACKEND)
        result = run_forwardtesting(voo, cutoff=cutoff, config=config, strategies=candidates)
        forecast = result.forecast
        fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True)
        for ax, column in zip(axes.flat, ["open", "high", "low", "close"]):
            ax.plot(actual.date, actual[column], color=QS["secondary"], marker="o", linewidth=1.8, label="Actual OOS")
            ax.plot(forecast.date, forecast[column], color=QS["primary"], marker="o", linestyle="--", linewidth=1.6, label=f"Forecast ({FORECAST_BACKEND})")
            ax.set_title(column.upper()); ax.grid(alpha=0.15)
        axes[0, 0].legend(); fig.suptitle("Paper-inspired 30-day OHLC forecast versus actual OOS", fontsize=15, fontweight="bold"); fig.tight_layout(); plt.show()

        forecast_error = pd.DataFrame({"OHLC": ["Open", "High", "Low", "Close"], "MAE": [np.mean(np.abs(actual[c].to_numpy() - forecast[c].to_numpy())) for c in ["open", "high", "low", "close"]]})
        display(forecast_error.style.format({"MAE": "${:,.2f}"}))
        '''),
        cell("markdown", r'''
        ## Step 7 — Select strategies on two different inputs

        - **Traditional selection:** scores each candidate on the final 30 actual historical bars before the cutoff.
        - **Forward selection:** scores each candidate on the forecasted OHLC horizon.

        This chart is the direct analogue of the paper's baseline comparison. It reports the scoring data, not performance on the actual OOS period.
        '''),
        cell("code", r'''
        def ranked(table: pd.DataFrame, path: str) -> pd.DataFrame:
            output = table[["strategy", "total_return", "sharpe", "max_drawdown", "trade_count"]].copy()
            output["selection_path"] = path
            return output

        selection_scores = pd.concat([ranked(result.forward_scores, "Forecast-scored"), ranked(result.traditional_scores, "Historical-scored")], ignore_index=True)
        display(selection_scores.sort_values(["selection_path", "sharpe"], ascending=[True, False]).style.format({"total_return": "{:.2%}", "sharpe": "{:.2f}", "max_drawdown": "{:.2%}", "trade_count": "{:.0f}"}))

        pivot = selection_scores.pivot(index="strategy", columns="selection_path", values="sharpe").fillna(0)
        fig, ax = plt.subplots(figsize=(12, 5.5)); x = np.arange(len(pivot)); width = 0.36
        ax.bar(x - width / 2, pivot.get("Forecast-scored", 0), width, label="Forecast-scored", color=QS["secondary"])
        ax.bar(x + width / 2, pivot.get("Historical-scored", 0), width, label="Historical-scored", color=QS["primary"])
        ax.axhline(0, color=QS["muted"], linewidth=0.8); ax.set_xticks(x, pivot.index, rotation=18, ha="right")
        ax.set_title("Candidate score comparison — selection-only Sharpe"); ax.set_ylabel("Annualized Sharpe (0% cash rate)"); ax.legend(); ax.grid(axis="y", alpha=0.15); fig.tight_layout(); plt.show()
        print({"forward_selected": result.selected_forward.name, "traditional_selected": result.selected_traditional.name})
        '''),
        cell("markdown", r'''
        ## Step 8 — Evaluate both selected rules on the identical actual OOS bars

        Only now do we inspect realized OOS performance. A Buy & Hold comparator uses the same capital and transaction-cost assumptions. This one 30-day, single-origin holdout can illustrate isolation but cannot establish a general performance advantage.
        '''),
        cell("code", r'''
        buy_hold = run_backtest(actual, pd.Series(1, index=actual.index), INITIAL_CAPITAL, FEE_BPS, SLIPPAGE_BPS)
        oos_metrics = pd.DataFrame([
            {"method": f"Forward-selected: {result.selected_forward.name}", **result.forward_actual.metrics},
            {"method": f"Traditional-selected: {result.selected_traditional.name}", **result.traditional_actual.metrics},
            {"method": "Buy & Hold", **buy_hold.metrics},
        ]).set_index("method")
        display(oos_metrics[["total_return", "sharpe", "sortino", "calmar", "max_drawdown", "trade_count", "total_cost"]].style.format({"total_return": "{:.2%}", "sharpe": "{:.2f}", "sortino": "{:.2f}", "calmar": "{:.2f}", "max_drawdown": "{:.2%}", "trade_count": "{:.0f}", "total_cost": "${:,.2f}"}))

        fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, height_ratios=[3, 1])
        series = [(result.forward_actual, f"Forward-selected: {result.selected_forward.name}", QS["secondary"]), (result.traditional_actual, f"Traditional-selected: {result.selected_traditional.name}", QS["primary"]), (buy_hold, "Buy & Hold", QS["benchmark"])]
        for backtest, label, color in series:
            axes[0].plot(backtest.equity.date, backtest.equity.equity, label=label, color=color, linewidth=2 if "Forward" in label else 1.5, linestyle="--" if label == "Buy & Hold" else "-")
            axes[1].plot(backtest.equity.date, backtest.equity.drawdown * 100, label=label, color=color, linewidth=1.4, linestyle="--" if label == "Buy & Hold" else "-")
        axes[0].set_title("Actual OOS NAV — identical dates, different selection paths"); axes[0].set_ylabel("Portfolio value ($)"); axes[0].legend(); axes[0].grid(alpha=0.15)
        axes[1].set_title("Actual OOS drawdown — do not annualize maximum drawdown"); axes[1].set_ylabel("Drawdown (%)"); axes[1].legend(ncol=3); axes[1].grid(alpha=0.15)
        fig.tight_layout(); plt.show()
        '''),
        cell("markdown", r'''
        ## Step 9 — More OOS diagnostics: daily returns, trade timing, and risk/return comparison

        A final diagnostic view shows what the short OOS result is made of. The risk-return plot is descriptive: no confidence interval is implied with only 30 daily observations.
        '''),
        cell("code", r'''
        oos_returns = pd.DataFrame({"date": actual.date, "Forward-selected": result.forward_actual.equity.strategy_return, "Traditional-selected": result.traditional_actual.equity.strategy_return, "Buy & Hold": buy_hold.equity.strategy_return})
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
        axes[0].bar(oos_returns.date, oos_returns["Forward-selected"] * 100, width=0.8, color=np.where(oos_returns["Forward-selected"] >= 0, QS["secondary"], QS["error"]))
        axes[0].set_title("Forward-selected daily OOS returns"); axes[0].set_ylabel("Return (%)"); axes[0].tick_params(axis="x", rotation=35); axes[0].grid(axis="y", alpha=0.15)
        trades = result.forward_actual.trades
        axes[1].plot(actual.date, actual.close, color=QS["text"], linewidth=1.2, label="VOO close")
        if not trades.empty:
            buys = trades.loc[trades.position_change > 0]; sells = trades.loc[trades.position_change < 0]
            axes[1].scatter(buys.date, buys.close, marker="^", color=QS["secondary"], label="Position on")
            axes[1].scatter(sells.date, sells.close, marker="v", color=QS["error"], label="Position off")
        axes[1].set_title("Forward-selected trade timing on actual OOS"); axes[1].legend(); axes[1].grid(alpha=0.15)
        risk = oos_metrics.assign(volatility=[result.forward_actual.equity.strategy_return.std(ddof=0) * np.sqrt(252), result.traditional_actual.equity.strategy_return.std(ddof=0) * np.sqrt(252), buy_hold.equity.strategy_return.std(ddof=0) * np.sqrt(252)])
        for label, row in risk.iterrows(): axes[2].scatter(row.volatility * 100, row.total_return * 100, s=90, label=label)
        axes[2].axhline(0, color=QS["muted"], linewidth=0.8); axes[2].set_title("OOS total return versus annualized volatility"); axes[2].set_xlabel("Annualized volatility (%)"); axes[2].set_ylabel("Total return (%)"); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.15)
        fig.tight_layout(); plt.show()
        '''),
        cell("markdown", r'''
        ## Step 10 — Audit and interpretation

        **What the charts establish:** the code cleanly separates forecast-scored selection, history-scored selection, and actual OOS evaluation, and it charges the configured next-bar transaction costs.

        **What they do not establish:** a flat last-value forecast is not a DNN; one VOO 30-day holdout cannot demonstrate durable superiority; historical bars do not predict future returns.

        The paper uses 30-day OHLC forecasts and compares candidate indicators such as EMA, MACD, RSI, TEMA, and ADX. This package currently implements the candidates shown here and makes the forecasting backend configurable rather than reproducing the paper's results as a claim.
        '''),
        cell("code", r'''
        expected_position = result.forward_actual.equity.signal.shift(1).fillna(0).astype(int).to_numpy()
        np.testing.assert_array_equal(result.forward_actual.equity.position.to_numpy(), expected_position)
        assert history.date.max() <= cutoff < actual.date.min()
        assert len(actual) == PAPER_HORIZON
        assert not voo.date.duplicated().any() and voo.date.is_monotonic_increasing
        assert (voo[["open", "high", "low", "close"]] > 0).all().all()
        print("Audit passed: clean OHLCV, one cutoff, untouched OOS dates, and next-bar positions.")
        print("Limit: this is a historical single-origin illustration, not a live-trading signal or performance guarantee.")
        '''),
        cell("markdown", r'''
        ## Exercise

        Change exactly one research decision, rerun all charts, then write down what changed and what did not:

        1. Set `PAPER_HORIZON = 60`.
        2. Switch the selection `objective` from `sharpe` to `sortino`.
        3. If PyTorch is installed, set `DNN_VISUAL_LAB_BACKEND=torch_mlp` before starting Jupyter.

        **Answer scaffold:** report the cutoff, selected strategies, forecast MAE by OHLC field, actual OOS dates, total return, maximum drawdown, and whether the conclusion survives a different horizon. Do not choose an answer based only on the best-looking plot.
        '''),
    ]
    notebook["metadata"].update({"language_info": {"name": "python", "version": "3.12"}, "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    NOTEBOOK.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
