# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "output/jupyter-notebook/voo-ema20-200-webull-tutorial.ipynb"
CELL_INDEX = 0


def cell(cell_type: str, source: str) -> dict:
    global CELL_INDEX
    CELL_INDEX += 1
    return {
        "id": f"cell-{CELL_INDEX:03d}",
        "cell_type": cell_type,
        "metadata": {},
        "source": dedent(source).strip("\n").splitlines(keepends=True),
        **({"execution_count": None, "outputs": []} if cell_type == "code" else {}),
    }


def build() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    notebook["cells"] = [
        cell("markdown", r'''
        # Tutorial: VOO EMA 20/200 with Webull API

        สร้างระบบทดลองแบบ **read-only historical research**:

        ```text
        Webull daily bars → EMA 20/EMA 200 → crossover signal → next-bar backtest → audit
        ```

        Notebook นี้ไม่ส่งคำสั่งซื้อขายและไม่ใช่คำแนะนำการลงทุน
        '''),
        cell("markdown", r'''
        ## Learning goals

        เมื่อจบ Notebook นี้จะสามารถ:

        1. ดึงข้อมูล daily bars ของ `VOO` จาก Webull OpenAPI
        2. คำนวณ EMA 20 และ EMA 200
        3. สร้างสัญญาณเมื่อ EMA 20 ตัด EMA 200
        4. Backtest แบบ signal-at-close และใช้ position ใน bar ถัดไป
        5. เปรียบเทียบกับ Buy & Hold และตรวจ leakage เบื้องต้น

        **Prerequisites:** Python 3.11+, พื้นฐาน pandas และ technical analysis เล็กน้อย

        **Outline:** setup → credentials → fetch VOO → EMA → backtest → audit → exercise
        '''),
        cell("markdown", r'''
        ## Step 0 — Install dependencies

        รันจาก root ของ repository:

        ```bash
        pip install -e ".[webull]"
        pip install matplotlib jupyterlab
        ```

        ถ้าต้องการทดสอบโดยไม่ใช้ credential ให้เปิด Notebook ได้ทันที เพราะค่าเริ่มต้นเป็น demo mode
        '''),
        cell("code", r'''
        from pathlib import Path
        import os
        import sys
        import logging
        import json

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from IPython.display import display

        def find_repo_root() -> Path:
            for candidate in (Path.cwd(), *Path.cwd().parents):
                if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
                    return candidate
            raise RuntimeError("Open Jupyter from the dnn-forwardtesting repository or install the package")

        REPO_ROOT = find_repo_root()
        sys.path.insert(0, str(REPO_ROOT / "src"))

        from dnn_forwardtesting import run_backtest, validate_ohlcv

        SYMBOL = "VOO"
        BAR_COUNT = 1200
        FAST_EMA = 20
        SLOW_EMA = 200
        INITIAL_CAPITAL = 10_000.0
        FEE_BPS = 5.0
        SLIPPAGE_BPS = 2.0
        USE_LIVE_WEBULL = os.getenv("WEBULL_TUTORIAL_LIVE", "0") == "1"

        DATA_DIR = REPO_ROOT / "data" / "private"
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        DATA_PATH = DATA_DIR / f"{SYMBOL.lower()}-webull-d.csv"

        print({
            "symbol": SYMBOL,
            "bar_count": BAR_COUNT,
            "mode": "live Webull API" if USE_LIVE_WEBULL else "offline demo",
            "data_path": str(DATA_PATH),
        })
        '''),
        cell("markdown", r'''
        ## Step 1 — Configure Webull credentials safely

        ตั้งค่าใน terminal ก่อนเปิด Jupyter:

        ```bash
        export WEBULL_TUTORIAL_LIVE=1
        # Webull Thailand App Key/App Secret
        export WEBULL_ENV=th
        export WEBULL_REGION=th
        export WEBULL_APP_KEY="your_app_key"
        export WEBULL_APP_SECRET="your_app_secret"
        export WEBULL_TOKEN_DIR="data/private/.webull-token-voo-th"
        jupyter lab
        ```

        สำหรับ App Key ที่สร้างจาก Webull Thailand ให้ใช้ `WEBULL_ENV=th` และ `WEBULL_REGION=th`.
        การสร้าง token ครั้งแรกอาจต้องยืนยันรหัส SMS ใน Webull App ก่อน จึงจะเรียก market-data ได้.
        `uat` และ `prod` คงไว้สำหรับ Webull US credentials ที่ตรงกับ endpoint นั้น ๆ.
        อย่าใส่ key หรือ secret ลงใน Notebook, Git, screenshot หรือ output
        '''),
        cell("code", r'''
        WEBULL_ENV = os.getenv("WEBULL_ENV", "th").lower()
        WEBULL_REGION = os.getenv("WEBULL_REGION", "th" if WEBULL_ENV == "th" else "us").lower()
        WEBULL_APP_KEY = os.getenv("WEBULL_APP_KEY", "")
        WEBULL_APP_SECRET = os.getenv("WEBULL_APP_SECRET", "")
        WEBULL_TOKEN_DIR = Path(os.getenv("WEBULL_TOKEN_DIR", str(DATA_DIR / ".webull-token-voo-th")))

        credential_status = pd.Series({
            "live_mode": USE_LIVE_WEBULL,
            "environment": WEBULL_ENV,
            "region": WEBULL_REGION,
            "app_key_present": bool(WEBULL_APP_KEY),
            "app_secret_present": bool(WEBULL_APP_SECRET),
            "token_dir": str(WEBULL_TOKEN_DIR),
        })
        display(credential_status.to_frame("value"))

        if USE_LIVE_WEBULL and not (WEBULL_APP_KEY and WEBULL_APP_SECRET):
            raise RuntimeError("Live mode requires WEBULL_APP_KEY and WEBULL_APP_SECRET")
        '''),
        cell("markdown", r'''
        ## Step 2 — Fetch daily bars from Webull

        โค้ดนี้ใช้ Webull Python SDK เฉพาะ market-data endpoint:

        - symbol: `VOO`
        - category: `US_ETF`
        - timespan: `D`
        - count: `1200`

        ถ้า API ตอบ `401`, `unauthorized` หรือ `insufficient permission` ให้ตรวจ environment, app permission และ market-data subscription ก่อน
        '''),
        cell("code", r'''
        WEBULL_ENDPOINTS = {
            "th": "api.webull.co.th",
            "uat": "us-openapi-alb.uat.webullbroker.com",
            "prod": "api.webull.com",
        }

        def silence_webull_logging(api_client=None):
            silent_logger = logging.getLogger("webull-notebook-silent")
            silent_logger.handlers.clear()
            silent_logger.addHandler(logging.NullHandler())
            silent_logger.setLevel(logging.CRITICAL + 1)
            silent_logger.propagate = False
            for logger_name in ("webull.core", "webull.data"):
                logger = logging.getLogger(logger_name)
                logger.handlers.clear()
                logger.addHandler(logging.NullHandler())
                logger.setLevel(logging.CRITICAL + 1)
            if api_client is not None:
                if hasattr(api_client, "set_logger"):
                    api_client.set_logger(silent_logger)
                api_client._stream_logger_set = True
                api_client._file_logger_set = True

        def fetch_webull_daily_bars(symbol: str = SYMBOL, count: int = BAR_COUNT) -> pd.DataFrame:
            try:
                from webull.core.client import ApiClient
                from webull.data.data_client import DataClient
                from webull.data.quotes.market_data import MarketData
            except ImportError as exc:
                raise RuntimeError("Install the optional Webull dependency: pip install -e '.[webull]'") from exc

            if WEBULL_ENV not in WEBULL_ENDPOINTS:
                raise ValueError("WEBULL_ENV must be 'th', 'uat', or 'prod'")
            WEBULL_TOKEN_DIR.mkdir(parents=True, exist_ok=True)
            api_client = ApiClient(WEBULL_APP_KEY, WEBULL_APP_SECRET, WEBULL_REGION)
            api_client.add_endpoint(WEBULL_REGION, WEBULL_ENDPOINTS[WEBULL_ENV])
            api_client.set_token_dir(str(WEBULL_TOKEN_DIR))
            silence_webull_logging(api_client)

            # Reuse a locally verified token when it exists. Calling DataClient always
            # requests token creation/verification again, which can trigger unnecessary
            # 2FA prompts in the Webull Thailand flow.
            token_file = WEBULL_TOKEN_DIR / "token.txt"
            token_lines = token_file.read_text(encoding="utf-8").splitlines() if token_file.exists() else []
            if len(token_lines) >= 3 and token_lines[0] and token_lines[2] == "NORMAL":
                api_client.set_token(token_lines[0])
                market_data = MarketData(api_client)
            else:
                market_data = DataClient(api_client).market_data

            response = market_data.get_history_bar(
                symbol.upper(),
                "US_ETF",
                "D",
                count=count,
                real_time_required="true",
            )
            if getattr(response, "status_code", None) != 200:
                raise RuntimeError(
                    f"Webull API returned status {getattr(response, 'status_code', 'unknown')}; "
                    "check credentials and market-data permission"
                )
            payload = response.json()
            if not isinstance(payload, list) or not payload:
                raise ValueError("Webull response is empty or has an unexpected shape")

            raw = pd.DataFrame(payload).rename(columns={"time": "date"})
            required = ["date", "open", "high", "low", "close"]
            missing = [column for column in required if column not in raw.columns]
            if missing:
                raise ValueError(f"Webull response missing columns: {missing}")
            columns = required + (["volume"] if "volume" in raw.columns else [])
            raw = raw.loc[:, columns]
            raw["date"] = pd.to_datetime(raw["date"], errors="coerce", format="mixed")
            for column in columns[1:]:
                raw[column] = pd.to_numeric(raw[column], errors="coerce")
            return validate_ohlcv(raw.sort_values("date").reset_index(drop=True), min_rows=250)
        '''),
        cell("markdown", r'''
        ## Step 3 — Load VOO data

        ค่าเริ่มต้นเป็น **offline demo mode** เพื่อให้ Notebook รันได้โดยไม่ต้องส่งข้อมูลส่วนตัวออกไป
        เมื่อตั้ง `WEBULL_TUTORIAL_LIVE=1` จะเรียก API จริงและบันทึกเฉพาะ OHLCV CSV ลงโฟลเดอร์ที่ถูก ignore
        '''),
        cell("code", r'''
        def make_demo_ohlcv(rows: int = BAR_COUNT, seed: int = 42) -> pd.DataFrame:
            rng = np.random.default_rng(seed)
            dates = pd.date_range("2019-01-02", periods=rows, freq="B")
            returns = 0.00025 + 0.009 * rng.normal(size=rows)
            close = 250 * np.exp(np.cumsum(returns))
            open_ = close * (1 + rng.normal(0, 0.002, rows))
            high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.012, rows))
            low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.012, rows))
            volume = rng.integers(1_000_000, 8_000_000, rows)
            return pd.DataFrame({
                "date": dates,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            })

        if USE_LIVE_WEBULL:
            voo = fetch_webull_daily_bars()
            voo.to_csv(DATA_PATH, index=False)
            print(f"Saved {len(voo):,} Webull bars to {DATA_PATH}")
        else:
            voo = validate_ohlcv(make_demo_ohlcv(), min_rows=250)
            print("Using deterministic demo data. Set WEBULL_TUTORIAL_LIVE=1 for live Webull data.")

        display(voo.head(3))
        display(voo.tail(3))
        print({"rows": len(voo), "start": str(voo["date"].min().date()), "end": str(voo["date"].max().date())})
        '''),
        cell("markdown", r'''
        ## Step 4 — Calculate EMA 20/200 and crossover signals

        กติกาของกลยุทธ์:

        - `signal = 1` เมื่อ EMA 20 อยู่เหนือ EMA 200 → ถือ VOO
        - `signal = 0` เมื่อ EMA 20 อยู่ต่ำกว่า EMA 200 → ถือ cash
        - ไม่ใช้ future data ในการสร้าง signal
        '''),
        cell("code", r'''
        voo = voo.copy()
        voo["ema20"] = voo["close"].ewm(span=FAST_EMA, adjust=False, min_periods=FAST_EMA).mean()
        voo["ema200"] = voo["close"].ewm(span=SLOW_EMA, adjust=False, min_periods=SLOW_EMA).mean()
        voo["signal"] = (voo["ema20"] > voo["ema200"]).fillna(False).astype(int)
        voo["cross_up"] = (voo["ema20"] > voo["ema200"]) & (voo["ema20"].shift(1) <= voo["ema200"].shift(1))
        voo["cross_down"] = (voo["ema20"] < voo["ema200"]) & (voo["ema20"].shift(1) >= voo["ema200"].shift(1))

        display(voo.loc[voo["cross_up"] | voo["cross_down"], ["date", "close", "ema20", "ema200", "cross_up", "cross_down"]].tail(10))
        '''),
        cell("code", r'''
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(voo["date"], voo["close"], label="VOO close", color="#607d8b", linewidth=1)
        ax.plot(voo["date"], voo["ema20"], label="EMA 20", color="#1565c0", linewidth=1.5)
        ax.plot(voo["date"], voo["ema200"], label="EMA 200", color="#ef6c00", linewidth=1.5)
        ax.scatter(voo.loc[voo["cross_up"], "date"], voo.loc[voo["cross_up"], "close"], marker="^", color="green", label="Cross up")
        ax.scatter(voo.loc[voo["cross_down"], "date"], voo.loc[voo["cross_down"], "close"], marker="v", color="red", label="Cross down")
        ax.set_title("VOO EMA 20/200")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        plt.show()
        '''),
        cell("markdown", r'''
        ## Step 5 — Backtest with next-bar execution

        `run_backtest` จะ shift signal ไป 1 bar ก่อนคำนวณ position:

        ```text
        signal ที่เกิดตอน close ของ bar t → position เริ่มใน bar t+1
        ```

        ค่า cost ในตัวอย่างนี้คือ fee 5 bps + slippage 2 bps ต่อ turnover
        '''),
        cell("code", r'''
        backtest = run_backtest(
            frame=voo,
            signal=voo["signal"],
            initial_capital=INITIAL_CAPITAL,
            fee_bps=FEE_BPS,
            slippage_bps=SLIPPAGE_BPS,
        )

        metrics = pd.Series(backtest.metrics, name="EMA20/200")
        display(metrics.to_frame())
        '''),
        cell("code", r'''
        voo["buy_hold_equity"] = INITIAL_CAPITAL * (1 + voo["close"].pct_change().fillna(0)).cumprod()
        comparison = backtest.equity[["date", "equity", "drawdown"]].copy()
        comparison["buy_hold_equity"] = voo["buy_hold_equity"].to_numpy()

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, height_ratios=[3, 1])
        axes[0].plot(comparison["date"], comparison["equity"], label="EMA 20/200 strategy", color="#00a86b")
        axes[0].plot(comparison["date"], comparison["buy_hold_equity"], label="Buy & Hold", color="#607d8b", linestyle="--")
        axes[0].set_ylabel("Equity")
        axes[0].legend()
        axes[0].grid(alpha=0.25)
        axes[1].fill_between(comparison["date"], comparison["drawdown"], 0, color="#e53935", alpha=0.25)
        axes[1].set_ylabel("Drawdown")
        axes[1].grid(alpha=0.25)
        fig.tight_layout()
        plt.show()
        '''),
        cell("markdown", r'''
        ## Step 6 — Inspect trades and run a leakage audit

        ตาราง trade แสดงวันที่ position เปลี่ยน, turnover และ cost
        '''),
        cell("code", r'''
        display(backtest.trades.tail(20))

        expected_position = voo["signal"].shift(1).fillna(0).astype(int).to_numpy()
        actual_position = backtest.equity["position"].to_numpy()
        np.testing.assert_array_equal(actual_position, expected_position)
        assert voo["date"].is_monotonic_increasing
        assert not voo["date"].duplicated().any()
        assert (voo[["open", "high", "low", "close"]] > 0).all().all()

        print("Audit passed: dates are clean and positions use the previous bar's signal.")
        print("Research boundary: this notebook does not place or preview live orders.")
        '''),
        cell("markdown", r'''
        ## Exercise — Change the moving-average pair

        ลองเปลี่ยนจาก EMA 20/200 เป็น EMA 50/200 แล้วตอบคำถาม:

        1. จำนวน trade เปลี่ยนอย่างไร?
        2. Total return และ maximum drawdown เปลี่ยนอย่างไร?
        3. ผลลัพธ์ยังคงเหมือนเดิมหรือไม่เมื่อเพิ่ม fee/slippage?
        '''),
        cell("code", r'''
        # Answer scaffold — complete this cell.
        exercise_fast = 50
        exercise_slow = 200
        voo["exercise_fast"] = voo["close"].ewm(span=exercise_fast, adjust=False, min_periods=exercise_fast).mean()
        voo["exercise_slow"] = voo["close"].ewm(span=exercise_slow, adjust=False, min_periods=exercise_slow).mean()
        voo["exercise_signal"] = (voo["exercise_fast"] > voo["exercise_slow"]).fillna(False).astype(int)

        exercise_result = run_backtest(
            voo,
            voo["exercise_signal"],
            initial_capital=INITIAL_CAPITAL,
            fee_bps=FEE_BPS,
            slippage_bps=SLIPPAGE_BPS,
        )
        display(pd.Series(exercise_result.metrics, name="EMA50/200").to_frame())
        '''),
        cell("markdown", r'''
        ## Pitfalls and extensions

        - **Lookahead:** อย่าใช้ `signal.shift(-1)` หรือใช้ราคาปิดของ bar อนาคตในการตัดสินใจ
        - **Warm-up:** EMA 200 ต้องมีข้อมูลก่อนหน้าอย่างน้อย 200 bars ก่อนตีความ signal
        - **Webull permission:** VOO อาจใช้ไม่ได้กับ UAT credential; ใช้ production market-data permission
        - **Return definition:** daily price bars ไม่ได้แปลว่า total-return series ที่รวมเงินปันผลเสมอไป
        - **Next step:** เพิ่ม rolling walk-forward หลาย cutoff และ model transaction costs ให้ตรงกับ execution venue

        ผลลัพธ์ทั้งหมดเป็น historical research เท่านั้น ไม่รับประกันผลตอบแทนในอนาคต
        '''),
    ]
    notebook["metadata"].setdefault("kernelspec", {"display_name": "Python 3", "language": "python", "name": "python3"})
    notebook["metadata"].setdefault("language_info", {"name": "python", "version": "3.12"})
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {NOTEBOOK}")


if __name__ == "__main__":
    build()
