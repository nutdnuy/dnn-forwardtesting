from __future__ import annotations

import numpy as np
import pandas as pd


def add_indicators(frame: pd.DataFrame, fast: int = 5, slow: int = 20) -> pd.DataFrame:
    data = frame.copy()
    close = data["close"].astype(float)
    data["sma_fast"] = close.rolling(fast).mean()
    data["sma_slow"] = close.rolling(slow).mean()
    data["ema_fast"] = close.ewm(span=fast, adjust=False).mean()
    data["ema_slow"] = close.ewm(span=slow, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    data["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data["macd"] = ema12 - ema26
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    return data
