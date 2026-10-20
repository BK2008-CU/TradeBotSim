"""Signal engine: technical score + ML score + news-sentiment score -> BUY / HOLD / SELL.

Every score lives in [-1, 1]: positive is bullish, negative is bearish.
This is a decision-support tool for paper trading, not financial advice.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

FEATURES = [
    "ret1", "ret5", "ret10", "rsi", "macd_hist",
    "bb_pos", "vol20", "vol_ratio", "sma_gap20", "sma_gap50",
]

_analyzer = SentimentIntensityAnalyzer()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = d["close"]

    d["sma20"] = c.rolling(20).mean()
    d["sma50"] = c.rolling(50).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi"] = 100 - 100 / (1 + rs)
    d.loc[loss == 0, "rsi"] = 100.0

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    d["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()

    std20 = c.rolling(20).std()
    d["bb_upper"] = d["sma20"] + 2 * std20
    d["bb_lower"] = d["sma20"] - 2 * std20
    width = (d["bb_upper"] - d["bb_lower"]).replace(0, np.nan)
    d["bb_pos"] = (c - d["bb_lower"]) / width

    d["ret1"] = c.pct_change()
    d["ret5"] = c.pct_change(5)
    d["ret10"] = c.pct_change(10)
    d["vol20"] = d["ret1"].rolling(20).std()
    d["vol_ratio"] = d["volume"] / d["volume"].rolling(20).mean()
    d["sma_gap20"] = c / d["sma20"] - 1
    d["sma_gap50"] = c / d["sma50"] - 1
    return d


def technical_score(d: pd.DataFrame) -> tuple[float, dict]:
    last = d.iloc[-1]
    needed = ["sma20", "sma50", "rsi", "macd_hist", "bb_pos"]
    if last[needed].isna().any():
        return 0.0, {}

    c = last["close"]
    trend = 0.5 * (1 if c > last["sma50"] else -1) + 0.5 * (1 if last["sma20"] > last["sma50"] else -1)
    rsi_s = float(np.clip((50 - last["rsi"]) / 25, -1, 1))  # oversold = bullish
    macd_s = float(np.clip(last["macd_hist"] / (c * 0.005), -1, 1))
    bb_s = float(np.clip((0.5 - last["bb_pos"]) * 2, -1, 1))  # near lower band = bullish

    parts = {"Trend": trend, "RSI": rsi_s, "MACD": macd_s, "Bollinger": bb_s}
    return float(np.mean(list(parts.values()))), parts


def ml_score(d: pd.DataFrame, horizon: int = 5) -> tuple[float, float | None]:
    """Gradient-boosting model predicting whether price is higher in `horizon` days.

    Returns (score, holdout_accuracy). The score is shrunk toward 0 when the model
    does not beat a coin flip on the time-ordered holdout set.
    """
    data = d.dropna(subset=FEATURES).copy()
    data["target"] = (data["close"].shift(-horizon) > data["close"]).astype(int)
    train = data.iloc[:-horizon]
    if len(train) < 150:
        return 0.0, None

    X, y = train[FEATURES], train["target"]
    if y.nunique() < 2:
        return 0.0, None

    split = int(len(train) * 0.8)
    model = GradientBoostingClassifier(n_estimators=120, max_depth=2, learning_rate=0.05, random_state=42)
    model.fit(X.iloc[:split], y.iloc[:split])
    acc = float(model.score(X.iloc[split:], y.iloc[split:]))

    model.fit(X, y)
    proba = model.predict_proba(data[FEATURES].iloc[[-1]])[0]
    p_up = float(proba[list(model.classes_).index(1)])

    reliability = float(np.clip((acc - 0.5) / 0.1, 0, 1))
    return (p_up - 0.5) * 2 * reliability, acc


def sentiment_score(headlines: list[str]) -> float:
    if not headlines:
        return 0.0
    vals = [_analyzer.polarity_scores(h)["compound"] for h in headlines]
    return float(np.clip(np.mean(vals) * 2, -1, 1))


def analyze(
    df: pd.DataFrame,
    headlines: list[str],
    weights: dict,
    buy_threshold: float = 0.25,
    sell_threshold: float = -0.25,
) -> dict:
    d = add_indicators(df)
    tech, parts = technical_score(d)
    ml, acc = ml_score(d)
    sent = sentiment_score(headlines)

    w_total = sum(weights.values()) or 1
    total = (weights["technical"] * tech + weights["ml"] * ml + weights["sentiment"] * sent) / w_total

    signal = "BUY" if total >= buy_threshold else "SELL" if total <= sell_threshold else "HOLD"
    return {
        "signal": signal,
        "score": round(total, 3),
        "technical": round(tech, 3),
        "ml": round(ml, 3),
        "sentiment": round(sent, 3),
        "ml_accuracy": None if acc is None else round(acc, 3),
        "components": parts,
        "price": float(df["close"].iloc[-1]),
        "change_1d": float(d["ret1"].iloc[-1]) if pd.notna(d["ret1"].iloc[-1]) else 0.0,
        "indicators": d,
    }