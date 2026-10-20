"""Alpaca market data: daily bars and recent news headlines."""

from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame


def fetch_bars(key: str, secret: str, symbols: list[str], days: int = 730) -> dict[str, pd.DataFrame]:
    """Return {symbol: DataFrame(open, high, low, close, volume)} of daily bars.

    Uses the free IEX feed, so volume reflects IEX trades only (fine for ratios).
    """
    client = StockHistoricalDataClient(key, secret)
    start = datetime.now(timezone.utc) - timedelta(days=days)
    out: dict[str, pd.DataFrame] = {}

    for i in range(0, len(symbols), 40):
        chunk = symbols[i : i + 40]
        req = StockBarsRequest(
            symbol_or_symbols=chunk,
            timeframe=TimeFrame.Day,
            start=start,
            feed=DataFeed.IEX,
        )
        df = client.get_stock_bars(req).df
        if df.empty:
            continue
        for sym, g in df.groupby(level=0):
            g = g.droplevel(0)
            g.index = pd.to_datetime(g.index).tz_localize(None)
            out[sym] = g[["open", "high", "low", "close", "volume"]]
    return out


def fetch_headlines(key: str, secret: str, symbol: str, limit: int = 15, days: int = 7) -> list[str]:
    """Recent news headlines for one symbol (empty list on any failure)."""
    try:
        client = NewsClient(key, secret)
        req = NewsRequest(
            symbols=symbol,
            start=datetime.now(timezone.utc) - timedelta(days=days),
            limit=limit,
        )
        res = client.get_news(req)
        items = getattr(res, "news", None) or res.data.get("news", [])
        return [n.headline for n in items if getattr(n, "headline", None)]
    except Exception:
        return []