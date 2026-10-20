"""Stock universe grouped by sector. Edit freely.

Only tickers tradable on Alpaca (US-listed) belong here. Foreign listings
such as SK Hynix are not tradable through Alpaca.
"""

UNIVERSE = {
    "Mega-cap Tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],
    "Semiconductors": ["AMD", "AVGO", "MU", "INTC", "QCOM", "TSM", "ASML", "AMAT", "LRCX"],
    "Software & Cloud": ["CRM", "ORCL", "ADBE", "NOW", "SNOW", "PLTR", "CRWV", "ANET", "NET"],
    "Financials": ["JPM", "BAC", "GS", "MS", "V", "MA", "BRK.B", "BLK"],
    "Healthcare": ["JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK", "TMO"],
    "Consumer": ["WMT", "COST", "HD", "NKE", "MCD", "SBUX", "KO", "PEP", "DIS"],
    "Energy": ["XOM", "CVX", "COP", "SLB"],
    "Industrials & Defense": ["LMT", "RTX", "NOC", "BA", "CAT", "GE", "HON"],
    "ETFs": ["SPY", "QQQ", "DIA", "IWM", "XLK", "XLF", "XLE"],
}


def all_tickers():
    return sorted({t for group in UNIVERSE.values() for t in group})